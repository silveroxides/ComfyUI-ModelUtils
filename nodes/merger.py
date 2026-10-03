import fnmatch
import os
import re
from contextlib import closing
import torch
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from .merger_ops import TWO_MODEL_MODES, THREE_MODEL_MODES, MissingTensorBehavior, MissingTensorError
from .device_utils import (
    estimate_model_size, prepare_for_large_operation, cleanup_after_operation
)

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned
from .quantization_guard import DiffusionQuantization, diffusion_key_map, inspect_low_bit_input, write_preserved_tensor
from .lora_resize import canonical_lora_block_name, is_direct_diff_key, layer_tensor_keys, parse_lora_layers
from .layer_parameters import parameter_input, resolve_layer_parameters
from .uel_io import atomic_uel_writer, stream_work_units
from .lora_alpha import lora_alpha_scale
from .artifact_paths import canonical_model_artifact_path


def load_documentation_from_file(filename):
    """Loads documentation from a markdown file in the ../docs/ directory."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    docs_path = os.path.join(current_dir, '..', 'docs', filename)
    try:
        with open(docs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"# Documentation File Not Found\n\nPlease ensure `{filename}` exists in the `docs` directory of the custom node."


def _compile_patterns(pattern_string, glob_mode=False):
    """Compiles whitespace-separated patterns into a list.

    In regex mode (default): returns compiled re.Pattern objects.
      Raises ValueError if any pattern is invalid regex.
    In glob mode: returns plain strings; fnmatch handles matching.
    """
    if not pattern_string or not pattern_string.strip():
        return []

    patterns = []
    for pattern in pattern_string.split():
        if not pattern:
            continue
        if glob_mode:
            patterns.append(pattern)
        else:
            try:
                patterns.append(re.compile(pattern))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    return patterns


def _matches_any_pattern(key, patterns, glob_mode=False):
    """Returns True if key matches any pattern.

    Glob mode: fnmatch substring match; dots are literal, * matches any sequence.
    Regex mode: compiled re.Pattern substring search.
    """
    if glob_mode:
        return any(fnmatch.fnmatch(key, f"*{p}*") for p in patterns)
    for pattern in patterns:
        if pattern.search(key):
            return True
    return False


class MergerLogic:
    """Shared merge execution logic for all merger nodes."""

    @staticmethod
    def execute_merge(model_names, calc_mode, all_modes, recipe_params, model_type):
        calc_mode_class = next((m for m in all_modes if m.name == calc_mode), None)
        if not calc_mode_class:
            raise ValueError(f"Calc mode '{calc_mode}' not found.")
        primary_model_name = model_names.get('model_a')
        if not primary_model_name or primary_model_name == "None":
            raise ValueError("Model A is required to run the merge.")

        # Prepare memory before heavy operation
        total_size_gb = 0
        model_paths = []
        for name in model_names.values():
            if name and name != "None":
                path = folder_paths.get_full_path(model_type, name)
                if path:
                    total_size_gb += estimate_model_size(path)
                    model_paths.append(path)

        if total_size_gb > 0:
            process_device = recipe_params.get('device', 'cpu')
            print(f"[Merger] Preparing memory for {total_size_gb:.2f}GB merge operation...")
            prepare_for_large_operation(total_size_gb * 1.2, torch.device(process_device))

        handlers = {}
        handler_paths = {}
        for name in model_names.values():
            if name and name != "None":
                path = folder_paths.get_full_path(model_type, name)
                if not path:
                    raise FileNotFoundError(f"Model '{name}' not found.")
                handlers[name] = MemoryEfficientSafeOpen(path, low_memory=True)
                handler_paths[name] = path

        primary_handler = handlers[primary_model_name]
        diffusion_quantizers = {}
        diffusion_maps = {}
        if model_type == "diffusion_models":
            try:
                diffusion_quantizers = {
                    name: DiffusionQuantization(handler, handler_paths[name], f"Model '{name}'")
                    for name, handler in handlers.items()
                }
                diffusion_maps = {
                    name: diffusion_key_map(quantizer.data_keys, f"Merger model {name!r}")
                    for name, quantizer in diffusion_quantizers.items()
                }
            except Exception:
                for handler in handlers.values():
                    handler.__exit__(None, None, None)
                raise
            all_keys = tuple(diffusion_maps[primary_model_name])
            metadata = diffusion_quantizers[primary_model_name].output_metadata()
        else:
            all_keys = primary_handler.keys()
            metadata = primary_handler.metadata()

        low_bit_keys_by_name = {}
        if model_type == "loras":
            try:
                for name, handler in handlers.items():
                    low_bit_keys_by_name[name] = inspect_low_bit_input(
                        handler, f"LoRA ({name})", "Generic LoRA Merge"
                    )
            except Exception:
                for handler in handlers.values():
                    handler.__exit__(None, None, None)
                raise

        # Convert mismatch_mode string to enum
        mismatch_mode_str = recipe_params.get('mismatch_mode', 'skip')
        mismatch_mode = MissingTensorBehavior(mismatch_mode_str)
        recipe_params['mismatch_mode'] = mismatch_mode
        alignment_mode = recipe_params.get('alignment_mode', 'pad/crop')

        # Compile filter patterns
        glob_mode = recipe_params.get('glob_patterns', False)
        include_mode = recipe_params.get('include_mode', False)
        exclude_patterns = _compile_patterns(recipe_params.get('exclude_patterns', ''), glob_mode=glob_mode)
        discard_patterns = _compile_patterns(recipe_params.get('discard_patterns', ''), glob_mode=glob_mode)

        # Pre-compute key differences for logging
        primary_keys = set(all_keys)
        for name, handler in handlers.items():
            if name != primary_model_name:
                secondary_keys = set(
                    diffusion_maps[name] if name in diffusion_maps else handler.keys()
                )
                missing = primary_keys - secondary_keys
                extra = secondary_keys - primary_keys
                if missing:
                    print(f"[Merger] {name} is missing {len(missing)} keys present in Model A")
                if extra:
                    print(f"[Merger] {name} has {len(extra)} extra keys not in Model A (ignored)")

        pbar = comfy.utils.ProgressBar(len(all_keys))
        save_dtype = recipe_params.pop('save_dtype')
        save_torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(save_dtype)
        override_dtype = recipe_params.pop('override_dtype', False)
        include_1d_diffs = recipe_params.pop('include_1d_diffs', False)

        def determine_target_dtype(original_dtypes, target_dtype, override):
            if override:
                return target_dtype
            priority_map = {
                torch.float32: 3,
                torch.bfloat16: 2,
                torch.float16: 2,
            }
            highest_dtype = target_dtype
            highest_priority = priority_map.get(highest_dtype, 1)
            for dtype in original_dtypes:
                p = priority_map.get(dtype, 1)
                if p > highest_priority:
                    highest_priority = p
                    highest_dtype = dtype
            return highest_dtype

        recipe_params.update({"handlers": handlers})
        process_device = recipe_params.get('device', 'cpu')
        process_dtype = recipe_params.get('dtype', torch.float32)

        skipped_keys = 0
        excluded_keys = 0
        discarded_keys = 0
        error_keys = []

        output_folder = "loras" if calc_mode == "SVD LoRA Extraction" else model_type
        output_filename = recipe_params.get("output_filename")
        output_path, output_name = canonical_model_artifact_path(output_folder, output_filename)

        alpha_normalization = {}
        alpha_keys = set()
        logical_layer_names = {}
        if model_type == "loras":
            for name, handler in handlers.items():
                pairs, _ = parse_lora_layers(handler.keys())
                mapping = {}
                for block_keys in pairs.values():
                    if all(role in block_keys for role in ("down", "up", "alpha")):
                        up_key = block_keys["up"]
                        alpha_key = block_keys["alpha"]
                        mapping[up_key] = (
                            alpha_key,
                            int(handler.get_shape(block_keys["down"])[0]),
                            block_keys["down"],
                        )
                        alpha_keys.add(alpha_key)
                alpha_normalization[name] = mapping
                for up_key, (_, _, down_key) in mapping.items():
                    if up_key in low_bit_keys_by_name[name] or down_key in low_bit_keys_by_name[name]:
                        raise ValueError(
                            f"Cannot alpha-normalize low-bit LoRA factors for '{up_key}'."
                        )

        if model_type == "loras":
            primary_pairs, _ = parse_lora_layers(primary_handler.keys())
            for block, roles in primary_pairs.items():
                logical_name = canonical_lora_block_name(block)
                for key in layer_tensor_keys(roles).values():
                    logical_layer_names[key] = logical_name
            for key in all_keys:
                logical_layer_names.setdefault(key, key)
        else:
            logical_layer_names = {key: key for key in all_keys}
        try:
            resolved_layer_parameters = resolve_layer_parameters(
                recipe_params.get("layer_parameters"),
                f"merge:{calc_mode}",
                sorted(set(logical_layer_names.values())),
                {name: recipe_params[name] for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta") if name in recipe_params},
                node_name=f"{model_type} {calc_mode} merger",
            )
        except Exception:
            for handler in handlers.values():
                handler.__exit__(None, None, None)
            raise

        work_units = []
        for key in all_keys:
            preserve_low_bit = model_type == "loras" and any(
                key in low_bit_keys for low_bit_keys in low_bit_keys_by_name.values()
            )
            discarded = _matches_any_pattern(key, discard_patterns, glob_mode=glob_mode)
            entries = {}
            if key in alpha_keys:
                discarded = True
            if not preserve_low_bit and not discarded:
                preserve_a = (
                    model_type == "loras"
                    and len(primary_handler.get_shape(key)) == 1
                    and not include_1d_diffs
                ) or (
                    not _matches_any_pattern(key, exclude_patterns, glob_mode=glob_mode)
                    if include_mode else
                    _matches_any_pattern(key, exclude_patterns, glob_mode=glob_mode)
                )
                requested_names = [primary_model_name]
                if not preserve_a:
                    requested_names.extend(
                        model_names.get(f"model_{label.lower()}")
                        for label in calc_mode_class.models_used
                        if label != "A"
                    )
                for name in dict.fromkeys(requested_names):
                    if name in handlers and key in (
                        diffusion_maps[name] if name in diffusion_maps else handlers[name].keys()
                    ):
                        source_key = diffusion_maps[name][key] if name in diffusion_maps else key
                        required = (
                            diffusion_quantizers[name].required_keys(source_key)
                            if name in diffusion_quantizers else [source_key]
                        )
                        entries[name] = list(dict.fromkeys(required))
                        alpha_info = alpha_normalization.get(name, {}).get(key)
                        if alpha_info is not None:
                            entries[name].append(alpha_info[0])
            work_units.append((key, entries))

        output_metadata = (metadata or {}).copy()
        if model_type == "loras":
            output_metadata["alpha_normalized"] = "true"
            output_metadata["alpha_normalization"] = (
                "lora_up := lora_up * (alpha / rank); alpha tensors removed"
            )
        with atomic_uel_writer(output_path, output_metadata) as writer, torch.no_grad(), closing(
            stream_work_units(
                handlers, work_units,
                pin_memory=str(process_device).startswith("cuda"),
            )
        ) as streamed:
            for key, loaded in tqdm(streamed, total=len(work_units), desc="Merging layers", unit="layers"):
                preserve_low_bit = model_type == "loras" and any(
                    key in low_bit_keys for low_bit_keys in low_bit_keys_by_name.values()
                )
                if key in alpha_keys:
                    discarded_keys += 1
                    pbar.update(1)
                    continue
                if preserve_low_bit:
                    write_preserved_tensor(writer, key, primary_handler, force_raw=True)
                    pbar.update(1)
                    continue
                # Check discard patterns first - skip entirely
                if _matches_any_pattern(key, discard_patterns, glob_mode=glob_mode):
                    discarded_keys += 1
                    pbar.update(1)
                    continue

                for name, mapping in alpha_normalization.items():
                    alpha_info = mapping.get(key)
                    if alpha_info is None or (name, key) not in loaded:
                        continue
                    alpha_key, source_rank, _ = alpha_info
                    alpha = loaded[(name, alpha_key)]
                    scale = lora_alpha_scale(alpha, source_rank, layer=key)
                    if scale != 1.0:
                        loaded[(name, key)] = loaded[(name, key)] * scale

                if diffusion_quantizers:
                    for name, quantizer in diffusion_quantizers.items():
                        source_key = diffusion_maps[name].get(key)
                        if source_key is None or (name, source_key) not in loaded:
                            continue
                        if source_key in quantizer.quantized_keys:
                            tensors = {
                                part: loaded[(name, part)]
                                for part in quantizer.required_keys(source_key)
                            }
                            loaded[(name, key)] = quantizer.decode(
                                source_key, tensors, process_dtype, device="cpu"
                            )
                        else:
                            loaded[(name, key)] = loaded[(name, source_key)]

                # Pre-load Model A's tensor with pinned memory for CUDA
                cpu_tensor = loaded[(primary_model_name, key)]

                # Determine original dtypes across all active models for this key
                original_dtypes = [
                    diffusion_quantizers[primary_model_name].logical_dtype(
                        diffusion_maps[primary_model_name][key], process_dtype
                    )
                    if primary_model_name in diffusion_quantizers
                    else primary_handler.get_dtype(key)
                ]
                for name, handler in handlers.items():
                    if name != primary_model_name and key in (
                        diffusion_maps[name] if name in diffusion_maps else handler.keys()
                    ):
                        original_dtypes.append(
                            diffusion_quantizers[name].logical_dtype(
                                diffusion_maps[name][key], process_dtype
                            )
                            if name in diffusion_quantizers
                            else handler.get_dtype(key)
                        )

                target_dtype = determine_target_dtype(original_dtypes, save_torch_dtype, override_dtype)
                preserve_model_a_1d = model_type == "loras" and cpu_tensor.ndim == 1 and not include_1d_diffs
                if preserve_model_a_1d:
                    target_dtype = determine_target_dtype(
                        [primary_handler.get_dtype(key)], save_torch_dtype, override_dtype
                    )
                elif model_type == "loras" and is_direct_diff_key(key) and cpu_tensor.ndim == 1:
                    target_dtype = torch.float32

                if process_device == 'cuda':
                    tensor_a = transfer_to_gpu_pinned(cpu_tensor, process_device, process_dtype)
                else:
                    tensor_a = cpu_tensor.to(device=process_device, dtype=process_dtype)
                del cpu_tensor

                # Check exclude patterns - use Model A only, no merge
                if preserve_model_a_1d or (
                    not _matches_any_pattern(key, exclude_patterns, glob_mode=glob_mode)
                    if include_mode else
                    _matches_any_pattern(key, exclude_patterns, glob_mode=glob_mode)
                ):
                    t = tensor_a.detach().to(target_dtype).cpu()

                    writer.write_batch([(key, t)])
                    pbar.update(1)
                    if not preserve_model_a_1d:
                        excluded_keys += 1
                    del tensor_a
                    continue

                # Pass tensor_a metadata to recipes for zeros mode and fallback
                layer_recipe_params = {
                    **recipe_params,
                    **resolved_layer_parameters.get(logical_layer_names.get(key, key), {}),
                    '_tensor_a': tensor_a,
                    '_tensor_a_shape': tensor_a.shape,
                    '_tensor_a_dtype': tensor_a.dtype,
                    'preloaded_tensors': loaded,
                }

                try:
                    recipe = calc_mode_class.create_recipe(key=key, **layer_recipe_params)
                    result = recipe.merge()
                except MissingTensorError as e:
                    if mismatch_mode == MissingTensorBehavior.ERROR:
                        raise ValueError(f"Layer mismatch error (mismatch_mode='error'): {e}")
                    result = None
                    error_keys.append(key)

                # Handle None result (mismatch occurred with skip mode)
                if result is None:
                    result = tensor_a
                    skipped_keys += 1

                if isinstance(result, dict):
                    outputs = []
                    for r_key, r_tensor in result.items():
                        t = r_tensor.detach().to(target_dtype).cpu()
                        outputs.append((r_key, t))
                else:
                    # Ensure compatibility with Model A's architecture.
                    # If alignment_mode is 'pad/crop', we crop results that were padded.
                    # If alignment_mode is 'interpolate', resizing happened during operators.
                    if alignment_mode == 'pad/crop':
                        target_shape = layer_recipe_params['_tensor_a_shape']
                        if result.shape != target_shape:
                            slices = tuple(slice(0, min(res_s, tgt_s)) for res_s, tgt_s in zip(result.shape, target_shape))
                            result = result[slices]

                    t = result.detach().to(target_dtype).cpu()

                    outputs = [(key, t)]

                writer.write_batch(outputs)
                pbar.update(1)

                # Clean up references to allow GC immediately.
                # Local loop variables must be explicitly deleted to prevent PyTorch from keeping tensors in VRAM.
                recipe.clean()
                del layer_recipe_params['_tensor_a']
                del layer_recipe_params['preloaded_tensors']
                del tensor_a
                del recipe
                del result

        # Log summary
        if excluded_keys > 0:
            print(f"[Merger] Excluded {excluded_keys} keys from merge (kept Model A only)")
        if discarded_keys > 0:
            print(f"[Merger] Discarded {discarded_keys} keys from output")
        if skipped_keys > 0:
            print(f"[Merger] Used Model A's values for {skipped_keys} keys due to mismatches")
        if error_keys:
            print(f"[Merger] Unexpected errors on {len(error_keys)} keys: {error_keys[:5]}{'...' if len(error_keys) > 5 else ''}")

        for handler in handlers.values():
            handler.__exit__(None, None, None)

        output_folder = "loras" if calc_mode == "SVD LoRA Extraction" else model_type
        output_filename = recipe_params.get("output_filename")
        output_path, output_name = canonical_model_artifact_path(output_folder, output_filename)
        if os.path.exists(output_path):
            print(f"[Merger] Output saved to {output_path}")
        # Cleanup after heavy operation
        cleanup_after_operation()

        return output_name


# --- Two-Model Merger Nodes ---

class CheckpointTwoMerger(io.ComfyNode):
    MODEL_TYPE = "checkpoints"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CheckpointTwoMerger",
            display_name="Merge Checkpoints (2 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("checkpoints"), tooltip="Primary checkpoint; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("checkpoints"), tooltip="Second checkpoint contributing to the selected calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES], tooltip="Two-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.99, min=-10.0, max=10.0, step=0.001, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_2_checkpoint", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force the entire model to be saved as the selected save_dtype. If False (default), higher precision dtypes are preserved."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
        }
        model_names = {"model_a": model_a, "model_b": model_b}
        filename = MergerLogic.execute_merge(model_names, calc_mode, TWO_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


class ModelTwoMerger(io.ComfyNode):
    MODEL_TYPE = "diffusion_models"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelTwoMerger",
            display_name="Merge Models (2 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("diffusion_models"), tooltip="Primary diffusion model; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("diffusion_models"), tooltip="Second diffusion model contributing to the selected calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES], tooltip="Two-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.99, min=-10.0, max=10.0, step=0.001, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_2_model", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force the entire model to be saved as the selected save_dtype. If False (default), higher precision dtypes are preserved."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
        }
        model_names = {"model_a": model_a, "model_b": model_b}
        filename = MergerLogic.execute_merge(model_names, calc_mode, TWO_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


class TextEncoderTwoMerger(io.ComfyNode):
    MODEL_TYPE = "text_encoders"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderTwoMerger",
            display_name="Merge Text Encoders (2 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("text_encoders"), tooltip="Primary text encoder; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("text_encoders"), tooltip="Second text encoder contributing to the selected calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES], tooltip="Two-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.99, min=-10.0, max=10.0, step=0.001, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_2_textencoder", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force the entire model to be saved as the selected save_dtype. If False (default), higher precision dtypes are preserved."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
        }
        model_names = {"model_a": model_a, "model_b": model_b}
        filename = MergerLogic.execute_merge(model_names, calc_mode, TWO_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


class LoRATwoMerger(io.ComfyNode):
    MODEL_TYPE = "loras"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRATwoMerger",
            display_name="Merge LoRAs (2 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("loras"), tooltip="Primary LoRA; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("loras"), tooltip="Second LoRA contributing to the selected calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES], tooltip="Two-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.99, min=-10.0, max=10.0, step=0.001, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_2_lora", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force merged non-1D tensors to save_dtype. Enabled 1D direct diffs remain FP32."),
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Merge 1D tensors. When disabled, preserve Model A's 1D tensors unchanged."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool,
                include_1d_diffs: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
            "include_1d_diffs": include_1d_diffs,
        }
        model_names = {"model_a": model_a, "model_b": model_b}
        filename = MergerLogic.execute_merge(model_names, calc_mode, TWO_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


class EmbeddingTwoMerger(io.ComfyNode):
    MODEL_TYPE = "embeddings"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmbeddingTwoMerger",
            display_name="Merge Embeddings (2 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("embeddings"), tooltip="Primary embedding; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("embeddings"), tooltip="Second embedding contributing to the selected calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES], tooltip="Two-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.99, min=-10.0, max=10.0, step=0.001, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_2_embedding", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force the entire model to be saved as the selected save_dtype. If False (default), higher precision dtypes are preserved."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
        }
        model_names = {"model_a": model_a, "model_b": model_b}
        filename = MergerLogic.execute_merge(model_names, calc_mode, TWO_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


# --- Three-Model Merger Nodes ---

class CheckpointThreeMerger(io.ComfyNode):
    MODEL_TYPE = "checkpoints"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CheckpointThreeMerger",
            display_name="Merge Checkpoints (3 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("checkpoints"), tooltip="Primary checkpoint; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("checkpoints"), tooltip="Second checkpoint contributing to the selected calculation mode."),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("checkpoints"), tooltip="Third checkpoint contributing to the selected three-model calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES], tooltip="Three-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_3_checkpoint", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force the entire model to be saved as the selected save_dtype. If False (default), higher precision dtypes are preserved."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
        }
        model_names = {"model_a": model_a, "model_b": model_b, "model_c": model_c}
        filename = MergerLogic.execute_merge(model_names, calc_mode, THREE_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


class ModelThreeMerger(io.ComfyNode):
    MODEL_TYPE = "diffusion_models"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelThreeMerger",
            display_name="Merge Models (3 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("diffusion_models"), tooltip="Primary diffusion model; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("diffusion_models"), tooltip="Second diffusion model contributing to the selected calculation mode."),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("diffusion_models"), tooltip="Third diffusion model contributing to the selected three-model calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES], tooltip="Three-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_3_model", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force the entire model to be saved as the selected save_dtype. If False (default), higher precision dtypes are preserved."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
        }
        model_names = {"model_a": model_a, "model_b": model_b, "model_c": model_c}
        filename = MergerLogic.execute_merge(model_names, calc_mode, THREE_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


class TextEncoderThreeMerger(io.ComfyNode):
    MODEL_TYPE = "text_encoders"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderThreeMerger",
            display_name="Merge Text Encoders (3 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("text_encoders"), tooltip="Primary text encoder; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("text_encoders"), tooltip="Second text encoder contributing to the selected calculation mode."),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("text_encoders"), tooltip="Third text encoder contributing to the selected three-model calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES], tooltip="Three-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.99, min=-10.0, max=10.0, step=0.001, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_3_textencoder", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force the entire model to be saved as the selected save_dtype. If False (default), higher precision dtypes are preserved."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
        }
        model_names = {"model_a": model_a, "model_b": model_b, "model_c": model_c}
        filename = MergerLogic.execute_merge(model_names, calc_mode, THREE_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


class LoRAThreeMerger(io.ComfyNode):
    MODEL_TYPE = "loras"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAThreeMerger",
            display_name="Merge LoRAs (3 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("loras"), tooltip="Primary LoRA; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("loras"), tooltip="Second LoRA contributing to the selected calculation mode."),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("loras"), tooltip="Third LoRA contributing to the selected three-model calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES], tooltip="Three-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_3_lora", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force merged non-1D tensors to save_dtype. Enabled 1D direct diffs remain FP32."),
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Merge 1D tensors. When disabled, preserve Model A's 1D tensors unchanged."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool,
                include_1d_diffs: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
            "include_1d_diffs": include_1d_diffs,
        }
        model_names = {"model_a": model_a, "model_b": model_b, "model_c": model_c}
        filename = MergerLogic.execute_merge(model_names, calc_mode, THREE_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)


class EmbeddingThreeMerger(io.ComfyNode):
    MODEL_TYPE = "embeddings"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmbeddingThreeMerger",
            display_name="Merge Embeddings (3 Models)",
            category="ModelUtils/Merging",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], tooltip="MERGE writes the selected result; DOCUMENTATION ONLY returns the operation reference without loading model files."),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("embeddings"), tooltip="Primary embedding; anchors metadata, tensor shapes, and values preserved by exclusions or skip handling."),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("embeddings"), tooltip="Second embedding contributing to the selected calculation mode."),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("embeddings"), tooltip="Third embedding contributing to the selected three-model calculation mode."),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES], tooltip="Three-model operation to apply per comparable tensor; DOCUMENTATION ONLY shows its formula and coefficient meanings."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Handle missing or incompatible tensors: preserve Model A, substitute zeros where supported, or abort with an error."),
                io.Combo.Input("alignment_mode", options=["pad/crop", "interpolate"], default="pad/crop", tooltip="Resolve compatible shape differences by zero-padding/cropping or by interpolating Model B and Model C to Model A shape."),
                io.Float.Input("alpha", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; use DOCUMENTATION ONLY for its exact role in the selected calculation mode."),
                io.Float.Input("beta", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("gamma", default=0.5, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("delta", default=2.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("epsilon", default=0.01, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Float.Input("zeta", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Mode-specific coefficient; some calculation modes ignore it. See DOCUMENTATION ONLY for the selected formula."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, tooltip="Random seed used only by calculation modes with stochastic behavior."),
                io.String.Input("output_filename", default="merged_3_embedding", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], tooltip="Output tensor dtype; when Override Dtype is disabled, source tensors with higher precision remain at that precision."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], tooltip="Device used for per-tensor merge arithmetic; CUDA out-of-memory retries the affected tensor on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors excluded from merging and preserved from Model A."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Newline-separated regex or glob patterns for tensors omitted entirely from the output."),
                io.Boolean.Input("glob_patterns", default=False,
                                 tooltip="When True, exclude/discard patterns use glob syntax (* = any sequence, dots are literal). "
                                         "When False (default), patterns are Python regex matched as substrings."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer"),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force the entire model to be saved as the selected save_dtype. If False (default), higher precision dtypes are preserved."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching tensors are merged; nonmatching tensors are preserved from Model A."),
                parameter_input("merge"),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alignment_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, epsilon: float, zeta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str,
                glob_patterns: bool, lazy_load: bool, force_clear_cache: bool, override_dtype: bool, include_mode: bool = False, layer_parameters=None) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode, "alignment_mode": alignment_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon, "zeta": zeta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype, "override_dtype": override_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
            "glob_patterns": glob_patterns,
            "include_mode": include_mode,
            "layer_parameters": layer_parameters,
            "lazy_load": lazy_load, "force_clear_cache": force_clear_cache,
        }
        model_names = {"model_a": model_a, "model_b": model_b, "model_c": model_c}
        filename = MergerLogic.execute_merge(model_names, calc_mode, THREE_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)
