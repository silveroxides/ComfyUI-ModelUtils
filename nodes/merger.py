import os
import re
import torch
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from safetensors.torch import save_file
from .merger_ops import TWO_MODEL_MODES, THREE_MODEL_MODES, MissingTensorBehavior, MissingTensorError
from .merger_utils import MemoryEfficientSafeOpen, transfer_to_gpu_pinned
from .device_utils import (
    estimate_model_size, prepare_for_large_operation, cleanup_after_operation
)


def load_documentation_from_file(filename):
    """Loads documentation from a markdown file in the ../docs/ directory."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    docs_path = os.path.join(current_dir, '..', 'docs', filename)
    try:
        with open(docs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"# Documentation File Not Found\n\nPlease ensure `{filename}` exists in the `docs` directory of the custom node."


def _compile_patterns(pattern_string):
    """Compiles whitespace-separated regex patterns into a list of compiled patterns.

    Returns empty list if pattern_string is empty or whitespace-only.
    Raises ValueError if any pattern is invalid regex.
    """
    if not pattern_string or not pattern_string.strip():
        return []

    patterns = []
    for pattern in pattern_string.split():
        try:
            patterns.append(re.compile(pattern))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    return patterns


def _matches_any_pattern(key, patterns):
    """Returns True if key matches any of the compiled regex patterns (substring search)."""
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
        for name in model_names.values():
            if name and name != "None":
                path = folder_paths.get_full_path(model_type, name)
                if not path:
                    raise FileNotFoundError(f"Model '{name}' not found.")
                handlers[name] = MemoryEfficientSafeOpen(path)

        primary_handler = handlers[primary_model_name]
        all_keys = primary_handler.keys()
        metadata = primary_handler.header.get("__metadata__", {})

        # Convert mismatch_mode string to enum
        mismatch_mode_str = recipe_params.get('mismatch_mode', 'skip')
        mismatch_mode = MissingTensorBehavior(mismatch_mode_str)
        recipe_params['mismatch_mode'] = mismatch_mode

        # Compile filter patterns
        exclude_patterns = _compile_patterns(recipe_params.get('exclude_patterns', ''))
        discard_patterns = _compile_patterns(recipe_params.get('discard_patterns', ''))

        # Pre-compute key differences for logging
        primary_keys = set(all_keys)
        for name, handler in handlers.items():
            if name != primary_model_name:
                secondary_keys = set(handler.keys())
                missing = primary_keys - secondary_keys
                extra = secondary_keys - primary_keys
                if missing:
                    print(f"[Merger] {name} is missing {len(missing)} keys present in Model A")
                if extra:
                    print(f"[Merger] {name} has {len(extra)} extra keys not in Model A (ignored)")

        merged_state_dict = {}
        pbar = comfy.utils.ProgressBar(len(all_keys))
        save_dtype = recipe_params.pop('save_dtype')
        save_torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(save_dtype)
        recipe_params.update({"handlers": handlers})
        process_device = recipe_params.get('device', 'cpu')
        process_dtype = recipe_params.get('dtype', torch.float32)

        skipped_keys = 0
        excluded_keys = 0
        discarded_keys = 0
        error_keys = []

        for key in tqdm(all_keys, desc="Merging layers", unit="layers"):
            # Check discard patterns first - skip entirely
            if _matches_any_pattern(key, discard_patterns):
                discarded_keys += 1
                pbar.update(1)
                continue

            # Pre-load Model A's tensor with pinned memory for CUDA
            cpu_tensor = primary_handler.get_tensor(key)
            if process_device == 'cuda':
                tensor_a = transfer_to_gpu_pinned(cpu_tensor, process_device, process_dtype)
            else:
                tensor_a = cpu_tensor.to(device=process_device, dtype=process_dtype)
            del cpu_tensor

            # Check exclude patterns - use Model A only, no merge
            if _matches_any_pattern(key, exclude_patterns):
                merged_state_dict[key] = tensor_a.to(save_torch_dtype).cpu().clone()
                excluded_keys += 1
                pbar.update(1)
                continue

            # Pass tensor_a metadata to recipes for zeros mode and fallback
            recipe_params['_tensor_a'] = tensor_a
            recipe_params['_tensor_a_shape'] = tensor_a.shape
            recipe_params['_tensor_a_dtype'] = tensor_a.dtype

            try:
                recipe = calc_mode_class.create_recipe(key=key, **recipe_params)
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
                for r_key, r_tensor in result.items():
                    merged_state_dict[r_key] = r_tensor.to(save_torch_dtype).cpu().clone()
            else:
                merged_state_dict[key] = result.to(save_torch_dtype).cpu().clone()

            # Clean up tensor_a reference to allow GC
            del recipe_params['_tensor_a']
            pbar.update(1)

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
        # Use [-1] for diffusion_models to get the actual diffusion_models folder, not legacy unet
        output_dir = folder_paths.get_folder_paths(output_folder)[-1]
        os.makedirs(output_dir, exist_ok=True)
        output_filename = recipe_params.get("output_filename")
        output_path = os.path.join(output_dir, f"{output_filename}.safetensors")
        save_file(merged_state_dict, output_path, metadata=metadata)

        # Cleanup after heavy operation
        del merged_state_dict
        cleanup_after_operation()

        return f"{output_filename}.safetensors"


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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("checkpoints")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("checkpoints")),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.99, min=0.0, max=1.0, step=0.001),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_2_checkpoint"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.99, min=0.0, max=1.0, step=0.001),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_2_model"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.99, min=0.0, max=1.0, step=0.001),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_2_textencoder"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.99, min=0.0, max=1.0, step=0.001),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_2_lora"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("embeddings")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("embeddings")),
                io.Combo.Input("calc_mode", options=[m.name for m in TWO_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.99, min=0.0, max=1.0, step=0.001),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_2_embedding"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("checkpoints")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("checkpoints")),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("checkpoints")),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("delta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_3_checkpoint"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("delta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_3_model"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("delta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_3_textencoder"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("delta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_3_lora"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
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
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
                io.Combo.Input("model_a", options=["None"] + folder_paths.get_filename_list("embeddings")),
                io.Combo.Input("model_b", options=["None"] + folder_paths.get_filename_list("embeddings")),
                io.Combo.Input("model_c", options=["None"] + folder_paths.get_filename_list("embeddings")),
                io.Combo.Input("calc_mode", options=[m.name for m in THREE_MODEL_MODES]),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
                io.Float.Input("alpha", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("beta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("gamma", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Float.Input("delta", default=0.5, min=-2.0, max=3.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_3_embedding"),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
                io.Combo.Input("process_device", options=["cuda", "cpu"]),
                io.String.Input("exclude_patterns", default="", multiline=True),
                io.String.Input("discard_patterns", default="", multiline=True),
            ],
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
        )

    @classmethod
    def execute(cls, execution_mode: str, model_a: str, model_b: str, model_c: str,
                calc_mode: str, mismatch_mode: str, alpha: float, beta: float,
                gamma: float, delta: float, seed: int, output_filename: str, save_dtype: str,
                process_device: str, exclude_patterns: str, discard_patterns: str) -> io.NodeOutput:
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "mismatch_mode": mismatch_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32,
            "exclude_patterns": exclude_patterns, "discard_patterns": discard_patterns,
        }
        model_names = {"model_a": model_a, "model_b": model_b, "model_c": model_c}
        filename = MergerLogic.execute_merge(model_names, calc_mode, THREE_MODEL_MODES, recipe_params, cls.MODEL_TYPE)
        return io.NodeOutput(filename, doc)