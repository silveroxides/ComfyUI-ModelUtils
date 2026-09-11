import fnmatch
import gc
import re

import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io
from tqdm import tqdm
from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned

from .artifact_paths import canonical_model_artifact_path
from .device_utils import cleanup_after_operation, estimate_model_size, prepare_for_large_operation
from .lora_alpha import normalize_lora_pair
from .lora_resize import build_lora_layer_map, canonical_lora_key, parse_lora_layers, select_output_dtype, validate_canonical_blocks
from .quantization_guard import inspect_low_bit_input
from .uel_io import AsyncTensorCursor, atomic_uel_writer


LODESTONE_METHODS = ("sum", "mean", "slotnorm", "normmatch", "slotnorm-normmatch")
_ALLOWED_ROLES = {
    "down", "up", "alpha", "diff", "format", "down_suffix", "up_suffix",
    "alpha_suffix",
}


def _matches(key, patterns, glob_mode):
    entries = [entry.strip() for entry in patterns.splitlines() if entry.strip()]
    if glob_mode:
        return any(fnmatch.fnmatchcase(key, entry) for entry in entries)
    return any(re.search(entry, key) is not None for entry in entries)


def factor_delta(down, up, layer):
    if down.ndim == 2 and up.ndim == 2:
        if down.shape[0] != up.shape[1]:
            raise ValueError(f"Lodestone rank mismatch for '{layer}'.")
        return up.float() @ down.float()
    if down.ndim == 4 and up.ndim == 4 and tuple(up.shape[2:]) == (1, 1):
        rank, channels, height, width = down.shape
        if up.shape[1] != rank:
            raise ValueError(f"Lodestone rank mismatch for '{layer}'.")
        delta = up.reshape(up.shape[0], rank).float() @ down.reshape(rank, -1).float()
        return delta.reshape(up.shape[0], channels, height, width)
    raise ValueError(
        f"Lodestone layer '{layer}' requires linear factors or convolutional "
        "factors with a 1x1 up kernel."
    )


def merge_lodestone_deltas(deltas, method):
    if method not in LODESTONE_METHODS:
        raise ValueError(f"Unknown Lodestone method '{method}'.")
    if not deltas:
        raise ValueError("Lodestone requires at least one delta.")
    shape = tuple(deltas[0].shape)
    if any(tuple(delta.shape) != shape for delta in deltas[1:]):
        raise ValueError("Lodestone source deltas must have matching shapes.")
    norms = torch.stack([torch.linalg.vector_norm(delta.float()) for delta in deltas])
    target = torch.median(norms)
    values = deltas
    if method in {"slotnorm", "slotnorm-normmatch"}:
        values = [
            delta * (target / norm.clamp_min(1e-12))
            for delta, norm in zip(deltas, norms)
        ]
    merged = torch.stack(values).sum(dim=0)
    if method in {"mean", "slotnorm"}:
        merged /= len(values)
    elif method in {"normmatch", "slotnorm-normmatch"}:
        merged_norm = torch.linalg.vector_norm(merged.float())
        if merged_norm > 0:
            merged *= target / merged_norm.clamp_min(1e-12)
    return merged


def _to_device(tensor, device):
    if device.type == "cuda":
        return transfer_to_gpu_pinned(tensor, device, torch.float32)
    return tensor.to(device=device, dtype=torch.float32)


def _release_cuda_oom():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _build_units(layer_map, infos, handlers, mismatch_mode, exclude_patterns, discard_patterns, glob_patterns, include_mode=False):
    units = []
    stream_keys = [[] for _ in handlers]
    for core, mapped in layer_map.items():
        if not any(index == 0 for index, _ in mapped):
            continue
        output_key = canonical_lora_key(core, "diff")
        if _matches(output_key, discard_patterns, glob_patterns):
            continue
        matched = _matches(output_key, exclude_patterns, glob_patterns)
        preserve = not matched if include_mode else matched
        sources = []
        for index, block in sorted(mapped):
            roles = infos[index]["pairs"][block]
            unsupported = set(roles) - _ALLOWED_ROLES
            if unsupported:
                raise ValueError(f"Lodestone layer '{block}' has unsupported roles: {', '.join(sorted(unsupported))}.")
            if "down" in roles or "up" in roles:
                if not all(role in roles for role in ("down", "up")):
                    raise ValueError(f"Lodestone layer '{block}' has an incomplete factor pair.")
                keys = [roles["down"], roles["up"]]
                if any(key in infos[index]["low_bit"] for key in keys):
                    raise ValueError(f"Lodestone cannot expand low-bit factors for '{block}'.")
                if "alpha" in roles:
                    keys.append(roles["alpha"])
                sources.append((index, block, "pair", keys))
            elif "diff" in roles:
                key = roles["diff"]
                if key in infos[index]["low_bit"]:
                    raise ValueError(f"Lodestone cannot merge low-bit delta '{block}'.")
                sources.append((index, block, "diff", [key]))
        if preserve:
            sources = [source for source in sources if source[0] == 0]
        present = {source[0] for source in sources}
        if len(present) < len(handlers) and not preserve:
            if mismatch_mode == "error":
                raise ValueError(f"Lodestone layer '{core}' is missing from one or more inputs.")
            if mismatch_mode == "skip":
                sources = [source for source in sources if source[0] == 0]
        if not sources:
            continue
        dtypes = []
        for index, _block, kind, keys in sources:
            stream_keys[index].extend(keys)
            dtypes.append(handlers[index].get_dtype(keys[0] if kind == "diff" else keys[1]))
        units.append({
            "core": core,
            "output_key": output_key,
            "sources": sources,
            "missing": 0 if preserve or mismatch_mode != "zeros" else len(handlers) - len(present),
            "dtypes": dtypes,
        })
    return units, stream_keys


def _process_on_device(unit, loaded, method, save_dtype, device):
    deltas = []
    for index, block, kind, keys in unit["sources"]:
        if kind == "diff":
            delta = _to_device(loaded[(index, keys[0])], device)
        else:
            down = loaded[(index, keys[0])]
            up = loaded[(index, keys[1])]
            alpha = loaded[(index, keys[2])] if len(keys) == 3 else None
            down, up = normalize_lora_pair(down, up, alpha, layer=block)
            delta = factor_delta(_to_device(down, device), _to_device(up, device), block)
        deltas.append(delta)
    if unit["missing"]:
        deltas.extend(torch.zeros_like(deltas[0]) for _ in range(unit["missing"]))
    merged = merge_lodestone_deltas(deltas, method)
    dtype = select_output_dtype(unit["dtypes"], save_dtype, is_1d_diff=merged.ndim == 1)
    return merged.to(dtype).cpu().contiguous()


def _process(unit, loaded, method, save_dtype, requested_device):
    device = torch.device(requested_device)
    try:
        return _process_on_device(unit, loaded, method, save_dtype, device), False
    except Exception as error:
        if device.type != "cuda" or not _is_cuda_oom(error):
            raise
        _release_cuda_oom()
        return _process_on_device(unit, loaded, method, save_dtype, torch.device("cpu")), True


def merge_lodestone_loras(lora_paths, method, device, save_dtype, output_filename,
                          mismatch_mode="skip", exclude_patterns="", discard_patterns="",
                          glob_patterns=False, verbose=True, include_mode=False):
    operation = "Lodestone LoRA Merge"
    prepare_for_large_operation(sum(estimate_model_size(path) for path in lora_paths) * 2.5, torch.device(device))
    handlers = [MemoryEfficientSafeOpen(path, low_memory=True) for path in lora_paths]
    cursors = []
    try:
        infos = []
        for index, handler in enumerate(handlers):
            low_bit = inspect_low_bit_input(handler, f"LoRA {index + 1}", operation)
            pairs, passthrough = parse_lora_layers(handler.keys())
            infos.append({"pairs": pairs, "passthrough": passthrough, "low_bit": low_bit})
        layer_map = build_lora_layer_map(infos, None)
        validate_canonical_blocks(layer_map, operation)
        units, stream_keys = _build_units(
            layer_map, infos, handlers, mismatch_mode, exclude_patterns, discard_patterns, glob_patterns, include_mode
        )
        cursors = [
            AsyncTensorCursor(handler, keys, pin_memory=torch.device(device).type == "cuda")
            for handler, keys in zip(handlers, stream_keys)
        ]
        output_path, _ = canonical_model_artifact_path("loras", output_filename)
        metadata = (handlers[0].metadata() or {}).copy()
        metadata.update({"merge_method": f"lodestone:{method}", "alpha_normalized": "true", "output_representation": "full_direct_difference"})
        fallback_count = 0
        pbar = comfy.utils.ProgressBar(len(units))
        with atomic_uel_writer(output_path, metadata) as writer, torch.no_grad():
            for unit in tqdm(units, desc=f"Lodestone layers ({method})", unit="layers"):
                loaded = {}
                try:
                    for index, _block, _kind, keys in unit["sources"]:
                        for key in keys:
                            loaded[(index, key)] = cursors[index].take(key)
                    output, fallback = _process(unit, loaded, method, save_dtype, device)
                    writer.write_batch([(unit["output_key"], output)])
                    fallback_count += int(fallback)
                    del output
                finally:
                    for index, key in list(loaded):
                        del loaded[(index, key)]
                        cursors[index].release(key)
                    loaded.clear()
                pbar.update(1)
        for cursor in cursors:
            cursor.finish()
        if verbose:
            print(f"[{operation}] CUDA OOM CPU fallbacks: {fallback_count}")
            print(f"[{operation}] Saved {len(units)} full-delta layers to {output_path}")
        return output_path
    finally:
        for cursor in cursors:
            cursor.close()
        for handler in handlers:
            handler.__exit__(None, None, None)
        cleanup_after_operation()


def _controls(default_filename):
    return [
        io.Combo.Input("calc_mode", options=list(LODESTONE_METHODS), default="sum", tooltip="How full LoRA weight changes are combined per layer. sum adds them; mean averages them; slotnorm equalizes each input's Frobenius magnitude before averaging; normmatch adds them and scales the result to a typical input magnitude; slotnorm-normmatch performs both normalization steps."),
        io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="What to do when a layer exists in LoRA 1 but is absent from another input. skip copies LoRA 1's layer unchanged; zeros treats each missing input as a zero update; error stops without replacing an existing output file."),
        io.String.Input("output_filename", default=default_filename, tooltip="Name of the new .safetensors file. It is saved in ComfyUI's loras folder; omit the extension."),
        io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], default="bf16", tooltip="Precision used to save the expanded full-weight difference tensors. FP32 is largest and most precise; FP16 is smallest; BF16 has wider numeric range than FP16."),
        io.Combo.Input("process_device", options=["cuda", "cpu"], default="cuda", tooltip="Where each full layer is expanded and merged. CUDA is faster; if a layer runs out of VRAM, that layer is automatically retried on CPU."),
        io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Optional layer-name patterns to copy from LoRA 1 instead of merging. Enter one pattern per line; useful for protecting specific blocks."),
        io.String.Input("discard_patterns", default="", multiline=True, tooltip="Optional layer-name patterns to leave out of the saved LoRA completely. Enter one pattern per line; discarded layers cannot affect the model."),
        io.Boolean.Input("glob_patterns", default=False, tooltip="Choose filter syntax. Off uses regular expressions; on uses shell-style globs where * matches any text and ? matches one character."),
        io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching layers are merged; nonmatching layers are preserved from LoRA 1."),
    ]


def _execute(kwargs, count):
    paths = [folder_paths.get_full_path_or_raise("loras", kwargs[f"lora_{index}"]) for index in range(1, count + 1)]
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[kwargs["save_dtype"]]
    merge_lodestone_loras(paths, kwargs["calc_mode"], kwargs["process_device"], dtype,
                          kwargs["output_filename"], kwargs["mismatch_mode"],
                          kwargs["exclude_patterns"], kwargs["discard_patterns"], kwargs["glob_patterns"], include_mode=kwargs.get("include_mode", False))
    return canonical_model_artifact_path("loras", kwargs["output_filename"])[1]


class LodestoneLoRATwoMerger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = folder_paths.get_filename_list("loras")
        return io.Schema(node_id="LodestoneLoRATwoMerger", display_name="Lodestone Merge LoRAs (2)",
                         category="ModelUtils/LoRA/Merge/Lodestone",
                         description="Expand two LoRAs into full per-layer weight changes, combine them with the selected Frobenius-norm method, and save a full-difference LoRA.",
                         inputs=[io.Combo.Input("lora_1", options=options, tooltip="First LoRA to merge. It also supplies metadata and is the layer kept when Missing Layer Handling is set to skip."), io.Combo.Input("lora_2", options=options, tooltip="Second LoRA to merge. It contributes equally with LoRA 1 wherever both contain the same layer."), *_controls("lodestone_merged_2_lora")],
                         outputs=[io.AnyType.Output(display_name="output_filename")], is_output_node=True)

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(_execute(kwargs, 2))


class LodestoneLoRAThreeMerger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = folder_paths.get_filename_list("loras")
        return io.Schema(node_id="LodestoneLoRAThreeMerger", display_name="Lodestone Merge LoRAs (3)",
                         category="ModelUtils/LoRA/Merge/Lodestone",
                         description="Expand three LoRAs into full per-layer weight changes, combine them with the selected Frobenius-norm method, and save a full-difference LoRA.",
                         inputs=[io.Combo.Input("lora_1", options=options, tooltip="First LoRA to merge. It supplies metadata, defines the output layer set, and is kept when Missing Layer Handling is skip."), io.Combo.Input("lora_2", options=options, tooltip="Second equal-strength LoRA contribution for every matching layer."), io.Combo.Input("lora_3", options=options, tooltip="Third equal-strength LoRA contribution for every matching layer."), *_controls("lodestone_merged_3_lora")],
                         outputs=[io.AnyType.Output(display_name="output_filename")], is_output_node=True)

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(_execute(kwargs, 3))


class LodestoneLoRAMultiMerger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = folder_paths.get_filename_list("loras")
        optional = ["None", *options]
        inputs = [io.Combo.Input("lora_count", options=[str(index) for index in range(2, 9)], default="2", tooltip="Number of consecutive LoRA selectors to merge, starting at LoRA 1. Every included LoRA has equal influence before the selected normalization method."),
                  io.Combo.Input("lora_1", options=options, tooltip="First LoRA to merge. It supplies metadata, defines the output layer set, and anchors missing-layer handling."), io.Combo.Input("lora_2", options=options, tooltip="Second equal-strength LoRA contribution. LoRA Count always includes this input.")]
        inputs.extend(io.Combo.Input(f"lora_{index}", options=optional, default="None", tooltip="Additional equal-strength LoRA contribution. Select a file for every input included by LoRA Count.") for index in range(3, 9))
        inputs.extend(_controls("lodestone_merged_multi_lora"))
        return io.Schema(node_id="LodestoneLoRAMultiMerger", display_name="Lodestone LoRA Multi-Merge",
                         category="ModelUtils/LoRA/Merge/Lodestone", description="Expand 2-8 equal-strength LoRAs into full per-layer weight changes, combine them with Frobenius-norm-aware math, and save a full-difference LoRA.", inputs=inputs,
                         outputs=[io.AnyType.Output(display_name="output_filename")], is_output_node=True)

    @classmethod
    def execute(cls, **kwargs):
        count = int(kwargs["lora_count"])
        missing = [index for index in range(1, count + 1) if not kwargs[f"lora_{index}"] or kwargs[f"lora_{index}"] == "None"]
        if missing:
            raise ValueError(f"LoRA Count includes unselected input(s): {', '.join(map(str, missing))}.")
        return io.NodeOutput(_execute(kwargs, count))


LODESTONE_MERGER_NODES = [LodestoneLoRATwoMerger, LodestoneLoRAThreeMerger, LodestoneLoRAMultiMerger]
