"""
LoRA Resize - Resize existing LoRAs to different ranks.

Based on kohya_ss resize_lora.py. Merges LoRA weights and re-extracts via SVD.
Supports fixed rank and dynamic methods (sv_ratio, sv_fro, sv_cumulative).
"""
import os
import re
import torch
import torch.linalg as linalg
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned, IncrementalSafetensorsWriter
from .quantization_guard import inspect_low_bit_input, layer_has_low_bit, write_preserved_tensor
from typing import Optional, Dict, Tuple, List



# Reuse rank functions from extraction module
from .lora_extract_svd import (
    _index_sv_ratio,
    _index_sv_cumulative,
    _index_sv_fro,
    _svd_extract_linear,
    _svd_extract_conv,
    _compile_patterns,
    _matches_any_pattern,
    _format_lora_key,
)


MIN_SV = 1e-6

LORA_PAIR_SUFFIXES = (
    (".lora_A.default.weight", ".lora_B.default.weight", ".alpha", "peft"),
    (".lora_down.weight", ".lora_up.weight", ".alpha", "comfy"),
    (".lora_A.weight", ".lora_B.weight", ".alpha", "diffusers"),
    (".lora.down.weight", ".lora.up.weight", ".alpha", "huggingface"),
)


def is_direct_diff_key(key: str) -> bool:
    return key.endswith(".diff") or key.endswith(".diff_b")


def select_output_dtype(
    source_dtypes: List[torch.dtype],
    requested_dtype: torch.dtype,
    force: bool = False,
    is_1d_diff: bool = False,
) -> torch.dtype:
    """Select an output dtype without losing precision-sensitive LoRA deltas."""
    if is_1d_diff:
        return torch.float32
    if not force and torch.float32 in source_dtypes:
        return torch.float32
    floating_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if source_dtypes and not any(dtype in floating_dtypes for dtype in source_dtypes):
        return source_dtypes[0]
    return requested_dtype


# =============================================================================
# LoRA Format Detection - Support various LoRA sources
# =============================================================================

def detect_lora_format(keys: List[str]) -> Dict:
    """
    Detect LoRA format and key patterns from various sources.

    Supports:
    - Kohya/A1111: lora_unet_*, lora_te_*
    - Diffusers: *.lora_A.weight, *.lora_B.weight
    - PEFT: base_model.model.*.lora_A/B
    - ComfyUI native: *.lora_down.weight, *.lora_up.weight
    - HuggingFace: *.lora.down.weight, *.lora.up.weight (dots instead of underscores)
    - Full diff: *.diff, *.diff_b (full weight differences, not low-rank)
    - Flux/SDXL: Various transformer key patterns

    Returns:
        Dict with format info and key patterns
    """
    format_info = {
        "format": "unknown",
        "down_suffix": ".lora_down.weight",
        "up_suffix": ".lora_up.weight",
        "alpha_suffix": ".alpha",
        "has_alpha": False,
        "key_count": 0,
        "is_full_diff": False,  # New flag for full diff format
    }

    # Check for alpha keys
    alpha_keys = [k for k in keys if '.alpha' in k or k.endswith('alpha')]
    format_info["has_alpha"] = len(alpha_keys) > 0

    detected = []
    for down_suffix, up_suffix, alpha_suffix, name in LORA_PAIR_SUFFIXES:
        if any(k.endswith(down_suffix) or k.endswith(up_suffix) for k in keys):
            detected.append((down_suffix, up_suffix, alpha_suffix, name))
    has_full_diff = any(is_direct_diff_key(k) for k in keys)
    if has_full_diff:
        detected.append((None, None, None, "full_diff"))

    if len(detected) > 1:
        format_info["format"] = "mixed"
    elif detected:
        down_suffix, up_suffix, alpha_suffix, name = detected[0]
        format_info.update({
            "format": name,
            "down_suffix": down_suffix,
            "up_suffix": up_suffix,
            "alpha_suffix": alpha_suffix,
        })
    format_info["is_full_diff"] = has_full_diff and not any(item[0] for item in detected)
    pairs, _ = parse_lora_layers(keys)
    format_info["key_count"] = len(pairs)

    return format_info



def parse_lora_layers(keys: List[str]) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Parse low-rank, direct-diff, and pass-through tensors in one full scan."""
    layers: Dict[str, Dict[str, str]] = {}
    consumed = set()

    for key in keys:
        if key.endswith(".diff_b"):
            block_name = key[:-7]
            layers.setdefault(block_name, {})["diff_b"] = key
            consumed.add(key)
            continue
        if key.endswith(".diff"):
            block_name = key[:-5]
            layers.setdefault(block_name, {})["diff"] = key
            consumed.add(key)
            continue
        for down_suffix, up_suffix, alpha_suffix, format_name in LORA_PAIR_SUFFIXES:
            if key.endswith(down_suffix):
                block_name = key[:-len(down_suffix)]
                layer = layers.setdefault(block_name, {})
                layer.update({
                    "down": key,
                    "down_suffix": down_suffix,
                    "up_suffix": up_suffix,
                    "alpha_suffix": alpha_suffix,
                    "format": format_name,
                })
                consumed.add(key)
                break
            if key.endswith(up_suffix):
                block_name = key[:-len(up_suffix)]
                layer = layers.setdefault(block_name, {})
                layer.update({
                    "up": key,
                    "down_suffix": down_suffix,
                    "up_suffix": up_suffix,
                    "alpha_suffix": alpha_suffix,
                    "format": format_name,
                })
                consumed.add(key)
                break

    for key in keys:
        if not key.endswith(".alpha"):
            continue
        block_name = key[:-6]
        if block_name in layers and ("down" in layers[block_name] or "up" in layers[block_name]):
            layers[block_name]["alpha"] = key
            consumed.add(key)

    # Incomplete low-rank groups cannot be resized, but their tensors must survive.
    for block_name, layer in list(layers.items()):
        if ("down" in layer) == ("up" in layer):
            continue
        for name in ("down", "up", "alpha"):
            key = layer.pop(name, None)
            if key is not None:
                consumed.discard(key)
        for name in ("down_suffix", "up_suffix", "alpha_suffix", "format"):
            layer.pop(name, None)
        if not layer:
            del layers[block_name]

    return layers, [key for key in keys if key not in consumed]


def extract_lora_pairs(keys: List[str], format_info: Optional[Dict] = None) -> Dict[str, Dict[str, str]]:
    """
    Group LoRA keys into down/up/alpha pairs.

    For full_diff format, groups .diff and .diff_b keys.

    Returns:
        Dict[block_name, {"down": key, "up": key, "alpha": key}]
        For full_diff: {"diff": key, "diff_b": key}
    """
    pairs, _ = parse_lora_layers(keys)
    return pairs


def detect_lora_rank(
    handler: MemoryEfficientSafeOpen,
    pairs: Dict,
    low_bit_keys: Optional[set[str]] = None,
) -> Tuple[int, float]:
    """
    Detect the rank and alpha of an existing LoRA.

    Returns:
        (network_dim, network_alpha)
    """
    network_dim = None
    network_alpha = None

    for block_name, block_keys in pairs.items():
        if "down" not in block_keys:
            continue

        # Get dim from down weight shape
        if network_dim is None:
            down_key = block_keys["down"]
            shape = handler.get_shape(down_key)
            # Linear: [rank, in_features] or Conv: [rank, in_ch, k, k]
            network_dim = shape[0]

        # Get alpha if present
        if (
            network_alpha is None
            and "alpha" in block_keys
            and block_keys["alpha"] not in (low_bit_keys or set())
        ):
            alpha_tensor = handler.get_tensor(block_keys["alpha"])
            network_alpha = float(alpha_tensor.item())

        if network_dim is not None and network_alpha is not None:
            break

    # Default alpha to dim if not found
    if network_alpha is None:
        network_alpha = float(network_dim) if network_dim else 1.0
    if network_dim is None:
        network_dim = 1

    return network_dim, network_alpha


# =============================================================================
# Merge Functions
# =============================================================================

def _merge_linear(lora_down: torch.Tensor, lora_up: torch.Tensor, device: str) -> torch.Tensor:
    """Merge linear LoRA weights: lora_up @ lora_down."""
    lora_down = lora_down.to(device=device, dtype=torch.float32)
    lora_up = lora_up.to(device=device, dtype=torch.float32)
    return lora_up @ lora_down


def _merge_conv(lora_down: torch.Tensor, lora_up: torch.Tensor, device: str) -> torch.Tensor:
    """Merge conv LoRA weights."""
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert in_rank == out_rank, f"rank mismatch: {in_rank} vs {out_rank}"

    lora_down = lora_down.to(device=device, dtype=torch.float32)
    lora_up = lora_up.to(device=device, dtype=torch.float32)

    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    return merged.reshape(out_size, in_size, kernel_size, kernel_size)


# =============================================================================
# Extract Functions (after merge)
# =============================================================================

def _extract_linear(
    weight: torch.Tensor,
    max_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    scale: float,
    niter: int = 2,
) -> Dict:
    """Extract LoRA from merged linear weight."""
    out_size, in_size = weight.shape

    if dynamic_method is None:
        # Fixed rank - use svd_lowrank for 10x speedup
        rank = min(max_rank, min(out_size, in_size) - 1)
        U, S, Vh = torch.svd_lowrank(weight, q=rank, niter=niter)
        Vh = Vh.T  # svd_lowrank returns V, not Vh
        new_rank = rank
        new_alpha = float(scale * new_rank)
        stats = {"sum_retained": 1.0, "fro_retained": 1.0}  # Not computed for lowrank
    else:
        # Dynamic methods need full SVD to compute rank from all singular values
        U, S, Vh = linalg.svd(weight, full_matrices=False)
        new_rank, new_alpha, stats = _compute_resize(S, max_rank, dynamic_method, dynamic_param, scale)
        U = U[:, :new_rank]
        S = S[:new_rank]
        Vh = Vh[:new_rank, :]

    lora_up = U @ torch.diag(S)
    lora_down = Vh

    return {
        "lora_down": lora_down.cpu().contiguous(),
        "lora_up": lora_up.cpu().contiguous(),
        "new_rank": new_rank,
        "new_alpha": new_alpha,
        **stats
    }


def _extract_conv(
    weight: torch.Tensor,
    max_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    scale: float,
    niter: int = 2,
) -> Dict:
    """Extract LoRA from merged conv weight."""
    out_ch, in_ch, kh, kw = weight.shape
    mat = weight.reshape(out_ch, -1)

    if dynamic_method is None:
        # Fixed rank - use svd_lowrank for 10x speedup
        rank = min(max_rank, min(mat.shape) - 1)
        U, S, Vh = torch.svd_lowrank(mat, q=rank, niter=niter)
        Vh = Vh.T  # svd_lowrank returns V, not Vh
        new_rank = rank
        new_alpha = float(scale * new_rank)
        stats = {"sum_retained": 1.0, "fro_retained": 1.0}  # Not computed for lowrank
    else:
        # Dynamic methods need full SVD to compute rank from all singular values
        U, S, Vh = linalg.svd(mat, full_matrices=False)
        new_rank, new_alpha, stats = _compute_resize(S, max_rank, dynamic_method, dynamic_param, scale)
        U = U[:, :new_rank]
        S = S[:new_rank]
        Vh = Vh[:new_rank, :]

    lora_up = (U @ torch.diag(S)).reshape(out_ch, new_rank, 1, 1)
    lora_down = Vh.reshape(new_rank, in_ch, kh, kw)

    return {
        "lora_down": lora_down.cpu().contiguous(),
        "lora_up": lora_up.cpu().contiguous(),
        "new_rank": new_rank,
        "new_alpha": new_alpha,
        **stats
    }


def _compute_resize(
    S: torch.Tensor,
    max_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    scale: float
) -> Tuple[int, float, Dict]:
    """Compute new rank and alpha based on resize method."""

    if dynamic_method == "sv_ratio" and dynamic_param is not None:
        # Use kohya convention: S[0]/ratio
        min_sv = S[0] / dynamic_param
        new_rank = max(1, int(torch.sum(S > min_sv).item()))
    elif dynamic_method == "sv_cumulative" and dynamic_param is not None:
        new_rank = _index_sv_cumulative(S, dynamic_param, max_rank)
    elif dynamic_method == "sv_fro" and dynamic_param is not None:
        new_rank = _index_sv_fro(S, dynamic_param, max_rank)
    else:
        new_rank = max_rank

    # Clamp rank
    if S[0] < MIN_SV:
        new_rank = 1
    else:
        new_rank = max(1, min(new_rank, max_rank, len(S) - 1))

    new_alpha = float(scale * new_rank)

    # Compute retention stats
    s_sum = float(torch.sum(torch.abs(S)))
    s_rank = float(torch.sum(torch.abs(S[:new_rank]))) if new_rank <= len(S) else s_sum

    S_sq = S.pow(2)
    s_fro = float(torch.sqrt(torch.sum(S_sq)))
    s_red_fro = float(torch.sqrt(torch.sum(S_sq[:new_rank]))) if new_rank <= len(S) else s_fro

    stats = {
        "sum_retained": s_rank / s_sum if s_sum > 0 else 1.0,
        "fro_retained": s_red_fro / s_fro if s_fro > 0 else 1.0,
    }

    return new_rank, new_alpha, stats


# =============================================================================
# Main Resize Function
# =============================================================================

def resize_lora_file(
    lora_path: str,
    new_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    device: str,
    save_dtype: torch.dtype,
    output_filename: str,
    verbose: bool = True,
    svd_niter: int = 2,
    lazy_load: bool = True,
    force_clear_cache: bool = False,
) -> str:
    """
    Resize a LoRA file to a new rank.

    Args:
        lora_path: Path to input LoRA
        new_rank: Target rank (max rank for dynamic methods)
        dynamic_method: None, "sv_ratio", "sv_fro", "sv_cumulative"
        dynamic_param: Parameter for dynamic method
        device: Processing device
        save_dtype: Output dtype
        output_filename: Output filename (without extension)
        verbose: Print progress info
        svd_niter: Power iterations for SVD accuracy
        lazy_load: Low memory mode: load tensors from disk on demand
        force_clear_cache: Clear CUDA cache after each layer

    Returns:
        Path to saved resized LoRA
    """
    # Prepare memory
    lora_size_gb = estimate_model_size(lora_path)
    prepare_for_large_operation(lora_size_gb * 2, torch.device(device))

    handler = MemoryEfficientSafeOpen(lora_path, low_memory=lazy_load)

    try:
        low_bit_keys = inspect_low_bit_input(handler, f"LoRA ({lora_path})", "LoRA Resize")
        metadata = (handler.metadata() or {}).copy()
        all_keys = handler.keys()

        # Detect format and extract pairs
        format_info = detect_lora_format(all_keys)
        pairs, passthrough_keys = parse_lora_layers(all_keys)
        network_dim, network_alpha = detect_lora_rank(handler, pairs, low_bit_keys)

        scale = network_alpha / network_dim if network_dim > 0 else 1.0

        if verbose:
            method_str = f"{dynamic_method}: {dynamic_param}" if dynamic_method else "fixed"
            print(f"[LoRA Resize] Format: {format_info['format']}, layers: {format_info['key_count']}")
            print(f"[LoRA Resize] Original dim={network_dim}, alpha={network_alpha:.1f}, scale={scale:.3f}")
            print(f"[LoRA Resize] Resizing with method={method_str}, max_rank={new_rank}")

        # Build metadata and output path before loop (all values known at this point)
        if dynamic_method:
            metadata["ss_training_comment"] = f"Dynamic resize with {dynamic_method}: {dynamic_param} from dim {network_dim}"
            metadata["ss_network_dim"] = "Dynamic"
            metadata["ss_network_alpha"] = "Dynamic"
        else:
            metadata["ss_training_comment"] = f"Resized from dim {network_dim} to {new_rank}"
            metadata["ss_network_dim"] = str(new_rank)
            metadata["ss_network_alpha"] = str(scale * new_rank)

        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")

        fro_list = []
        pbar = comfy.utils.ProgressBar(len(pairs) + len(passthrough_keys))

        writer = IncrementalSafetensorsWriter(output_path, metadata=metadata)
        writer.__enter__()
        try:
            preserved_keys = set()
            with torch.no_grad():
                for block_name, block_keys in tqdm(pairs.items(), desc="Resizing layers", unit="layers"):
                    if layer_has_low_bit(block_keys, low_bit_keys):
                        layer_keys = [
                            block_keys[name]
                            for name in ("down", "up", "alpha", "diff", "diff_b")
                            if name in block_keys
                        ]
                        layer_keys.extend(
                            key for key in passthrough_keys if key.startswith(f"{block_name}.")
                        )
                        for key in layer_keys:
                            if key not in preserved_keys:
                                write_preserved_tensor(writer, key, handler)
                                preserved_keys.add(key)
                        pbar.update(1)
                        continue
                    direct_keys = [block_keys[name] for name in ("diff", "diff_b") if name in block_keys]
                    layer_source_keys = [
                        block_keys[name]
                        for name in ("down", "up", "alpha", "diff", "diff_b")
                        if name in block_keys
                    ]
                    layer_source_dtypes = [handler.get_dtype(key) for key in layer_source_keys]
                    if direct_keys:
                        for key in direct_keys:
                            tensor = handler.get_tensor(key)
                            target_dtype = select_output_dtype(
                                layer_source_dtypes,
                                save_dtype,
                                is_1d_diff=tensor.ndim == 1,
                            )
                            writer.write(key, tensor.to(target_dtype).cpu().contiguous())
                        if "down" not in block_keys or "up" not in block_keys:
                            pbar.update(1)
                            continue

                    if "down" not in block_keys or "up" not in block_keys:
                        pbar.update(1)
                        continue

                    lora_down = handler.get_tensor(block_keys["down"])
                    lora_up = handler.get_tensor(block_keys["up"])

                    is_conv = len(lora_down.shape) == 4

                    # Transfer to GPU with pinned memory if available
                    if device == 'cuda':
                        lora_down = transfer_to_gpu_pinned(lora_down, device, torch.float32)
                        lora_up = transfer_to_gpu_pinned(lora_up, device, torch.float32)

                    # Merge and re-extract
                    if is_conv:
                        weight = _merge_conv(lora_down, lora_up, device)
                        result = _extract_conv(weight, new_rank, dynamic_method, dynamic_param, scale, svd_niter)
                    else:
                        weight = _merge_linear(lora_down, lora_up, device)
                        result = _extract_linear(weight, new_rank, dynamic_method, dynamic_param, scale, svd_niter)

                    del weight, lora_down, lora_up

                    fro_list.append(result['fro_retained'])

                    layer_dtype = select_output_dtype(
                        layer_source_dtypes,
                        save_dtype,
                    )

                    block_sd = {
                        block_keys["down"]: result["lora_down"].to(layer_dtype),
                        block_keys["up"]: result["lora_up"].to(layer_dtype),
                    }
                    alpha_key = block_keys.get("alpha", f"{block_name}{block_keys['alpha_suffix']}")
                    block_sd[alpha_key] = torch.tensor(result["new_alpha"], dtype=layer_dtype)

                    writer.write_dict(block_sd)

                    del result
                    if force_clear_cache:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    pbar.update(1)

                for key in tqdm(passthrough_keys, desc="Copying auxiliary tensors", unit="layers"):
                    if key in preserved_keys:
                        pass
                    elif key in low_bit_keys:
                        write_preserved_tensor(writer, key, handler)
                    else:
                        tensor = handler.get_tensor(key)
                        target_dtype = select_output_dtype([handler.get_dtype(key)], save_dtype)
                        writer.write(key, tensor.to(target_dtype).cpu().contiguous())
                    pbar.update(1)
        finally:
            writer.__exit__(None, None, None)

        if verbose and fro_list:
            import numpy as np
            avg_fro = np.mean(fro_list)
            std_fro = np.std(fro_list)
            print(f"[LoRA Resize] Average Frobenius retention: {avg_fro:.1%} ± {std_fro:.3f}")

        print(f"[LoRA Resize] Saved to {output_path}")
        return output_path

    finally:
        handler.__exit__(None, None, None)
        cleanup_after_operation()


# =============================================================================
# Node Definitions
# =============================================================================

class LoRAResizeFixed(io.ComfyNode):
    """Resize LoRA to a fixed rank."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAResizeFixed",
            display_name="LoRA Resize (Fixed Rank)",
            category="ModelUtils/LoRA/Resize",
            description="Resize existing LoRA to a specific rank by merging and re-extracting via SVD.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                              tooltip="LoRA to resize"),
                io.Int.Input("new_rank", default=64, min=1, max=3072,
                            tooltip="Target rank"),
                io.Int.Input("svd_niter", default=2, min=0, max=10,
                            tooltip="SVD power iterations (higher = more accurate but slower)"),
                io.String.Input("output_filename", default="resized_lora"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, new_rank, svd_niter, output_filename, save_dtype, device, lazy_load, force_clear_cache) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = resize_lora_file(
            lora_path, new_rank, None, None, device, dtype, output_filename,
            svd_niter=svd_niter, lazy_load=lazy_load, force_clear_cache=force_clear_cache
        )
        return io.NodeOutput(path)


class LoRAResizeRatio(io.ComfyNode):
    """Resize LoRA keeping SVs above ratio threshold."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAResizeRatio",
            display_name="LoRA Resize (SV Ratio)",
            category="ModelUtils/LoRA/Resize",
            description="Dynamically resize LoRA, keeping singular values where S[i] > S[0]/ratio.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                              tooltip="LoRA to resize"),
                io.Int.Input("max_rank", default=128, min=1, max=3072,
                            tooltip="Maximum allowed rank"),
                io.Float.Input("ratio", default=2.0, min=1.0, max=100.0, step=0.1,
                              tooltip="Keep SVs where S[i] > S[0]/ratio"),
                io.String.Input("output_filename", default="resized_lora_ratio"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, ratio, output_filename, save_dtype, device, lazy_load, force_clear_cache) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = resize_lora_file(
            lora_path, max_rank, "sv_ratio", ratio, device, dtype, output_filename,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache
        )
        return io.NodeOutput(path)


class LoRAResizeFrobenius(io.ComfyNode):
    """Resize LoRA to preserve Frobenius norm target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAResizeFrobenius",
            display_name="LoRA Resize (Frobenius)",
            category="ModelUtils/LoRA/Resize",
            description="Dynamically resize LoRA to preserve target fraction of Frobenius norm.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                              tooltip="LoRA to resize"),
                io.Int.Input("max_rank", default=128, min=1, max=3072,
                            tooltip="Maximum allowed rank"),
                io.Float.Input("target", default=0.9, min=0.1, max=1.0, step=0.01,
                              tooltip="Target Frobenius norm retention (0.9 = 90%)"),
                io.String.Input("output_filename", default="resized_lora_fro"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, target, output_filename, save_dtype, device, lazy_load, force_clear_cache) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = resize_lora_file(
            lora_path, max_rank, "sv_fro", target, device, dtype, output_filename,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache
        )
        return io.NodeOutput(path)


class LoRAResizeCumulative(io.ComfyNode):
    """Resize LoRA to preserve cumulative SV target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAResizeCumulative",
            display_name="LoRA Resize (Cumulative)",
            category="ModelUtils/LoRA/Resize",
            description="Dynamically resize LoRA to preserve target fraction of cumulative singular values.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                              tooltip="LoRA to resize"),
                io.Int.Input("max_rank", default=128, min=1, max=3072,
                            tooltip="Maximum allowed rank"),
                io.Float.Input("target", default=0.9, min=0.1, max=1.0, step=0.01,
                              tooltip="Target cumulative SV retention (0.9 = 90%)"),
                io.String.Input("output_filename", default="resized_lora_cumulative"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, target, output_filename, save_dtype, device, lazy_load, force_clear_cache) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = resize_lora_file(
            lora_path, max_rank, "sv_cumulative", target, device, dtype, output_filename,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache
        )
        return io.NodeOutput(path)




# =============================================================================
# LoRA Merge To Model (Save merged model, skip extraction)
# =============================================================================

def merge_loras_to_model(
    lora_paths: List[str],
    lora_weights: List[float],
    base_model_path: str,
    device: str,
    save_dtype: torch.dtype,
    output_filename: str,
    skip_patterns_str: str = "",
    verbose: bool = True,
    lazy_load: bool = True,
    force_clear_cache: bool = False,
    include_1d_diffs: bool = False,
) -> str:
    """
    Merge multiple LoRAs into a base model and save the result directly.

    Unlike merge_multi_loras_via_base, this function:
    - Does NOT extract the result back to a LoRA
    - Saves the merged full model to the base model's directory

    Args:
        lora_paths: List of paths to LoRA files
        lora_weights: List of weight strengths (0.0-2.0) for each LoRA
        base_model_path: Path to base model
        device: Processing device
        save_dtype: Output dtype
        output_filename: Output filename (without extension)
        skip_patterns_str: Regex patterns for layers to skip
        verbose: Print progress info
        lazy_load: Low memory mode: load tensors from disk on demand
        force_clear_cache: Clear CUDA cache after each layer

    Returns:
        Path to saved merged model
    """
    # Estimate memory and prepare
    total_size_gb = estimate_model_size(base_model_path)
    for lp in lora_paths:
        total_size_gb += estimate_model_size(lp)

    if verbose:
        print(f"[LoRA Merge To Model] Preparing memory for {total_size_gb:.2f}GB operation...")
        print(f"[LoRA Merge To Model] Merging {len(lora_paths)} LoRAs with weights: {lora_weights}")
    prepare_for_large_operation(total_size_gb * 1.5, torch.device(device))

    # Open all files - use low_memory for base model to avoid OS page caching
    base_handler = MemoryEfficientSafeOpen(base_model_path, low_memory=lazy_load)
    lora_handlers = [MemoryEfficientSafeOpen(lp, low_memory=lazy_load) for lp in lora_paths]


    try:
        base_low_bit_keys = inspect_low_bit_input(
            base_handler, f"Base model ({base_model_path})", "LoRA Merge To Model"
        )
        lora_low_bit_keys = [
            inspect_low_bit_input(
                handler, f"LoRA {i + 1} ({lora_paths[i]})", "LoRA Merge To Model"
            )
            for i, handler in enumerate(lora_handlers)
        ]
        # Detect format and extract pairs for each LoRA
        lora_infos = []

        for i, handler in enumerate(lora_handlers):
            keys = handler.keys()
            format_info = detect_lora_format(keys)
            pairs, _ = parse_lora_layers(keys)
            network_dim, network_alpha = detect_lora_rank(
                handler, pairs, lora_low_bit_keys[i]
            )
            lora_infos.append({
                "handler": handler,
                "format_info": format_info,
                "pairs": pairs,
                "network_dim": network_dim,
                "network_alpha": network_alpha,
                "weight": lora_weights[i],
                "low_bit_keys": lora_low_bit_keys[i],
            })
            if verbose:
                print(f"[LoRA Merge To Model] LoRA {i+1}: {format_info['format']}, {len(pairs)} layers, dim={network_dim}")

        # Common prefixes
        BASE_PREFIXES = ["model.diffusion_model.", "diffusion_model.", "transformer.", "model.", "net."]
        LORA_PREFIXES = [
            "lora_unet_", "lora_transformer_", "lora_te1_", "lora_te2_", "lora_te_",
            "lycoris_", "diffusion_model.", "transformer.", "unet."
        ]

        def extract_core_layer_base(key: str) -> str:
            result = key
            if result.endswith(".weight"):
                result = result[:-7]
            elif result.endswith(".bias"):
                result = result[:-5]
            for prefix in BASE_PREFIXES:
                if result.startswith(prefix):
                    result = result[len(prefix):]
                    break
            return result

        def extract_core_layer_lora(block_name: str) -> str:
            result = block_name
            for prefix in LORA_PREFIXES:
                if result.startswith(prefix):
                    result = result[len(prefix):]
                    break
            # Strip trailing .lora suffix (HuggingFace format leaves this after suffix removal)
            if result.endswith(".lora"):
                result = result[:-5]
            return result.replace(".", "_")


        base_keys = list(base_handler.keys())
        base_aliases = set()
        for key in base_keys:
            normalized = key
            for prefix in BASE_PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
            base_aliases.add(normalized)
            base_aliases.add(normalized.replace(".", "_"))

        # Build lookups for low-rank layers and exact direct patches.
        lora_lookup = {}
        direct_lookup = {}
        for info in lora_infos:
            for block_name, block_keys in info["pairs"].items():
                if "down" in block_keys or "up" in block_keys:
                    core = extract_core_layer_lora(block_name)
                    lora_lookup.setdefault(core, []).append((info, block_keys))

                for direct_name in ("diff", "diff_b"):
                    if direct_name not in block_keys:
                        continue
                    direct_key = block_keys[direct_name]
                    core = block_name
                    for prefix in LORA_PREFIXES:
                        if core.startswith(prefix):
                            core = core[len(prefix):]
                            break
                    if direct_name == "diff_b":
                        targets = [f"{core}.bias"]
                    else:
                        # Exact non-weight state keys win; module diffs fall back to .weight.
                        exact_aliases = {core, core.replace(".", "_")}
                        targets = [core] if exact_aliases & base_aliases else [f"{core}.weight"]
                    contribution = (info, direct_key, block_keys)
                    for target in targets:
                        direct_lookup.setdefault(target, []).append(contribution)
                        underscored_target = target.replace(".", "_")
                        if underscored_target != target:
                            direct_lookup.setdefault(underscored_target, []).append(contribution)

        # Compile skip patterns
        skip_patterns = _compile_patterns(skip_patterns_str)

        # Preserve metadata from base model
        base_metadata = base_handler.metadata().copy() if base_handler.metadata() else {}
        base_metadata["merge_comment"] = f"Merged {len(lora_paths)} LoRAs with weights: {lora_weights}"

        # Build output path before loop
        base_dir = os.path.dirname(base_model_path)
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, f"{output_filename.strip()}.safetensors")

        stats = {"merged": 0, "copied": 0, "skipped": 0}
        pbar = comfy.utils.ProgressBar(len(base_keys))

        if verbose:
            print(f"[LoRA Merge To Model] Processing {len(base_keys)} base model keys...")

        writer = IncrementalSafetensorsWriter(output_path, metadata=base_metadata)
        writer.__enter__()
        try:
            with torch.no_grad():
                for base_key in tqdm(base_keys, desc="Merging to model", unit="keys"):
                    if base_key in base_low_bit_keys:
                        write_preserved_tensor(writer, base_key, base_handler)
                        stats["copied"] += 1
                        pbar.update(1)
                        continue
                    # Check skip patterns
                    if _matches_any_pattern(base_key, skip_patterns):
                        stats["skipped"] += 1
                        pbar.update(1)
                        continue

                    # Load base weight
                    cpu_base = base_handler.get_tensor(base_key)

                    core_with_suffix = base_key
                    for prefix in BASE_PREFIXES:
                        if core_with_suffix.startswith(prefix):
                            core_with_suffix = core_with_suffix[len(prefix):]
                            break
                    direct_contributions = direct_lookup.get(core_with_suffix, [])
                    if not direct_contributions:
                        direct_contributions = direct_lookup.get(core_with_suffix.replace(".", "_"), [])

                    low_rank_contributions = []
                    if base_key.endswith(".weight"):
                        core = extract_core_layer_base(base_key)
                        low_rank_contributions = lora_lookup.get(core.replace(".", "_"), [])

                    if direct_contributions or low_rank_contributions:
                            # Transfer to GPU for computation
                            if device == 'cuda':
                                base_weight = transfer_to_gpu_pinned(cpu_base, device, torch.float32)
                            else:
                                base_weight = cpu_base.to(device=device, dtype=torch.float32)
                            del cpu_base

                            source_dtypes = [base_handler.get_dtype(base_key)]
                            applied_1d_diff = False

                            for info, direct_key, block_keys in direct_contributions:
                                if layer_has_low_bit(block_keys, info["low_bit_keys"]):
                                    continue
                                cpu_diff = info["handler"].get_tensor(direct_key)
                                if cpu_diff.ndim == 1 and not include_1d_diffs:
                                    del cpu_diff
                                    continue
                                if tuple(cpu_diff.shape) != tuple(base_weight.shape):
                                    print(
                                        f"[LoRA Merge To Model] Shape mismatch for {direct_key}: "
                                        f"{tuple(cpu_diff.shape)} != {tuple(base_weight.shape)}; skipped"
                                    )
                                    del cpu_diff
                                    continue
                                source_dtypes.append(info["handler"].get_dtype(direct_key))
                                applied_1d_diff = applied_1d_diff or cpu_diff.ndim == 1
                                if device == 'cuda':
                                    delta = transfer_to_gpu_pinned(cpu_diff, device, torch.float32)
                                else:
                                    delta = cpu_diff.to(device=device, dtype=torch.float32)
                                del cpu_diff
                                base_weight = base_weight + info["weight"] * delta
                                del delta

                            # Accumulate low-rank deltas from all contributing LoRAs.
                            for info, block_keys in low_rank_contributions:
                                if "down" not in block_keys or "up" not in block_keys:
                                    continue
                                if layer_has_low_bit(block_keys, info["low_bit_keys"]):
                                    continue
                                source_keys = [block_keys["down"], block_keys["up"]]
                                if "alpha" in block_keys:
                                    source_keys.append(block_keys["alpha"])
                                source_dtypes.extend(info["handler"].get_dtype(key) for key in source_keys)

                                cpu_down = info["handler"].get_tensor(block_keys["down"])
                                cpu_up = info["handler"].get_tensor(block_keys["up"])
                                if device == 'cuda':
                                    lora_down = transfer_to_gpu_pinned(cpu_down, device, torch.float32)
                                    lora_up = transfer_to_gpu_pinned(cpu_up, device, torch.float32)
                                else:
                                    lora_down = cpu_down.to(device=device, dtype=torch.float32)
                                    lora_up = cpu_up.to(device=device, dtype=torch.float32)
                                del cpu_down, cpu_up

                                layer_rank = lora_down.shape[0]
                                if "alpha" in block_keys:
                                    alpha_tensor = info["handler"].get_tensor(block_keys["alpha"])
                                    layer_alpha = float(alpha_tensor.item())
                                else:
                                    layer_alpha = float(layer_rank)
                                effective_scale = (layer_alpha / layer_rank if layer_rank > 0 else 1.0) * info["weight"]

                                is_conv = len(lora_down.shape) == 4
                                if is_conv:
                                    in_rank, in_size, kernel_size, _ = lora_down.shape
                                    out_size = lora_up.shape[0]
                                    delta = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
                                    delta = delta.reshape(out_size, in_size, kernel_size, kernel_size)
                                else:
                                    delta = lora_up @ lora_down
                                del lora_down, lora_up
                                base_weight = base_weight + effective_scale * delta
                                del delta

                            target_dtype = select_output_dtype(
                                source_dtypes,
                                save_dtype,
                                is_1d_diff=applied_1d_diff,
                            )
                            writer.write(base_key, base_weight.to(target_dtype).cpu().contiguous())
                            del base_weight
                            stats["merged"] += 1
                    else:
                        target_dtype = select_output_dtype(
                            [base_handler.get_dtype(base_key)],
                            save_dtype,
                        )
                        writer.write(base_key, cpu_base.to(target_dtype).contiguous())
                        del cpu_base
                        stats["copied"] += 1

                    if force_clear_cache:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    pbar.update(1)
        finally:
            writer.__exit__(None, None, None)

        if verbose:
            print(f"[LoRA Merge To Model] Done: {stats['merged']} merged, {stats['copied']} copied, {stats['skipped']} skipped")

        print(f"[LoRA Merge To Model] Saved to {output_path}")
        return output_path

    finally:
        base_handler.__exit__(None, None, None)
        for handler in lora_handlers:
            handler.__exit__(None, None, None)
        cleanup_after_operation()


class LoRAMergeToModel(io.ComfyNode):
    """Merge multiple LoRAs into base model and save as full model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAMergeToModel",
            display_name="LoRA Merge To Model",
            category="ModelUtils/LoRA/Merge",
            description="Merge 1-8 LoRAs into a base model and save the result. Saves to base model directory.",
            inputs=[
                io.Combo.Input("base_model", options=folder_paths.get_filename_list("diffusion_models"),
                              tooltip="Base model the LoRAs were trained on"),
                io.Combo.Input("lora_count", options=["1", "2", "3", "4", "5", "6", "7", "8"], default="2",
                              tooltip="Number of LoRAs to merge"),
                # LoRA 1
                io.Combo.Input("lora_1", options=folder_paths.get_filename_list("loras"),
                              tooltip="First LoRA"),
                io.Float.Input("weight_1", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 1"),
                # LoRA 2
                io.Combo.Input("lora_2", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Second LoRA"),
                io.Float.Input("weight_2", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 2"),
                # LoRA 3
                io.Combo.Input("lora_3", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Third LoRA"),
                io.Float.Input("weight_3", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 3"),
                # LoRA 4
                io.Combo.Input("lora_4", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Fourth LoRA"),
                io.Float.Input("weight_4", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 4"),
                # LoRA 5
                io.Combo.Input("lora_5", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Fifth LoRA"),
                io.Float.Input("weight_5", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 5"),
                # LoRA 6
                io.Combo.Input("lora_6", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Sixth LoRA"),
                io.Float.Input("weight_6", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 6"),
                # LoRA 7
                io.Combo.Input("lora_7", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Seventh LoRA"),
                io.Float.Input("weight_7", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 7"),
                # LoRA 8
                io.Combo.Input("lora_8", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Eighth LoRA"),
                io.Float.Input("weight_8", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 8"),
                # Settings
                io.String.Input("skip_patterns", default="", multiline=True,
                               tooltip="Regex patterns for layers to skip"),
                io.String.Input("output_filename", default="merged_model"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Apply 1D direct-diff tensors as FP32. Disabled preserves prior behavior."),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, base_model, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                skip_patterns, output_filename, save_dtype, device, lazy_load, force_clear_cache,
                include_1d_diffs) -> io.NodeOutput:

        # Build LoRA list based on count
        count = int(lora_count)
        lora_names = [lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8][:count]
        lora_weights = [weight_1, weight_2, weight_3, weight_4, weight_5, weight_6, weight_7, weight_8][:count]

        # Filter out "None" entries
        valid_loras = [(name, weight) for name, weight in zip(lora_names, lora_weights) if name != "None"]
        if not valid_loras:
            raise ValueError("At least one LoRA must be selected")

        lora_names, lora_weights = zip(*valid_loras)
        lora_paths = [folder_paths.get_full_path_or_raise("loras", name) for name in lora_names]
        base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = merge_loras_to_model(
            lora_paths=list(lora_paths),
            lora_weights=list(lora_weights),
            base_model_path=base_path,
            device=device,
            save_dtype=dtype,
            output_filename=output_filename,
            skip_patterns_str=skip_patterns,
            lazy_load=lazy_load,
            force_clear_cache=force_clear_cache,
            include_1d_diffs=include_1d_diffs,
        )
        return io.NodeOutput(path)
