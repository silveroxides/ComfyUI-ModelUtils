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
from safetensors.torch import save_file
from typing import Optional, Dict, Tuple, List
from .merger_utils import MemoryEfficientSafeOpen, transfer_to_gpu_pinned
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation

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

    # Detect format by key patterns
    sample_keys = keys[:50]  # Check more keys for full_diff detection

    # Full diff format: *.diff (NOT .lora_down, just straight weight diff)
    if any(k.endswith('.diff') for k in sample_keys):
        format_info["format"] = "full_diff"
        format_info["down_suffix"] = ".diff"  # Used for pairing, but no up
        format_info["up_suffix"] = None  # No up weight in full diff
        format_info["alpha_suffix"] = None
        format_info["is_full_diff"] = True

    # Kohya/A1111 format: lora_unet_down_blocks_0_*.lora_down.weight
    elif any('lora_unet_' in k or 'lora_te' in k for k in sample_keys):
        format_info["format"] = "kohya"
        format_info["down_suffix"] = ".lora_down.weight"
        format_info["up_suffix"] = ".lora_up.weight"

    # Diffusers format: transformer.*.lora_A.weight
    elif any('.lora_A.weight' in k for k in sample_keys):
        format_info["format"] = "diffusers"
        format_info["down_suffix"] = ".lora_A.weight"
        format_info["up_suffix"] = ".lora_B.weight"
        format_info["alpha_suffix"] = ".alpha"

    # PEFT format: base_model.model.*.lora_A.default.weight
    elif any('base_model.model.' in k and 'lora_' in k for k in sample_keys):
        format_info["format"] = "peft"
        # Handle nested structure
        format_info["down_suffix"] = ".lora_A.default.weight"
        format_info["up_suffix"] = ".lora_B.default.weight"

    # HuggingFace format: transformer.*.lora.down.weight (dots instead of underscores)
    elif any('.lora.down.weight' in k for k in sample_keys):
        format_info["format"] = "huggingface"
        format_info["down_suffix"] = ".lora.down.weight"
        format_info["up_suffix"] = ".lora.up.weight"
        format_info["alpha_suffix"] = ".alpha"

    # ComfyUI native / standard
    elif any('.lora_down.weight' in k for k in sample_keys):
        format_info["format"] = "comfy"
        format_info["down_suffix"] = ".lora_down.weight"
        format_info["up_suffix"] = ".lora_up.weight"

    # Count LoRA layers
    down_keys = [k for k in keys if format_info["down_suffix"] in k]
    format_info["key_count"] = len(down_keys)

    return format_info



def extract_lora_pairs(keys: List[str], format_info: Dict) -> Dict[str, Dict[str, str]]:
    """
    Group LoRA keys into down/up/alpha pairs.

    For full_diff format, groups .diff and .diff_b keys.

    Returns:
        Dict[block_name, {"down": key, "up": key, "alpha": key}]
        For full_diff: {"diff": key, "diff_b": key}
    """
    down_suffix = format_info["down_suffix"]
    up_suffix = format_info["up_suffix"]
    alpha_suffix = format_info["alpha_suffix"]
    is_full_diff = format_info.get("is_full_diff", False)

    pairs = {}

    if is_full_diff:
        # Full diff format: group .diff and .diff_b
        for key in keys:
            if key.endswith('.diff'):
                block_name = key[:-5]  # Remove .diff
                if block_name not in pairs:
                    pairs[block_name] = {}
                pairs[block_name]["diff"] = key
            elif key.endswith('.diff_b'):
                block_name = key[:-7]  # Remove .diff_b
                if block_name not in pairs:
                    pairs[block_name] = {}
                pairs[block_name]["diff_b"] = key
    else:
        # Standard LoRA format
        for key in keys:
            if down_suffix and down_suffix in key:
                block_name = key.replace(down_suffix, "")
                if block_name not in pairs:
                    pairs[block_name] = {}
                pairs[block_name]["down"] = key
            elif up_suffix and up_suffix in key:
                block_name = key.replace(up_suffix, "")
                if block_name not in pairs:
                    pairs[block_name] = {}
                pairs[block_name]["up"] = key
            elif alpha_suffix and (alpha_suffix in key or key.endswith('.alpha')):
                # Handle various alpha key formats
                block_name = key.replace(alpha_suffix, "").replace(".alpha", "")
                if block_name not in pairs:
                    pairs[block_name] = {}
                pairs[block_name]["alpha"] = key

    return pairs


def detect_lora_rank(handler: MemoryEfficientSafeOpen, pairs: Dict) -> Tuple[int, float]:
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
            shape = handler.header[down_key]["shape"]
            # Linear: [rank, in_features] or Conv: [rank, in_ch, k, k]
            network_dim = shape[0]

        # Get alpha if present
        if network_alpha is None and "alpha" in block_keys:
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

    Returns:
        Path to saved resized LoRA
    """
    # Prepare memory
    lora_size_gb = estimate_model_size(lora_path)
    prepare_for_large_operation(lora_size_gb * 2, torch.device(device))

    handler = MemoryEfficientSafeOpen(lora_path)

    try:
        metadata = handler.metadata().copy()
        all_keys = handler.keys()

        # Detect format and extract pairs
        format_info = detect_lora_format(all_keys)
        pairs = extract_lora_pairs(all_keys, format_info)
        network_dim, network_alpha = detect_lora_rank(handler, pairs)

        scale = network_alpha / network_dim if network_dim > 0 else 1.0

        if verbose:
            method_str = f"{dynamic_method}: {dynamic_param}" if dynamic_method else "fixed"
            print(f"[LoRA Resize] Format: {format_info['format']}, layers: {format_info['key_count']}")
            print(f"[LoRA Resize] Original dim={network_dim}, alpha={network_alpha:.1f}, scale={scale:.3f}")
            print(f"[LoRA Resize] Resizing with method={method_str}, max_rank={new_rank}")

        output_sd = {}
        fro_list = []
        pbar = comfy.utils.ProgressBar(len(pairs))

        with torch.no_grad():
            for block_name, block_keys in tqdm(pairs.items(), desc="Resizing layers", unit="layers"):
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

                # Store using same format suffixes as input
                down_suffix = format_info["down_suffix"]
                up_suffix = format_info["up_suffix"]
                alpha_suffix = format_info["alpha_suffix"]

                # Standardize prefix using heuristics if possible
                new_block_name = _format_lora_key(block_name)

                output_sd[f"{new_block_name}{down_suffix}"] = result["lora_down"].to(save_dtype)
                output_sd[f"{new_block_name}{up_suffix}"] = result["lora_up"].to(save_dtype)
                if alpha_suffix:
                    output_sd[f"{new_block_name}{alpha_suffix}"] = torch.tensor(result["new_alpha"]).to(save_dtype)

                pbar.update(1)

        if verbose and fro_list:
            import numpy as np
            avg_fro = np.mean(fro_list)
            std_fro = np.std(fro_list)
            print(f"[LoRA Resize] Average Frobenius retention: {avg_fro:.1%} ± {std_fro:.3f}")

        # Update metadata
        if dynamic_method:
            metadata["ss_training_comment"] = f"Dynamic resize with {dynamic_method}: {dynamic_param} from dim {network_dim}"
            metadata["ss_network_dim"] = "Dynamic"
            metadata["ss_network_alpha"] = "Dynamic"
        else:
            metadata["ss_training_comment"] = f"Resized from dim {network_dim} to {new_rank}"
            metadata["ss_network_dim"] = str(new_rank)
            metadata["ss_network_alpha"] = str(scale * new_rank)

        # Save
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")

        save_file(output_sd, output_path, metadata)
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
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, new_rank, svd_niter, output_filename, save_dtype, device) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = resize_lora_file(lora_path, new_rank, None, None, device, dtype, output_filename, svd_niter=svd_niter)
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
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, ratio, output_filename, save_dtype, device) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = resize_lora_file(lora_path, max_rank, "sv_ratio", ratio, device, dtype, output_filename)
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
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, target, output_filename, save_dtype, device) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = resize_lora_file(lora_path, max_rank, "sv_fro", target, device, dtype, output_filename)
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
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, target, output_filename, save_dtype, device) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = resize_lora_file(lora_path, max_rank, "sv_cumulative", target, device, dtype, output_filename)
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
    base_handler = MemoryEfficientSafeOpen(base_model_path, low_memory=True)
    lora_handlers = [MemoryEfficientSafeOpen(lp) for lp in lora_paths]


    try:
        # Detect format and extract pairs for each LoRA
        lora_infos = []

        for i, handler in enumerate(lora_handlers):
            keys = handler.keys()
            format_info = detect_lora_format(keys)
            pairs = extract_lora_pairs(keys, format_info)
            network_dim, network_alpha = detect_lora_rank(handler, pairs)
            lora_infos.append({
                "handler": handler,
                "format_info": format_info,
                "pairs": pairs,
                "network_dim": network_dim,
                "network_alpha": network_alpha,
                "weight": lora_weights[i],
            })
            if verbose:
                print(f"[LoRA Merge To Model] LoRA {i+1}: {format_info['format']}, {len(pairs)} layers, dim={network_dim}")

        # Common prefixes
        BASE_PREFIXES = ["model.diffusion_model.", "diffusion_model.", "transformer.", "model."]
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


        # Build LoRA lookup: core layer name (underscored) -> list of (info, block_keys)
        lora_lookup = {}
        for info in lora_infos:
            for block_name, block_keys in info["pairs"].items():
                core = extract_core_layer_lora(block_name)
                if core not in lora_lookup:
                    lora_lookup[core] = []
                lora_lookup[core].append((info, block_keys))

        # Compile skip patterns
        skip_patterns = _compile_patterns(skip_patterns_str)

        # Preserve metadata from base model
        base_metadata = base_handler.metadata().copy() if base_handler.metadata() else {}
        base_metadata["merge_comment"] = f"Merged {len(lora_paths)} LoRAs with weights: {lora_weights}"

        output_sd = {}
        stats = {"merged": 0, "copied": 0, "skipped": 0}
        base_keys = list(base_handler.keys())
        pbar = comfy.utils.ProgressBar(len(base_keys))

        if verbose:
            print(f"[LoRA Merge To Model] Processing {len(base_keys)} base model keys...")

        with torch.no_grad():
            for base_key in tqdm(base_keys, desc="Merging to model", unit="keys"):
                # Check skip patterns
                if _matches_any_pattern(base_key, skip_patterns):
                    stats["skipped"] += 1
                    pbar.update(1)
                    continue

                # Load base weight
                cpu_base = base_handler.get_tensor(base_key)

                # Only process weight tensors for LoRA merging
                if base_key.endswith(".weight"):
                    core = extract_core_layer_base(base_key)
                    core_underscored = core.replace(".", "_")

                    # Check if any LoRA contributes to this layer
                    if core_underscored in lora_lookup:
                        # Transfer to GPU for computation
                        if device == 'cuda':
                            base_weight = transfer_to_gpu_pinned(cpu_base, device, torch.float32)
                        else:
                            base_weight = cpu_base.to(device=device, dtype=torch.float32)
                        del cpu_base

                        # Accumulate deltas from all contributing LoRAs
                        for info, block_keys in lora_lookup[core_underscored]:
                            is_full_diff = info["format_info"].get("is_full_diff", False)

                            if is_full_diff:
                                # Full diff format
                                if "diff" not in block_keys:
                                    continue
                                cpu_diff = info["handler"].get_tensor(block_keys["diff"])
                                if device == 'cuda':
                                    delta = transfer_to_gpu_pinned(cpu_diff, device, torch.float32)
                                else:
                                    delta = cpu_diff.to(device=device, dtype=torch.float32)
                                del cpu_diff
                                effective_scale = info["weight"]
                            else:
                                # Standard LoRA format
                                if "down" not in block_keys or "up" not in block_keys:
                                    continue

                                cpu_down = info["handler"].get_tensor(block_keys["down"])
                                cpu_up = info["handler"].get_tensor(block_keys["up"])
                                if device == 'cuda':
                                    lora_down = transfer_to_gpu_pinned(cpu_down, device, torch.float32)
                                    lora_up = transfer_to_gpu_pinned(cpu_up, device, torch.float32)
                                else:
                                    lora_down = cpu_down.to(device=device, dtype=torch.float32)
                                    lora_up = cpu_up.to(device=device, dtype=torch.float32)
                                del cpu_down, cpu_up

                                # Get alpha
                                if "alpha" in block_keys:
                                    alpha_tensor = info["handler"].get_tensor(block_keys["alpha"])
                                    layer_alpha = float(alpha_tensor.item())
                                else:
                                    layer_alpha = float(info["network_dim"])
                                layer_scale = layer_alpha / info["network_dim"] if info["network_dim"] > 0 else 1.0
                                effective_scale = layer_scale * info["weight"]

                                # Compute delta
                                is_conv = len(lora_down.shape) == 4
                                if is_conv:
                                    in_rank, in_size, kernel_size, k_ = lora_down.shape
                                    out_size, out_rank, _, _ = lora_up.shape
                                    delta = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
                                    delta = delta.reshape(out_size, in_size, kernel_size, kernel_size)
                                else:
                                    delta = lora_up @ lora_down
                                del lora_down, lora_up

                            # Apply delta to base weight
                            base_weight = base_weight + effective_scale * delta
                            del delta

                        # Store merged weight
                        output_sd[base_key] = base_weight.to(save_dtype).cpu().contiguous()
                        del base_weight
                        stats["merged"] += 1
                    else:
                        # No LoRA contribution, copy as-is
                        output_sd[base_key] = cpu_base.to(save_dtype).contiguous()
                        del cpu_base
                        stats["copied"] += 1
                else:
                    # Non-weight tensor (bias, norm, etc.), copy as-is
                    output_sd[base_key] = cpu_base.to(save_dtype).contiguous()
                    del cpu_base
                    stats["copied"] += 1


                pbar.update(1)

        if verbose:
            print(f"[LoRA Merge To Model] Done: {stats['merged']} merged, {stats['copied']} copied, {stats['skipped']} skipped")

        # Save to base model directory
        base_dir = os.path.dirname(base_model_path)
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, f"{output_filename.strip()}.safetensors")

        save_file(output_sd, output_path, base_metadata)
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
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, base_model, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                skip_patterns, output_filename, save_dtype, device) -> io.NodeOutput:

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
        )
        return io.NodeOutput(path)
