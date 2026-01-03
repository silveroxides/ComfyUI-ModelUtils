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
    }
    
    # Check for alpha keys
    alpha_keys = [k for k in keys if '.alpha' in k or k.endswith('alpha')]
    format_info["has_alpha"] = len(alpha_keys) > 0
    
    # Detect format by key patterns
    sample_keys = keys[:20]  # Check first 20 keys
    
    # Kohya/A1111 format: lora_unet_down_blocks_0_*.lora_down.weight
    if any('lora_unet_' in k or 'lora_te' in k for k in sample_keys):
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
    
    Returns:
        Dict[block_name, {"down": key, "up": key, "alpha": key}]
    """
    down_suffix = format_info["down_suffix"]
    up_suffix = format_info["up_suffix"]
    alpha_suffix = format_info["alpha_suffix"]
    
    pairs = {}
    
    for key in keys:
        if down_suffix in key:
            block_name = key.replace(down_suffix, "")
            if block_name not in pairs:
                pairs[block_name] = {}
            pairs[block_name]["down"] = key
        elif up_suffix in key:
            block_name = key.replace(up_suffix, "")
            if block_name not in pairs:
                pairs[block_name] = {}
            pairs[block_name]["up"] = key
        elif alpha_suffix in key or key.endswith('.alpha'):
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
) -> Dict:
    """Extract LoRA from merged linear weight."""
    out_size, in_size = weight.shape
    
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
) -> Dict:
    """Extract LoRA from merged conv weight."""
    out_ch, in_ch, kh, kw = weight.shape
    mat = weight.reshape(out_ch, -1)
    
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
        # Note: _index_sv_ratio uses S[0]*ratio, kohya uses S[0]/ratio
        # Use kohya convention here for consistency with their tool
        min_sv = S[0] / dynamic_param
        new_rank = max(1, int(torch.sum(S > min_sv).item()))
    elif dynamic_method == "sv_cumulative" and dynamic_param is not None:
        new_rank = _index_sv_cumulative(S, dynamic_param)
    elif dynamic_method == "sv_fro" and dynamic_param is not None:
        new_rank = _index_sv_fro(S, dynamic_param)
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
    verbose: bool = True
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
                    result = _extract_conv(weight, new_rank, dynamic_method, dynamic_param, scale)
                else:
                    weight = _merge_linear(lora_down, lora_up, device)
                    result = _extract_linear(weight, new_rank, dynamic_method, dynamic_param, scale)
                
                del weight, lora_down, lora_up
                
                fro_list.append(result['fro_retained'])
                
                # Store using same format suffixes as input
                down_suffix = format_info["down_suffix"]
                up_suffix = format_info["up_suffix"]
                alpha_suffix = format_info["alpha_suffix"]
                
                output_sd[f"{block_name}{down_suffix}"] = result["lora_down"].to(save_dtype)
                output_sd[f"{block_name}{up_suffix}"] = result["lora_up"].to(save_dtype)
                output_sd[f"{block_name}{alpha_suffix}"] = torch.tensor([result["new_alpha"]]).to(save_dtype)
                
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
                io.Int.Input("new_rank", default=64, min=1, max=1024,
                            tooltip="Target rank"),
                io.String.Input("output_filename", default="resized_lora"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls, lora_name, new_rank, output_filename, save_dtype, device) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]
        
        path = resize_lora_file(lora_path, new_rank, None, None, device, dtype, output_filename)
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
                io.Int.Input("max_rank", default=128, min=1, max=1024,
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
                io.Int.Input("max_rank", default=128, min=1, max=1024,
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
                io.Int.Input("max_rank", default=128, min=1, max=1024,
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
