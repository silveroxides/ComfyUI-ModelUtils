"""
LoRA Extraction via SVD decomposition.

Native implementation of Low-Rank Adapter extraction with multiple rank selection modes.
No external dependencies required.
"""
import fnmatch
import os
import re
import torch
import torch.linalg as linalg
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from .device_utils import (
    estimate_model_size, prepare_for_large_operation,
    cleanup_after_operation, get_device_capabilities
)

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned, IncrementalSafetensorsWriter




# =============================================================================
# Rank Index Functions - Determine LoRA rank from singular values
# =============================================================================

def _index_sv_fixed(S: torch.Tensor, dim: int) -> int:
    """Fixed rank mode - use specified dimension."""
    return max(1, min(dim, len(S) - 1))


def _index_sv_threshold(S: torch.Tensor, threshold: float) -> int:
    """Threshold mode - keep singular values above absolute threshold."""
    rank = int(torch.sum(S > threshold).item())
    return max(1, min(rank, len(S) - 1))


def _index_sv_ratio(S: torch.Tensor, ratio: float) -> int:
    """Ratio mode - keep singular values > max(S) / ratio (kohya convention)."""
    if ratio <= 0:
        return len(S) - 1
    min_sv = S[0] / ratio
    rank = int(torch.sum(S > min_sv).item())
    return max(1, min(rank, len(S) - 1))


def _index_sv_cumulative(S: torch.Tensor, target: float, max_rank: int = None) -> int:
    """Cumulative mode - keep enough SVs to reach target % of total.

    Calculates relative to max_rank if provided, otherwise relative to full.
    """
    if max_rank is not None and max_rank < len(S):
        total = torch.sum(S[:max_rank])
    else:
        total = torch.sum(S)

    if total < 1e-8:
        return 1
    cumsum = torch.cumsum(S, dim=0) / total
    rank = int(torch.searchsorted(cumsum, target).item()) + 1
    return max(1, min(rank, len(S) - 1))


def _index_sv_fro(S: torch.Tensor, target: float, max_rank: int = None) -> int:
    """Frobenius norm mode - preserve target fraction of Frobenius norm.

    Calculates relative to max_rank if provided, otherwise relative to full.
    This means "retain target% of what's achievable within max_rank".
    """
    if max_rank is not None and max_rank < len(S):
        # Calculate relative to what's achievable within max_rank
        S_capped = S[:max_rank]
        S_sq = S_capped.pow(2)
        total_sq = torch.sum(S_sq)
    else:
        S_sq = S.pow(2)
        total_sq = torch.sum(S_sq)

    if total_sq < 1e-8:
        return 1

    # Cumsum of all S (not capped) to find where we reach target
    cumsum = torch.cumsum(S.pow(2), dim=0) / total_sq
    rank = int(torch.searchsorted(cumsum, target ** 2).item()) + 1
    return max(1, min(rank, len(S) - 1))


def _index_sv_knee(S: torch.Tensor, min_sv: float = 1e-8) -> int:
    """Knee detection - find elbow point on singular value curve."""
    n = len(S)
    if n < 3:
        return 1
    s_max, s_min = S[0].item(), S[-1].item()
    if s_max - s_min < min_sv:
        return 1
    # Normalize to [0, 1]
    s_norm = (S - s_min) / (s_max - s_min)
    x_norm = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)
    # Distance from diagonal
    distances = (x_norm + s_norm - 1).abs()
    rank = torch.argmax(distances).item() + 1
    return max(1, min(rank, n - 1))


def _index_sv_cumulative_knee(S: torch.Tensor, min_sv: float = 1e-8) -> int:
    """Knee detection on cumulative singular value curve."""
    n = len(S)
    if n < 3:
        return 1
    total = torch.sum(S)
    if total < min_sv:
        return 1
    cumsum = torch.cumsum(S, dim=0) / total
    y_min, y_max = cumsum[0].item(), cumsum[-1].item()
    if y_max - y_min < min_sv:
        return 1
    y_norm = (cumsum - y_min) / (y_max - y_min)
    x_norm = torch.linspace(0, 1, n, device=S.device, dtype=S.dtype)
    distances = (y_norm - x_norm).abs()
    rank = torch.argmax(distances).item() + 1
    return max(1, min(rank, n - 1))


def _index_sv_rel_decrease(S: torch.Tensor, tau: float = 0.1) -> int:
    """Relative decrease mode - stop when consecutive SV ratio drops below tau."""
    if len(S) < 2:
        return 1
    ratios = S[1:] / (S[:-1] + 1e-8)
    for k in range(len(ratios)):
        if ratios[k] < tau:
            return k + 1
    return len(S)


def _compute_rank(S: torch.Tensor, mode: str, mode_param: float,
                  max_rank: int = None) -> int:
    """Compute rank based on mode and parameters."""
    if mode == "fixed":
        rank = _index_sv_fixed(S, int(mode_param))
    elif mode == "threshold":
        rank = _index_sv_threshold(S, mode_param)
    elif mode == "ratio":
        rank = _index_sv_ratio(S, mode_param)
    elif mode == "quantile" or mode == "sv_cumulative":
        rank = _index_sv_cumulative(S, mode_param, max_rank)
    elif mode == "sv_fro":
        rank = _index_sv_fro(S, mode_param, max_rank)
    elif mode == "sv_knee":
        rank = _index_sv_knee(S)
    elif mode == "sv_cumulative_knee":
        rank = _index_sv_cumulative_knee(S)
    elif mode == "sv_rel_decrease":
        rank = _index_sv_rel_decrease(S, mode_param)
    else:
        rank = _index_sv_fixed(S, int(mode_param) if mode_param else 64)

    if max_rank is not None:
        rank = min(rank, max_rank)
    return rank


# =============================================================================
# SVD Extraction Functions
# =============================================================================

def _svd_extract_linear_lowrank(
    weight: torch.Tensor,
    rank: int,
    device: str,
    clamp_quantile: float = 0.99,
    niter: int = 2,
) -> tuple:
    """
    Low-rank SVD decomposition using svd_lowrank for fixed rank extraction.

    Much faster than full SVD when rank << min(m, n) because it only computes
    the top-k singular values using randomized algorithms.

    Args:
        niter: Power iterations for SVD accuracy (higher = more accurate but slower)

    Returns:
        (lora_down, lora_up, diff), "low rank" or (weight, "full")
    """
    weight = weight.to(device=device, dtype=torch.float32)
    out_dim, in_dim = weight.shape

    # Clamp rank to valid range
    max_possible_rank = min(out_dim, in_dim)
    rank = max(1, min(rank, max_possible_rank))

    # Check if decomposition is worthwhile
    if rank >= max_possible_rank:
        return weight, "full"

    # Use svd_lowrank for faster low-rank approximation
    try:
        U, S, V = torch.svd_lowrank(weight, q=rank, niter=niter)
        # U: [out_dim, rank], S: [rank], V: [in_dim, rank]
        Vh = V.T  # [rank, in_dim]
    except Exception as e:
        raise RuntimeError(f"svd_lowrank failed: {e}")

    # Clamp outliers
    if clamp_quantile < 1.0 and len(S) > 0:
        try:
            max_val = torch.quantile(S, clamp_quantile)
            S = S.clamp(max=max_val)
        except RuntimeError:
            pass

    # Construct LoRA matrices
    lora_up = U @ torch.diag(S)  # [out_dim, rank]
    lora_down = Vh               # [rank, in_dim]

    # Compute reconstruction diff
    diff = weight - (lora_up @ lora_down)

    return (lora_down, lora_up, diff), "low rank"


def _svd_extract_linear(
    weight: torch.Tensor,
    mode: str,
    mode_param: float,
    device: str,
    max_rank: int = None,
    clamp_quantile: float = 0.99,
    niter: int = 2,
) -> tuple:
    """
    SVD decomposition for linear layers.

    For fixed rank mode, uses svd_lowrank which is much faster.
    For adaptive modes, uses full SVD to analyze all singular values.

    Returns:
        (lora_down, lora_up, rank), "low rank" or (weight, "full")
    """
    weight = weight.to(device=device, dtype=torch.float32)
    out_dim, in_dim = weight.shape

    # For fixed rank, use the faster svd_lowrank
    if mode == "fixed" and mode_param > 0:
        target_rank = int(mode_param)
        if max_rank is not None:
            target_rank = min(target_rank, max_rank)
        return _svd_extract_linear_lowrank(weight, target_rank, device, clamp_quantile, niter)

    # For adaptive modes, use full SVD to analyze singular values
    try:
        U, S, Vh = linalg.svd(weight, full_matrices=False)
    except Exception as e:
        raise RuntimeError(f"SVD failed: {e}")

    # Compute rank based on mode
    rank = _compute_rank(S, mode, mode_param, max_rank)

    # Check if decomposition is worthwhile
    if rank >= min(out_dim, in_dim):
        return weight, "full"

    # Truncate to rank
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    # Clamp outliers
    if clamp_quantile < 1.0 and len(S) > 0:
        try:
            max_val = torch.quantile(S, clamp_quantile)
            S = S.clamp(max=max_val)
        except RuntimeError:
            pass  # Skip clamping if quantile fails

    # Construct LoRA matrices
    # lora_up = U @ diag(S), lora_down = Vh
    lora_up = U @ torch.diag(S)  # [out_dim, rank]
    lora_down = Vh               # [rank, in_dim]

    # Compute reconstruction diff for optional sparse bias
    diff = weight - (lora_up @ lora_down)

    return (lora_down, lora_up, diff), "low rank"


def _svd_extract_conv(
    weight: torch.Tensor,
    mode: str,
    mode_param: float,
    device: str,
    max_rank: int = None,
    clamp_quantile: float = 0.99,
) -> tuple:
    """
    SVD decomposition for conv2d layers.

    Returns:
        (lora_down, lora_up, rank), "low rank" or (weight, "full")
    """
    out_ch, in_ch, kh, kw = weight.shape
    is_1x1 = (kh == 1 and kw == 1)

    # Flatten for SVD
    if is_1x1:
        mat = weight.view(out_ch, in_ch)  # [out_ch, in_ch]
    else:
        mat = weight.reshape(out_ch, -1)  # [out_ch, in_ch*k*k]

    result, mode_str = _svd_extract_linear(mat, mode, mode_param, device, max_rank, clamp_quantile)

    if mode_str == "full":
        return weight, "full"

    lora_down, lora_up, diff = result
    rank = lora_down.shape[0]

    # Reshape for conv
    if is_1x1:
        lora_down = lora_down.unsqueeze(-1).unsqueeze(-1)  # [rank, in_ch, 1, 1]
        lora_up = lora_up.unsqueeze(-1).unsqueeze(-1)      # [out_ch, rank, 1, 1]
    else:
        lora_down = lora_down.reshape(rank, in_ch, kh, kw)
        lora_up = lora_up.reshape(out_ch, rank, 1, 1)

    return (lora_down, lora_up, diff), "low rank"


# =============================================================================
# Chunked Extraction for Large Layers
# =============================================================================

def _detect_fused_layer(key: str, shape: tuple) -> int:
    """Detect fused QKV/MLP layers and return number of chunks."""
    if len(shape) != 2:
        return 1

    out_dim, in_dim = shape
    key_lower = key.lower()

    # QKV layers (3 chunks)
    if "qkv" in key_lower and out_dim % 3 == 0 and out_dim // 3 >= in_dim // 4:
        return 3

    # KV layers (2 chunks)
    if "kv" in key_lower and "qkv" not in key_lower and out_dim % 2 == 0:
        return 2

    # Fused MLP (2 chunks)
    if ("linear1" in key_lower or "gate_up" in key_lower) and out_dim % 2 == 0 and out_dim >= in_dim * 2:
        return 2

    return 1


def _extract_chunked_layer(
    weight_diff: torch.Tensor,
    num_chunks: int,
    mode: str,
    mode_param: float,
    device: str,
    max_rank: int = None,
) -> tuple:
    """Extract LoRA from fused layer by chunking."""
    out_dim, in_dim = weight_diff.shape
    chunk_size = out_dim // num_chunks

    all_lora_up = []
    all_lora_down = []

    for i in range(num_chunks):
        chunk = weight_diff[i * chunk_size:(i + 1) * chunk_size, :]

        try:
            result, mode_str = _svd_extract_linear(chunk, mode, mode_param, device, max_rank)
            if mode_str == "full":
                return None, None, 0
            lora_down, lora_up, _ = result
            all_lora_down.append(lora_down)
            all_lora_up.append(lora_up)
        except Exception:
            return None, None, 0

    # Combine chunks
    # Note: Using the first chunk's down matrix as the basis for all chunks
    # is a better approximation than a simple mean if the chunks share the same input space (like QKV).
    # Ideally, we would use a more sophisticated joint SVD, but this is a reasonable fallback.
    combined_lora_up = torch.cat(all_lora_up, dim=0)
    combined_lora_down = all_lora_down[0]
    rank = combined_lora_down.shape[0]

    return combined_lora_up, combined_lora_down, rank


# =============================================================================
# Pattern Matching
# =============================================================================

def _compile_patterns(pattern_string: str, glob_mode: bool = False) -> list:
    """Compile patterns from whitespace-separated string.

    In regex mode (default) returns compiled re.Pattern objects.
    In glob mode returns plain strings; fnmatch handles matching.
    """
    if not pattern_string or not pattern_string.strip():
        return []
    patterns = []
    for p in pattern_string.split():
        p = p.strip()
        if not p:
            continue
        if glob_mode:
            patterns.append(p)
        else:
            try:
                patterns.append(re.compile(p))
            except re.error:
                pass
    return patterns


def _matches_any_pattern(key: str, patterns: list, glob_mode: bool = False) -> bool:
    """Check if key matches any pattern.

    Glob mode: each pattern is matched as a substring glob using fnmatch
    (dots are literal, * matches any sequence of characters).
    Regex mode: each pattern is a compiled re.Pattern, matched via search().
    """
    if glob_mode:
        return any(fnmatch.fnmatch(key, f"*{p}*") for p in patterns)
    return any(p.search(key) for p in patterns)


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_lora_from_files(
    model_a_path: str,
    model_b_path: str,
    mode: str,
    linear_param: float,
    conv_param: float,
    device: str,
    save_dtype: str,
    output_path: str,
    linear_max_rank: int = None,
    conv_max_rank: int = None,
    clamp_quantile: float = 0.99,
    min_diff: float = 0.0,
    skip_patterns_str: str = "",
    mismatch_mode: str = "skip",
    chunk_large_layers: bool = True,
    svd_niter: int = 2,
    lazy_load: bool = True,
    force_clear_cache: bool = True,
    glob_skip_patterns: bool = False,
) -> None:
    """
    Extract LoRA from difference between two models, writing incrementally to disk.

    Args:
        model_a_path: Finetuned model path
        model_b_path: Base model path
        mode: Rank selection mode
        linear_param: Mode parameter for linear layers
        conv_param: Mode parameter for conv layers
        device: Computation device
        save_dtype: Output dtype
        output_path: Full path to output safetensors file
        linear_max_rank: Max rank for linear layers
        conv_max_rank: Max rank for conv layers
        clamp_quantile: Quantile for weight clamping
        min_diff: Minimum difference threshold
        skip_patterns_str: Patterns for layers to skip
        mismatch_mode: How to handle mismatches
        chunk_large_layers: Enable chunked extraction
        glob_skip_patterns: When True, treat skip_patterns as glob (* wildcard).
                            When False (default), treat as Python regex.
    """
    save_torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(save_dtype, torch.float16)

    skip_patterns = _compile_patterns(skip_patterns_str, glob_mode=glob_skip_patterns)

    # Prepare memory before heavy operation
    total_size_gb = estimate_model_size(model_a_path) + estimate_model_size(model_b_path)
    print(f"[LoRA Extract] Preparing memory for {total_size_gb:.2f}GB operation...")
    prepare_for_large_operation(total_size_gb * 1.5, torch.device(device))  # 1.5x for SVD headroom

    handler_a = MemoryEfficientSafeOpen(model_a_path, low_memory=lazy_load)
    handler_b = MemoryEfficientSafeOpen(model_b_path, low_memory=lazy_load)

    try:
        keys_a = set(handler_a.keys())
        keys_b = set(handler_b.keys())
        weight_keys = [k for k in keys_a if k.endswith(".weight")]
        pbar = comfy.utils.ProgressBar(len(weight_keys))
        stats = {"extracted": 0, "full": 0, "skipped": 0, "chunked": 0}

        def _process_layer(key):
            lora_name = _format_lora_key(key)

            if _matches_any_pattern(key, skip_patterns, glob_mode=glob_skip_patterns):
                return "skipped", None

            # Load tensors with pinned memory for CUDA
            use_pinned = device == 'cuda'

            if key not in keys_b:
                if mismatch_mode == "skip":
                    return "skipped", None
                if mismatch_mode == "error":
                    raise ValueError(f"Key {key} not found in model B")

                cpu_a = handler_a.get_tensor(key)
                if use_pinned:
                    weight_diff = transfer_to_gpu_pinned(cpu_a, device, torch.float32)
                else:
                    weight_diff = cpu_a.to(device=device, dtype=torch.float32)
                del cpu_a

                if mismatch_mode == "zeros":
                    # For zeros mode, we treat the missing base as a zeroed tensor of same shape
                    # which is already captured by weight_diff = tensor_a
                    pass
            else:
                cpu_a = handler_a.get_tensor(key)
                cpu_b = handler_b.get_tensor(key)

                if use_pinned:
                    tensor_a = transfer_to_gpu_pinned(cpu_a, device, torch.float32)
                    tensor_b = transfer_to_gpu_pinned(cpu_b, device, torch.float32)
                else:
                    tensor_a = cpu_a.to(device=device, dtype=torch.float32)
                    tensor_b = cpu_b.to(device=device, dtype=torch.float32)
                del cpu_a, cpu_b

                if tensor_a.shape != tensor_b.shape:
                    if mismatch_mode == "skip":
                        del tensor_a, tensor_b
                        return "skipped", None
                    if mismatch_mode == "error":
                        raise ValueError(f"Shape mismatch for {key}: {tensor_a.shape} vs {tensor_b.shape}")

                    # For zeros/fallback, use tensor_a as the difference
                    weight_diff = tensor_a
                    del tensor_b
                else:
                    weight_diff = tensor_a - tensor_b
                    del tensor_a, tensor_b

            # Skip small differences
            if min_diff > 0 and weight_diff.abs().max() < min_diff:
                del weight_diff
                return "skipped", None

            # Skip 1D tensors
            if weight_diff.ndim < 2:
                del weight_diff
                return "skipped", None

            is_conv = weight_diff.ndim == 4
            layer_results = {}

            try:
                if is_conv:
                    result, mode_str = _svd_extract_conv(
                        weight_diff, mode, conv_param, device, conv_max_rank, clamp_quantile
                    )
                else:
                    result, mode_str = _svd_extract_linear(
                        weight_diff, mode, linear_param, device, linear_max_rank, clamp_quantile, svd_niter
                    )
            except Exception as e:
                # Try chunked extraction for large tensors
                if chunk_large_layers and not is_conv:
                    num_chunks = _detect_fused_layer(key, weight_diff.shape)
                    if num_chunks > 1:
                        print(f"[LoRA Extract] Chunked: {key} ({num_chunks} chunks)")
                        lora_up, lora_down, rank = _extract_chunked_layer(
                            weight_diff, num_chunks, mode, linear_param, device, linear_max_rank
                        )
                        if lora_up is not None:
                            layer_results[f"{lora_name}.lora_B.weight"] = lora_up.to(save_torch_dtype).cpu().contiguous()
                            layer_results[f"{lora_name}.lora_A.weight"] = lora_down.to(save_torch_dtype).cpu().contiguous()
                            del weight_diff
                            return "chunked", layer_results

                print(f"[LoRA Extract] Failed: {key}: {e}")
                del weight_diff
                return "skipped", None

            # Store result
            if mode_str == "full":
                layer_results[f"{lora_name}.diff"] = weight_diff.to(save_torch_dtype).cpu().contiguous()
                status = "full"
            else:
                lora_down, lora_up, _ = result
                layer_results[f"{lora_name}.lora_A.weight"] = lora_down.to(save_torch_dtype).cpu().contiguous()
                layer_results[f"{lora_name}.lora_B.weight"] = lora_up.to(save_torch_dtype).cpu().contiguous()
                status = "extracted"

            del weight_diff
            return status, layer_results

        writer = IncrementalSafetensorsWriter(output_path)
        writer.__enter__()
        try:
            for key in tqdm(weight_keys, desc="Extracting LoRA", unit="layers"):
                status, layer_sd = _process_layer(key)
                stats[status] += 1
                if layer_sd:
                    writer.write_dict(layer_sd)

                if force_clear_cache:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                pbar.update(1)
        finally:
            writer.__exit__(None, None, None)

        print(f"[LoRA Extract] Done: {stats['extracted']} extracted, {stats['chunked']} chunked, "
              f"{stats['full']} full, {stats['skipped']} skipped")

    finally:
        handler_a.__exit__(None, None, None)
        handler_b.__exit__(None, None, None)
        cleanup_after_operation()


def _build_lora_output_path(output_filename: str) -> str:
    """Build output path for LoRA file."""
    output_dir = folder_paths.get_folder_paths("loras")[0]
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{output_filename.strip()}.safetensors")


# =============================================================================
# Node Definitions
# =============================================================================

def _get_model_inputs():
    return [
        io.Combo.Input("model_a", options=folder_paths.get_filename_list("diffusion_models"),
                      tooltip="Finetuned model (A - B = LoRA)"),
        io.Combo.Input("model_b", options=folder_paths.get_filename_list("diffusion_models"),
                      tooltip="Base model (A - B = LoRA)"),
    ]


def _reconstruct_dots(key: str) -> str:
    """Reconstruct dot structure from underscored key using heuristics."""
    # Block indices: input_blocks_1 -> input_blocks.1
    blocks = [
        "input_blocks", "output_blocks", "middle_block",
        "transformer_blocks", "single_transformer_blocks",
        "double_blocks", "single_blocks",
        "down_blocks", "up_blocks", "mid_block",
        "attentions", "resnets", "upsamplers", "downsamplers"
    ]
    for b in blocks:
        key = re.sub(f"{b}_(\\d+)", f"{b}.\\1", key)

    # Sequential numbers: .1_1 -> .1.1
    key = re.sub(r"(\.\d+)_(\d+)", r"\1.\2", key)

    # Separators that should be preceded by dot
    separators = [
        "transformer_blocks", "single_transformer_blocks",
        "attn1", "attn2", "attn",
        "img_attn", "txt_attn",
        "to_q", "to_k", "to_v", "to_out",
        "q_proj", "k_proj", "v_proj", "out_proj",
        "qkv", "proj", "ff", "net",
        "norm1", "norm2", "norm3", "norm",
        "time_emb_proj"
    ]
    for s in separators:
        key = key.replace(f"_{s}", f".{s}")

    # Special case: to_out_0 -> to_out.0
    key = re.sub(r"to_out_(\d+)", r"to_out.\1", key)

    return key


def _format_lora_key(key: str) -> str:
    """
    Format the key for LoRA saving.
    Always standardizes to 'diffusion_model.' prefix for maximum compatibility.

    ComfyUI maps 'diffusion_model.<layer>' generically for all model types, making
    it universally compatible. The 'transformer.' prefix only works for a subset of
    model types via model-specific mapping code and is intentionally not used here.
    """
    if key.endswith(".weight"):
        key = key[:-7]

    # Handle ComfyUI Checkpoint wrapper (ldm/sgm format: model.diffusion_model.*)
    if key.startswith("model.diffusion_model."):
        inner_key = key[22:]  # strip len("model.diffusion_model.")
        return f"diffusion_model.{inner_key}"

    # Handle net.* wrapper
    if key.startswith("net."):
        inner_key = key[4:]  # strip len("net.")
        return f"diffusion_model.{inner_key}"

    # Handle direct Diffusers keys (transformer_blocks.* / single_transformer_blocks.* without prefix)
    if key.startswith("transformer_blocks") or key.startswith("single_transformer_blocks"):
        return f"diffusion_model.{key}"

    # Handle legacy lora_unet_ prefix (for resizing without base)
    if key.startswith("lora_unet_"):
        core = key[10:]
        dotted = _reconstruct_dots(core)
        return f"diffusion_model.{dotted}"

    # Handle already-prefixed keys
    if key.startswith("diffusion_model."):
        return key

    # Absorb now-invalid transformer. prefix into diffusion_model. for compatibility
    if key.startswith("transformer."):
        return f"diffusion_model.{key[12:]}"

    # Handle known Diffusers UNet blocks
    if any(key.startswith(p) for p in ["down_blocks", "up_blocks", "mid_block", "conv_in", "conv_out", "time_embedding", "class_embedding"]):
        return f"diffusion_model.{key}"

    # Default fallback
    return f"diffusion_model.{key}"


def _get_common_inputs():
    return [
        io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
        io.Boolean.Input("force_clear_cache", default=True, tooltip="Clear CUDA cache after each layer"),
        io.Boolean.Input("chunk_large_layers", default=False,
                        tooltip="Split large fused layers (QKV, MLP) into chunks"),
        io.Float.Input("clamp_quantile", default=0.99, min=0.5, max=1.0, step=0.01,
                      tooltip="Clamp outlier singular values"),
        io.Float.Input("min_diff", default=0.0, min=0.0, max=1.0, step=0.001,
                      tooltip="Skip layers with max difference below this"),
        io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
        io.String.Input("output_filename", default="extracted_lora"),
        io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
        io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
        io.String.Input("skip_patterns", default="", multiline=True,
                       tooltip="Patterns for layers to skip (regex or glob depending on glob_skip_patterns)"),
        io.Boolean.Input("glob_skip_patterns", default=False,
                        tooltip="When True, skip_patterns use glob syntax (* = any sequence, ? = any char, dots are literal). "
                                "When False (default), patterns are Python regex matched as substrings."),
    ]


class LoRAExtractFixed(io.ComfyNode):
    """Extract LoRA with fixed rank."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAExtractFixed",
            display_name="LoRA Extract (Fixed Rank)",
            category="ModelUtils/LoRA",
            description="Extract LoRA with specified fixed rank for each layer type.",
            inputs=[
                *_get_model_inputs(),
                io.Int.Input("linear_dim", default=64, min=1, max=16384,
                            tooltip="Rank for linear/attention layers"),
                io.Int.Input("conv_dim", default=32, min=1, max=16384,
                            tooltip="Rank for conv layers"),
                io.Int.Input("svd_niter", default=2, min=0, max=10,
                            tooltip="SVD power iterations (higher = more accurate but slower)"),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_dim, conv_dim, svd_niter, chunk_large_layers,
                clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_lora_from_files(
            model_a_path, model_b_path, "fixed", linear_dim, conv_dim,
            device, save_dtype, output_path, linear_dim, conv_dim,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers, svd_niter,
            lazy_load, force_clear_cache, glob_skip_patterns
        )

        return io.NodeOutput(output_path)


class LoRAExtractRatio(io.ComfyNode):
    """Extract LoRA with singular value ratio threshold."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAExtractRatio",
            display_name="LoRA Extract (Ratio)",
            category="ModelUtils/LoRA",
            description="Keep singular values > max(S) / ratio.",
            inputs=[
                *_get_model_inputs(),
                io.Float.Input("linear_ratio", default=2.0, min=1.0, max=100.0, step=0.1,
                              tooltip="Ratio threshold for linear layers (higher = more SVs kept)"),
                io.Float.Input("conv_ratio", default=2.0, min=1.0, max=100.0, step=0.1,
                              tooltip="Ratio threshold for conv layers (higher = more SVs kept)"),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384,
                            tooltip="Maximum rank for linear layers"),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384,
                            tooltip="Maximum rank for conv layers"),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_ratio, conv_ratio, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_lora_from_files(
            model_a_path, model_b_path, "ratio", linear_ratio, conv_ratio,
            device, save_dtype, output_path, linear_max_rank, conv_max_rank,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache,
            glob_skip_patterns=glob_skip_patterns
        )

        return io.NodeOutput(output_path)


class LoRAExtractQuantile(io.ComfyNode):
    """Extract LoRA with cumulative singular value quantile."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAExtractQuantile",
            display_name="LoRA Extract (Quantile)",
            category="ModelUtils/LoRA",
            description="Keep enough singular values to reach target percentage.",
            inputs=[
                *_get_model_inputs(),
                io.Float.Input("linear_quantile", default=0.9, min=0.0, max=1.0, step=0.01,
                              tooltip="Target cumulative % for linear layers"),
                io.Float.Input("conv_quantile", default=0.9, min=0.0, max=1.0, step=0.01,
                              tooltip="Target cumulative % for conv layers"),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384,
                            tooltip="Maximum rank for linear layers"),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384,
                            tooltip="Maximum rank for conv layers"),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_quantile, conv_quantile, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_lora_from_files(
            model_a_path, model_b_path, "quantile", linear_quantile, conv_quantile,
            device, save_dtype, output_path, linear_max_rank, conv_max_rank,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache,
            glob_skip_patterns=glob_skip_patterns
        )

        return io.NodeOutput(output_path)


class LoRAExtractKnee(io.ComfyNode):
    """Extract LoRA with automatic knee detection."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAExtractKnee",
            display_name="LoRA Extract (Knee Detection)",
            category="ModelUtils/LoRA",
            description="Automatically find optimal rank using knee detection on singular value curve.",
            inputs=[
                *_get_model_inputs(),
                io.Combo.Input("knee_method", options=["sv_knee", "sv_cumulative_knee"],
                              default="sv_knee", tooltip="Knee detection method"),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384,
                            tooltip="Maximum rank for linear layers"),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384,
                            tooltip="Maximum rank for conv layers"),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, knee_method, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_lora_from_files(
            model_a_path, model_b_path, knee_method, 0, 0,
            device, save_dtype, output_path, linear_max_rank, conv_max_rank,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache,
            glob_skip_patterns=glob_skip_patterns
        )

        return io.NodeOutput(output_path)


class LoRAExtractFrobenius(io.ComfyNode):
    """Extract LoRA preserving target fraction of Frobenius norm."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAExtractFrobenius",
            display_name="LoRA Extract (Frobenius)",
            category="ModelUtils/LoRA",
            description="Preserve target fraction of Frobenius norm.",
            inputs=[
                *_get_model_inputs(),
                io.Float.Input("linear_target", default=0.9, min=0.0, max=1.0, step=0.01,
                              tooltip="Target Frobenius norm fraction for linear"),
                io.Float.Input("conv_target", default=0.9, min=0.0, max=1.0, step=0.01,
                              tooltip="Target Frobenius norm fraction for conv"),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384,
                            tooltip="Maximum rank for linear layers"),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384,
                            tooltip="Maximum rank for conv layers"),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_target, conv_target, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_lora_from_files(
            model_a_path, model_b_path, "sv_fro", linear_target, conv_target,
            device, save_dtype, output_path, linear_max_rank, conv_max_rank,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache,
            glob_skip_patterns=glob_skip_patterns
        )

        return io.NodeOutput(output_path)
