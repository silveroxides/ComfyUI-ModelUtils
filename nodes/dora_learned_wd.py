"""
Learned DoRA Extraction via Optimization.

Extracts Weight-Decomposed Low-Rank Adapters using gradient descent optimization
to minimize the reconstruction MSE loss, matching the true non-linear DoRA formulation.
Uses analytical SVD purely as the initialization seed.
"""
import fnmatch
import os
import re
import math
import torch
import torch.linalg as linalg
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from .device_utils import (
    estimate_model_size, prepare_for_large_operation,
    cleanup_after_operation
)

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned, IncrementalSafetensorsWriter

# Import the existing SV ranking and chunking logic so we don't have to duplicate it
from .dora_extract_wd import (
    _compute_rank, _svd_extract_linear_lowrank, _svd_extract_linear,
    _svd_extract_conv, _detect_fused_layer, _compile_patterns,
    _matches_any_pattern, _build_lora_output_path, _get_model_inputs,
    _format_lora_key, _get_common_inputs
)


def _adaptive_lr_update_cosine(curr_lr: float, improved: bool, worse_loss_counter: int, iteration: int, tensor_shape: tuple, min_lr: float, early_stop_stall: int, lr_factor: float = 0.95, lr_cooldown: int = 1) -> tuple:
    M, N = tensor_shape
    shape_ratio = abs(M - N) / max(M, N) if max(M, N) > 0 else 0

    t = min(worse_loss_counter / max(early_stop_stall, 1), 1.0)
    u_factor = (1 + math.cos(2 * math.pi * t)) / 2

    if improved:
        base_boost = 1.25
        distance = base_boost - 1.0
        scaled_distance = distance * (1.0 + 0.5 * shape_ratio)
        boost_mult = 1.0 + scaled_distance * (1.0 - u_factor)
        return min(curr_lr * boost_mult, 100.0), True
    else:
        cooldown = max(lr_cooldown, 1)
        if iteration % cooldown != 0:
            return curr_lr, False

        min_decay = lr_factor - 0.03 * shape_ratio
        max_decay = 0.995
        decay_mult = min_decay + (max_decay - min_decay) * u_factor
        return max(curr_lr * decay_mult, min_lr), True


def _compute_shape_aware_plateau_params(M: int, N: int, lr_patience: int, lr_factor: float, lr_cooldown: int, lr_shape_influence: float = 1.0) -> tuple:
    if lr_shape_influence > 0:
        aspect_ratio = max(M, N) / min(M, N) if min(M, N) > 0 else 1.0
        ar_factor = math.sqrt(aspect_ratio)
        blend = lr_shape_influence

        effective_patience = lr_patience
        raw_factor = lr_factor
        aggressive_factor = raw_factor**ar_factor
        effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend
        effective_cooldown = lr_cooldown
    else:
        effective_patience = lr_patience
        effective_factor = lr_factor
        effective_cooldown = lr_cooldown

    return effective_patience, effective_factor, effective_cooldown


def _optimize_dora(
    tensor_base: torch.Tensor,
    tensor_ft: torch.Tensor,
    init_down: torch.Tensor,
    init_up: torch.Tensor,
    optimize_iters: int,
    learning_rate: float,
    optimizer_type: str = "prodigy",
    lr_schedule: str = "plateau",
    lr_patience: int = 2,
    lr_factor: float = 0.9,
    lr_cooldown: int = 2,
    early_stop_loss: float = 1e-6,
    early_stop_stall: int = 2000,
    early_stop_lr: float = 9.01e-9,
    layer_name: str = "Layer",
) -> tuple:
    """
    Run gradient descent to optimize lora_down and lora_up against the actual DoRA formula.

    Args:
        tensor_base: Base model weight
        tensor_ft: Finetuned model weight
        init_down: SVD-initialized lora_down matrix
        init_up: SVD-initialized lora_up matrix
        optimize_iters: Number of iterations for AdamW
        learning_rate: Learning rate for AdamW
        optimizer_type: Optimizer choice (adamw, radam, prodigy)
        lr_schedule: LR Schedule choice (exponential, plateau, adaptive)
        early_stop_loss: Target loss to stop optimizing
        early_stop_stall: Number of non-improving iterations before stopping

    Returns:
        (optimized_down, optimized_up)
    """
    if optimize_iters <= 0:
        return init_down, init_up

    is_conv = tensor_base.ndim == 4
    if is_conv:
        out_ch, in_ch, kh, kw = tensor_base.shape
        flat_base = tensor_base.reshape(out_ch, -1)
        flat_ft = tensor_ft.reshape(out_ch, -1)

        # Flatten init matrices for matmul: (out, rank) @ (rank, in*k*k)
        rank = init_down.shape[0]
        flat_up = init_up.reshape(out_ch, rank)
        flat_down = init_down.reshape(rank, -1)
    else:
        flat_base = tensor_base
        flat_ft = tensor_ft
        flat_up = init_up
        flat_down = init_down

    # Magnitude of the finetuned model (fixed DoRA scale)
    m = torch.linalg.norm(flat_ft, dim=1, keepdim=True).detach()
    eps = 1e-8

    # Set up learnable parameters
    with torch.inference_mode(False), torch.enable_grad():
        # Clone inference tensors to normal tensors for autograd
        flat_base_opt = flat_base.clone()
        flat_ft_opt = flat_ft.clone()
        m_opt = m.clone()

        # Free original tensors to save memory
        del flat_base, flat_ft, m

        B = torch.nn.Parameter(flat_up.clone())
        A = torch.nn.Parameter(flat_down.clone())

        opt_type = optimizer_type.lower()
        if opt_type == "radam":
            optimizer = torch.optim.RAdam([B, A], lr=learning_rate)
        elif opt_type == "prodigy":
            try:
                from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
                optimizer = ProdigyPlusScheduleFree([B, A], lr=learning_rate, use_schedulefree=False)
            except ImportError:
                print("ProdigyPlusScheduleFree optimizer not found. Falling back to AdamW. To install: pip install prodigyplus")
                optimizer = torch.optim.AdamW([B, A], lr=learning_rate)
        else:
            optimizer = torch.optim.AdamW([B, A], lr=learning_rate)

        # Learning rate scheduling and early stopping state
        best_loss = float('inf')
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0
        current_lr = learning_rate
        schedule_type = lr_schedule.lower()

        M_dim, N_dim = flat_base_opt.shape[0], flat_base_opt.shape[1]
        eff_patience, eff_factor, eff_cooldown = _compute_shape_aware_plateau_params(M_dim, N_dim, lr_patience, lr_factor, lr_cooldown)

        # Optimization loop
        opt_pbar = tqdm(range(optimize_iters), desc=f"  └─ Optimizing {layer_name}", leave=False, dynamic_ncols=True, mininterval=0.5)
        for i in opt_pbar:
            optimizer.zero_grad()

            # 1. Reconstruct directional weight: W_dir = W_base + B @ A
            w_approx = flat_base_opt + (B @ A)

            # 2. Normalize directional weight
            approx_norm = torch.linalg.norm(w_approx, dim=1, keepdim=True) + eps

            # 3. Scale by target magnitude
            w_reconstructed = m_opt * (w_approx / approx_norm)

            # 4. Compute MSE loss against actual finetuned weight
            loss = torch.nn.functional.mse_loss(w_reconstructed, flat_ft_opt)
            loss_val = loss.item()

            if i == 0:
                initial_loss = loss_val
                best_loss = loss_val

            loss.backward()
            optimizer.step()

            # --- Early Stopping & Schedulers ---
            improved = False
            if loss_val < best_loss * 0.9999: # 0.01% improvement threshold
                best_loss = loss_val
                worse_loss_counter = 0
                plateau_counter = 0
                improved = True
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            if loss_val <= early_stop_loss:
                break

            if worse_loss_counter > early_stop_stall:
                break

            if current_lr <= early_stop_lr:
                break

            # LR Schedulers
            if schedule_type == "adaptive":
                current_lr, updated = _adaptive_lr_update_cosine(
                    current_lr, improved, worse_loss_counter, i, (M_dim, N_dim), early_stop_lr, early_stop_stall, lr_factor, lr_cooldown
                )
                if updated:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
            elif schedule_type == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= max(eff_patience, 1): # Decay if patience threshold reached
                    current_lr = max(current_lr * eff_factor, early_stop_lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    plateau_counter = 0 # reset counter after decay
                    cooldown_counter = eff_cooldown
            elif schedule_type == "exponential":
                current_lr = max(current_lr * 0.99, early_stop_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            if i % 10 == 0 or i == optimize_iters - 1:
                if schedule_type == "plateau":
                    opt_pbar.set_postfix({"loss": f"{loss_val:.4e}", "lr": f"{current_lr:.2e}", "plat": f"{plateau_counter}/{eff_patience}"}, refresh=False)
                else:
                    opt_pbar.set_postfix({"loss": f"{loss_val:.4e}", "lr": f"{current_lr:.2e}", "stall": f"{worse_loss_counter}"}, refresh=False)

        final_loss = loss.item()
        opt_pbar.close()
        print(f"  └─ {layer_name} | Loss: {initial_loss:.6f} -> {final_loss:.6f} (stopped at iter {i+1})")

        final_up = B.detach()
        final_down = A.detach()

    # Reshape back to original dimensions
    if is_conv:
        final_up = final_up.reshape(out_ch, rank, 1, 1)
        final_down = final_down.reshape(rank, in_ch, kh, kw)

    return final_down, final_up


def _extract_chunked_learned_layer(
    tensor_base: torch.Tensor,
    tensor_ft: torch.Tensor,
    num_chunks: int,
    mode: str,
    mode_param: float,
    device: str,
    max_rank: int = None,
    optimize_iters: int = 500,
    learning_rate: float = 1.0,
    optimizer_type: str = "prodigy",
    lr_schedule: str = "plateau",
    lr_patience: int = 2,
    lr_factor: float = 0.9,
    lr_cooldown: int = 2,
    early_stop_loss: float = 1e-6,
    early_stop_stall: int = 2000,
    early_stop_lr: float = 9.01e-9,
    layer_name: str = "Layer",
) -> tuple:
    """Extract Learned DoRA from fused layer by chunking."""
    out_dim, in_dim = tensor_base.shape
    chunk_size = out_dim // num_chunks

    all_lora_up = []
    all_lora_down = []

    for i in range(num_chunks):
        chunk_base = tensor_base[i * chunk_size:(i + 1) * chunk_size, :]
        chunk_ft = tensor_ft[i * chunk_size:(i + 1) * chunk_size, :]

        norm_base = torch.linalg.norm(chunk_base, dim=1, keepdim=True)
        norm_ft = torch.linalg.norm(chunk_ft, dim=1, keepdim=True)
        eps = 1e-8

        w_dir = chunk_ft * (norm_base / (norm_ft + eps))
        weight_diff = w_dir - chunk_base

        try:
            result, mode_str = _svd_extract_linear(weight_diff, mode, mode_param, device, max_rank)
            if mode_str == "full":
                return None, None, 0
            lora_down, lora_up, _ = result

            # Apply learned optimization
            lora_down, lora_up = _optimize_dora(
                chunk_base, chunk_ft, lora_down, lora_up, optimize_iters, learning_rate,
                optimizer_type, lr_schedule, lr_patience, lr_factor, lr_cooldown,
                early_stop_loss, early_stop_stall, early_stop_lr,
                layer_name=f"{layer_name} (Chunk {i+1}/{num_chunks})"
            )

            all_lora_down.append(lora_down)
            all_lora_up.append(lora_up)
        except Exception:
            return None, None, 0

    combined_lora_up = torch.cat(all_lora_up, dim=0)
    combined_lora_down = all_lora_down[0]
    rank = combined_lora_down.shape[0]

    return combined_lora_up, combined_lora_down, rank


def extract_dora_learned_from_files(
    model_a_path: str,
    model_b_path: str,
    mode: str,
    linear_param: float,
    conv_param: float,
    device: str,
    save_dtype: str,
    output_path: str,
    optimize_iters: int = 500,
    learning_rate: float = 1.0,
    optimizer_type: str = "prodigy",
    lr_schedule: str = "plateau",
    lr_patience: int = 2,
    lr_factor: float = 0.9,
    lr_cooldown: int = 2,
    early_stop_loss: float = 1e-6,
    early_stop_stall: int = 2000,
    early_stop_lr: float = 9.01e-9,
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
    Extract Learned DoRA from difference between two models.
    """
    save_torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(save_dtype, torch.float16)

    skip_patterns = _compile_patterns(skip_patterns_str, glob_mode=glob_skip_patterns)

    total_size_gb = estimate_model_size(model_a_path) + estimate_model_size(model_b_path)
    print(f"[Learned DoRA Extract] Preparing memory for {total_size_gb:.2f}GB operation...")
    prepare_for_large_operation(total_size_gb * 1.5, torch.device(device))

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

            use_pinned = device == 'cuda'

            if key not in keys_b:
                if mismatch_mode == "skip":
                    return "skipped", None
                if mismatch_mode == "error":
                    raise ValueError(f"Key {key} not found in model B")

                # Cannot extract true DoRA without base. Fallback to diff.
                cpu_a = handler_a.get_tensor(key)
                tensor_a = transfer_to_gpu_pinned(cpu_a, device, torch.float32) if use_pinned else cpu_a.to(device=device, dtype=torch.float32)
                del cpu_a
                weight_diff = tensor_a
                tensor_base = torch.zeros_like(tensor_a)
                tensor_ft = tensor_a
                dora_scale = None
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

                    weight_diff = tensor_a
                    tensor_base = torch.zeros_like(tensor_a)
                    tensor_ft = tensor_a
                    dora_scale = None
                    del tensor_b
                else:
                    tensor_base = tensor_b
                    tensor_ft = tensor_a

                    is_conv_dim = tensor_a.ndim == 4
                    if is_conv_dim:
                        out_ch = tensor_a.shape[0]
                        flat_a = tensor_a.reshape(out_ch, -1)
                        flat_b = tensor_b.reshape(out_ch, -1)
                        norm_a = torch.linalg.norm(flat_a, dim=1, keepdim=True)
                        norm_b = torch.linalg.norm(flat_b, dim=1, keepdim=True)
                    else:
                        norm_a = torch.linalg.norm(tensor_a, dim=1, keepdim=True)
                        norm_b = torch.linalg.norm(tensor_b, dim=1, keepdim=True)

                    eps = 1e-8
                    norm_ratio = norm_b / (norm_a + eps)
                    if is_conv_dim:
                        norm_ratio = norm_ratio.view(-1, 1, 1, 1)
                        dora_scale = norm_a.view(-1, 1, 1, 1)
                    else:
                        dora_scale = norm_a

                    w_dir = tensor_a * norm_ratio
                    weight_diff = w_dir - tensor_b
                    del w_dir

            if min_diff > 0 and weight_diff.abs().max() < min_diff:
                del weight_diff, tensor_base, tensor_ft
                return "skipped", None

            if weight_diff.ndim < 2:
                del weight_diff, tensor_base, tensor_ft
                return "skipped", None

            is_conv = weight_diff.ndim == 4
            layer_results = {}

            try:
                # 1. Run Analytical SVD (Initialization)
                if is_conv:
                    result, mode_str = _svd_extract_conv(
                        weight_diff, mode, conv_param, device, conv_max_rank, clamp_quantile
                    )
                else:
                    result, mode_str = _svd_extract_linear(
                        weight_diff, mode, linear_param, device, linear_max_rank, clamp_quantile, svd_niter
                    )

                # 2. Run Optimization (Learned Rounding)
                if mode_str != "full" and dora_scale is not None:
                    lora_down, lora_up, _ = result
                    lora_down, lora_up = _optimize_dora(
                        tensor_base, tensor_ft, lora_down, lora_up, optimize_iters, learning_rate,
                        optimizer_type, lr_schedule, lr_patience, lr_factor, lr_cooldown,
                        early_stop_loss, early_stop_stall, early_stop_lr,
                        layer_name=lora_name
                    )
                    result = (lora_down, lora_up, _)

            except Exception as e:
                if chunk_large_layers and not is_conv:
                    num_chunks = _detect_fused_layer(key, weight_diff.shape)
                    if num_chunks > 1:
                        print(f"[Learned DoRA] Chunked: {key} ({num_chunks} chunks)")
                        lora_up, lora_down, rank = _extract_chunked_learned_layer(
                            tensor_base, tensor_ft, num_chunks, mode, linear_param, device, linear_max_rank, optimize_iters, learning_rate,
                            optimizer_type, lr_schedule, lr_patience, lr_factor, lr_cooldown,
                            early_stop_loss, early_stop_stall, early_stop_lr,
                            layer_name=lora_name
                        )
                        if lora_up is not None:
                            layer_results[f"{lora_name}.lora_up.weight"] = lora_up.to(save_torch_dtype).cpu().contiguous()
                            layer_results[f"{lora_name}.lora_down.weight"] = lora_down.to(save_torch_dtype).cpu().contiguous()
                            layer_results[f"{lora_name}.alpha"] = torch.tensor(rank, dtype=save_torch_dtype)
                            if dora_scale is not None:
                                layer_results[f"{lora_name}.dora_scale"] = dora_scale.to(save_torch_dtype).cpu().contiguous()
                            del weight_diff, tensor_base, tensor_ft
                            return "chunked", layer_results

                print(f"[Learned DoRA] Failed: {key}: {e}")
                del weight_diff, tensor_base, tensor_ft
                return "skipped", None

            if mode_str == "full":
                layer_results[f"{lora_name}.diff"] = weight_diff.to(save_torch_dtype).cpu().contiguous()
                status = "full"
            else:
                lora_down, lora_up, _ = result
                rank = lora_down.shape[0]
                layer_results[f"{lora_name}.lora_down.weight"] = lora_down.to(save_torch_dtype).cpu().contiguous()
                layer_results[f"{lora_name}.lora_up.weight"] = lora_up.to(save_torch_dtype).cpu().contiguous()
                layer_results[f"{lora_name}.alpha"] = torch.tensor(rank, dtype=save_torch_dtype)
                if dora_scale is not None:
                    layer_results[f"{lora_name}.dora_scale"] = dora_scale.to(save_torch_dtype).cpu().contiguous()
                status = "extracted"

            del weight_diff, tensor_base, tensor_ft
            return status, layer_results

        writer = IncrementalSafetensorsWriter(output_path)
        writer.__enter__()
        try:
            for key in tqdm(weight_keys, desc="Extracting Learned DoRA", unit="layers"):
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

        print(f"[Learned DoRA] Done: {stats['extracted']} extracted, {stats['chunked']} chunked, "
              f"{stats['full']} full, {stats['skipped']} skipped")

    finally:
        handler_a.__exit__(None, None, None)
        handler_b.__exit__(None, None, None)
        cleanup_after_operation()


# =============================================================================
# Node Definitions
# =============================================================================

def _get_learned_inputs():
    return [
        io.Int.Input("optimize_iters", default=500, min=0, max=10000,
                    tooltip="Number of gradient descent iterations (0 = skip optimization)"),
        io.Float.Input("learning_rate", default=1.0, min=0.0001, max=100.0, step=0.001,
                      tooltip="Base learning rate. AdamW prefers ~0.01, Prodigy prefers ~1.0"),
        io.Combo.Input("optimizer", options=["prodigy", "adamw", "radam"], default="prodigy",
                      tooltip="Optimization algorithm"),
        io.Combo.Input("lr_schedule", options=["plateau", "adaptive", "exponential", "constant"], default="plateau",
                      tooltip="Learning rate scheduling strategy"),
        io.Int.Input("lr_patience", default=2, min=0, max=1000, tooltip="Steps to wait before decaying LR (Plateau)"),
        io.Float.Input("lr_factor", default=0.9, min=0.01, max=1.0, step=0.01, tooltip="Factor to decay LR by (Plateau)"),
        io.Int.Input("lr_cooldown", default=2, min=0, max=1000, tooltip="Steps to wait after decay before checking again (Plateau)"),
        io.Float.Input("early_stop_loss", default=1e-6, min=0.0, max=1.0, step=1e-9,
                      tooltip="Stop early if MSE loss drops below this value"),
        io.Int.Input("early_stop_stall", default=2000, min=0, max=10000,
                      tooltip="Stop early if loss doesn't improve for this many iterations"),
        io.Float.Input("early_stop_lr", default=9.01e-9, min=0.0, max=1.0, step=1e-10,
                      tooltip="Stop early if learning rate drops below this value"),
    ]


class DoRALearnedExtractFixed(io.ComfyNode):
    """Extract Learned DoRA with fixed rank."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DoRALearnedExtractFixed",
            display_name="DoRA Learned Extract (Fixed Rank) (WIP)",
            category="ModelUtils/DoRA",
            description="Extract DoRA via MSE Loss minimization with fixed rank.",
            is_experimental=True,
            inputs=[
                *_get_model_inputs(),
                io.Int.Input("linear_dim", default=64, min=1, max=16384, tooltip="Rank for linear/attention layers"),
                io.Int.Input("conv_dim", default=32, min=1, max=16384, tooltip="Rank for conv layers"),
                *_get_learned_inputs(),
                io.Int.Input("svd_niter", default=2, min=0, max=10, tooltip="SVD iterations for initialization"),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_dim, conv_dim, optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, svd_niter, chunk_large_layers,
                clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_dora_learned_from_files(
            model_a_path, model_b_path, "fixed", linear_dim, conv_dim,
            device, save_dtype, output_path, optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, linear_dim, conv_dim,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers, svd_niter,
            lazy_load, force_clear_cache, glob_skip_patterns
        )

        return io.NodeOutput(output_path)


class DoRALearnedExtractRatio(io.ComfyNode):
    """Extract Learned DoRA with singular value ratio threshold."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DoRALearnedExtractRatio",
            display_name="DoRA Learned Extract (Ratio) (WIP)",
            category="ModelUtils/DoRA",
            description="Learned extraction keeping SVs > max(S) / ratio.",
            is_experimental=True,
            inputs=[
                *_get_model_inputs(),
                io.Float.Input("linear_ratio", default=2.0, min=1.0, max=100.0, step=0.1),
                io.Float.Input("conv_ratio", default=2.0, min=1.0, max=100.0, step=0.1),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_learned_inputs(),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_ratio, conv_ratio, linear_max_rank, conv_max_rank,
                optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, chunk_large_layers, clamp_quantile, min_diff, mismatch_mode,
                output_filename, save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_dora_learned_from_files(
            model_a_path, model_b_path, "ratio", linear_ratio, conv_ratio,
            device, save_dtype, output_path, optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, linear_max_rank, conv_max_rank,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )

        return io.NodeOutput(output_path)


class DoRALearnedExtractQuantile(io.ComfyNode):
    """Extract Learned DoRA with cumulative singular value quantile."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DoRALearnedExtractQuantile",
            display_name="DoRA Learned Extract (Quantile) (WIP)",
            category="ModelUtils/DoRA",
            description="Learned extraction keeping enough SVs to reach target percentage.",
            is_experimental=True,
            inputs=[
                *_get_model_inputs(),
                io.Float.Input("linear_quantile", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Float.Input("conv_quantile", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_learned_inputs(),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_quantile, conv_quantile, linear_max_rank, conv_max_rank,
                optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, chunk_large_layers, clamp_quantile, min_diff, mismatch_mode,
                output_filename, save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_dora_learned_from_files(
            model_a_path, model_b_path, "quantile", linear_quantile, conv_quantile,
            device, save_dtype, output_path, optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, linear_max_rank, conv_max_rank,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )

        return io.NodeOutput(output_path)


class DoRALearnedExtractKnee(io.ComfyNode):
    """Extract Learned DoRA with automatic knee detection."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DoRALearnedExtractKnee",
            display_name="DoRA Learned Extract (Knee Detection) (WIP)",
            category="ModelUtils/DoRA",
            description="Learned extraction automatically finding optimal rank using knee detection.",
            is_experimental=True,
            inputs=[
                *_get_model_inputs(),
                io.Combo.Input("knee_method", options=["sv_knee", "sv_cumulative_knee"], default="sv_knee"),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_learned_inputs(),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, knee_method, linear_max_rank, conv_max_rank,
                optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, chunk_large_layers, clamp_quantile, min_diff, mismatch_mode,
                output_filename, save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_dora_learned_from_files(
            model_a_path, model_b_path, knee_method, 0, 0,
            device, save_dtype, output_path, optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, linear_max_rank, conv_max_rank,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )

        return io.NodeOutput(output_path)


class DoRALearnedExtractFrobenius(io.ComfyNode):
    """Extract Learned DoRA preserving target fraction of Frobenius norm."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DoRALearnedExtractFrobenius",
            display_name="DoRA Learned Extract (Frobenius) (WIP)",
            category="ModelUtils/DoRA",
            description="Learned extraction preserving target fraction of Frobenius norm.",
            is_experimental=True,
            inputs=[
                *_get_model_inputs(),
                io.Float.Input("linear_target", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Float.Input("conv_target", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_learned_inputs(),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_target, conv_target, linear_max_rank, conv_max_rank,
                optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, chunk_large_layers, clamp_quantile, min_diff, mismatch_mode,
                output_filename, save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_dora_learned_from_files(
            model_a_path, model_b_path, "sv_fro", linear_target, conv_target,
            device, save_dtype, output_path, optimize_iters, learning_rate, optimizer, lr_schedule, lr_patience, lr_factor, lr_cooldown, early_stop_loss, early_stop_stall, early_stop_lr, linear_max_rank, conv_max_rank,
            clamp_quantile, min_diff, skip_patterns, mismatch_mode, chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )

        return io.NodeOutput(output_path)
