"""
Text Encoder LoRA and DoRA Extraction.

Specifically targets extracting text encoder deltas as LoRA/DoRA with custom mapping rules
for layer name prefixes and PEFT weight formatting.
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
    cleanup_after_operation
)

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned, IncrementalSafetensorsWriter

# Import SVD rank utilities from standard lora extract file to avoid duplication
from .lora_extract_svd import (
    _compute_rank,
    _svd_extract_linear,
    _svd_extract_conv,
    _extract_chunked_layer,
    _detect_fused_layer,
    _compile_patterns,
    _matches_any_pattern
)


def _format_te_lora_key(key: str) -> str:
    """
    Format key specifically for Text Encoder LoRA/DoRA saving.
    Corrects layer name prefixing based on specific target mappings.
    """
    if key.endswith(".weight"):
        key = key[:-7]

    rules = [
        ("model.language_model.layers", "text_encoders.transformer.model.layers"),
        ("model.visual.blocks", "text_encoders.transformer.visual.blocks"),
        ("model.visual.deepstack_merger_list", "text_encoders.transformer.visual.deepstack_merger_list"),
        ("model.visual.merger", "text_encoders.transformer.visual.merger"),
        ("model.visual.pos_embed", "text_encoders.transformer.visual.pos_embed"),
        ("model.layers", "text_encoders.transformer.model.layers"),
        ("vision_model.embeddings.position_embedding", "text_encoders.transformer.vision_model.embeddings.position_embedding"),
        ("vision_model.encoder.layers", "text_encoders.transformer.vision_model.encoder.layers"),
    ]

    for prefix, target in rules:
        if key.startswith(prefix):
            return target + key[len(prefix):]

    # Prepend text_encoders. as default generic fallback
    if key.startswith("text_encoders."):
        return key
    return f"text_encoders.{key}"


def extract_te_from_files(
    model_a_path: str,
    model_b_path: str,
    mode: str,
    linear_param: float,
    conv_param: float,
    device: str,
    save_dtype: str,
    output_path: str,
    is_dora: bool = False,
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
    Extract LoRA/DoRA from difference between two Text Encoder models, writing incrementally to disk.
    """
    save_torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(save_dtype, torch.float16)

    skip_patterns = _compile_patterns(skip_patterns_str, glob_mode=glob_skip_patterns)

    # Prepare memory before heavy operation
    total_size_gb = estimate_model_size(model_a_path) + estimate_model_size(model_b_path)
    print(f"[TE Extract] Preparing memory for {total_size_gb:.2f}GB operation...")
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
            lora_name = _format_te_lora_key(key)

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
                    dora_scale = None
                    del tensor_b
                else:
                    # Skip 1D weights
                    if tensor_a.ndim < 2:
                        del tensor_a, tensor_b
                        return "skipped", None

                    if is_dora:
                        # Weight Decomposition calculation
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

                        # Compute directional diff
                        eps = 1e-8
                        norm_ratio = norm_b / (norm_a + eps)
                        if is_conv_dim:
                            norm_ratio = norm_ratio.view(-1, 1, 1, 1)
                            dora_scale = norm_a.view(-1, 1, 1, 1)
                        else:
                            dora_scale = norm_a

                        w_dir = tensor_a * norm_ratio
                        weight_diff = w_dir - tensor_b
                        del tensor_a, tensor_b, w_dir
                    else:
                        weight_diff = tensor_a - tensor_b
                        dora_scale = None
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
                        print(f"[TE Extract] Chunked: {key} ({num_chunks} chunks)")
                        lora_up, lora_down, rank = _extract_chunked_layer(
                            weight_diff, num_chunks, mode, linear_param, device, linear_max_rank
                        )
                        if lora_up is not None:
                            # PEFT suffixes
                            layer_results[f"{lora_name}.lora_B.weight"] = lora_up.to(save_torch_dtype).cpu().contiguous()
                            layer_results[f"{lora_name}.lora_A.weight"] = lora_down.to(save_torch_dtype).cpu().contiguous()
                            if is_dora and dora_scale is not None:
                                layer_results[f"{lora_name}.dora_scale"] = dora_scale.to(save_torch_dtype).cpu().contiguous()
                            del weight_diff
                            return "chunked", layer_results

                print(f"[TE Extract] Failed: {key}: {e}")
                del weight_diff
                return "skipped", None

            # Store result
            if mode_str == "full":
                layer_results[f"{lora_name}.diff"] = weight_diff.to(save_torch_dtype).cpu().contiguous()
                status = "full"
            else:
                lora_down, lora_up, _ = result
                # Write to PEFT weight suffixes
                layer_results[f"{lora_name}.lora_A.weight"] = lora_down.to(save_torch_dtype).cpu().contiguous()
                layer_results[f"{lora_name}.lora_B.weight"] = lora_up.to(save_torch_dtype).cpu().contiguous()
                if is_dora and dora_scale is not None:
                    layer_results[f"{lora_name}.dora_scale"] = dora_scale.to(save_torch_dtype).cpu().contiguous()
                status = "extracted"

            del weight_diff
            return status, layer_results

        writer = IncrementalSafetensorsWriter(output_path)
        writer.__enter__()
        try:
            for key in tqdm(weight_keys, desc="Extracting TE Layers", unit="layers"):
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

        print(f"[TE Extract] Done: {stats['extracted']} extracted, {stats['chunked']} chunked, "
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


def _get_te_model_inputs():
    return [
        io.Combo.Input("model_a", options=folder_paths.get_filename_list("text_encoders"),
                      tooltip="Finetuned Text Encoder model (A - B = LoRA)"),
        io.Combo.Input("model_b", options=folder_paths.get_filename_list("text_encoders"),
                      tooltip="Base Text Encoder model (A - B = LoRA)"),
    ]


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
        io.String.Input("output_filename", default="extracted_te_lora"),
        io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
        io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
        io.String.Input("skip_patterns", default="", multiline=True,
                       tooltip="Patterns for layers to skip (regex or glob depending on glob_skip_patterns)"),
        io.Boolean.Input("glob_skip_patterns", default=False,
                        tooltip="When True, skip_patterns use glob syntax (* = any sequence, ? = any char, dots are literal). "
                                "When False (default), patterns are Python regex matched as substrings."),
    ]


# =============================================================================
# LoRA Text Encoder Extract Nodes
# =============================================================================

class TextEncoderLoRAExtractFixed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderLoRAExtractFixed",
            display_name="TE LoRA Extract (Fixed Rank)",
            category="ModelUtils/LoRA Extract (TE)",
            description="Extract Text Encoder LoRA with specified fixed rank for each layer type.",
            inputs=[
                *_get_te_model_inputs(),
                io.Int.Input("linear_dim", default=64, min=1, max=16384, tooltip="Rank for linear/attention layers"),
                io.Int.Input("conv_dim", default=32, min=1, max=16384, tooltip="Rank for conv layers"),
                io.Int.Input("svd_niter", default=2, min=0, max=10, tooltip="SVD power iterations"),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_dim, conv_dim, svd_niter, chunk_large_layers,
                clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, "fixed", linear_dim, conv_dim,
            device, save_dtype, output_path, is_dora=False,
            linear_max_rank=linear_dim, conv_max_rank=conv_dim,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers, svd_niter=svd_niter,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


class TextEncoderLoRAExtractRatio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderLoRAExtractRatio",
            display_name="TE LoRA Extract (Ratio)",
            category="ModelUtils/LoRA Extract (TE)",
            description="Keep singular values > max(S) / ratio.",
            inputs=[
                *_get_te_model_inputs(),
                io.Float.Input("linear_ratio", default=2.0, min=1.0, max=100.0, step=0.1),
                io.Float.Input("conv_ratio", default=2.0, min=1.0, max=100.0, step=0.1),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_ratio, conv_ratio, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, "ratio", linear_ratio, conv_ratio,
            device, save_dtype, output_path, is_dora=False,
            linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


class TextEncoderLoRAExtractQuantile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderLoRAExtractQuantile",
            display_name="TE LoRA Extract (Quantile)",
            category="ModelUtils/LoRA Extract (TE)",
            description="Keep enough singular values to reach target percentage.",
            inputs=[
                *_get_te_model_inputs(),
                io.Float.Input("linear_quantile", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Float.Input("conv_quantile", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_quantile, conv_quantile, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, "quantile", linear_quantile, conv_quantile,
            device, save_dtype, output_path, is_dora=False,
            linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


class TextEncoderLoRAExtractKnee(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderLoRAExtractKnee",
            display_name="TE LoRA Extract (Knee)",
            category="ModelUtils/LoRA Extract (TE)",
            description="Automatically find optimal rank using knee detection on singular value curve.",
            inputs=[
                *_get_te_model_inputs(),
                io.Combo.Input("knee_method", options=["sv_knee", "sv_cumulative_knee"], default="sv_knee"),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, knee_method, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, knee_method, 0, 0,
            device, save_dtype, output_path, is_dora=False,
            linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


class TextEncoderLoRAExtractFrobenius(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderLoRAExtractFrobenius",
            display_name="TE LoRA Extract (Frobenius)",
            category="ModelUtils/LoRA Extract (TE)",
            description="Preserve target fraction of Frobenius norm.",
            inputs=[
                *_get_te_model_inputs(),
                io.Float.Input("linear_target", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Float.Input("conv_target", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_target, conv_target, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, "sv_fro", linear_target, conv_target,
            device, save_dtype, output_path, is_dora=False,
            linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


# =============================================================================
# DoRA Text Encoder Extract Nodes
# =============================================================================

class TextEncoderDoRAExtractFixed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderDoRAExtractFixed",
            display_name="TE DoRA Extract (Fixed Rank)",
            category="ModelUtils/DoRA Extract (TE)",
            description="Extract Text Encoder DoRA with specified fixed rank for each layer type.",
            inputs=[
                *_get_te_model_inputs(),
                io.Int.Input("linear_dim", default=64, min=1, max=16384, tooltip="Rank for linear/attention layers"),
                io.Int.Input("conv_dim", default=32, min=1, max=16384, tooltip="Rank for conv layers"),
                io.Int.Input("svd_niter", default=2, min=0, max=10, tooltip="SVD power iterations"),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_dim, conv_dim, svd_niter, chunk_large_layers,
                clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, "fixed", linear_dim, conv_dim,
            device, save_dtype, output_path, is_dora=True,
            linear_max_rank=linear_dim, conv_max_rank=conv_dim,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers, svd_niter=svd_niter,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


class TextEncoderDoRAExtractRatio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderDoRAExtractRatio",
            display_name="TE DoRA Extract (Ratio)",
            category="ModelUtils/DoRA Extract (TE)",
            description="Keep singular values > max(S) / ratio.",
            inputs=[
                *_get_te_model_inputs(),
                io.Float.Input("linear_ratio", default=2.0, min=1.0, max=100.0, step=0.1),
                io.Float.Input("conv_ratio", default=2.0, min=1.0, max=100.0, step=0.1),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_ratio, conv_ratio, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, "ratio", linear_ratio, conv_ratio,
            device, save_dtype, output_path, is_dora=True,
            linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


class TextEncoderDoRAExtractQuantile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderDoRAExtractQuantile",
            display_name="TE DoRA Extract (Quantile)",
            category="ModelUtils/DoRA Extract (TE)",
            description="Keep enough singular values to reach target percentage.",
            inputs=[
                *_get_te_model_inputs(),
                io.Float.Input("linear_quantile", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Float.Input("conv_quantile", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_quantile, conv_quantile, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, "quantile", linear_quantile, conv_quantile,
            device, save_dtype, output_path, is_dora=True,
            linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


class TextEncoderDoRAExtractKnee(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderDoRAExtractKnee",
            display_name="TE DoRA Extract (Knee)",
            category="ModelUtils/DoRA Extract (TE)",
            description="Automatically find optimal rank using knee detection on singular value curve.",
            inputs=[
                *_get_te_model_inputs(),
                io.Combo.Input("knee_method", options=["sv_knee", "sv_cumulative_knee"], default="sv_knee"),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, knee_method, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, knee_method, 0, 0,
            device, save_dtype, output_path, is_dora=True,
            linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)


class TextEncoderDoRAExtractFrobenius(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderDoRAExtractFrobenius",
            display_name="TE DoRA Extract (Frobenius)",
            category="ModelUtils/DoRA Extract (TE)",
            description="Preserve target fraction of Frobenius norm.",
            inputs=[
                *_get_te_model_inputs(),
                io.Float.Input("linear_target", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Float.Input("conv_target", default=0.9, min=0.0, max=1.0, step=0.01),
                io.Int.Input("linear_max_rank", default=128, min=1, max=16384),
                io.Int.Input("conv_max_rank", default=128, min=1, max=16384),
                *_get_common_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_target, conv_target, linear_max_rank, conv_max_rank,
                chunk_large_layers, clamp_quantile, min_diff, mismatch_mode, output_filename,
                save_dtype, device, skip_patterns, glob_skip_patterns, lazy_load, force_clear_cache) -> io.NodeOutput:

        model_a_path = folder_paths.get_full_path_or_raise("text_encoders", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("text_encoders", model_b)
        output_path = _build_lora_output_path(output_filename)

        extract_te_from_files(
            model_a_path, model_b_path, "sv_fro", linear_target, conv_target,
            device, save_dtype, output_path, is_dora=True,
            linear_max_rank=linear_max_rank, conv_max_rank=conv_max_rank,
            clamp_quantile=clamp_quantile, min_diff=min_diff, skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode, chunk_large_layers=chunk_large_layers,
            lazy_load=lazy_load, force_clear_cache=force_clear_cache, glob_skip_patterns=glob_skip_patterns
        )
        return io.NodeOutput(output_path)
