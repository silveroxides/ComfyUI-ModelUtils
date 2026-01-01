"""
LyCORIS-based LoRA extraction nodes.

Uses the LyCORIS library for advanced extraction modes with our memory-efficient tensor streaming.
Provides separate nodes for each extraction mode with mode-specific parameters.
"""
import os
import re
import torch
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from safetensors.torch import save_file
from .merger_utils import MemoryEfficientSafeOpen

try:
    from lycoris.utils import extract_linear, extract_conv
    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    print("[LyCORIS Extract] Warning: lycoris not installed. Install with: pip install lycoris-lora")


def _compile_patterns(pattern_string: str) -> list:
    """Compiles newline/whitespace-separated regex patterns into a list of compiled patterns."""
    if not pattern_string or not pattern_string.strip():
        return []
    
    patterns = []
    for pattern in pattern_string.split():
        pattern = pattern.strip()
        if not pattern:
            continue
        try:
            patterns.append(re.compile(pattern))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    return patterns


def _matches_any_pattern(key: str, patterns: list) -> bool:
    """Returns True if key matches any of the compiled regex patterns."""
    for pattern in patterns:
        if pattern.search(key):
            return True
    return False


def _detect_fused_layer(key: str, shape: tuple) -> int:
    """
    Detect if a layer is a fused QKV/KV/MLP layer and return the number of chunks.
    
    Returns:
        Number of chunks (2, 3, 4) or 1 if not a fused layer
    """
    if len(shape) != 2:
        return 1
    
    out_dim, in_dim = shape
    key_lower = key.lower()
    
    # QKV layers (3 chunks: Q, K, V)
    if "qkv" in key_lower:
        if out_dim % 3 == 0 and out_dim // 3 >= in_dim // 4:
            return 3
    
    # KV layers (2 chunks: K, V) - some architectures use this
    if "kv" in key_lower and "qkv" not in key_lower:
        if out_dim % 2 == 0:
            return 2
    
    # MLP layers with fused gate/up (2 chunks) - e.g., linear1 in Flux
    if "linear1" in key_lower or "gate_up" in key_lower or "mlp.0" in key_lower:
        if out_dim % 2 == 0 and out_dim >= in_dim * 2:
            return 2
    
    # Large layers that might benefit from chunking (heuristic)
    # If output dim is very large relative to input
    if out_dim > in_dim * 2 and out_dim % 2 == 0:
        return 2
    
    return 1


def _extract_chunked_layer(
    weight_diff: torch.Tensor,
    num_chunks: int,
    mode: str,
    mode_param: float,
    process_device: str,
    save_torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Extract LoRA from a fused layer by chunking, extracting each chunk, and recombining.
    
    Returns:
        (lora_up, lora_down, rank) concatenated from all chunks
    """
    out_dim, in_dim = weight_diff.shape
    chunk_size = out_dim // num_chunks
    
    all_lora_up = []
    all_lora_down = []
    
    for i in range(num_chunks):
        chunk = weight_diff[i * chunk_size:(i + 1) * chunk_size, :]
        
        # Extract this chunk with fixed mode (dynamic modes fail on large tensors)
        try:
            result, decompose_mode = extract_linear(
                chunk,
                mode=mode,
                mode_param=int(mode_param) if mode == "fixed" else mode_param,
                device=process_device,
            )
        except RuntimeError:
            # Fallback to fixed mode
            fallback_rank = min(64, min(chunk.shape[0], chunk.shape[1]) // 4)
            fallback_rank = max(1, fallback_rank)
            result, decompose_mode = extract_linear(
                chunk,
                mode="fixed",
                mode_param=fallback_rank,
                device=process_device,
            )
        
        if decompose_mode == "full":
            # Can't concatenate full diffs as LoRA, use as-is
            return None, None, 0
        
        extract_a, extract_b, _ = result
        all_lora_down.append(extract_a)  # [rank, in_dim]
        all_lora_up.append(extract_b)    # [chunk_size, rank]
    
    # Concatenate lora_up along output dimension
    # Each lora_up is [chunk_size, rank], concat to [out_dim, rank]
    combined_lora_up = torch.cat(all_lora_up, dim=0)
    
    # For lora_down, we need to handle differently
    # Each lora_down is [rank, in_dim] - we can average them or use the first
    # Using average maintains information from all chunks
    combined_lora_down = torch.stack(all_lora_down, dim=0).mean(dim=0)
    
    rank = combined_lora_down.shape[0]
    
    return combined_lora_up, combined_lora_down, rank


def lycoris_extract_from_files(
    model_a_path: str,
    model_b_path: str,
    mode: str,
    linear_param: float,
    conv_param: float,
    prefix_lora: str,
    process_device: str,
    save_dtype: str,
    skip_patterns_str: str = "",
    mismatch_mode: str = "skip",
    use_cp: bool = True,
    use_sparse_bias: bool = False,
    sparsity: float = 0.98,
    chunk_fused_layers: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Memory-efficient LyCORIS extraction by streaming tensors from safetensors files.
    
    Args:
        model_a_path: Path to the finetuned/modified model
        model_b_path: Path to the base/original model  
        mode: Extraction mode - "fixed", "threshold", "ratio", "quantile", "full"
        linear_param: Mode parameter for linear layers
        conv_param: Mode parameter for conv layers
        prefix_lora: Prefix for LoRA keys
        process_device: Device for processing
        save_dtype: Output dtype
        skip_patterns_str: Regex patterns for layers to skip
        mismatch_mode: How to handle mismatches
        use_cp: Use CP decomposition for 3x3 convolutions
        use_sparse_bias: Enable sparse bias storage
        sparsity: Sparsity threshold for sparse bias (0-1)
        chunk_fused_layers: Split large fused layers (QKV, MLP) into chunks for extraction
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError("lycoris library not installed. Install with: pip install lycoris-lora")
    
    output_sd = {}
    
    save_torch_dtype = {
        "fp32": torch.float32, 
        "fp16": torch.float16, 
        "bf16": torch.bfloat16
    }.get(save_dtype, torch.float16)
    
    skip_patterns = _compile_patterns(skip_patterns_str)
    
    handler_a = MemoryEfficientSafeOpen(model_a_path)
    handler_b = MemoryEfficientSafeOpen(model_b_path)
    
    try:
        keys_a = set(handler_a.keys())
        keys_b = set(handler_b.keys())
        
        all_weight_keys = [k for k in keys_a if k.endswith(".weight")]
        
        missing_in_b = keys_a - keys_b
        extra_in_b = keys_b - keys_a
        if missing_in_b:
            print(f"[LyCORIS Extract] {len(missing_in_b)} keys in model A not in model B")
        if extra_in_b:
            print(f"[LyCORIS Extract] {len(extra_in_b)} keys in model B not in model A (ignored)")
        
        pbar = comfy.utils.ProgressBar(len(all_weight_keys))
        
        skipped_by_pattern = 0
        skipped_by_mismatch = 0
        skipped_1d_weights = 0
        shape_mismatch_keys = 0
        full_weights = 0
        low_rank_weights = 0
        
        for key in tqdm(all_weight_keys, desc="Extracting LyCORIS weights", unit="layers"):
            lora_key = key[:-7]
            lora_name = prefix_lora + lora_key.replace(".", "_")
            
            if _matches_any_pattern(key, skip_patterns):
                skipped_by_pattern += 1
                pbar.update(1)
                continue
            
            if key not in keys_b:
                if mismatch_mode == "error":
                    raise ValueError(f"Key '{key}' not found in model B")
                elif mismatch_mode == "skip":
                    skipped_by_mismatch += 1
                    pbar.update(1)
                    continue
                tensor_a = handler_a.get_tensor(key).to(device=process_device, dtype=torch.float32)
                weight_diff = tensor_a
                del tensor_a
            else:
                tensor_a = handler_a.get_tensor(key).to(device=process_device, dtype=torch.float32)
                tensor_b = handler_b.get_tensor(key).to(device=process_device, dtype=torch.float32)
                
                if tensor_a.shape != tensor_b.shape:
                    del tensor_a, tensor_b
                    if mismatch_mode == "error":
                        raise ValueError(f"Shape mismatch for '{key}'")
                    elif mismatch_mode == "skip":
                        shape_mismatch_keys += 1
                        pbar.update(1)
                        continue
                    tensor_a = handler_a.get_tensor(key).to(device=process_device, dtype=torch.float32)
                    weight_diff = tensor_a
                    del tensor_a
                else:
                    weight_diff = tensor_a - tensor_b
                    del tensor_a, tensor_b
            
            if weight_diff.ndim < 2:
                skipped_1d_weights += 1
                del weight_diff
                pbar.update(1)
                continue
            
            is_conv = weight_diff.ndim == 4
            is_linear_conv = is_conv and weight_diff.shape[2] == 1 and weight_diff.shape[3] == 1
            
            try:
                if is_conv:
                    mode_param = linear_param if is_linear_conv else conv_param
                    result, decompose_mode = extract_conv(
                        weight_diff,
                        mode=mode,
                        mode_param=int(mode_param) if mode == "fixed" else mode_param,
                        device=process_device,
                        is_cp=use_cp and not is_linear_conv,
                    )
                else:
                    result, decompose_mode = extract_linear(
                        weight_diff,
                        mode=mode,
                        mode_param=int(linear_param) if mode == "fixed" else linear_param,
                        device=process_device,
                    )
            except Exception as e:
                error_str = str(e).lower()
                if "quantile" in error_str and "too large" in error_str:
                    # Check if we should use chunked extraction
                    if chunk_fused_layers and not is_conv:
                        num_chunks = _detect_fused_layer(key, weight_diff.shape)
                        if num_chunks > 1:
                            print(f"[LyCORIS Extract] Chunked extraction for {key}: {num_chunks} chunks")
                            try:
                                lora_up, lora_down, rank = _extract_chunked_layer(
                                    weight_diff, num_chunks, mode, linear_param,
                                    process_device, save_torch_dtype
                                )
                                if lora_up is not None:
                                    output_sd[f"{lora_name}.lora_up.weight"] = lora_up.to(save_torch_dtype).cpu()
                                    output_sd[f"{lora_name}.lora_down.weight"] = lora_down.to(save_torch_dtype).cpu()
                                    output_sd[f"{lora_name}.alpha"] = torch.tensor([rank]).to(save_torch_dtype)
                                    low_rank_weights += 1
                                    del weight_diff
                                    pbar.update(1)
                                    continue
                            except Exception as e_chunk:
                                print(f"[LyCORIS Extract] Chunked extraction failed: {e_chunk}")
                    
                    # Fallback: use fixed mode with reasonable rank for large tensors
                    fallback_rank = min(64, min(weight_diff.shape[0], weight_diff.shape[1]) // 4)
                    fallback_rank = max(1, fallback_rank)
                    print(f"[LyCORIS Extract] Large tensor fallback for {key}: using fixed rank {fallback_rank}")
                    try:
                        if is_conv:
                            result, decompose_mode = extract_conv(
                                weight_diff,
                                mode="fixed",
                                mode_param=fallback_rank,
                                device=process_device,
                                is_cp=use_cp and not is_linear_conv,
                            )
                        else:
                            result, decompose_mode = extract_linear(
                                weight_diff,
                                mode="fixed",
                                mode_param=fallback_rank,
                                device=process_device,
                            )
                    except Exception as e2:
                        print(f"[LyCORIS Extract] Could not extract for {key}: {e2}")
                        del weight_diff
                        pbar.update(1)
                        continue
                else:
                    print(f"[LyCORIS Extract] Could not extract for {key}: {e}")
                    del weight_diff
                    pbar.update(1)
                    continue
            
            # Process extraction result
            if decompose_mode == "full":
                output_sd[f"{lora_name}.diff"] = weight_diff.to(save_torch_dtype).cpu()
                full_weights += 1
            elif decompose_mode == "low rank":
                extract_a, extract_b, diff = result
                
                # CP decomposition for 3x3 convs
                if is_conv and not is_linear_conv and use_cp:
                    if extract_a.ndim == 4 and extract_a.shape[2] > 1:
                        dim = extract_a.size(0)
                        try:
                            (extract_c, extract_a_new, _), _ = extract_conv(
                                extract_a.transpose(0, 1),
                                "fixed",
                                dim,
                                process_device,
                                True,
                            )
                            extract_a_new = extract_a_new.transpose(0, 1)
                            extract_c = extract_c.transpose(0, 1)
                            output_sd[f"{lora_name}.lora_mid.weight"] = extract_c.to(save_torch_dtype).cpu()
                            extract_a = extract_a_new
                        except Exception:
                            pass  # Skip CP decomposition on error
                
                output_sd[f"{lora_name}.lora_down.weight"] = extract_a.to(save_torch_dtype).cpu()
                output_sd[f"{lora_name}.lora_up.weight"] = extract_b.to(save_torch_dtype).cpu()
                output_sd[f"{lora_name}.alpha"] = torch.tensor([extract_a.shape[0]]).to(save_torch_dtype)
                
                # Sparse bias storage
                if use_sparse_bias and diff is not None:
                    try:
                        diff_flat = diff.detach().cpu().reshape(extract_b.size(0), -1)
                        abs_diff = torch.abs(diff_flat)
                        # Use numpy for large tensor quantile
                        if abs_diff.numel() > 10_000_000:
                            import numpy as np
                            threshold = float(np.quantile(abs_diff.numpy().flatten(), sparsity))
                        else:
                            threshold = float(torch.quantile(abs_diff, sparsity))
                        sparse_diff = diff_flat.masked_fill(abs_diff < threshold, 0).to_sparse().coalesce()
                        
                        if sparse_diff._nnz() > 0:
                            output_sd[f"{lora_name}.bias_indices"] = sparse_diff.indices().to(torch.int16)
                            output_sd[f"{lora_name}.bias_values"] = sparse_diff.values().to(save_torch_dtype)
                            output_sd[f"{lora_name}.bias_size"] = torch.tensor(diff_flat.shape).to(torch.int16)
                    except Exception:
                        pass  # Skip sparse bias on error
                
                low_rank_weights += 1
                del extract_a, extract_b, diff
            
            del weight_diff
            pbar.update(1)
        
        print(f"[LyCORIS Extract] Extracted {low_rank_weights} low-rank, {full_weights} full weights")
        if skipped_by_pattern > 0:
            print(f"[LyCORIS Extract] Skipped {skipped_by_pattern} keys (matched skip patterns)")
        if skipped_by_mismatch > 0:
            print(f"[LyCORIS Extract] Skipped {skipped_by_mismatch} keys (missing in model B)")
        if shape_mismatch_keys > 0:
            print(f"[LyCORIS Extract] Skipped {shape_mismatch_keys} keys (shape mismatch)")
        if skipped_1d_weights > 0:
            print(f"[LyCORIS Extract] Skipped {skipped_1d_weights} 1D weights")
    
    finally:
        handler_a.__exit__(None, None, None)
        handler_b.__exit__(None, None, None)
    
    return output_sd


def _save_lycoris(output_sd: dict, output_filename: str) -> str:
    """Save LyCORIS state dict to loras folder."""
    if not output_sd:
        raise ValueError("No LyCORIS weights extracted - models may be identical or incompatible")
    
    output_dir = folder_paths.get_folder_paths("loras")[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")
    
    save_file(output_sd, output_path)
    print(f"[LyCORIS Extract] Saved {len(output_sd)} tensors to {output_path}")
    
    return output_path


# Common inputs shared by all extraction nodes
def _get_common_inputs():
    return [
        io.Combo.Input(
            "model_a", 
            options=folder_paths.get_filename_list("diffusion_models"),
            tooltip="The finetuned/modified model (A - B = LoRA)"
        ),
        io.Combo.Input(
            "model_b", 
            options=folder_paths.get_filename_list("diffusion_models"),
            tooltip="The base/original model (A - B = LoRA)"
        ),
    ]


def _get_common_optional_inputs():
    return [
        io.Boolean.Input("use_cp", default=True,
                        tooltip="Use CP decomposition for 3x3 convolutions (reduces file size)"),
        io.Boolean.Input("use_sparse_bias", default=False,
                        tooltip="Enable sparse bias storage for residual error compensation"),
        io.Float.Input("sparsity", default=0.98, min=0.0, max=1.0, step=0.01,
                      tooltip="Sparsity threshold for sparse bias (higher = more sparse, smaller file)"),
        io.Boolean.Input("chunk_fused_layers", default=True,
                        tooltip="Split large fused layers (QKV, MLP) into chunks for extraction. "
                                "Helps with large models like Flux. Disable for simple fixed-rank fallback."),
        io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip",
                      tooltip="How to handle missing keys or shape mismatches"),
        io.String.Input("output_filename", default="extracted_lycoris",
                       tooltip="Output filename (without extension)"),
        io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
        io.Combo.Input("process_device", options=["cuda", "cpu"], default="cuda",
                      tooltip="Device for SVD computation"),
        io.String.Input("skip_patterns", default="", multiline=True,
                       tooltip="Regex patterns for layers to skip (whitespace-separated)"),
    ]


class LyCORISExtractFixed(io.ComfyNode):
    """Extract LyCORIS with fixed rank for linear and conv layers."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LyCORISExtractFixed",
            display_name="LyCORIS Extract (Fixed Rank)",
            category="ModelUtils/LoRA",
            description="Extracts LyCORIS with fixed LoRA rank. "
                        "Specify separate ranks for linear and conv layers.",
            inputs=[
                *_get_common_inputs(),
                io.Int.Input("linear_dim", default=64, min=1, max=4096, step=1,
                            tooltip="LoRA rank for linear/attention layers"),
                io.Int.Input("conv_dim", default=32, min=1, max=4096, step=1,
                            tooltip="LoRA rank for conv layers"),
                *_get_common_optional_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_dim, conv_dim, use_cp, use_sparse_bias,
                sparsity, chunk_fused_layers, mismatch_mode, output_filename, save_dtype,
                process_device, skip_patterns) -> io.NodeOutput:
        if not LYCORIS_AVAILABLE:
            raise ImportError("lycoris library not installed. Install with: pip install lycoris-lora")
        
        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        
        output_sd = lycoris_extract_from_files(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            mode="fixed",
            linear_param=linear_dim,
            conv_param=conv_dim,
            prefix_lora="lora_unet_",
            process_device=process_device,
            save_dtype=save_dtype,
            skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode,
            use_cp=use_cp,
            use_sparse_bias=use_sparse_bias,
            sparsity=sparsity,
            chunk_fused_layers=chunk_fused_layers,
        )
        
        return io.NodeOutput(_save_lycoris(output_sd, output_filename))


class LyCORISExtractThreshold(io.ComfyNode):
    """Extract LyCORIS with singular value threshold."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LyCORISExtractThreshold",
            display_name="LyCORIS Extract (Threshold)",
            category="ModelUtils/LoRA",
            description="Extracts LyCORIS using singular value threshold. "
                        "Ranks are determined dynamically based on threshold.",
            inputs=[
                *_get_common_inputs(),
                io.Float.Input("linear_threshold", default=0.01, min=0.0, max=100.0, step=0.001,
                              tooltip="Singular value threshold for linear layers (absolute value)"),
                io.Float.Input("conv_threshold", default=0.01, min=0.0, max=100.0, step=0.001,
                              tooltip="Singular value threshold for conv layers (absolute value)"),
                *_get_common_optional_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_threshold, conv_threshold, use_cp, 
                use_sparse_bias, sparsity, chunk_fused_layers, mismatch_mode, output_filename,
                save_dtype, process_device, skip_patterns) -> io.NodeOutput:
        if not LYCORIS_AVAILABLE:
            raise ImportError("lycoris library not installed. Install with: pip install lycoris-lora")
        
        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        
        output_sd = lycoris_extract_from_files(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            mode="threshold",
            linear_param=linear_threshold,
            conv_param=conv_threshold,
            prefix_lora="lora_unet_",
            process_device=process_device,
            save_dtype=save_dtype,
            skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode,
            use_cp=use_cp,
            use_sparse_bias=use_sparse_bias,
            sparsity=sparsity,
            chunk_fused_layers=chunk_fused_layers,
        )
        
        return io.NodeOutput(_save_lycoris(output_sd, output_filename))


class LyCORISExtractRatio(io.ComfyNode):
    """Extract LyCORIS with singular value ratio."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LyCORISExtractRatio",
            display_name="LyCORIS Extract (Ratio)",
            category="ModelUtils/LoRA",
            description="Extracts LyCORIS using singular value ratio. "
                        "Includes singular values >= ratio * max_singular_value.",
            inputs=[
                *_get_common_inputs(),
                io.Float.Input("linear_ratio", default=0.5, min=0.0, max=1.0, step=0.01,
                              tooltip="Ratio of max singular value for linear layers (0-1)"),
                io.Float.Input("conv_ratio", default=0.5, min=0.0, max=1.0, step=0.01,
                              tooltip="Ratio of max singular value for conv layers (0-1)"),
                *_get_common_optional_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_ratio, conv_ratio, use_cp, use_sparse_bias,
                sparsity, chunk_fused_layers, mismatch_mode, output_filename, save_dtype,
                process_device, skip_patterns) -> io.NodeOutput:
        if not LYCORIS_AVAILABLE:
            raise ImportError("lycoris library not installed. Install with: pip install lycoris-lora")
        
        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        
        output_sd = lycoris_extract_from_files(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            mode="ratio",
            linear_param=linear_ratio,
            conv_param=conv_ratio,
            prefix_lora="lora_unet_",
            process_device=process_device,
            save_dtype=save_dtype,
            skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode,
            use_cp=use_cp,
            use_sparse_bias=use_sparse_bias,
            sparsity=sparsity,
            chunk_fused_layers=chunk_fused_layers,
        )
        
        return io.NodeOutput(_save_lycoris(output_sd, output_filename))


class LyCORISExtractQuantile(io.ComfyNode):
    """Extract LyCORIS with cumulative singular value quantile."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LyCORISExtractQuantile",
            display_name="LyCORIS Extract (Quantile)",
            category="ModelUtils/LoRA",
            description="Extracts LyCORIS using cumulative singular value quantile. "
                        "Includes enough singular values to reach the specified percentage of total.",
            inputs=[
                *_get_common_inputs(),
                io.Float.Input("linear_quantile", default=0.75, min=0.0, max=1.0, step=0.01,
                              tooltip="Cumulative quantile for linear layers (0-1, e.g. 0.75 = 75% of variance)"),
                io.Float.Input("conv_quantile", default=0.75, min=0.0, max=1.0, step=0.01,
                              tooltip="Cumulative quantile for conv layers (0-1, e.g. 0.75 = 75% of variance)"),
                *_get_common_optional_inputs(),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, linear_quantile, conv_quantile, use_cp, 
                use_sparse_bias, sparsity, chunk_fused_layers, mismatch_mode, output_filename,
                save_dtype, process_device, skip_patterns) -> io.NodeOutput:
        if not LYCORIS_AVAILABLE:
            raise ImportError("lycoris library not installed. Install with: pip install lycoris-lora")
        
        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        
        output_sd = lycoris_extract_from_files(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            mode="quantile",
            linear_param=linear_quantile,
            conv_param=conv_quantile,
            prefix_lora="lora_unet_",
            process_device=process_device,
            save_dtype=save_dtype,
            skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode,
            use_cp=use_cp,
            use_sparse_bias=use_sparse_bias,
            sparsity=sparsity,
            chunk_fused_layers=chunk_fused_layers,
        )
        
        return io.NodeOutput(_save_lycoris(output_sd, output_filename))


class LyCORISExtractFull(io.ComfyNode):
    """Extract full weight differences (no decomposition)."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LyCORISExtractFull",
            display_name="LyCORIS Extract (Full Diff)",
            category="ModelUtils/LoRA",
            description="Extracts full weight differences without SVD decomposition. "
                        "Larger file size but perfect reconstruction.",
            inputs=[
                *_get_common_inputs(),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip",
                              tooltip="How to handle missing keys or shape mismatches"),
                io.String.Input("output_filename", default="extracted_full_diff",
                               tooltip="Output filename (without extension)"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("process_device", options=["cuda", "cpu"], default="cuda",
                              tooltip="Device for computation"),
                io.String.Input("skip_patterns", default="", multiline=True,
                               tooltip="Regex patterns for layers to skip"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_a, model_b, mismatch_mode, output_filename, save_dtype,
                process_device, skip_patterns) -> io.NodeOutput:
        if not LYCORIS_AVAILABLE:
            raise ImportError("lycoris library not installed. Install with: pip install lycoris-lora")
        
        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        
        output_sd = lycoris_extract_from_files(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            mode="full",
            linear_param=0,
            conv_param=0,
            prefix_lora="lora_unet_",
            process_device=process_device,
            save_dtype=save_dtype,
            skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode,
            use_cp=False,
            use_sparse_bias=False,
            sparsity=0.0,
        )
        
        return io.NodeOutput(_save_lycoris(output_sd, output_filename))
