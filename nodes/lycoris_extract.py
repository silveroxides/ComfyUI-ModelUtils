"""
LyCORIS-based LoRA extraction node.

Uses the LyCORIS library for advanced extraction modes (threshold, ratio, quantile)
with our memory-efficient tensor streaming.
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
    print("[LyCORIS Extract] Warning: lycoris not installed. Install with: pip install lycoris")


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
    """Returns True if key matches any of the compiled regex patterns (substring search)."""
    for pattern in patterns:
        if pattern.search(key):
            return True
    return False


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
) -> dict[str, torch.Tensor]:
    """
    Memory-efficient LyCORIS extraction by streaming tensors from safetensors files.
    
    Uses LyCORIS's extract_linear and extract_conv functions for proper decomposition.
    
    Args:
        model_a_path: Path to the finetuned/modified model
        model_b_path: Path to the base/original model  
        mode: Extraction mode - "fixed", "threshold", "ratio", "quantile", "full"
        linear_param: Mode parameter for linear layers (rank/threshold/ratio/quantile)
        conv_param: Mode parameter for conv layers (rank/threshold/ratio/quantile)
        prefix_lora: Prefix for LoRA keys (e.g., "lora_unet_")
        process_device: Device for processing ("cuda" or "cpu")
        save_dtype: Output dtype ("fp16", "bf16", "fp32")
        skip_patterns_str: Regex patterns for layers to skip
        mismatch_mode: "skip" (ignore), "zeros" (use model A), "error" (raise error)
        use_cp: Use CP decomposition for 3x3 convolutions
    
    Returns:
        Dictionary of LyCORIS tensors
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError("lycoris library not installed. Install with: pip install lycoris")
    
    output_sd = {}
    
    # Get dtype for saving
    save_torch_dtype = {
        "fp32": torch.float32, 
        "fp16": torch.float16, 
        "bf16": torch.bfloat16
    }.get(save_dtype, torch.float16)
    
    # Compile patterns
    skip_patterns = _compile_patterns(skip_patterns_str)
    
    # Open both files with memory-efficient handler
    handler_a = MemoryEfficientSafeOpen(model_a_path)
    handler_b = MemoryEfficientSafeOpen(model_b_path)
    
    try:
        keys_a = set(handler_a.keys())
        keys_b = set(handler_b.keys())
        
        # Get all weight keys from model A (primary)
        all_weight_keys = [k for k in keys_a if k.endswith(".weight")]
        
        # Log key differences
        missing_in_b = keys_a - keys_b
        extra_in_b = keys_b - keys_a
        if missing_in_b:
            print(f"[LyCORIS Extract] {len(missing_in_b)} keys in model A not in model B")
        if extra_in_b:
            print(f"[LyCORIS Extract] {len(extra_in_b)} keys in model B not in model A (ignored)")
        
        pbar = comfy.utils.ProgressBar(len(all_weight_keys))
        
        # Stats
        skipped_by_pattern = 0
        skipped_by_mismatch = 0
        skipped_1d_weights = 0
        shape_mismatch_keys = 0
        full_weights = 0
        low_rank_weights = 0
        
        for key in tqdm(all_weight_keys, desc="Extracting LyCORIS weights", unit="layers"):
            lora_key = key[:-7]  # Remove ".weight"
            # Convert key format: double_blocks.0.img_attn.proj -> lora_unet_double_blocks_0_img_attn_proj
            lora_name = prefix_lora + lora_key.replace(".", "_")
            
            # Check skip patterns
            if _matches_any_pattern(key, skip_patterns):
                skipped_by_pattern += 1
                pbar.update(1)
                continue
            
            # Check if key exists in model B
            if key not in keys_b:
                if mismatch_mode == "error":
                    raise ValueError(f"Key '{key}' not found in model B (mismatch_mode='error')")
                elif mismatch_mode == "skip":
                    skipped_by_mismatch += 1
                    pbar.update(1)
                    continue
                # zeros mode: diff = tensor_a - 0 = tensor_a
                tensor_a = handler_a.get_tensor(key).to(device=process_device, dtype=torch.float32)
                weight_diff = tensor_a
                del tensor_a
            else:
                # Load both tensors
                tensor_a = handler_a.get_tensor(key).to(device=process_device, dtype=torch.float32)
                tensor_b = handler_b.get_tensor(key).to(device=process_device, dtype=torch.float32)
                
                # Check shape mismatch
                if tensor_a.shape != tensor_b.shape:
                    del tensor_a, tensor_b
                    if mismatch_mode == "error":
                        raise ValueError(f"Shape mismatch for '{key}': {tensor_a.shape} vs {tensor_b.shape}")
                    elif mismatch_mode == "skip":
                        shape_mismatch_keys += 1
                        pbar.update(1)
                        continue
                    # zeros mode
                    tensor_a = handler_a.get_tensor(key).to(device=process_device, dtype=torch.float32)
                    weight_diff = tensor_a
                    del tensor_a
                else:
                    # Compute difference: finetuned - base
                    weight_diff = tensor_a - tensor_b
                    del tensor_a, tensor_b
            
            # Skip 1D weights (norms, biases can't be decomposed)
            if weight_diff.ndim < 2:
                skipped_1d_weights += 1
                del weight_diff
                pbar.update(1)
                continue
            
            # Determine if this is a conv or linear layer
            is_conv = weight_diff.ndim == 4
            is_linear_conv = is_conv and weight_diff.shape[2] == 1 and weight_diff.shape[3] == 1
            
            try:
                if is_conv:
                    # Use conv extraction
                    mode_param = linear_param if is_linear_conv else conv_param
                    result, decompose_mode = extract_conv(
                        weight_diff,
                        mode=mode,
                        mode_param=int(mode_param) if mode == "fixed" else mode_param,
                        device=process_device,
                        is_cp=use_cp and not is_linear_conv,
                    )
                else:
                    # Use linear extraction
                    result, decompose_mode = extract_linear(
                        weight_diff,
                        mode=mode,
                        mode_param=int(linear_param) if mode == "fixed" else linear_param,
                        device=process_device,
                    )
                
                if decompose_mode == "full":
                    # Store as full diff
                    output_sd[f"{lora_name}.diff"] = weight_diff.to(save_torch_dtype).cpu()
                    full_weights += 1
                elif decompose_mode == "low rank":
                    extract_a, extract_b, diff = result
                    
                    # Check if CP decomposition was applied (has 3 components for conv)
                    if is_conv and not is_linear_conv and use_cp and hasattr(result, '__len__') and len(result) == 3:
                        # Check if extract_a has been further decomposed
                        if extract_a.ndim == 4 and extract_a.shape[2] > 1:
                            # Re-do with CP decomposition
                            dim = extract_a.size(0)
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
                    
                    output_sd[f"{lora_name}.lora_down.weight"] = extract_a.to(save_torch_dtype).cpu()
                    output_sd[f"{lora_name}.lora_up.weight"] = extract_b.to(save_torch_dtype).cpu()
                    output_sd[f"{lora_name}.alpha"] = torch.tensor([extract_a.shape[0]]).to(save_torch_dtype)
                    
                    low_rank_weights += 1
                    del extract_a, extract_b, diff
                
            except Exception as e:
                print(f"[LyCORIS Extract] Could not extract for {key}: {e}")
            
            del weight_diff
            pbar.update(1)
        
        # Log summary
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


class LyCORISExtractFromFile(io.ComfyNode):
    """
    Memory-efficient LyCORIS extraction from safetensors files.
    
    Uses the LyCORIS library for advanced extraction modes while streaming tensors
    one at a time for memory efficiency.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LyCORISExtractFromFile",
            display_name="Extract LyCORIS from Files (Memory Efficient)",
            category="ModelUtils/LoRA",
            description="Extracts a LyCORIS/LoCoN by computing the difference between two model files. "
                        "Uses LyCORIS library for advanced extraction modes (threshold, ratio, quantile).",
            inputs=[
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
                io.Combo.Input("mode", options=["fixed", "threshold", "ratio", "quantile", "full"], default="fixed",
                              tooltip="Extraction mode: fixed=specific rank, threshold/ratio/quantile=dynamic rank, full=no decomposition"),
                io.Int.Input("linear_dim", default=64, min=1, max=4096, step=1,
                            tooltip="Rank for linear layers (fixed mode) or ignored in other modes"),
                io.Int.Input("conv_dim", default=32, min=1, max=4096, step=1,
                            tooltip="Rank for conv layers (fixed mode) or ignored in other modes"),
                io.Float.Input("linear_threshold", default=0.0, min=0.0, max=1.0, step=0.01,
                              tooltip="Threshold/ratio/quantile for linear layers (non-fixed modes)"),
                io.Float.Input("conv_threshold", default=0.0, min=0.0, max=1.0, step=0.01,
                              tooltip="Threshold/ratio/quantile for conv layers (non-fixed modes)"),
                io.Boolean.Input("use_cp", default=True,
                              tooltip="Use CP decomposition for 3x3 convolutions (reduces size)"),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip",
                              tooltip="How to handle missing keys or shape mismatches between models"),
                io.String.Input("output_filename", default="extracted_lycoris",
                               tooltip="Output filename (without extension)"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("process_device", options=["cuda", "cpu"], default="cuda",
                              tooltip="Device for SVD computation"),
                io.String.Input("skip_patterns", default="", multiline=True,
                               tooltip="Regex patterns for layers to skip"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model_a: str,
        model_b: str,
        mode: str,
        linear_dim: int,
        conv_dim: int,
        linear_threshold: float,
        conv_threshold: float,
        use_cp: bool,
        mismatch_mode: str,
        output_filename: str,
        save_dtype: str,
        process_device: str,
        skip_patterns: str,
    ) -> io.NodeOutput:
        if not LYCORIS_AVAILABLE:
            raise ImportError("lycoris library not installed. Install with: pip install lycoris")
        
        # Get full paths
        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        
        # Determine parameters based on mode
        if mode == "fixed":
            linear_param = linear_dim
            conv_param = conv_dim
        else:
            linear_param = linear_threshold
            conv_param = conv_threshold
        
        # Extract LyCORIS
        output_sd = lycoris_extract_from_files(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            mode=mode,
            linear_param=linear_param,
            conv_param=conv_param,
            prefix_lora="lora_unet_",
            process_device=process_device,
            save_dtype=save_dtype,
            skip_patterns_str=skip_patterns,
            mismatch_mode=mismatch_mode,
            use_cp=use_cp,
        )
        
        if not output_sd:
            raise ValueError("No LyCORIS weights extracted - models may be identical or incompatible")
        
        # Save to loras folder
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")
        
        save_file(output_sd, output_path)
        print(f"[LyCORIS Extract] Saved {len(output_sd)} tensors to {output_path}")
        
        return io.NodeOutput(output_path)
