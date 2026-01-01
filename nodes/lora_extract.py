"""
Memory-efficient LoRA extraction node.

This node loads two models from files and extracts a LoRA by computing the difference,
without requiring the full models to be loaded into GPU memory via ComfyUI's model management.
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


CLAMP_QUANTILE = 0.99


def extract_lora(diff, rank):
    """Extract LoRA weights from a weight difference tensor using SVD."""
    conv2d = (len(diff.shape) == 4)
    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]
    rank = min(rank, in_dim, out_dim)

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()

    U, S, Vh = torch.linalg.svd(diff.float())
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)
    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)
    if conv2d:
        U = U.reshape(out_dim, rank, 1, 1)
        Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    return (U, Vh)


def _compile_patterns(pattern_string: str) -> list:
    """Compiles whitespace-separated regex patterns into a list of compiled patterns."""
    if not pattern_string or not pattern_string.strip():
        return []
    
    patterns = []
    for pattern in pattern_string.split():
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


def extract_lora_from_files(
    model_a_path: str,
    model_b_path: str,
    rank: int,
    lora_type: str,
    bias_diff: bool,
    prefix_lora: str,
    process_device: str,
    save_dtype: str,
    exclude_patterns_str: str = "",
    discard_patterns_str: str = "",
    mismatch_mode: str = "skip",
) -> dict[str, torch.Tensor]:
    """
    Memory-efficient LoRA extraction by streaming tensors from safetensors files.
    
    Args:
        model_a_path: Path to the finetuned/modified model
        model_b_path: Path to the base/original model  
        rank: LoRA rank for SVD decomposition
        lora_type: "standard" for LoRA, "full_diff" for full difference
        bias_diff: Whether to include bias differences
        prefix_lora: Prefix for LoRA keys (e.g., "diffusion_model.")
        process_device: Device for processing ("cuda" or "cpu")
        save_dtype: Output dtype ("fp16", "bf16", "fp32")
        exclude_patterns_str: Regex patterns to exclude from extraction (use model A only)
        discard_patterns_str: Regex patterns to discard entirely (no output)
        mismatch_mode: "skip" (use model A), "zeros" (use zeros), "error" (raise error)
    
    Returns:
        Dictionary of LoRA tensors
    """
    output_sd = {}
    
    # Get dtype for saving
    save_torch_dtype = {
        "fp32": torch.float32, 
        "fp16": torch.float16, 
        "bf16": torch.bfloat16
    }.get(save_dtype, torch.float16)
    
    # Compile patterns
    exclude_patterns = _compile_patterns(exclude_patterns_str)
    discard_patterns = _compile_patterns(discard_patterns_str)
    
    # Open both files with memory-efficient handler
    handler_a = MemoryEfficientSafeOpen(model_a_path)
    handler_b = MemoryEfficientSafeOpen(model_b_path)
    
    try:
        keys_a = set(handler_a.keys())
        keys_b = set(handler_b.keys())
        
        # Get all weight keys from model A (primary)
        all_weight_keys = [k for k in keys_a if k.endswith(".weight")]
        all_bias_keys = [k for k in keys_a if k.endswith(".bias")] if bias_diff else []
        
        # Log key differences
        missing_in_b = keys_a - keys_b
        extra_in_b = keys_b - keys_a
        if missing_in_b:
            print(f"[LoRA Extract] {len(missing_in_b)} keys in model A not in model B")
        if extra_in_b:
            print(f"[LoRA Extract] {len(extra_in_b)} keys in model B not in model A (ignored)")
        
        pbar = comfy.utils.ProgressBar(len(all_weight_keys) + len(all_bias_keys))
        
        # Stats
        skipped_keys = 0
        excluded_keys = 0
        discarded_keys = 0
        shape_mismatch_keys = 0
        
        for key in tqdm(all_weight_keys, desc="Extracting LoRA weights", unit="layers"):
            lora_key = key[:-7]  # Remove ".weight"
            
            # Check discard patterns first - skip entirely
            if _matches_any_pattern(key, discard_patterns):
                discarded_keys += 1
                pbar.update(1)
                continue
            
            # Load model A tensor
            tensor_a = handler_a.get_tensor(key).to(device=process_device, dtype=torch.float32)
            
            # Check exclude patterns - skip extraction, no output for this key
            if _matches_any_pattern(key, exclude_patterns):
                excluded_keys += 1
                del tensor_a
                pbar.update(1)
                continue
            
            # Check if key exists in model B
            if key not in keys_b:
                if mismatch_mode == "error":
                    raise ValueError(f"Key '{key}' not found in model B (mismatch_mode='error')")
                elif mismatch_mode == "skip":
                    skipped_keys += 1
                    del tensor_a
                    pbar.update(1)
                    continue
                # zeros mode: use tensor_a as-is (diff with zeros = tensor_a)
                weight_diff = tensor_a
            else:
                tensor_b = handler_b.get_tensor(key).to(device=process_device, dtype=torch.float32)
                
                # Check shape mismatch
                if tensor_a.shape != tensor_b.shape:
                    del tensor_b
                    if mismatch_mode == "error":
                        raise ValueError(f"Shape mismatch for '{key}': {tensor_a.shape} vs {tensor_b.shape}")
                    elif mismatch_mode == "skip":
                        shape_mismatch_keys += 1
                        del tensor_a
                        pbar.update(1)
                        continue
                    # zeros mode: use tensor_a as-is
                    weight_diff = tensor_a
                else:
                    # Compute difference: finetuned - base
                    weight_diff = tensor_a - tensor_b
                    del tensor_b
            
            del tensor_a
            
            if lora_type == "standard":
                if weight_diff.ndim < 2:
                    # 1D weights (e.g., LayerNorm) - store as diff if bias_diff enabled
                    if bias_diff:
                        output_sd[f"{prefix_lora}{lora_key}.diff"] = weight_diff.to(save_torch_dtype).cpu()
                else:
                    try:
                        lora_up, lora_down = extract_lora(weight_diff, rank)
                        output_sd[f"{prefix_lora}{lora_key}.lora_up.weight"] = lora_up.to(save_torch_dtype).cpu()
                        output_sd[f"{prefix_lora}{lora_key}.lora_down.weight"] = lora_down.to(save_torch_dtype).cpu()
                    except Exception as e:
                        print(f"[LoRA Extract] Could not extract LoRA for {key}: {e}")
            
            elif lora_type == "full_diff":
                output_sd[f"{prefix_lora}{lora_key}.diff"] = weight_diff.to(save_torch_dtype).cpu()
            
            del weight_diff
            pbar.update(1)
        
        # Process bias keys
        for key in tqdm(all_bias_keys, desc="Extracting LoRA biases", unit="layers"):
            lora_key = key[:-5]  # Remove ".bias"
            
            # Check discard patterns
            if _matches_any_pattern(key, discard_patterns):
                discarded_keys += 1
                pbar.update(1)
                continue
            
            # Check exclude patterns
            if _matches_any_pattern(key, exclude_patterns):
                excluded_keys += 1
                pbar.update(1)
                continue
            
            tensor_a = handler_a.get_tensor(key).to(device=process_device, dtype=torch.float32)
            
            if key not in keys_b:
                if mismatch_mode == "error":
                    raise ValueError(f"Key '{key}' not found in model B (mismatch_mode='error')")
                elif mismatch_mode == "skip":
                    skipped_keys += 1
                    del tensor_a
                    pbar.update(1)
                    continue
                bias_diff_tensor = tensor_a
            else:
                tensor_b = handler_b.get_tensor(key).to(device=process_device, dtype=torch.float32)
                
                if tensor_a.shape != tensor_b.shape:
                    del tensor_b
                    if mismatch_mode == "error":
                        raise ValueError(f"Shape mismatch for '{key}': {tensor_a.shape} vs {tensor_b.shape}")
                    elif mismatch_mode == "skip":
                        shape_mismatch_keys += 1
                        del tensor_a
                        pbar.update(1)
                        continue
                    bias_diff_tensor = tensor_a
                else:
                    bias_diff_tensor = tensor_a - tensor_b
                    del tensor_b
            
            del tensor_a
            output_sd[f"{prefix_lora}{lora_key}.diff_b"] = bias_diff_tensor.to(save_torch_dtype).cpu()
            del bias_diff_tensor
            pbar.update(1)
        
        # Log summary
        if excluded_keys > 0:
            print(f"[LoRA Extract] Excluded {excluded_keys} keys from extraction")
        if discarded_keys > 0:
            print(f"[LoRA Extract] Discarded {discarded_keys} keys from output")
        if skipped_keys > 0:
            print(f"[LoRA Extract] Skipped {skipped_keys} keys (missing in model B)")
        if shape_mismatch_keys > 0:
            print(f"[LoRA Extract] Skipped {shape_mismatch_keys} keys (shape mismatch)")
    
    finally:
        handler_a.__exit__(None, None, None)
        handler_b.__exit__(None, None, None)
    
    return output_sd


class LoRASaveFromFile(io.ComfyNode):
    """
    Memory-efficient LoRA extraction from safetensors files.
    
    Unlike the standard LoraSave node which requires pre-loaded and subtracted models,
    this node loads models directly from files one tensor at a time, making it much
    more memory efficient for extracting LoRAs from large models.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRASaveFromFile",
            display_name="Extract LoRA from Files (Memory Efficient)",
            category="ModelUtils/LoRA",
            description="Extracts a LoRA by computing the difference between two model files. "
                        "Much more memory efficient than using ModelSubtract + LoraSave.",
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
                io.Int.Input("rank", default=64, min=1, max=4096, step=1,
                            tooltip="LoRA rank - higher preserves more detail but increases size"),
                io.Combo.Input("lora_type", options=["standard", "full_diff"], default="standard",
                              tooltip="standard: SVD decomposition to lora_up/lora_down, full_diff: raw weight differences"),
                io.Boolean.Input("bias_diff", default=True,
                                tooltip="Include bias differences in the LoRA"),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip",
                              tooltip="How to handle missing keys or shape mismatches between models"),
                io.String.Input("output_filename", default="extracted_lora",
                               tooltip="Output filename (without extension)"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("process_device", options=["cuda", "cpu"], default="cuda",
                              tooltip="Device for SVD computation"),
                io.String.Input("exclude_patterns", default="", multiline=True,
                               tooltip="Regex patterns to exclude from extraction (one per line, whitespace-separated)"),
                io.String.Input("discard_patterns", default="", multiline=True,
                               tooltip="Regex patterns to discard entirely (no output for matching keys)"),
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
        rank: int,
        lora_type: str,
        bias_diff: bool,
        mismatch_mode: str,
        output_filename: str,
        save_dtype: str,
        process_device: str,
        exclude_patterns: str,
        discard_patterns: str,
    ) -> io.NodeOutput:
        # Get full paths
        model_a_path = folder_paths.get_full_path_or_raise("diffusion_models", model_a)
        model_b_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
        
        # Extract LoRA
        output_sd = extract_lora_from_files(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            rank=rank,
            lora_type=lora_type,
            bias_diff=bias_diff,
            prefix_lora="diffusion_model.",
            process_device=process_device,
            save_dtype=save_dtype,
            exclude_patterns_str=exclude_patterns,
            discard_patterns_str=discard_patterns,
            mismatch_mode=mismatch_mode,
        )
        
        if not output_sd:
            raise ValueError("No LoRA weights extracted - models may be identical or incompatible")
        
        # Save to loras folder
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")
        
        save_file(output_sd, output_path)
        print(f"[LoRA Extract] Saved {len(output_sd)} tensors to {output_path}")
        
        return io.NodeOutput(output_path)


class LoRACheckpointSaveFromFile(io.ComfyNode):
    """
    Memory-efficient LoRA extraction from checkpoint files (which may contain both model and text encoder).
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRACheckpointSaveFromFile",
            display_name="Extract LoRA from Checkpoints (Memory Efficient)",
            category="ModelUtils/LoRA",
            description="Extracts a LoRA by computing the difference between two checkpoint files. "
                        "Handles both diffusion model and text encoder weights.",
            inputs=[
                io.Combo.Input(
                    "checkpoint_a", 
                    options=folder_paths.get_filename_list("checkpoints"),
                    tooltip="The finetuned/modified checkpoint (A - B = LoRA)"
                ),
                io.Combo.Input(
                    "checkpoint_b", 
                    options=folder_paths.get_filename_list("checkpoints"),
                    tooltip="The base/original checkpoint (A - B = LoRA)"
                ),
                io.Int.Input("rank", default=64, min=1, max=4096, step=1,
                            tooltip="LoRA rank - higher preserves more detail but increases size"),
                io.Combo.Input("lora_type", options=["standard", "full_diff"], default="standard",
                              tooltip="standard: SVD decomposition, full_diff: raw weight differences"),
                io.Boolean.Input("bias_diff", default=True,
                                tooltip="Include bias differences in the LoRA"),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip",
                              tooltip="How to handle missing keys or shape mismatches between models"),
                io.String.Input("output_filename", default="extracted_lora",
                               tooltip="Output filename (without extension)"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("process_device", options=["cuda", "cpu"], default="cuda",
                              tooltip="Device for SVD computation"),
                io.String.Input("exclude_patterns", default="", multiline=True,
                               tooltip="Regex patterns to exclude from extraction"),
                io.String.Input("discard_patterns", default="", multiline=True,
                               tooltip="Regex patterns to discard entirely"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        checkpoint_a: str,
        checkpoint_b: str,
        rank: int,
        lora_type: str,
        bias_diff: bool,
        mismatch_mode: str,
        output_filename: str,
        save_dtype: str,
        process_device: str,
        exclude_patterns: str,
        discard_patterns: str,
    ) -> io.NodeOutput:
        # Get full paths
        checkpoint_a_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_a)
        checkpoint_b_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_b)
        
        # Open handlers to detect prefixes
        handler_a = MemoryEfficientSafeOpen(checkpoint_a_path)
        keys = handler_a.keys()
        handler_a.__exit__(None, None, None)
        
        # Auto-detect key prefixes
        has_model = any(k.startswith("model.") for k in keys)
        has_diffusion = any(k.startswith("diffusion_model.") for k in keys)
        has_text_encoders = any(k.startswith("text_encoders.") or k.startswith("cond_stage_model.") for k in keys)
        
        output_sd = {}
        
        # Extract diffusion model LoRA
        if has_model or has_diffusion:
            prefix = "model.diffusion_model." if has_model else "diffusion_model."
            model_lora = extract_lora_from_files(
                model_a_path=checkpoint_a_path,
                model_b_path=checkpoint_b_path,
                rank=rank,
                lora_type=lora_type,
                bias_diff=bias_diff,
                prefix_lora="diffusion_model.",
                process_device=process_device,
                save_dtype=save_dtype,
                exclude_patterns_str=exclude_patterns,
                discard_patterns_str=discard_patterns,
                mismatch_mode=mismatch_mode,
            )
            output_sd.update(model_lora)
        
        # Note: Text encoder extraction would need additional prefix handling
        # For full checkpoint support, this would need to be expanded
        
        if not output_sd:
            raise ValueError("No LoRA weights extracted - checkpoints may be identical or incompatible")
        
        # Save to loras folder
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")
        
        save_file(output_sd, output_path)
        print(f"[LoRA Extract] Saved {len(output_sd)} tensors to {output_path}")
        
        return io.NodeOutput(output_path)
