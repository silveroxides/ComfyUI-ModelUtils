"""
LoRA Multi-Merge - Merge multiple LoRAs into a single LoRA file.
Resolves different naming conventions and ranks via concatenation.
"""
import os
import torch
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from safetensors.torch import save_file
from typing import List, Dict, Tuple, Optional
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation
from .lora_resize import (
    detect_lora_format,
    extract_lora_pairs,
    detect_lora_rank,
    _compile_patterns,
    _matches_any_pattern,
    _format_lora_key
)

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned



def merge_multi_loras(
    lora_paths: List[str],
    lora_weights: List[float],
    merge_mode: str,
    device: str,
    save_dtype: torch.dtype,
    output_filename: str,
    verbose: bool = True,
) -> str:
    """
    Merge multiple LoRAs into a single LoRA file.

    Modes:
    - concatenate: Mathematically sound merge by stacking ranks.
                   New rank = sum(input ranks). Alpha set to new rank.
    - weighted_sum: Weighted sum of A and B weights (Kohya style).
                    If ranks differ, smaller ones are zero-padded to match the largest rank.
                    New rank = max(input ranks).
    """
    # Estimate memory
    total_size_gb = sum(estimate_model_size(p) for p in lora_paths)
    if verbose:
        print(f"[LoRA Multi-Merge] Preparing memory for {total_size_gb:.2f}GB operation...")
        print(f"[LoRA Multi-Merge] Mode: {merge_mode}")

    prepare_for_large_operation(total_size_gb * 2.5, torch.device(device))

    handlers = [MemoryEfficientSafeOpen(p, low_memory=True) for p in lora_paths]

    try:
        # 1. Analyze all LoRAs
        layer_map = {}
        lora_infos = []

        for i, handler in enumerate(handlers):
            keys = handler.keys()
            fmt = detect_lora_format(keys)
            pairs = extract_lora_pairs(keys, fmt)
            rank, alpha = detect_lora_rank(handler, pairs)

            info = {
                "name": os.path.basename(lora_paths[i]),
                "format": fmt,
                "pairs": pairs,
                "rank": rank,
                "alpha": alpha,
                "weight": lora_weights[i],
                "scale": alpha / rank if rank > 0 else 1.0
            }
            lora_infos.append(info)

            if verbose:
                print(f"[LoRA Multi-Merge] LoRA {i+1} ({info['name']}): {len(pairs)} layers, dim={rank}, format={fmt['format']}")

            for block_name in pairs.keys():
                core = _format_lora_key(block_name)
                if core not in layer_map:
                    layer_map[core] = []
                layer_map[core].append(i)

        if verbose:
            print(f"[LoRA Multi-Merge] Total unique layers after resolving naming: {len(layer_map)}")

        output_sd = {}
        pbar = comfy.utils.ProgressBar(len(layer_map))

        # 2. Merge each core layer
        with torch.no_grad():
            for core, indices in tqdm(layer_map.items(), desc="Merging layers", unit="layers"):
                downs = []
                ups = []
                is_conv = False
                max_rank = 0

                # First pass: load and determine max rank
                for idx in indices:
                    info = lora_infos[idx]
                    handler = handlers[idx]
                    orig_block = next(b for b in info["pairs"].keys() if _format_lora_key(b) == core)
                    block_keys = info["pairs"][orig_block]

                    if "down" not in block_keys or "up" not in block_keys:
                        continue

                    t_down = handler.get_tensor(block_keys["down"])
                    t_up = handler.get_tensor(block_keys["up"])

                    if device == 'cuda':
                        t_down = transfer_to_gpu_pinned(t_down, device, torch.float32)
                        t_up = transfer_to_gpu_pinned(t_up, device, torch.float32)
                    else:
                        t_down = t_down.to(device=device, dtype=torch.float32)
                        t_up = t_up.to(device=device, dtype=torch.float32)

                    # Store with original info for padding/weighting
                    is_conv = len(t_down.shape) == 4
                    current_rank = t_down.shape[0]
                    max_rank = max(max_rank, current_rank)

                    # Apply scale immediately for concatenate mode, or keep for later
                    if merge_mode == "concatenate":
                        effective_weight = info["weight"] * info["scale"]
                        t_up = t_up * effective_weight

                    downs.append((t_down, info))
                    ups.append((t_up, info))

                if not downs:
                    pbar.update(1)
                    continue

                if merge_mode == "concatenate":
                    # Stack ranks
                    merged_down = torch.cat([d[0] for d in downs], dim=0)
                    merged_up = torch.cat([u[0] for u in ups], dim=1)
                    new_rank = merged_down.shape[0]
                    new_alpha = float(new_rank)
                else:
                    # weighted_sum (Kohya style)
                    # Result = sum( weight_i * Padded(B_i) ), sum( weight_i * Padded(A_i) )
                    # Note: We apply scale to weights here to normalize different alphas
                    merged_down = torch.zeros_like(downs[0][0])
                    # Need to handle different ranks via padding
                    target_down_shape = list(downs[0][0].shape)
                    target_down_shape[0] = max_rank
                    target_up_shape = list(ups[0][0].shape)
                    target_up_shape[1] = max_rank

                    merged_down = torch.zeros(target_down_shape, device=device, dtype=torch.float32)
                    merged_up = torch.zeros(target_up_shape, device=device, dtype=torch.float32)

                    for (t_d, info_d), (t_u, info_u) in zip(downs, ups):
                        r = t_d.shape[0]
                        w = info_d["weight"]
                        # We also apply sqrt(scale) to both to distribute the scale factor?
                        # Kohya just uses weights. But to be safe with different alphas,
                        # we apply scale to the final delta.
                        # For direct weight merge, we'll just use weights.

                        if r < max_rank:
                            # Pad down: [r, in...] -> [max_rank, in...]
                            pad_d = [0] * (len(t_d.shape) * 2)
                            pad_d[-1] = max_rank - r # last dim in pad is first dim in tensor (reversed)
                            # Wait, F.pad uses reverse order of dims.
                            # For [r, in], padding is (0,0, 0, max_rank-r)
                            padding_d = [0, 0] * (len(t_d.shape) - 1) + [0, max_rank - r]
                            t_d = torch.nn.functional.pad(t_d, tuple(padding_d))

                            # Pad up: [out, r...] -> [out, max_rank...]
                            # For [out, r], padding is (0, max_rank-r, 0, 0)
                            padding_u = [0, max_rank - r] + [0, 0] * (len(t_u.shape) - 1)
                            t_u = torch.nn.functional.pad(t_u, tuple(padding_u))

                        merged_down += w * t_d
                        merged_up += w * t_u

                    new_rank = max_rank
                    # Alpha is usually max of alphas or same as rank
                    new_alpha = float(max_rank)

                # Store
                output_sd[f"{core}.lora_down.weight"] = merged_down.to(save_dtype).cpu().contiguous()
                output_sd[f"{core}.lora_up.weight"] = merged_up.to(save_dtype).cpu().contiguous()
                output_sd[f"{core}.alpha"] = torch.tensor(new_alpha, dtype=save_dtype)

                # Cleanup
                del merged_down, merged_up
                for d, _ in downs: del d
                for u, _ in ups: del u
                downs.clear()
                ups.clear()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pbar.update(1)

        # 3. Final Summary
        if verbose:
            matched_stats = {i: 0 for i in range(len(lora_paths))}
            for indices in layer_map.values():
                for idx in indices:
                    matched_stats[idx] += 1

            print(f"[LoRA Multi-Merge] --- Merge Summary ---")
            for i, info in enumerate(lora_infos):
                print(f"[LoRA Multi-Merge] LoRA {i+1}: {matched_stats[i]}/{len(info['pairs'])} layers used in merge")
            print(f"[LoRA Multi-Merge] Output state dict has {len(output_sd)} tensors")

        # 4. Metadata
        metadata = {
            "ss_training_comment": f"Merged {len(lora_paths)} LoRAs via concatenation",
            "ss_network_module": "networks.lora",
        }

        # 4. Save
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")

        save_file(output_sd, output_path, metadata)
        if verbose:
            print(f"[LoRA Multi-Merge] Saved merged LoRA to {output_path}")

        return output_path

    finally:
        for h in handlers:
            h.__exit__(None, None, None)
        cleanup_after_operation()


class LoRAMultiMerge(io.ComfyNode):
    """Merge multiple LoRAs into a single LoRA file."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAMultiMerge",
            display_name="LoRA Multi-Merge",
            category="ModelUtils/LoRA/Merge",
            description="Merge 1-8 LoRAs into a single LoRA file. This resolves different ranks and naming conventions.",
            inputs=[
                io.Combo.Input("lora_count", options=[str(i) for i in range(1, 9)], default="2",
                              tooltip="Number of LoRAs to merge"),
                # LoRA 1
                io.Combo.Input("lora_1", options=folder_paths.get_filename_list("loras"), tooltip="First LoRA"),
                io.Float.Input("weight_1", default=1.0, min=-10.0, max=10.0, step=0.01),
                # LoRA 2
                io.Combo.Input("lora_2", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_2", default=1.0, min=-10.0, max=10.0, step=0.01),
                # LoRA 3
                io.Combo.Input("lora_3", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_3", default=1.0, min=-10.0, max=10.0, step=0.01),
                # LoRA 4
                io.Combo.Input("lora_4", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_4", default=1.0, min=-10.0, max=10.0, step=0.01),
                # LoRA 5
                io.Combo.Input("lora_5", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_5", default=1.0, min=-10.0, max=10.0, step=0.01),
                # LoRA 6
                io.Combo.Input("lora_6", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_6", default=1.0, min=-10.0, max=10.0, step=0.01),
                # LoRA 7
                io.Combo.Input("lora_7", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_7", default=1.0, min=-10.0, max=10.0, step=0.01),
                # LoRA 8
                io.Combo.Input("lora_8", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_8", default=1.0, min=-10.0, max=10.0, step=0.01),

                io.Combo.Input("merge_mode", options=["concatenate", "weighted_sum"], default="concatenate",
                              tooltip="concatenate: safe, increases rank. weighted_sum: fixed rank (to max input rank), mathematically lossy but standard in Kohya."),
                io.String.Input("output_filename", default="merged_lora"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                merge_mode, output_filename, save_dtype, device) -> io.NodeOutput:

        count = int(lora_count)
        names = [lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8]
        weights = [weight_1, weight_2, weight_3, weight_4, weight_5, weight_6, weight_7, weight_8]

        valid_paths = []
        valid_weights = []

        for i in range(count):
            if names[i] and names[i] != "None":
                path = folder_paths.get_full_path_or_raise("loras", names[i])
                valid_paths.append(path)
                valid_weights.append(weights[i])

        if not valid_paths:
            raise ValueError("No LoRAs selected for merging.")

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = merge_multi_loras(
            lora_paths=valid_paths,
            lora_weights=valid_weights,
            merge_mode=merge_mode,
            device=device,
            save_dtype=dtype,
            output_filename=output_filename
        )

        return io.NodeOutput(path)


class LoRAMultiMergeDARE(io.ComfyNode):
    """Merge multiple LoRAs into a single LoRA file using DARE-Ties method."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAMultiMergeDARE",
            display_name="LoRA Multi-Merge (DARE-Ties)",
            category="ModelUtils/LoRA/Merge",
            description="Merge 1-8 LoRAs into a single LoRA file using DARE-Ties. Drops small values and resolves sign conflicts.",
            inputs=[
                io.Combo.Input("lora_count", options=[str(i) for i in range(1, 9)], default="2"),
                # LoRA 1
                io.Combo.Input("lora_1", options=folder_paths.get_filename_list("loras"), tooltip="First LoRA"),
                io.Float.Input("weight_1", default=1.0, min=-10.0, max=10.0, step=0.01),
                # ...
                io.Combo.Input("lora_2", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_2", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_3", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_3", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_4", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_4", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_5", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_5", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_6", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_6", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_7", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_7", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_8", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_8", default=1.0, min=-10.0, max=10.0, step=0.01),

                io.Float.Input("drop_rate", default=0.1, min=0.0, max=1.0, step=0.01, tooltip="DARE drop rate"),
                io.Float.Input("trim_quantile", default=0.2, min=0.0, max=1.0, step=0.01, tooltip="TIES trim quantile (drops smallest values)"),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_lora_dare"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                drop_rate, trim_quantile, seed, output_filename, save_dtype, device) -> io.NodeOutput:

        count = int(lora_count)
        names = [lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8]
        weights = [weight_1, weight_2, weight_3, weight_4, weight_5, weight_6, weight_7, weight_8]

        valid_paths = []
        valid_weights = []

        for i in range(count):
            if names[i] and names[i] != "None":
                path = folder_paths.get_full_path_or_raise("loras", names[i])
                valid_paths.append(path)
                valid_weights.append(weights[i])

        if not valid_paths:
            raise ValueError("No LoRAs selected for merging.")

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = merge_multi_loras_dare(
            lora_paths=valid_paths,
            lora_weights=valid_weights,
            drop_rate=drop_rate,
            trim_quantile=trim_quantile,
            seed=seed,
            device=device,
            save_dtype=dtype,
            output_filename=output_filename
        )

        return io.NodeOutput(path)

def merge_multi_loras_dare(
    lora_paths: List[str],
    lora_weights: List[float],
    drop_rate: float,
    trim_quantile: float,
    seed: int,
    device: str,
    save_dtype: torch.dtype,
    output_filename: str,
    verbose: bool = True,
) -> str:
    """
    Merge multiple LoRAs using DARE-Ties method applied to weights.
    Different ranks are handled via zero-padding to the max rank.
    """
    total_size_gb = sum(estimate_model_size(p) for p in lora_paths)
    if verbose:
        print(f"[LoRA Multi-Merge DARE] Preparing memory for {total_size_gb:.2f}GB operation...")
        print(f"[LoRA Multi-Merge DARE] Drop rate: {drop_rate}, Trim quantile: {trim_quantile}")

    prepare_for_large_operation(total_size_gb * 2.5, torch.device(device))
    handlers = [MemoryEfficientSafeOpen(p, low_memory=True) for p in lora_paths]

    rng = torch.Generator(device=device).manual_seed(seed)

    try:
        # 1. Analyze
        layer_map = {}
        lora_infos = []
        for i, handler in enumerate(handlers):
            keys = handler.keys()
            fmt = detect_lora_format(keys)
            pairs = extract_lora_pairs(keys, fmt)
            rank, alpha = detect_lora_rank(handler, pairs)
            info = {
                "name": os.path.basename(lora_paths[i]),
                "pairs": pairs,
                "rank": rank,
                "alpha": alpha,
                "weight": lora_weights[i]
            }
            lora_infos.append(info)

            if verbose:
                print(f"[LoRA Multi-Merge DARE] LoRA {i+1} ({info['name']}): {len(pairs)} layers, dim={rank}")

            for block_name in pairs.keys():
                core = _format_lora_key(block_name)
                if core not in layer_map: layer_map[core] = []
                layer_map[core].append(i)

        if verbose:
            print(f"[LoRA Multi-Merge DARE] Total unique layers after resolving naming: {len(layer_map)}")

        output_sd = {}
        pbar = comfy.utils.ProgressBar(len(layer_map))

        # 2. Merge
        with torch.no_grad():
            for core, indices in tqdm(layer_map.items(), desc="Merging layers (DARE)", unit="layers"):
                downs = []
                ups = []
                max_rank = 0

                for idx in indices:
                    info = lora_infos[idx]
                    orig_block = next(b for b in info["pairs"].keys() if _format_lora_key(b) == core)
                    block_keys = info["pairs"][orig_block]
                    if "down" not in block_keys or "up" not in block_keys: continue

                    t_d = handlers[idx].get_tensor(block_keys["down"]).to(device=device, dtype=torch.float32)
                    t_u = handlers[idx].get_tensor(block_keys["up"]).to(device=device, dtype=torch.float32)

                    max_rank = max(max_rank, t_d.shape[0])
                    downs.append((t_d, info["weight"]))
                    ups.append((t_u, info["weight"]))

                if not downs:
                    pbar.update(1)
                    continue

                def process_ties_dare(tensors, weights, dim_to_pad):
                    # Pad all to max_rank
                    padded = []
                    for t, w in tensors:
                        if t.shape[dim_to_pad] < max_rank:
                            padding = [0] * (len(t.shape) * 2)
                            # dim_to_pad 0 (down) -> last pair in pad
                            # dim_to_pad 1 (up) -> second to last pair in pad
                            rev_dim = len(t.shape) - 1 - dim_to_pad
                            padding[rev_dim*2 + 1] = max_rank - t.shape[dim_to_pad]
                            t = torch.nn.functional.pad(t, tuple(padding))
                        padded.append(t * w)

                    # DARE
                    if drop_rate > 0:
                        for i in range(len(padded)):
                            mask = (torch.rand(padded[i].shape, generator=rng, device=device) > drop_rate).float()
                            padded[i] = (padded[i] * mask) / (1 - drop_rate)

                    # TIES
                    # 1. Trim
                    if trim_quantile > 0:
                        for i in range(len(padded)):
                            flat = padded[i].abs().flatten()
                            k = int(len(flat) * trim_quantile)
                            if k > 0:
                                threshold = torch.kthvalue(flat, k).values
                                padded[i] = torch.where(padded[i].abs() < threshold, torch.zeros_like(padded[i]), padded[i])

                    # 2. Elect & Merge
                    stacked = torch.stack(padded) # [N, ...]
                    signs = torch.sign(stacked)
                    sum_signs = signs.sum(dim=0)
                    dominant_sign = torch.sign(sum_signs)

                    # Filter those matching dominant sign
                    mask = (signs == dominant_sign) & (dominant_sign != 0)
                    filtered = torch.where(mask, stacked, torch.zeros_like(stacked))

                    # Average matching signs
                    count = mask.sum(dim=0)
                    result = filtered.sum(dim=0) / torch.clamp(count, min=1.0)
                    return result

                merged_down = process_ties_dare(downs, [1.0]*len(downs), 0)
                merged_up = process_ties_dare(ups, [1.0]*len(ups), 1)

                output_sd[f"{core}.lora_down.weight"] = merged_down.to(save_dtype).cpu().contiguous()
                output_sd[f"{core}.lora_up.weight"] = merged_up.to(save_dtype).cpu().contiguous()
                output_sd[f"{core}.alpha"] = torch.tensor(float(max_rank), dtype=save_dtype)

                del merged_down, merged_up
                for d, _ in downs: del d
                for u, _ in ups: del u
                downs.clear()
                ups.clear()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pbar.update(1)

        # Final Summary
        if verbose:
            matched_stats = {i: 0 for i in range(len(lora_paths))}
            for indices in layer_map.values():
                for idx in indices:
                    matched_stats[idx] += 1

            print(f"[LoRA Multi-Merge DARE] --- Merge Summary ---")
            for i, info in enumerate(lora_infos):
                print(f"[LoRA Multi-Merge DARE] LoRA {i+1}: {matched_stats[i]}/{len(info['pairs'])} layers used in merge")
            print(f"[LoRA Multi-Merge DARE] Output state dict has {len(output_sd)} tensors")

        # Save
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")
        save_file(output_sd, output_path, {"ss_training_comment": "Merged via DARE-Ties"})
        return output_path

    finally:
        for h in handlers: h.__exit__(None, None, None)
        cleanup_after_operation()


class LoRAMultiMergeDAREEnhanced(io.ComfyNode):
    """Merge multiple LoRAs into a single LoRA file using Enhanced DARE-Ties method."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAMultiMergeDAREEnhanced",
            display_name="LoRA Multi-Merge (Enhanced DARE-Ties)",
            category="ModelUtils/LoRA/Merge",
            description="Merge 1-8 LoRAs into a single LoRA file using Enhanced DARE-Ties. Uses dynamic probability masking based on value magnitudes.",
            inputs=[
                io.Combo.Input("lora_count", options=[str(i) for i in range(1, 9)], default="2"),
                # LoRA 1
                io.Combo.Input("lora_1", options=folder_paths.get_filename_list("loras"), tooltip="First LoRA"),
                io.Float.Input("weight_1", default=1.0, min=-10.0, max=10.0, step=0.01),
                # ...
                io.Combo.Input("lora_2", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_2", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_3", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_3", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_4", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_4", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_5", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_5", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_6", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_6", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_7", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_7", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Combo.Input("lora_8", options=["None"] + folder_paths.get_filename_list("loras"), default="None"),
                io.Float.Input("weight_8", default=1.0, min=-10.0, max=10.0, step=0.01),

                io.Float.Input("mask_power", default=2.0, min=0.001, max=10.0, step=0.01, tooltip="Mask power (Curve. 2.0 = quadratic)"),
                io.Float.Input("min_keep_prob", default=0.01, min=0.0, max=1.0, step=0.01, tooltip="Minimum keep probability (Floor to prevent explosion)"),
                io.Float.Input("mask_smooth", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Mask smoothness factor (0.0 = pure dropout, 1.0 = soft scale)"),
                io.Float.Input("trim_quantile", default=0.2, min=0.0, max=1.0, step=0.01, tooltip="TIES trim quantile (drops smallest values)"),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff),
                io.String.Input("output_filename", default="merged_lora_dare_enhanced"),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16"),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                mask_power, min_keep_prob, mask_smooth, trim_quantile, seed, output_filename, save_dtype, device) -> io.NodeOutput:

        count = int(lora_count)
        names = [lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8]
        weights = [weight_1, weight_2, weight_3, weight_4, weight_5, weight_6, weight_7, weight_8]

        valid_paths = []
        valid_weights = []

        for i in range(count):
            if names[i] and names[i] != "None":
                path = folder_paths.get_full_path_or_raise("loras", names[i])
                valid_paths.append(path)
                valid_weights.append(weights[i])

        if not valid_paths:
            raise ValueError("No LoRAs selected for merging.")

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path = merge_multi_loras_dare_enhanced(
            lora_paths=valid_paths,
            lora_weights=valid_weights,
            mask_power=mask_power,
            min_keep_prob=min_keep_prob,
            mask_smooth=mask_smooth,
            trim_quantile=trim_quantile,
            seed=seed,
            device=device,
            save_dtype=dtype,
            output_filename=output_filename
        )

        return io.NodeOutput(path)


def merge_multi_loras_dare_enhanced(
    lora_paths: List[str],
    lora_weights: List[float],
    mask_power: float,
    min_keep_prob: float,
    mask_smooth: float,
    trim_quantile: float,
    seed: int,
    device: str,
    save_dtype: torch.dtype,
    output_filename: str,
    verbose: bool = True,
) -> str:
    """
    Merge multiple LoRAs using Enhanced DARE-Ties method applied to weights.
    Different ranks are handled via zero-padding to the max rank.
    """
    total_size_gb = sum(estimate_model_size(p) for p in lora_paths)
    if verbose:
        print(f"[LoRA Multi-Merge Enhanced DARE] Preparing memory for {total_size_gb:.2f}GB operation...")

    prepare_for_large_operation(total_size_gb * 2.5, torch.device(device))
    handlers = [MemoryEfficientSafeOpen(p, low_memory=True) for p in lora_paths]

    rng = torch.Generator(device=device).manual_seed(seed)

    try:
        # 1. Analyze
        layer_map = {}
        lora_infos = []
        for i, handler in enumerate(handlers):
            keys = handler.keys()
            fmt = detect_lora_format(keys)
            pairs = extract_lora_pairs(keys, fmt)
            rank, alpha = detect_lora_rank(handler, pairs)
            info = {
                "name": os.path.basename(lora_paths[i]),
                "pairs": pairs,
                "rank": rank,
                "alpha": alpha,
                "weight": lora_weights[i]
            }
            lora_infos.append(info)

            if verbose:
                print(f"[LoRA Multi-Merge Enhanced DARE] LoRA {i+1} ({info['name']}): {len(pairs)} layers, dim={rank}")

            for block_name in pairs.keys():
                core = _format_lora_key(block_name)
                if core not in layer_map: layer_map[core] = []
                layer_map[core].append(i)

        if verbose:
            print(f"[LoRA Multi-Merge Enhanced DARE] Total unique layers after resolving naming: {len(layer_map)}")

        output_sd = {}
        pbar = comfy.utils.ProgressBar(len(layer_map))

        # 2. Merge
        with torch.no_grad():
            for core, indices in tqdm(layer_map.items(), desc="Merging layers (Enhanced DARE)", unit="layers"):
                downs = []
                ups = []
                max_rank = 0

                for idx in indices:
                    info = lora_infos[idx]
                    orig_block = next(b for b in info["pairs"].keys() if _format_lora_key(b) == core)
                    block_keys = info["pairs"][orig_block]
                    if "down" not in block_keys or "up" not in block_keys: continue

                    t_d = handlers[idx].get_tensor(block_keys["down"]).to(device=device, dtype=torch.float32)
                    t_u = handlers[idx].get_tensor(block_keys["up"]).to(device=device, dtype=torch.float32)

                    max_rank = max(max_rank, t_d.shape[0])
                    downs.append((t_d, info["weight"]))
                    ups.append((t_u, info["weight"]))

                if not downs:
                    pbar.update(1)
                    continue

                def process_ties_dare_enhanced(tensors, weights, dim_to_pad):
                    padded = []
                    for t, w in tensors:
                        if t.shape[dim_to_pad] < max_rank:
                            padding = [0] * (len(t.shape) * 2)
                            rev_dim = len(t.shape) - 1 - dim_to_pad
                            padding[rev_dim*2 + 1] = max_rank - t.shape[dim_to_pad]
                            t = torch.nn.functional.pad(t, tuple(padding))
                        padded.append(t * w)

                    # Enhanced DARE
                    for i in range(len(padded)):
                        t_val = padded[i]
                        abs_t = torch.abs(t_val)
                        max_val = torch.max(abs_t)
                        if max_val > 0:
                            prob = torch.clamp((abs_t / max_val) ** max(mask_power, 0.001), min=min_keep_prob, max=1.0)
                            prob = torch.nan_to_num(prob)

                            random_mask = torch.bernoulli(prob, generator=rng)
                            interpolated_mask = torch.lerp(random_mask, prob, mask_smooth)

                            padded[i] = (t_val * interpolated_mask) / prob

                    # TIES
                    if trim_quantile > 0:
                        for i in range(len(padded)):
                            flat = padded[i].abs().flatten()
                            k = max(1, int(len(flat) * trim_quantile))
                            if k > 0:
                                threshold = torch.kthvalue(flat, k).values
                                padded[i] = torch.where(padded[i].abs() < threshold, torch.zeros_like(padded[i]), padded[i])

                    stacked = torch.stack(padded)
                    signs = torch.sign(stacked)
                    sum_signs = signs.sum(dim=0)
                    dominant_sign = torch.sign(sum_signs)

                    mask = (signs == dominant_sign) & (dominant_sign != 0)
                    filtered = torch.where(mask, stacked, torch.zeros_like(stacked))

                    count = mask.sum(dim=0)
                    result = filtered.sum(dim=0) / torch.clamp(count, min=1.0)
                    return result

                merged_down = process_ties_dare_enhanced(downs, [1.0]*len(downs), 0)
                merged_up = process_ties_dare_enhanced(ups, [1.0]*len(ups), 1)

                output_sd[f"{core}.lora_down.weight"] = merged_down.to(save_dtype).cpu().contiguous()
                output_sd[f"{core}.lora_up.weight"] = merged_up.to(save_dtype).cpu().contiguous()
                output_sd[f"{core}.alpha"] = torch.tensor(float(max_rank), dtype=save_dtype)

                del merged_down, merged_up
                for d, _ in downs: del d
                for u, _ in ups: del u
                downs.clear()
                ups.clear()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pbar.update(1)

        # Final Summary
        if verbose:
            matched_stats = {i: 0 for i in range(len(lora_paths))}
            for indices in layer_map.values():
                for idx in indices:
                    matched_stats[idx] += 1

            print(f"[LoRA Multi-Merge Enhanced DARE] --- Merge Summary ---")
            for i, info in enumerate(lora_infos):
                print(f"[LoRA Multi-Merge Enhanced DARE] LoRA {i+1}: {matched_stats[i]}/{len(info['pairs'])} layers used in merge")
            print(f"[LoRA Multi-Merge Enhanced DARE] Output state dict has {len(output_sd)} tensors")

        # Save
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")
        save_file(output_sd, output_path, {"ss_training_comment": "Merged via Enhanced DARE-Ties"})
        return output_path

    finally:
        for h in handlers: h.__exit__(None, None, None)
        cleanup_after_operation()

