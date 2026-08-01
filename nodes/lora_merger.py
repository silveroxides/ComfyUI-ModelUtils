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
from typing import List, Dict, Tuple, Optional
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation
from .lora_resize import (
    detect_lora_format,
    detect_lora_rank,
    parse_lora_layers,
    select_output_dtype,
)
from .quantization_guard import inspect_low_bit_input, layer_has_low_bit, write_preserved_tensor

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned, IncrementalSafetensorsWriter


BASE_PREFIXES = ["model.diffusion_model.", "diffusion_model.", "transformer.", "model.", "net."]
LORA_PREFIXES = [
    "lora_unet_", "lora_transformer_", "lora_te1_", "lora_te2_", "lora_te_",
    "lycoris_", "diffusion_model.", "transformer.", "unet.",
]


def _strip_prefix(value: str, prefixes: List[str]) -> str:
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _lora_core(block_name: str) -> str:
    core = _strip_prefix(block_name, LORA_PREFIXES)
    if core.endswith(".lora"):
        core = core[:-5]
    return core


def _build_layer_map(lora_infos: List[Dict], base_model_path: Optional[str]) -> Dict[str, List[Tuple[int, str]]]:
    """Build a union map while optionally normalizing names against a reference model."""
    if not base_model_path:
        layer_map = {}
        for idx, info in enumerate(lora_infos):
            for block_name in info["pairs"]:
                layer_map.setdefault(block_name, []).append((idx, block_name))
        return layer_map

    with MemoryEfficientSafeOpen(base_model_path, low_memory=True) as base_handler:
        base_keys = list(base_handler.keys())

    normalized_base = {}
    for base_key in base_keys:
        normalized = _strip_prefix(base_key, BASE_PREFIXES)
        normalized_base[normalized] = base_key
        normalized_base[normalized.replace(".", "_")] = base_key

    layer_map = {}
    mapped = set()
    for idx, info in enumerate(lora_infos):
        for block_name, block_keys in info["pairs"].items():
            core = _lora_core(block_name)
            target = None
            if "down" in block_keys or "up" in block_keys:
                candidate = f"{core}.weight"
                target = normalized_base.get(candidate) or normalized_base.get(candidate.replace(".", "_"))
            elif "diff_b" in block_keys:
                candidate = f"{core}.bias"
                target = normalized_base.get(candidate) or normalized_base.get(candidate.replace(".", "_"))
            elif "diff" in block_keys:
                target = normalized_base.get(core) or normalized_base.get(core.replace(".", "_"))
                if target is None:
                    candidate = f"{core}.weight"
                    target = normalized_base.get(candidate) or normalized_base.get(candidate.replace(".", "_"))

            if target is None:
                continue
            normalized_target = _strip_prefix(target, BASE_PREFIXES)
            if "diff_b" in block_keys and normalized_target.endswith(".bias"):
                output_core = normalized_target[:-5]
            elif "diff" in block_keys and normalized_target.endswith(".weight"):
                output_core = normalized_target[:-7]
            elif "diff" in block_keys:
                output_core = normalized_target
            else:
                output_core = normalized_target[:-7] if normalized_target.endswith(".weight") else normalized_target
            output_core = f"diffusion_model.{output_core}"
            layer_map.setdefault(output_core, []).append((idx, block_name))
            mapped.add((idx, block_name))

    # Direct patches must not disappear merely because a reference model cannot resolve them.
    for idx, info in enumerate(lora_infos):
        for block_name, block_keys in info["pairs"].items():
            if (idx, block_name) not in mapped and ("diff" in block_keys or "diff_b" in block_keys):
                layer_map.setdefault(block_name, []).append((idx, block_name))
    return layer_map


def _load_direct_inputs(indices, lora_infos, handlers, kind, device, include_1d_diffs):
    tensors = []
    source_dtypes = []
    expected_shape = None
    for idx, orig_block in indices:
        block_keys = lora_infos[idx]["pairs"][orig_block]
        if kind not in block_keys:
            continue  # Missing direct layers are implicit zero contributions.
        key = block_keys[kind]
        tensor = handlers[idx].get_tensor(key)
        if tensor.ndim == 1 and not include_1d_diffs:
            continue
        if expected_shape is None:
            expected_shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Direct LoRA shape mismatch for {key}: {tuple(tensor.shape)} != {expected_shape}"
            )
        source_dtypes.append(handlers[idx].get_dtype(key))
        tensors.append((tensor.to(device=device, dtype=torch.float32), lora_infos[idx]["weight"]))
    return tensors, source_dtypes


def _logical_layer_dtypes(indices, lora_infos, handlers):
    source_dtypes = []
    for idx, orig_block in indices:
        block_keys = lora_infos[idx]["pairs"][orig_block]
        for name in ("down", "up", "alpha", "diff", "diff_b"):
            if name in block_keys:
                source_dtypes.append(handlers[idx].get_dtype(block_keys[name]))
    return source_dtypes


def _merge_direct_weighted(tensors):
    result = torch.zeros_like(tensors[0][0])
    for tensor, weight in tensors:
        result.add_(tensor, alpha=weight)
    return result


def _inspect_lora_inputs(handlers, lora_paths, operation):
    return [
        inspect_low_bit_input(
            handler, f"LoRA {i + 1} ({lora_paths[i]})", operation
        )
        for i, handler in enumerate(handlers)
    ]


def _indices_have_low_bit(indices, lora_infos):
    return any(
        layer_has_low_bit(
            lora_infos[idx]["pairs"][block_name],
            lora_infos[idx]["low_bit_keys"],
        )
        for idx, block_name in indices
    )


def _preserve_earliest_layer(writer, indices, lora_infos, handlers, written_keys):
    """Write the earliest selected adapter's original representation unchanged."""
    idx, block_name = min(indices, key=lambda item: item[0])
    block_keys = lora_infos[idx]["pairs"][block_name]
    count = 0
    for name in ("down", "up", "alpha", "diff", "diff_b"):
        key = block_keys.get(name)
        if key is None or key in written_keys:
            continue
        write_preserved_tensor(writer, key, handlers[idx])
        written_keys.add(key)
        count += 1
    for key in lora_infos[idx]["passthrough_keys"]:
        if not key.startswith(f"{block_name}.") or key in written_keys:
            continue
        write_preserved_tensor(writer, key, handlers[idx])
        written_keys.add(key)
        count += 1
    return count


def _preserve_low_bit_auxiliaries(writer, lora_infos, handlers, written_keys):
    """Preserve isolated unparsed low-bit tensors from the earliest adapter."""
    count = 0
    affected_keys = {
        key
        for info in lora_infos
        for key in info["passthrough_keys"]
        if key in info["low_bit_keys"]
    }
    for key in sorted(affected_keys):
        for idx, info in enumerate(lora_infos):
            if key not in info["passthrough_keys"]:
                continue
            if key not in written_keys:
                write_preserved_tensor(writer, key, handlers[idx])
                written_keys.add(key)
                count += 1
            break
    return count


def _ensure_guarded_layers_mapped(layer_map, lora_infos):
    mapped = {
        (idx, block_name)
        for indices in layer_map.values()
        for idx, block_name in indices
    }
    for idx, info in enumerate(lora_infos):
        for block_name, block_keys in info["pairs"].items():
            entry = (idx, block_name)
            if entry not in mapped and layer_has_low_bit(block_keys, info["low_bit_keys"]):
                layer_map.setdefault(block_name, []).append(entry)



def merge_multi_loras(
    lora_paths: List[str],
    lora_weights: List[float],
    merge_mode: str,
    device: str,
    save_dtype: torch.dtype,
    output_filename: str,
    base_model_path: Optional[str] = None,
    verbose: bool = True,
    include_1d_diffs: bool = False,
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
        low_bit_keys = _inspect_lora_inputs(handlers, lora_paths, "LoRA Multi-Merge")

        for i, handler in enumerate(handlers):
            keys = handler.keys()
            fmt = detect_lora_format(keys)
            pairs, passthrough_keys = parse_lora_layers(keys)
            rank, alpha = detect_lora_rank(handler, pairs, low_bit_keys[i])

            info = {
                "name": os.path.basename(lora_paths[i]),
                "format": fmt,
                "pairs": pairs,
                "rank": rank,
                "alpha": alpha,
                "weight": lora_weights[i],
                "scale": alpha / rank if rank > 0 else 1.0,
                "low_bit_keys": low_bit_keys[i],
                "passthrough_keys": passthrough_keys,
            }
            lora_infos.append(info)

            if verbose:
                print(f"[LoRA Multi-Merge] LoRA {i+1} ({info['name']}): {len(pairs)} layers, dim={rank}, format={fmt['format']}")

        layer_map = _build_layer_map(lora_infos, base_model_path)
        _ensure_guarded_layers_mapped(layer_map, lora_infos)

        if verbose:
            print(f"[LoRA Multi-Merge] Total unique layers after resolving naming: {len(layer_map)}")

        # Build metadata and output path before loop
        metadata = {
            "ss_training_comment": f"Merged {len(lora_paths)} LoRAs via concatenation",
            "ss_network_module": "networks.lora",
        }
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")

        pbar = comfy.utils.ProgressBar(len(layer_map))
        tensor_count = 0
        written_keys = set()

        writer = IncrementalSafetensorsWriter(output_path, metadata=metadata)
        writer.__enter__()
        try:
            # 2. Merge each core layer
            with torch.no_grad():
                for core, indices in tqdm(layer_map.items(), desc="Merging layers", unit="layers"):
                    if _indices_have_low_bit(indices, lora_infos):
                        tensor_count += _preserve_earliest_layer(
                            writer, indices, lora_infos, handlers, written_keys
                        )
                        pbar.update(1)
                        continue
                    logical_source_dtypes = _logical_layer_dtypes(indices, lora_infos, handlers)
                    downs = []
                    ups = []
                    max_rank = 0

                    # First pass: load and determine max rank
                    for idx, orig_block in indices:
                        info = lora_infos[idx]
                        handler = handlers[idx]
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
                        current_rank = t_down.shape[0]
                        max_rank = max(max_rank, current_rank)

                        # Apply scale immediately for concatenate mode, or keep for later
                        if merge_mode == "concatenate":
                            effective_weight = info["weight"] * info["scale"]
                            t_up = t_up * effective_weight

                        downs.append((t_down, info))
                        ups.append((t_up, info))

                    if downs and merge_mode == "concatenate":
                        # Stack ranks
                        merged_down = torch.cat([d[0] for d in downs], dim=0)
                        merged_up = torch.cat([u[0] for u in ups], dim=1)
                        new_rank = merged_down.shape[0]
                        new_alpha = float(new_rank)
                    elif downs:
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

                    if downs:
                        layer_dtype = select_output_dtype(logical_source_dtypes, save_dtype)
                        writer.write_dict({
                            f"{core}.lora_down.weight": merged_down.to(layer_dtype).cpu().contiguous(),
                            f"{core}.lora_up.weight": merged_up.to(layer_dtype).cpu().contiguous(),
                            f"{core}.alpha": torch.tensor(new_alpha, dtype=layer_dtype),
                        })
                        tensor_count += 3
                        written_keys.update({
                            f"{core}.lora_down.weight", f"{core}.lora_up.weight", f"{core}.alpha"
                        })
                        del merged_down, merged_up
                        for d, _ in downs: del d
                        for u, _ in ups: del u
                        downs.clear()
                        ups.clear()

                    for kind in ("diff", "diff_b"):
                        direct_inputs, _ = _load_direct_inputs(
                            indices, lora_infos, handlers, kind, device, include_1d_diffs
                        )
                        if not direct_inputs:
                            continue
                        merged_direct = _merge_direct_weighted(direct_inputs)
                        direct_dtype = select_output_dtype(
                            logical_source_dtypes,
                            save_dtype,
                            is_1d_diff=merged_direct.ndim == 1,
                        )
                        writer.write(
                            f"{core}.{kind}",
                            merged_direct.to(direct_dtype).cpu().contiguous(),
                        )
                        tensor_count += 1
                        written_keys.add(f"{core}.{kind}")
                        del merged_direct
                        for tensor, _ in direct_inputs:
                            del tensor

                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pbar.update(1)
                tensor_count += _preserve_low_bit_auxiliaries(
                    writer, lora_infos, handlers, written_keys
                )
        finally:
            writer.__exit__(None, None, None)

        # 3. Final Summary
        if verbose:
            matched_stats = {i: 0 for i in range(len(lora_paths))}
            for indices in layer_map.values():
                for idx, block_name in indices:
                    matched_stats[idx] += 1

            print(f"[LoRA Multi-Merge] --- Merge Summary ---")
            for i, info in enumerate(lora_infos):
                print(f"[LoRA Multi-Merge] LoRA {i+1}: {matched_stats[i]}/{len(info['pairs'])} layers used in merge")
            print(f"[LoRA Multi-Merge] Output state dict has {tensor_count} tensors")
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
                io.Combo.Input("base_model", options=["None"] + folder_paths.get_filename_list("diffusion_models"), default="None",
                              tooltip="Optional reference model to resolve key naming issues across formats. If None, input keys are preserved verbatim."),
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
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Include and merge 1D direct-diff tensors as FP32."),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, base_model, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                merge_mode, output_filename, save_dtype, device, include_1d_diffs) -> io.NodeOutput:

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
        base_path = None
        if base_model and base_model != "None":
            base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)

        path = merge_multi_loras(
            lora_paths=valid_paths,
            lora_weights=valid_weights,
            merge_mode=merge_mode,
            device=device,
            save_dtype=dtype,
            output_filename=output_filename,
            base_model_path=base_path,
            include_1d_diffs=include_1d_diffs,
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
                io.Combo.Input("base_model", options=["None"] + folder_paths.get_filename_list("diffusion_models"), default="None",
                              tooltip="Optional reference model to resolve key naming issues across formats. If None, input keys are preserved verbatim."),
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
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Include and merge 1D direct-diff tensors as FP32."),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, base_model, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                drop_rate, trim_quantile, seed, output_filename, save_dtype, device,
                include_1d_diffs) -> io.NodeOutput:

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
        base_path = None
        if base_model and base_model != "None":
            base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)

        path = merge_multi_loras_dare(
            lora_paths=valid_paths,
            lora_weights=valid_weights,
            drop_rate=drop_rate,
            trim_quantile=trim_quantile,
            seed=seed,
            device=device,
            save_dtype=dtype,
            output_filename=output_filename,
            base_model_path=base_path,
            include_1d_diffs=include_1d_diffs,
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
    base_model_path: Optional[str] = None,
    verbose: bool = True,
    include_1d_diffs: bool = False,
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
        low_bit_keys = _inspect_lora_inputs(
            handlers, lora_paths, "LoRA Multi-Merge DARE"
        )
        for i, handler in enumerate(handlers):
            keys = handler.keys()
            fmt = detect_lora_format(keys)
            pairs, passthrough_keys = parse_lora_layers(keys)
            rank, alpha = detect_lora_rank(handler, pairs, low_bit_keys[i])
            info = {
                "name": os.path.basename(lora_paths[i]),
                "pairs": pairs,
                "rank": rank,
                "alpha": alpha,
                "weight": lora_weights[i],
                "low_bit_keys": low_bit_keys[i],
                "passthrough_keys": passthrough_keys,
            }
            lora_infos.append(info)

            if verbose:
                print(f"[LoRA Multi-Merge DARE] LoRA {i+1} ({info['name']}): {len(pairs)} layers, dim={rank}")

        layer_map = _build_layer_map(lora_infos, base_model_path)
        _ensure_guarded_layers_mapped(layer_map, lora_infos)

        if verbose:
            print(f"[LoRA Multi-Merge DARE] Total unique layers after resolving naming: {len(layer_map)}")

        # Build output path before loop
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")

        pbar = comfy.utils.ProgressBar(len(layer_map))
        tensor_count = 0
        written_keys = set()

        writer = IncrementalSafetensorsWriter(output_path, metadata={"ss_training_comment": "Merged via DARE-Ties"})
        writer.__enter__()
        try:
            # 2. Merge
            with torch.no_grad():
                for core, indices in tqdm(layer_map.items(), desc="Merging layers (DARE)", unit="layers"):
                    if _indices_have_low_bit(indices, lora_infos):
                        tensor_count += _preserve_earliest_layer(
                            writer, indices, lora_infos, handlers, written_keys
                        )
                        pbar.update(1)
                        continue
                    logical_source_dtypes = _logical_layer_dtypes(indices, lora_infos, handlers)
                    downs = []
                    ups = []
                    max_rank = 0

                    for idx, orig_block in indices:
                        info = lora_infos[idx]
                        block_keys = info["pairs"][orig_block]
                        if "down" not in block_keys or "up" not in block_keys: continue

                        t_d = handlers[idx].get_tensor(block_keys["down"]).to(device=device, dtype=torch.float32)
                        t_u = handlers[idx].get_tensor(block_keys["up"]).to(device=device, dtype=torch.float32)

                        max_rank = max(max_rank, t_d.shape[0])
                        downs.append((t_d, info["weight"]))
                        ups.append((t_u, info["weight"]))

                    def process_ties_dare(tensors, weights, dim_to_pad):
                        # Pad all to max_rank
                        padded = []
                        for t, w in tensors:
                            if dim_to_pad is not None and t.shape[dim_to_pad] < max_rank:
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

                    if downs:
                        merged_down = process_ties_dare(downs, [1.0]*len(downs), 0)
                        merged_up = process_ties_dare(ups, [1.0]*len(ups), 1)
                        layer_dtype = select_output_dtype(logical_source_dtypes, save_dtype)
                        writer.write_dict({
                            f"{core}.lora_down.weight": merged_down.to(layer_dtype).cpu().contiguous(),
                            f"{core}.lora_up.weight": merged_up.to(layer_dtype).cpu().contiguous(),
                            f"{core}.alpha": torch.tensor(float(max_rank), dtype=layer_dtype),
                        })
                        tensor_count += 3
                        written_keys.update({
                            f"{core}.lora_down.weight", f"{core}.lora_up.weight", f"{core}.alpha"
                        })
                        del merged_down, merged_up
                        for d, _ in downs: del d
                        for u, _ in ups: del u
                        downs.clear()
                        ups.clear()

                    for kind in ("diff", "diff_b"):
                        direct_inputs, _ = _load_direct_inputs(
                            indices, lora_infos, handlers, kind, device, include_1d_diffs
                        )
                        if not direct_inputs:
                            continue
                        merged_direct = process_ties_dare(
                            direct_inputs, [1.0] * len(direct_inputs), None
                        )
                        direct_dtype = select_output_dtype(
                            logical_source_dtypes,
                            save_dtype,
                            is_1d_diff=merged_direct.ndim == 1,
                        )
                        writer.write(f"{core}.{kind}", merged_direct.to(direct_dtype).cpu().contiguous())
                        tensor_count += 1
                        written_keys.add(f"{core}.{kind}")
                        del merged_direct
                        for tensor, _ in direct_inputs:
                            del tensor
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pbar.update(1)
                tensor_count += _preserve_low_bit_auxiliaries(
                    writer, lora_infos, handlers, written_keys
                )
        finally:
            writer.__exit__(None, None, None)

        # Final Summary
        if verbose:
            matched_stats = {i: 0 for i in range(len(lora_paths))}
            for indices in layer_map.values():
                for idx, block_name in indices:
                    matched_stats[idx] += 1

            print(f"[LoRA Multi-Merge DARE] --- Merge Summary ---")
            for i, info in enumerate(lora_infos):
                print(f"[LoRA Multi-Merge DARE] LoRA {i+1}: {matched_stats[i]}/{len(info['pairs'])} layers used in merge")
            print(f"[LoRA Multi-Merge DARE] Output state dict has {tensor_count} tensors")

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
                io.Combo.Input("base_model", options=["None"] + folder_paths.get_filename_list("diffusion_models"), default="None",
                              tooltip="Optional reference model to resolve key naming issues across formats. If None, input keys are preserved verbatim."),
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
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Include and merge 1D direct-diff tensors as FP32."),
            ],
            outputs=[io.String.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, base_model, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                mask_power, min_keep_prob, mask_smooth, trim_quantile, seed, output_filename, save_dtype, device,
                include_1d_diffs) -> io.NodeOutput:

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
        base_path = None
        if base_model and base_model != "None":
            base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)

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
            output_filename=output_filename,
            base_model_path=base_path,
            include_1d_diffs=include_1d_diffs,
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
    base_model_path: Optional[str] = None,
    verbose: bool = True,
    include_1d_diffs: bool = False,
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
        low_bit_keys = _inspect_lora_inputs(
            handlers, lora_paths, "LoRA Multi-Merge Enhanced DARE"
        )
        for i, handler in enumerate(handlers):
            keys = handler.keys()
            fmt = detect_lora_format(keys)
            pairs, passthrough_keys = parse_lora_layers(keys)
            rank, alpha = detect_lora_rank(handler, pairs, low_bit_keys[i])
            info = {
                "name": os.path.basename(lora_paths[i]),
                "pairs": pairs,
                "rank": rank,
                "alpha": alpha,
                "weight": lora_weights[i],
                "low_bit_keys": low_bit_keys[i],
                "passthrough_keys": passthrough_keys,
            }
            lora_infos.append(info)

            if verbose:
                print(f"[LoRA Multi-Merge Enhanced DARE] LoRA {i+1} ({info['name']}): {len(pairs)} layers, dim={rank}")

        layer_map = _build_layer_map(lora_infos, base_model_path)
        _ensure_guarded_layers_mapped(layer_map, lora_infos)

        if verbose:
            print(f"[LoRA Multi-Merge Enhanced DARE] Total unique layers after resolving naming: {len(layer_map)}")

        # Build output path before loop
        output_dir = folder_paths.get_folder_paths("loras")[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename.strip()}.safetensors")

        pbar = comfy.utils.ProgressBar(len(layer_map))
        tensor_count = 0
        written_keys = set()

        writer = IncrementalSafetensorsWriter(output_path, metadata={"ss_training_comment": "Merged via Enhanced DARE-Ties"})
        writer.__enter__()
        try:
            # 2. Merge
            with torch.no_grad():
                for core, indices in tqdm(layer_map.items(), desc="Merging layers (Enhanced DARE)", unit="layers"):
                    if _indices_have_low_bit(indices, lora_infos):
                        tensor_count += _preserve_earliest_layer(
                            writer, indices, lora_infos, handlers, written_keys
                        )
                        pbar.update(1)
                        continue
                    logical_source_dtypes = _logical_layer_dtypes(indices, lora_infos, handlers)
                    downs = []
                    ups = []
                    max_rank = 0

                    for idx, orig_block in indices:
                        info = lora_infos[idx]
                        block_keys = info["pairs"][orig_block]
                        if "down" not in block_keys or "up" not in block_keys: continue

                        t_d = handlers[idx].get_tensor(block_keys["down"]).to(device=device, dtype=torch.float32)
                        t_u = handlers[idx].get_tensor(block_keys["up"]).to(device=device, dtype=torch.float32)

                        max_rank = max(max_rank, t_d.shape[0])
                        downs.append((t_d, info["weight"]))
                        ups.append((t_u, info["weight"]))

                    def process_ties_dare_enhanced(tensors, weights, dim_to_pad):
                        processed = []
                        for t, w in tensors:
                            t_val = t * w

                            # Enhanced DARE
                            abs_t = torch.abs(t_val)
                            max_val = torch.max(abs_t)
                            if max_val > 0:
                                prob = torch.clamp((abs_t / max_val) ** max(mask_power, 0.001), min=min_keep_prob, max=1.0)
                                prob = torch.nan_to_num(prob)

                                random_mask = torch.bernoulli(prob, generator=rng)
                                interpolated_mask = torch.lerp(random_mask, prob, mask_smooth)

                                t_val = t_val * interpolated_mask

                            # TIES
                            if trim_quantile > 0:
                                flat = t_val.abs().flatten()
                                k = max(1, int(len(flat) * trim_quantile))
                                if k > 0:
                                    threshold = torch.kthvalue(flat, k).values
                                    t_val = torch.where(t_val.abs() < threshold, torch.zeros_like(t_val), t_val)

                            # Pad
                            if dim_to_pad is not None and t_val.shape[dim_to_pad] < max_rank:
                                padding = [0] * (len(t_val.shape) * 2)
                                rev_dim = len(t_val.shape) - 1 - dim_to_pad
                                padding[rev_dim*2 + 1] = max_rank - t_val.shape[dim_to_pad]
                                t_val = torch.nn.functional.pad(t_val, tuple(padding))

                            processed.append(t_val)

                        stacked = torch.stack(processed)
                        signs = torch.sign(stacked)
                        sum_signs = signs.sum(dim=0)
                        dominant_sign = torch.sign(sum_signs)

                        mask = (signs == dominant_sign) & (dominant_sign != 0)
                        filtered = torch.where(mask, stacked, torch.zeros_like(stacked))

                        count = mask.sum(dim=0)
                        result = filtered.sum(dim=0) / torch.clamp(count, min=1.0)
                        return result

                    if downs:
                        merged_down = process_ties_dare_enhanced(downs, [1.0]*len(downs), 0)
                        merged_up = process_ties_dare_enhanced(ups, [1.0]*len(ups), 1)
                        layer_dtype = select_output_dtype(logical_source_dtypes, save_dtype)
                        writer.write_dict({
                            f"{core}.lora_down.weight": merged_down.to(layer_dtype).cpu().contiguous(),
                            f"{core}.lora_up.weight": merged_up.to(layer_dtype).cpu().contiguous(),
                            f"{core}.alpha": torch.tensor(float(max_rank), dtype=layer_dtype),
                        })
                        tensor_count += 3
                        written_keys.update({
                            f"{core}.lora_down.weight", f"{core}.lora_up.weight", f"{core}.alpha"
                        })
                        del merged_down, merged_up
                        for d, _ in downs: del d
                        for u, _ in ups: del u
                        downs.clear()
                        ups.clear()

                    for kind in ("diff", "diff_b"):
                        direct_inputs, _ = _load_direct_inputs(
                            indices, lora_infos, handlers, kind, device, include_1d_diffs
                        )
                        if not direct_inputs:
                            continue
                        merged_direct = process_ties_dare_enhanced(
                            direct_inputs, [1.0] * len(direct_inputs), None
                        )
                        direct_dtype = select_output_dtype(
                            logical_source_dtypes,
                            save_dtype,
                            is_1d_diff=merged_direct.ndim == 1,
                        )
                        writer.write(f"{core}.{kind}", merged_direct.to(direct_dtype).cpu().contiguous())
                        tensor_count += 1
                        written_keys.add(f"{core}.{kind}")
                        del merged_direct
                        for tensor, _ in direct_inputs:
                            del tensor
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pbar.update(1)
                tensor_count += _preserve_low_bit_auxiliaries(
                    writer, lora_infos, handlers, written_keys
                )
        finally:
            writer.__exit__(None, None, None)

        # Final Summary
        if verbose:
            matched_stats = {i: 0 for i in range(len(lora_paths))}
            for indices in layer_map.values():
                for idx, block_name in indices:
                    matched_stats[idx] += 1

            print(f"[LoRA Multi-Merge Enhanced DARE] --- Merge Summary ---")
            for i, info in enumerate(lora_infos):
                print(f"[LoRA Multi-Merge Enhanced DARE] LoRA {i+1}: {matched_stats[i]}/{len(info['pairs'])} layers used in merge")
            print(f"[LoRA Multi-Merge Enhanced DARE] Output state dict has {tensor_count} tensors")

        return output_path

    finally:
        for h in handlers: h.__exit__(None, None, None)
        cleanup_after_operation()

