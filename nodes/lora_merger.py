"""
LoRA Multi-Merge - Merge multiple LoRAs into a single LoRA file.
Resolves different naming conventions and ranks via concatenation.
"""
import os
import gc
from collections import Counter
import torch
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from typing import List, Optional
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation
from .lora_resize import (
    build_lora_layer_map,
    canonical_lora_key,
    detect_lora_format,
    layer_has_companions,
    layer_tensor_keys,
    parse_lora_layers,
    select_output_dtype,
    validate_canonical_blocks,
)
from .quantization_guard import inspect_low_bit_input, layer_has_low_bit, write_preserved_tensor
from .uel_io import AsyncTensorCursor as _AsyncTensorCursor, atomic_uel_writer
from .lora_alpha import normalize_lora_pair
from .artifact_paths import canonical_model_artifact_path

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned


def _logical_output_dtypes(indices, lora_infos, handlers):
    pair_dtypes = []
    direct_dtypes = {role: [] for role in _DIRECT_MERGE_ROLES}
    for idx, orig_block in indices:
        block_keys = lora_infos[idx]["pairs"][orig_block]
        for name in ("down", "up"):
            if name in block_keys:
                pair_dtypes.append(handlers[idx].get_dtype(block_keys[name]))
        for role in _DIRECT_MERGE_ROLES:
            if role in block_keys:
                direct_dtypes[role].append(
                    handlers[idx].get_dtype(block_keys[role])
                )
    return pair_dtypes, direct_dtypes


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


def _indices_have_companions(indices, lora_infos):
    return any(
        layer_has_companions(lora_infos[idx]["pairs"][block_name])
        for idx, block_name in indices
    )


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


_DIRECT_MERGE_ROLES = ("diff", "diff_b", "w_norm", "b_norm")


def _rank_distribution(handler, pairs, low_bit_keys):
    ranks = Counter()
    for block_keys in pairs.values():
        if (
            "down" in block_keys
            and "up" in block_keys
            and not layer_has_low_bit(block_keys, low_bit_keys)
        ):
            ranks[int(handler.get_shape(block_keys["down"])[0])] += 1
    return ranks


def _format_rank_distribution(ranks):
    if not ranks:
        return "none"
    return ", ".join(f"{rank}x{count}" for rank, count in sorted(ranks.items()))


def _append_work_entry(unit, claimed_sources, idx, key, output_key, role):
    source = (idx, key)
    if source in claimed_sources:
        return
    unit["entries"].append({
        "idx": idx,
        "key": key,
        "output_key": output_key,
        "role": role,
    })
    claimed_sources.add(source)


def _build_merge_work_units(
    layer_map, lora_infos, handlers, include_1d_diffs
):
    units = []
    claimed_sources = set()
    claimed_outputs = set()

    for core, indices in layer_map.items():
        guarded = _indices_have_companions(indices, lora_infos)
        raw = _indices_have_low_bit(indices, lora_infos)
        if guarded or raw:
            idx, block_name = min(indices, key=lambda item: item[0])
            block_keys = lora_infos[idx]["pairs"][block_name]
            if raw and "alpha" in block_keys and any(
                block_keys.get(role) in lora_infos[idx]["low_bit_keys"]
                for role in ("down", "up")
            ):
                raise ValueError(
                    f"Cannot alpha-normalize low-bit LoRA factors for '{block_name}'."
                )
            unit = {
                "kind": "raw_copy" if raw else "copy",
                "core": core,
                "entries": [],
            }
            for role, key in layer_tensor_keys(block_keys).items():
                output_key = canonical_lora_key(core, role)
                if output_key in claimed_outputs:
                    continue
                _append_work_entry(
                    unit, claimed_sources, idx, key, output_key, role
                )
                claimed_outputs.add(output_key)
            for key in lora_infos[idx]["passthrough_keys"]:
                if not key.startswith(f"{block_name}.") or key in claimed_outputs:
                    continue
                _append_work_entry(unit, claimed_sources, idx, key, key, None)
                claimed_outputs.add(key)
            if unit["entries"]:
                units.append(unit)
            continue

        pair_dtypes, direct_dtypes = _logical_output_dtypes(
            indices, lora_infos, handlers
        )
        unit = {
            "kind": "merge",
            "core": core,
            "indices": list(indices),
            "entries": [],
            "pair_source_dtypes": pair_dtypes,
            "direct_source_dtypes": direct_dtypes,
        }
        for idx, block_name in indices:
            block_keys = lora_infos[idx]["pairs"][block_name]
            if "down" in block_keys and "up" in block_keys:
                for role in ("down", "up", "alpha"):
                    if role in block_keys:
                        _append_work_entry(
                            unit,
                            claimed_sources,
                            idx,
                            block_keys[role],
                            None,
                            role,
                        )
            for role in _DIRECT_MERGE_ROLES:
                if role not in block_keys:
                    continue
                key = block_keys[role]
                if (
                    len(handlers[idx].get_shape(key)) == 1
                    and not include_1d_diffs
                ):
                    continue
                _append_work_entry(
                    unit, claimed_sources, idx, key, None, role
                )
        if unit["entries"]:
            units.append(unit)

    affected_low_bit_passthrough = {
        key
        for info in lora_infos
        for key in info["passthrough_keys"]
        if key in info["low_bit_keys"]
    }
    for key in sorted(affected_low_bit_passthrough):
        if key in claimed_outputs:
            continue
        for idx, info in enumerate(lora_infos):
            if key not in info["passthrough_keys"]:
                continue
            unit = {"kind": "raw_copy", "core": key, "entries": []}
            _append_work_entry(unit, claimed_sources, idx, key, key, None)
            claimed_outputs.add(key)
            units.append(unit)
            break

    for key in lora_infos[0]["passthrough_keys"]:
        if (0, key) in claimed_sources or key in claimed_outputs:
            continue
        unit = {
            "kind": (
                "raw_copy"
                if key in lora_infos[0]["low_bit_keys"]
                else "copy"
            ),
            "core": key,
            "entries": [],
        }
        _append_work_entry(unit, claimed_sources, 0, key, key, None)
        claimed_outputs.add(key)
        units.append(unit)

    stream_keys = [[] for _ in handlers]
    for unit in units:
        if unit["kind"] == "raw_copy":
            continue
        for entry in unit["entries"]:
            stream_keys[entry["idx"]].append(entry["key"])
    return units, stream_keys


def _is_cuda_oom_error(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
        and "cuda" in str(error).lower()
    )


def _release_cuda_oom_state():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_processing_device(tensor, process_device):
    if process_device.type == "cuda":
        return transfer_to_gpu_pinned(tensor, process_device, torch.float32)
    return tensor.to(device=process_device, dtype=torch.float32)


def _pad_rank(tensor, rank_dimension, target_rank):
    if rank_dimension is None or tensor.shape[rank_dimension] == target_rank:
        return tensor
    padding = [0] * (tensor.ndim * 2)
    reverse_dimension = tensor.ndim - 1 - rank_dimension
    padding[reverse_dimension * 2 + 1] = (
        target_rank - tensor.shape[rank_dimension]
    )
    return torch.nn.functional.pad(tensor, tuple(padding))


def _ties_result(values):
    stacked = torch.stack(values)
    signs = torch.sign(stacked)
    dominant_sign = torch.sign(signs.sum(dim=0))
    mask = (signs == dominant_sign) & (dominant_sign != 0)
    filtered = torch.where(mask, stacked, torch.zeros_like(stacked))
    return filtered.sum(dim=0) / torch.clamp(mask.sum(dim=0), min=1.0)


def _dare_ties_merge(
    contributions,
    target_rank,
    rank_dimension,
    drop_rate,
    trim_quantile,
    generator,
):
    values = []
    for tensor, weight in contributions:
        value = _pad_rank(tensor, rank_dimension, target_rank) * weight
        if drop_rate > 0:
            mask = (
                torch.rand(
                    value.shape,
                    generator=generator,
                    device=value.device,
                )
                > drop_rate
            ).to(value.dtype)
            value = (value * mask) / (1 - drop_rate)
        if trim_quantile > 0:
            flat = value.abs().flatten()
            k = int(flat.numel() * trim_quantile)
            if k > 0:
                threshold = torch.kthvalue(flat, k).values
                value = torch.where(
                    value.abs() < threshold, torch.zeros_like(value), value
                )
        values.append(value)
    return _ties_result(values)


def _enhanced_dare_ties_merge(
    contributions,
    target_rank,
    rank_dimension,
    mask_power,
    min_keep_prob,
    mask_smooth,
    trim_quantile,
    generator,
):
    values = []
    for tensor, weight in contributions:
        value = tensor * weight
        absolute = value.abs()
        maximum = absolute.max()
        if maximum > 0:
            probability = torch.clamp(
                (absolute / maximum) ** max(mask_power, 0.001),
                min=min_keep_prob,
                max=1.0,
            )
            probability = torch.nan_to_num(probability)
            random_mask = torch.bernoulli(probability, generator=generator)
            value = value * torch.lerp(random_mask, probability, mask_smooth)
        if trim_quantile > 0:
            flat = value.abs().flatten()
            k = max(1, int(flat.numel() * trim_quantile))
            threshold = torch.kthvalue(flat, k).values
            value = torch.where(
                value.abs() < threshold, torch.zeros_like(value), value
            )
        values.append(_pad_rank(value, rank_dimension, target_rank))
    return _ties_result(values)


def _process_merge_unit_on_device(
    unit,
    loaded,
    lora_infos,
    strategy,
    settings,
    save_dtype,
    process_device,
    work_index,
):
    pair_contributions = []
    direct_contributions = {role: [] for role in _DIRECT_MERGE_ROLES}
    maximum_rank = 0

    for idx, block_name in unit["indices"]:
        info = lora_infos[idx]
        block_keys = info["pairs"][block_name]
        if "down" in block_keys and "up" in block_keys:
            down_cpu = loaded[(idx, block_keys["down"])]
            up_cpu = loaded[(idx, block_keys["up"])]
            source_rank = int(down_cpu.shape[0])
            if up_cpu.ndim < 2 or int(up_cpu.shape[1]) != source_rank:
                raise ValueError(
                    f"LoRA rank mismatch for {block_keys['down']} and {block_keys['up']}"
                )
            maximum_rank = max(maximum_rank, source_rank)
            alpha = (
                loaded[(idx, block_keys["alpha"])]
                if "alpha" in block_keys else None
            )
            down_cpu, up_cpu = normalize_lora_pair(
                down_cpu,
                up_cpu,
                alpha,
                layer=block_name,
            )
            loaded[(idx, block_keys["down"])] = down_cpu
            loaded[(idx, block_keys["up"])] = up_cpu
            pair_contributions.append(
                (
                    _to_processing_device(down_cpu, process_device),
                    _to_processing_device(up_cpu, process_device),
                    float(info["weight"]),
                )
            )
        for role in _DIRECT_MERGE_ROLES:
            if role not in block_keys:
                continue
            key = block_keys[role]
            source = loaded.get((idx, key))
            if source is None:
                continue
            direct_contributions[role].append(
                (
                    _to_processing_device(source, process_device),
                    float(info["weight"]),
                )
            )

    generator = None
    if strategy in {"dare", "enhanced_dare"}:
        generator = torch.Generator(device=process_device).manual_seed(
            (int(settings["seed"]) + work_index) % (2**63 - 1)
        )

    outputs = []
    output_rank = None
    if pair_contributions:
        if strategy == "concatenate":
            merged_down = torch.cat(
                [down for down, _, _ in pair_contributions], dim=0
            )
            merged_up = torch.cat(
                [up * weight for _, up, weight in pair_contributions], dim=1
            )
            output_rank = int(merged_down.shape[0])
        elif strategy == "weighted_sum":
            first_down, first_up, _ = pair_contributions[0]
            target_down_shape = list(first_down.shape)
            target_down_shape[0] = maximum_rank
            target_up_shape = list(first_up.shape)
            target_up_shape[1] = maximum_rank
            merged_down = torch.zeros(
                target_down_shape,
                device=process_device,
                dtype=torch.float32,
            )
            merged_up = torch.zeros(
                target_up_shape,
                device=process_device,
                dtype=torch.float32,
            )
            for down, up, weight in pair_contributions:
                merged_down.add_(_pad_rank(down, 0, maximum_rank), alpha=weight)
                merged_up.add_(_pad_rank(up, 1, maximum_rank), alpha=weight)
            output_rank = maximum_rank
        elif strategy == "dare":
            merged_down = _dare_ties_merge(
                [(down, weight) for down, _, weight in pair_contributions],
                maximum_rank,
                0,
                settings["drop_rate"],
                settings["trim_quantile"],
                generator,
            )
            merged_up = _dare_ties_merge(
                [(up, weight) for _, up, weight in pair_contributions],
                maximum_rank,
                1,
                settings["drop_rate"],
                settings["trim_quantile"],
                generator,
            )
            output_rank = maximum_rank
        else:
            merged_down = _enhanced_dare_ties_merge(
                [(down, weight) for down, _, weight in pair_contributions],
                maximum_rank,
                0,
                settings["mask_power"],
                settings["min_keep_prob"],
                settings["mask_smooth"],
                settings["trim_quantile"],
                generator,
            )
            merged_up = _enhanced_dare_ties_merge(
                [(up, weight) for _, up, weight in pair_contributions],
                maximum_rank,
                1,
                settings["mask_power"],
                settings["min_keep_prob"],
                settings["mask_smooth"],
                settings["trim_quantile"],
                generator,
            )
            output_rank = maximum_rank

        layer_dtype = select_output_dtype(
            unit["pair_source_dtypes"], save_dtype
        )
        outputs.extend([
            (
                canonical_lora_key(unit["core"], "down"),
                merged_down.to(layer_dtype).cpu().contiguous(),
            ),
            (
                canonical_lora_key(unit["core"], "up"),
                merged_up.to(layer_dtype).cpu().contiguous(),
            ),
        ])

    for role, contributions in direct_contributions.items():
        if not contributions:
            continue
        expected_shape = tuple(contributions[0][0].shape)
        for tensor, _ in contributions[1:]:
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Direct LoRA shape mismatch for {unit['core']}: "
                    f"{tuple(tensor.shape)} != {expected_shape}"
                )
        if strategy in {"concatenate", "weighted_sum"}:
            merged_direct = _merge_direct_weighted(contributions)
        elif strategy == "dare":
            merged_direct = _dare_ties_merge(
                contributions,
                0,
                None,
                settings["drop_rate"],
                settings["trim_quantile"],
                generator,
            )
        else:
            merged_direct = _enhanced_dare_ties_merge(
                contributions,
                0,
                None,
                settings["mask_power"],
                settings["min_keep_prob"],
                settings["mask_smooth"],
                settings["trim_quantile"],
                generator,
            )
        direct_dtype = select_output_dtype(
            unit["direct_source_dtypes"][role],
            save_dtype,
            is_1d_diff=merged_direct.ndim == 1,
        )
        outputs.append(
            (
                canonical_lora_key(unit["core"], role),
                merged_direct.to(direct_dtype).cpu().contiguous(),
            )
        )
    return outputs, output_rank


def _process_merge_unit(
    unit,
    loaded,
    lora_infos,
    strategy,
    settings,
    save_dtype,
    device,
    work_index,
):
    process_device = torch.device(device)
    try:
        outputs, rank = _process_merge_unit_on_device(
            unit,
            loaded,
            lora_infos,
            strategy,
            settings,
            save_dtype,
            process_device,
            work_index,
        )
        return outputs, rank, False
    except Exception as error:
        if process_device.type != "cuda" or not _is_cuda_oom_error(error):
            raise
        _release_cuda_oom_state()
        outputs, rank = _process_merge_unit_on_device(
            unit,
            loaded,
            lora_infos,
            strategy,
            settings,
            save_dtype,
            torch.device("cpu"),
            work_index,
        )
        return outputs, rank, True


def _run_multi_lora_merge(
    lora_paths,
    lora_weights,
    strategy,
    settings,
    device,
    save_dtype,
    output_filename,
    base_model_path,
    verbose,
    include_1d_diffs,
):
    operation_names = {
        "concatenate": "LoRA Multi-Merge",
        "weighted_sum": "LoRA Multi-Merge",
        "dare": "LoRA Multi-Merge DARE",
        "enhanced_dare": "LoRA Multi-Merge Enhanced DARE",
    }
    operation = operation_names[strategy]
    total_size_gb = sum(estimate_model_size(path) for path in lora_paths)
    if verbose:
        print(f"[{operation}] Preparing memory for {total_size_gb:.2f}GB operation...")

    prepare_for_large_operation(total_size_gb * 2.5, torch.device(device))
    handlers = [MemoryEfficientSafeOpen(path, low_memory=True) for path in lora_paths]
    cursors = []
    try:
        low_bit_sets = _inspect_lora_inputs(handlers, lora_paths, operation)
        lora_infos = []
        for idx, handler in enumerate(handlers):
            keys = handler.keys()
            pairs, passthrough_keys = parse_lora_layers(keys)
            ranks = _rank_distribution(handler, pairs, low_bit_sets[idx])
            info = {
                "name": os.path.basename(lora_paths[idx]),
                "format": detect_lora_format(keys),
                "pairs": pairs,
                "weight": lora_weights[idx],
                "low_bit_keys": low_bit_sets[idx],
                "passthrough_keys": passthrough_keys,
                "ranks": ranks,
            }
            lora_infos.append(info)
            if verbose:
                print(
                    f"[{operation}] LoRA {idx + 1} ({info['name']}): "
                    f"{len(pairs)} target groups, "
                    f"factor-pair ranks={_format_rank_distribution(ranks)}, "
                    f"format={info['format']['format']}"
                )

        layer_map = build_lora_layer_map(lora_infos, base_model_path)
        _ensure_guarded_layers_mapped(layer_map, lora_infos)
        validate_canonical_blocks(layer_map, operation)
        units, stream_keys = _build_merge_work_units(
            layer_map, lora_infos, handlers, include_1d_diffs
        )
        if verbose:
            coverage = Counter(len(indices) for indices in layer_map.values())
            coverage_text = ", ".join(
                f"{contributors}-input={count}"
                for contributors, count in sorted(coverage.items())
            )
            print(
                f"[{operation}] Merge plan: {len(layer_map)} unique target groups; "
                f"contributor coverage: {coverage_text or 'none'}; "
                f"execution work units: {len(units)}"
            )
        cursors = [
            _AsyncTensorCursor(
                handler,
                keys,
                pin_memory=torch.device(device).type == "cuda",
            )
            for handler, keys in zip(handlers, stream_keys)
        ]

        output_path, _ = canonical_model_artifact_path("loras", output_filename)
        metadata = {
            "ss_training_comment": f"Merged {len(lora_paths)} LoRAs via {strategy}",
            "ss_network_module": "networks.lora",
            "alpha_normalized": "true",
            "alpha_normalization": (
                "lora_up := lora_up * (alpha / rank); alpha tensors removed"
            ),
        }
        output_ranks = Counter()
        fallback_count = 0
        tensor_count = 0
        pbar = comfy.utils.ProgressBar(len(units))

        with atomic_uel_writer(output_path, metadata) as writer:
            with torch.no_grad():
                for work_index, unit in enumerate(
                    tqdm(
                        units,
                        desc=f"Merging target groups ({strategy})",
                        unit="units",
                    )
                ):
                    if unit["kind"] == "raw_copy":
                        for entry in unit["entries"]:
                            write_preserved_tensor(
                                writer,
                                entry["key"],
                                handlers[entry["idx"]],
                                entry["output_key"],
                                force_raw=True,
                            )
                            tensor_count += 1
                        pbar.update(1)
                        continue

                    loaded = {}
                    try:
                        for entry in unit["entries"]:
                            loaded[(entry["idx"], entry["key"])] = cursors[
                                entry["idx"]
                            ].take(entry["key"])
                        if unit["kind"] == "copy":
                            outputs = [
                                (
                                    entry["output_key"],
                                    loaded[(entry["idx"], entry["key"])]
                                    .cpu()
                                    .contiguous(),
                                )
                                for entry in unit["entries"]
                            ]
                            output_rank = None
                            cpu_fallback = False
                        else:
                            outputs, output_rank, cpu_fallback = _process_merge_unit(
                                unit,
                                loaded,
                                lora_infos,
                                strategy,
                                settings,
                                save_dtype,
                                device,
                                work_index,
                            )
                        if outputs:
                            writer.write_batch(outputs)
                            tensor_count += len(outputs)
                        if output_rank is not None:
                            output_ranks[output_rank] += 1
                        fallback_count += int(cpu_fallback)
                        outputs.clear()
                    finally:
                        for entry in unit["entries"]:
                            source = (entry["idx"], entry["key"])
                            if source not in loaded:
                                continue
                            del loaded[source]
                            cursors[entry["idx"]].release(entry["key"])
                        loaded.clear()
                    pbar.update(1)

        for cursor in cursors:
            cursor.finish()
        if verbose:
            print(
                f"[{operation}] Output factor-pair ranks: "
                f"{_format_rank_distribution(output_ranks)}"
            )
            print(f"[{operation}] CUDA OOM CPU fallbacks: {fallback_count}")
            print(f"[{operation}] Output file contains {tensor_count} tensors")
            print(f"[{operation}] Saved merged LoRA to {output_path}")
        return output_path
    finally:
        for cursor in cursors:
            cursor.close()
        for handler in handlers:
            handler.__exit__(None, None, None)
        cleanup_after_operation()



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
    return _run_multi_lora_merge(
        lora_paths,
        lora_weights,
        merge_mode,
        {},
        device,
        save_dtype,
        output_filename,
        base_model_path,
        verbose,
        include_1d_diffs,
    )
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
                io.Float.Input("weight_1", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 1's represented delta before merging; negative values subtract it."),
                # LoRA 2
                io.Combo.Input("lora_2", options=["None"] + folder_paths.get_filename_list("loras"), default="None",
                               tooltip="Optional LoRA 2; used only when included by LoRA Count."),
                io.Float.Input("weight_2", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 2's represented delta before merging."),
                # LoRA 3
                io.Combo.Input("lora_3", options=["None"] + folder_paths.get_filename_list("loras"), default="None",
                               tooltip="Optional LoRA 3; used only when included by LoRA Count."),
                io.Float.Input("weight_3", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 3's represented delta before merging."),
                # LoRA 4
                io.Combo.Input("lora_4", options=["None"] + folder_paths.get_filename_list("loras"), default="None",
                               tooltip="Optional LoRA 4; used only when included by LoRA Count."),
                io.Float.Input("weight_4", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 4's represented delta before merging."),
                # LoRA 5
                io.Combo.Input("lora_5", options=["None"] + folder_paths.get_filename_list("loras"), default="None",
                               tooltip="Optional LoRA 5; used only when included by LoRA Count."),
                io.Float.Input("weight_5", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 5's represented delta before merging."),
                # LoRA 6
                io.Combo.Input("lora_6", options=["None"] + folder_paths.get_filename_list("loras"), default="None",
                               tooltip="Optional LoRA 6; used only when included by LoRA Count."),
                io.Float.Input("weight_6", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 6's represented delta before merging."),
                # LoRA 7
                io.Combo.Input("lora_7", options=["None"] + folder_paths.get_filename_list("loras"), default="None",
                               tooltip="Optional LoRA 7; used only when included by LoRA Count."),
                io.Float.Input("weight_7", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 7's represented delta before merging."),
                # LoRA 8
                io.Combo.Input("lora_8", options=["None"] + folder_paths.get_filename_list("loras"), default="None",
                               tooltip="Optional LoRA 8; used only when included by LoRA Count."),
                io.Float.Input("weight_8", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 8's represented delta before merging."),

                io.Combo.Input("merge_mode", options=["concatenate", "weighted_sum"], default="concatenate",
                              tooltip="concatenate: safe, increases rank. weighted_sum: fixed rank (to max input rank), mathematically lossy but standard in Kohya."),
                io.String.Input("output_filename", default="merged_lora",
                                tooltip="Output filename without extension, written under ComfyUI's LoRA directory."),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16",
                               tooltip="Requested dtype for generated factors; each logical layer's source factor precision is considered independently of scalar alpha."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda",
                               tooltip="Device for per-target arithmetic; a CUDA OOM retries only the affected target on CPU."),
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Include and merge 1D direct-diff tensors as FP32."),
            ],
            outputs=[io.AnyType.Output(display_name="output_path")],
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

        _, output_name = canonical_model_artifact_path("loras", output_filename)
        return io.NodeOutput(output_name)


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
                io.Combo.Input("lora_count", options=[str(i) for i in range(1, 9)], default="2",
                               tooltip="Use the first N LoRA selectors; selectors beyond this count are ignored."),
                # LoRA 1
                io.Combo.Input("lora_1", options=folder_paths.get_filename_list("loras"), tooltip="First LoRA"),
                io.Float.Input("weight_1", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Scale LoRA 1's represented delta before DARE-Ties merging; negative values subtract it."),
                # ...
                io.Combo.Input("lora_2", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 2; used only when included by LoRA Count."),
                io.Float.Input("weight_2", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 2's represented delta before DARE-Ties merging."),
                io.Combo.Input("lora_3", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 3; used only when included by LoRA Count."),
                io.Float.Input("weight_3", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 3's represented delta before DARE-Ties merging."),
                io.Combo.Input("lora_4", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 4; used only when included by LoRA Count."),
                io.Float.Input("weight_4", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 4's represented delta before DARE-Ties merging."),
                io.Combo.Input("lora_5", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 5; used only when included by LoRA Count."),
                io.Float.Input("weight_5", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 5's represented delta before DARE-Ties merging."),
                io.Combo.Input("lora_6", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 6; used only when included by LoRA Count."),
                io.Float.Input("weight_6", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 6's represented delta before DARE-Ties merging."),
                io.Combo.Input("lora_7", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 7; used only when included by LoRA Count."),
                io.Float.Input("weight_7", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 7's represented delta before DARE-Ties merging."),
                io.Combo.Input("lora_8", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 8; used only when included by LoRA Count."),
                io.Float.Input("weight_8", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 8's represented delta before DARE-Ties merging."),

                io.Float.Input("drop_rate", default=0.1, min=0.0, max=1.0, step=0.01, tooltip="Fraction of factor values randomly dropped before rescaling; higher values increase sparsification."),
                io.Float.Input("trim_quantile", default=0.2, min=0.0, max=1.0, step=0.01, tooltip="Remove the lowest-magnitude fraction before resolving sign conflicts with TIES."),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, tooltip="Random seed for reproducible DARE masks."),
                io.String.Input("output_filename", default="merged_lora_dare", tooltip="Output filename without extension, written under ComfyUI's LoRA directory."),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16", tooltip="Requested dtype for generated factors; scalar alpha tensors do not determine factor precision."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", tooltip="Device for per-target arithmetic; a CUDA OOM retries only the affected target on CPU."),
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Include and merge 1D direct-diff tensors as FP32."),
            ],
            outputs=[io.AnyType.Output(display_name="output_path")],
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

        _, output_name = canonical_model_artifact_path("loras", output_filename)
        return io.NodeOutput(output_name)

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
    return _run_multi_lora_merge(
        lora_paths,
        lora_weights,
        "dare",
        {
            "drop_rate": drop_rate,
            "trim_quantile": trim_quantile,
            "seed": seed,
        },
        device,
        save_dtype,
        output_filename,
        base_model_path,
        verbose,
        include_1d_diffs,
    )
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
                io.Combo.Input("lora_count", options=[str(i) for i in range(1, 9)], default="2", tooltip="Use the first N LoRA selectors; selectors beyond this count are ignored."),
                # LoRA 1
                io.Combo.Input("lora_1", options=folder_paths.get_filename_list("loras"), tooltip="First LoRA"),
                io.Float.Input("weight_1", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 1's represented delta before enhanced DARE-Ties merging; negative values subtract it."),
                # ...
                io.Combo.Input("lora_2", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 2; used only when included by LoRA Count."),
                io.Float.Input("weight_2", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 2's represented delta before enhanced DARE-Ties merging."),
                io.Combo.Input("lora_3", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 3; used only when included by LoRA Count."),
                io.Float.Input("weight_3", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 3's represented delta before enhanced DARE-Ties merging."),
                io.Combo.Input("lora_4", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 4; used only when included by LoRA Count."),
                io.Float.Input("weight_4", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 4's represented delta before enhanced DARE-Ties merging."),
                io.Combo.Input("lora_5", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 5; used only when included by LoRA Count."),
                io.Float.Input("weight_5", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 5's represented delta before enhanced DARE-Ties merging."),
                io.Combo.Input("lora_6", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 6; used only when included by LoRA Count."),
                io.Float.Input("weight_6", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 6's represented delta before enhanced DARE-Ties merging."),
                io.Combo.Input("lora_7", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 7; used only when included by LoRA Count."),
                io.Float.Input("weight_7", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 7's represented delta before enhanced DARE-Ties merging."),
                io.Combo.Input("lora_8", options=["None"] + folder_paths.get_filename_list("loras"), default="None", tooltip="Optional LoRA 8; used only when included by LoRA Count."),
                io.Float.Input("weight_8", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Scale LoRA 8's represented delta before enhanced DARE-Ties merging."),

                io.Float.Input("mask_power", default=2.0, min=0.001, max=10.0, step=0.01, tooltip="Exponent shaping magnitude-based keep probabilities; larger values favor high-magnitude values more strongly."),
                io.Float.Input("min_keep_prob", default=0.01, min=0.0, max=1.0, step=0.01, tooltip="Lower bound for each value's keep probability, limiting inverse-probability rescaling."),
                io.Float.Input("mask_smooth", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Blend between stochastic masking at 0 and smooth probability scaling at 1."),
                io.Float.Input("trim_quantile", default=0.2, min=0.0, max=1.0, step=0.01, tooltip="Remove the lowest-magnitude fraction before resolving sign conflicts with TIES."),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, tooltip="Random seed for reproducible enhanced DARE masks."),
                io.String.Input("output_filename", default="merged_lora_dare_enhanced", tooltip="Output filename without extension, written under ComfyUI's LoRA directory."),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16", tooltip="Requested dtype for generated factors; scalar alpha tensors do not determine factor precision."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", tooltip="Device for per-target arithmetic; a CUDA OOM retries only the affected target on CPU."),
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Include and merge 1D direct-diff tensors as FP32."),
            ],
            outputs=[io.AnyType.Output(display_name="output_path")],
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

        _, output_name = canonical_model_artifact_path("loras", output_filename)
        return io.NodeOutput(output_name)


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
    return _run_multi_lora_merge(
        lora_paths,
        lora_weights,
        "enhanced_dare",
        {
            "mask_power": mask_power,
            "min_keep_prob": min_keep_prob,
            "mask_smooth": mask_smooth,
            "trim_quantile": trim_quantile,
            "seed": seed,
        },
        device,
        save_dtype,
        output_filename,
        base_model_path,
        verbose,
        include_1d_diffs,
    )
