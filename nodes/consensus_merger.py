"""Dedicated streaming Consensus-Weighted Blending merger nodes."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from tqdm import tqdm
from unifiedefficientloader import (
    IncrementalSafetensorsWriter,
    MemoryEfficientSafeOpen,
    transfer_to_gpu_pinned,
)

from .device_utils import (
    cleanup_after_operation,
    estimate_model_size,
    prepare_for_large_operation,
)
from .lora_resize import (
    canonical_lora_key,
    layer_has_companions,
    layer_tensor_keys,
    parse_lora_layers,
    select_output_dtype,
    validate_canonical_blocks,
)
from .merger import (
    _compile_patterns,
    _matches_any_pattern,
    load_documentation_from_file,
)
from .quantization_guard import (
    inspect_low_bit_input,
    layer_has_low_bit,
    write_preserved_tensor,
)


CWB_PRESETS = {
    "baseline": {
        "consensus_type": "median", "power_alpha": 2.0,
        "similarity_threshold": 0.0, "diversity_beta": 0.0,
        "rescale_norm": False, "global_scale": 1.0,
    },
    "power_blend": {
        "consensus_type": "median", "alignment_method": "similarity",
        "alignment_threshold": 0.9, "power_alpha": 8.0,
        "similarity_threshold": 0.75, "diversity_beta": 0.0,
        "rescale_norm": True, "global_scale": 1.0,
        "dynamic_similarity_contrast": True,
    },
    "high_clarity": {
        "consensus_type": "median", "power_alpha": 3.0,
        "similarity_threshold": 0.3, "diversity_beta": 0.0,
        "rescale_norm": False, "global_scale": 1.0,
    },
    "smooth": {
        "consensus_type": "mean", "power_alpha": 1.5,
        "similarity_threshold": 0.0, "diversity_beta": 0.0,
        "rescale_norm": False, "global_scale": 1.0,
    },
    "varied_merge": {
        "consensus_type": "median", "power_alpha": 2.0,
        "similarity_threshold": 0.0, "diversity_beta": 0.0,
        "rescale_norm": True, "global_scale": 0.7,
    },
    "diverse_concept": {
        "consensus_type": "median", "power_alpha": 2.0,
        "similarity_threshold": 0.0, "diversity_beta": 1.0,
        "rescale_norm": True, "global_scale": 0.7,
    },
    "high_diversity_concept": {
        "consensus_type": "median", "power_alpha": 2.0,
        "similarity_threshold": 0.0, "diversity_beta": 2.0,
        "rescale_norm": True, "global_scale": 0.7,
    },
    "dsc_baseline": {
        "consensus_type": "median", "power_alpha": 2.0,
        "similarity_threshold": 0.0, "diversity_beta": 0.0,
        "rescale_norm": False, "global_scale": 1.0,
        "dynamic_similarity_contrast": True, "soft_comfort_bandpass": True,
    },
    "dsc_high_clarity": {
        "consensus_type": "median", "power_alpha": 4.0,
        "similarity_threshold": 0.3, "diversity_beta": 0.0,
        "rescale_norm": False, "global_scale": 1.0,
        "dynamic_similarity_contrast": True, "soft_comfort_bandpass": True,
    },
    "dsc_smooth": {
        "consensus_type": "mean", "power_alpha": 1.0,
        "similarity_threshold": 0.0, "diversity_beta": 0.0,
        "rescale_norm": False, "global_scale": 1.0,
        "dynamic_similarity_contrast": True, "soft_comfort_bandpass": True,
    },
    "dsc_varied_merge": {
        "consensus_type": "median", "power_alpha": 2.5,
        "similarity_threshold": 0.0, "diversity_beta": 0.0,
        "rescale_norm": True, "global_scale": 0.7,
        "dynamic_similarity_contrast": True, "soft_comfort_bandpass": True,
    },
    "dsc_diverse_concept": {
        "consensus_type": "median", "power_alpha": 2.0,
        "similarity_threshold": 0.0, "diversity_beta": 1.5,
        "rescale_norm": True, "global_scale": 0.7,
        "dynamic_similarity_contrast": True, "soft_comfort_bandpass": True,
    },
    "dsc_high_diversity_concept": {
        "consensus_type": "median", "power_alpha": 2.0,
        "similarity_threshold": 0.0, "diversity_beta": 3.0,
        "rescale_norm": True, "global_scale": 0.7,
        "dynamic_similarity_contrast": True, "soft_comfort_bandpass": True,
    },
}

CWB_PRESET_OPTIONS = ["custom", *CWB_PRESETS]
LORA_PREFIXES = (
    "base_model.model.",
    "lora_unet_", "lora_transformer_", "lora_te1_", "lora_te2_",
    "lora_te_", "lycoris_", "diffusion_model.", "transformer.", "unet.",
)


@dataclass(frozen=True)
class CWBSettings:
    consensus_type: str
    alignment_method: str
    alignment_threshold: float
    similarity_threshold: float
    power_alpha: float
    diversity_beta: float
    rescale_norm: bool
    global_scale: float
    dynamic_similarity_contrast: bool
    soft_comfort_bandpass: bool
    position_weight: float
    preserve_common_prefix: bool


def resolve_cwb_settings(**values) -> CWBSettings:
    """Resolve a named preset without hiding position/prefix controls."""
    preset_name = values.get("cwb_preset", "baseline")
    resolved = {
        "consensus_type": values.get("consensus_type", "median"),
        "alignment_method": values.get("alignment_method", "similarity"),
        "alignment_threshold": float(values.get("alignment_threshold", 0.4)),
        "similarity_threshold": float(values.get("similarity_threshold", 0.0)),
        "power_alpha": float(values.get("power_alpha", 2.0)),
        "diversity_beta": float(values.get("diversity_beta", 0.0)),
        "rescale_norm": bool(values.get("rescale_norm", False)),
        "global_scale": float(values.get("global_scale", 1.0)),
        "dynamic_similarity_contrast": bool(
            values.get("dynamic_similarity_contrast", False)
        ),
        "soft_comfort_bandpass": bool(values.get("soft_comfort_bandpass", False)),
        "position_weight": float(values.get("position_weight", 0.0)),
        "preserve_common_prefix": bool(values.get("preserve_common_prefix", False)),
    }
    if not 0.0 <= resolved["position_weight"] <= 1.0:
        raise ValueError("Position weight must be between 0.0 and 1.0.")
    if preset_name != "custom":
        try:
            preset = CWB_PRESETS[preset_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported CWB preset: {preset_name}") from exc
        user_scale = resolved["global_scale"]
        resolved.update(
            alignment_method="similarity",
            dynamic_similarity_contrast=False,
            soft_comfort_bandpass=False,
        )
        resolved.update(preset)
        if user_scale != 1.0:
            resolved["global_scale"] = user_scale
    return CWBSettings(**resolved)


def _common_prefix_length(tensors: list[torch.Tensor]) -> int:
    if not tensors:
        return 0
    limit = min(tensor.shape[0] for tensor in tensors)
    if limit == 0:
        return 0
    reference = tensors[0][:limit]
    common = torch.ones(limit, dtype=torch.bool, device=reference.device)
    for tensor in tensors[1:]:
        comparison = torch.isclose(reference, tensor[:limit], rtol=1e-5, atol=1e-6)
        common &= comparison.reshape(limit, -1).all(dim=1)
    mismatch = torch.nonzero(~common, as_tuple=False)
    return limit if mismatch.numel() == 0 else int(mismatch[0].item())


def _position_biased_scores(scores: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return scores
    n_ref, n_source = scores.shape
    if n_ref == 0 or n_source == 0:
        return scores
    ref_positions = torch.linspace(0.0, 1.0, n_ref, device=scores.device)
    source_positions = torch.linspace(0.0, 1.0, n_source, device=scores.device)
    distance = ref_positions[:, None] - source_positions[None, :]
    sigma = max(1.0 - min(weight, 1.0), 1.0 / max(n_ref, n_source, 1))
    affinity = torch.exp(-0.5 * (distance / sigma) ** 2).to(scores)
    return scores * (1.0 - weight) + affinity * weight


def merge_consensus_group(
    stacked: torch.Tensor,
    settings: CWBSettings,
    *,
    apply_global_scale: bool = True,
) -> torch.Tensor:
    """Merge one aligned vector group using CWB."""
    if stacked.shape[0] == 1:
        merged = stacked[0].clone()
    else:
        consensus = (
            torch.median(stacked, dim=0).values
            if settings.consensus_type == "median"
            else torch.mean(stacked, dim=0)
        )
        stacked_norm = F.normalize(stacked, p=2, dim=1, eps=1e-8)
        consensus_norm = F.normalize(consensus, p=2, dim=0, eps=1e-8)
        similarities = torch.mv(stacked_norm, consensus_norm)
        weighted = similarities
        if settings.dynamic_similarity_contrast:
            min_sim = similarities.min()
            max_sim = similarities.max()
            if max_sim > min_sim:
                weighted = 0.7 + 0.3 * (
                    similarities - min_sim
                ) / (max_sim - min_sim + 1e-8)

        row_weights = torch.zeros_like(similarities)
        mask = similarities >= settings.similarity_threshold
        if mask.any():
            safe = weighted[mask].clamp(min=0.0, max=1.0)
            row_weights[mask] = torch.pow(safe, settings.power_alpha)
            if settings.diversity_beta > 0.0:
                distance_base = 1.5 if settings.soft_comfort_bandpass else 1.001
                row_weights[mask] *= torch.pow(
                    (distance_base - safe).clamp(min=0.0),
                    settings.diversity_beta,
                )
            weight_sum = row_weights.sum()
            if weight_sum > 0:
                row_weights /= weight_sum
            else:
                row_weights.fill_(1.0 / len(similarities))
        else:
            row_weights.fill_(1.0 / len(similarities))
        merged = (stacked * row_weights.unsqueeze(1)).sum(dim=0)

        if settings.rescale_norm:
            average_norm = torch.norm(stacked, p=2, dim=1).mean()
            merged_norm = torch.norm(merged, p=2)
            if merged_norm > 0:
                merged = (merged / merged_norm) * average_norm
    if apply_global_scale and settings.global_scale != 1.0:
        merged *= settings.global_scale
    return merged


def _greedy_similarity_matches(
    similarities: torch.Tensor,
    settings: CWBSettings,
) -> list[int]:
    """Match source rows to reference rows without duplicating the score matrix."""
    scores = _position_biased_scores(similarities, settings.position_weight)
    scores.masked_fill_(similarities < settings.alignment_threshold, -100.0)
    matched = [-1] * similarities.shape[0]
    for _ in range(min(similarities.shape)):
        flat_index = torch.argmax(scores)
        best = scores.flatten()[flat_index].item()
        if best <= -100.0:
            break
        ref_row = int((flat_index // similarities.shape[1]).item())
        source_row = int((flat_index % similarities.shape[1]).item())
        matched[ref_row] = source_row
        scores[ref_row, :] = -100.0
        scores[:, source_row] = -100.0
    return matched


def merge_cwb_tensors(
    tensors: list[torch.Tensor],
    settings: CWBSettings,
    *,
    reference_index: int = 0,
    allow_similarity_alignment: bool = True,
) -> torch.Tensor:
    """CWB tensors with a shared rank and trailing vector dimensions."""
    if not tensors:
        raise ValueError("CWB requires at least one tensor.")
    if any(tensor.ndim != tensors[0].ndim for tensor in tensors):
        raise ValueError("CWB source tensors must have matching ranks.")

    if tensors[0].ndim == 0:
        stacked = torch.stack(tensors)
        merged = (
            torch.median(stacked, dim=0).values
            if settings.consensus_type == "median"
            else torch.mean(stacked, dim=0)
        )
        return merged * settings.global_scale

    if tensors[0].ndim == 1:
        target = tensors[reference_index].shape[0]
        aligned = []
        for tensor in tensors:
            if tensor.shape[0] < target:
                tensor = F.pad(tensor, (0, target - tensor.shape[0]))
            else:
                tensor = tensor[:target]
            aligned.append(tensor)
        stacked = torch.stack(aligned)
        merged = (
            torch.median(stacked, dim=0).values
            if settings.consensus_type == "median"
            else torch.mean(stacked, dim=0)
        )
        return merged * settings.global_scale

    trailing_shapes = {tuple(tensor.shape[1:]) for tensor in tensors}
    if len(trailing_shapes) != 1:
        raise ValueError("CWB vector dimensions must match after alignment.")

    prefix_length = _common_prefix_length(tensors) if settings.preserve_common_prefix else 0
    preserved_prefix = tensors[0][:prefix_length]
    bodies = [tensor[prefix_length:] for tensor in tensors]
    adjusted_reference = reference_index
    reference = bodies[adjusted_reference]
    if reference.shape[0] == 0:
        return preserved_prefix.clone()

    groups: list[list[torch.Tensor]] = [[reference[row]] for row in range(reference.shape[0])]
    reference_flat = reference.reshape(reference.shape[0], -1)

    for index, tensor in enumerate(bodies):
        if index == adjusted_reference or tensor.shape[0] == 0:
            continue
        flat = tensor.reshape(tensor.shape[0], -1)
        if settings.alignment_method == "index" or not allow_similarity_alignment:
            for row in range(min(reference.shape[0], tensor.shape[0])):
                groups[row].append(tensor[row])
            continue

        similarities = torch.mm(
            F.normalize(reference_flat, p=2, dim=1, eps=1e-8),
            F.normalize(flat, p=2, dim=1, eps=1e-8).t(),
        )
        matched = _greedy_similarity_matches(similarities, settings)
        for ref_row, source_row in enumerate(matched):
            if source_row >= 0:
                groups[ref_row].append(tensor[source_row])

    merged_rows = []
    for group in groups:
        stacked = torch.stack([row.reshape(-1) for row in group])
        merged_rows.append(merge_consensus_group(stacked, settings).reshape(group[0].shape))
    merged = torch.stack(merged_rows)
    if prefix_length:
        merged = torch.cat([preserved_prefix, merged], dim=0)
    return merged


def _common_lora_prefix_length(
    downs: list[torch.Tensor],
    ups: list[torch.Tensor],
    reference_index: int,
) -> int:
    reference_down = downs[reference_index]
    reference_up = ups[reference_index]
    limit = reference_down.shape[0]
    common = torch.ones(limit, dtype=torch.bool, device=reference_down.device)
    for down, up in zip(downs, ups):
        down_equal = torch.isclose(
            reference_down, down, rtol=1e-5, atol=1e-6
        ).reshape(limit, -1).all(dim=1)
        up_equal = torch.isclose(
            reference_up.movedim(1, 0),
            up.movedim(1, 0),
            rtol=1e-5,
            atol=1e-6,
        ).reshape(limit, -1).all(dim=1)
        common &= down_equal & up_equal
    mismatch = torch.nonzero(~common, as_tuple=False)
    return limit if mismatch.numel() == 0 else int(mismatch[0].item())


def merge_cwb_lora_pairs(
    downs: list[torch.Tensor],
    ups: list[torch.Tensor],
    settings: CWBSettings,
    *,
    reference_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CWB LoRA rank components while keeping A rows paired with B columns."""
    if not downs or len(downs) != len(ups):
        raise ValueError("CWB requires matching LoRA A/B source lists.")
    rank = downs[reference_index].shape[0]
    if any(down.shape[0] != rank or up.shape[1] != rank for down, up in zip(downs, ups)):
        raise ValueError("CWB LoRA sources must be padded to one shared rank.")

    prefix = (
        _common_lora_prefix_length(downs, ups, reference_index)
        if settings.preserve_common_prefix else 0
    )
    reference_down = downs[reference_index]
    reference_up = ups[reference_index]
    reference_up_components = reference_up.movedim(1, 0)
    down_groups = [[reference_down[row]] for row in range(prefix, rank)]
    up_groups = [[reference_up_components[row]] for row in range(prefix, rank)]

    ref_down_body = reference_down[prefix:].reshape(rank - prefix, -1)
    ref_up_body = reference_up_components[prefix:].reshape(rank - prefix, -1)
    for index, (down, up) in enumerate(zip(downs, ups)):
        if index == reference_index or not down_groups:
            continue
        source_down_components = down[prefix:]
        source_up_components = up.movedim(1, 0)[prefix:]
        source_down = source_down_components.reshape(rank - prefix, -1)
        source_up = source_up_components.reshape(rank - prefix, -1)
        if settings.alignment_method == "index":
            matched = list(range(source_down.shape[0]))
            down_similarities = up_similarities = None
        else:
            down_similarities = torch.mm(
                F.normalize(ref_down_body, p=2, dim=1, eps=1e-8),
                F.normalize(source_down, p=2, dim=1, eps=1e-8).T,
            )
            up_similarities = torch.mm(
                F.normalize(ref_up_body, p=2, dim=1, eps=1e-8),
                F.normalize(source_up, p=2, dim=1, eps=1e-8).T,
            )
            contribution_similarities = down_similarities * up_similarities
            matched = _greedy_similarity_matches(contribution_similarities, settings)

        for ref_row, source_row in enumerate(matched):
            if source_row < 0 or ref_row >= len(down_groups):
                continue
            source_down_row = source_down_components[source_row]
            source_up_column = source_up_components[source_row]
            if (
                down_similarities is not None
                and down_similarities[ref_row, source_row] < 0
                and up_similarities[ref_row, source_row] < 0
            ):
                source_down_row = -source_down_row
                source_up_column = -source_up_column
            down_groups[ref_row].append(source_down_row)
            up_groups[ref_row].append(source_up_column)

    merged_down_rows = [
        merge_consensus_group(
            torch.stack([component.reshape(-1) for component in group]),
            settings,
            apply_global_scale=False,
        ).reshape(group[0].shape)
        for group in down_groups
    ]
    merged_up_columns = [
        merge_consensus_group(
            torch.stack([component.reshape(-1) for component in group]), settings
        ).reshape(group[0].shape)
        for group in up_groups
    ]
    if prefix:
        merged_down_rows = [*reference_down[:prefix], *merged_down_rows]
        merged_up_columns = [*reference_up_components[:prefix], *merged_up_columns]
    return torch.stack(merged_down_rows), torch.stack(merged_up_columns).movedim(0, 1)


def _copy_to_target_shape(tensor: torch.Tensor, target: torch.Size) -> torch.Tensor | None:
    if tensor.ndim != len(target):
        return None
    if tensor.ndim == 0:
        return tensor if tensor.shape == target else None
    output_shape = list(target)
    output = tensor.new_zeros(output_shape)
    slices = tuple(slice(0, min(source, wanted)) for source, wanted in zip(tensor.shape, output_shape))
    output[slices] = tensor[slices]
    return output


def _to_compute(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if device == "cuda":
        return transfer_to_gpu_pinned(tensor, device, torch.float32)
    return tensor.to(device=device, dtype=torch.float32)


def _release_failed_cuda_operation() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _merge_tensors_to_cpu(
    tensors: list[torch.Tensor],
    settings: CWBSettings,
    device: str,
    target_dtype: torch.dtype,
    *,
    reference_index: int = 0,
    allow_similarity_alignment: bool,
    operation_label: str = "tensor",
) -> torch.Tensor:
    """Own one tensor operation's GPU lifetime and return only its CPU result."""
    def execute(target_device: str) -> torch.Tensor:
        compute_tensors = [_to_compute(tensor, target_device) for tensor in tensors]
        try:
            merged = merge_cwb_tensors(
                compute_tensors,
                settings,
                reference_index=reference_index,
                allow_similarity_alignment=allow_similarity_alignment,
            )
            return merged.to(target_dtype).cpu().contiguous()
        finally:
            del compute_tensors

    try:
        return execute(device)
    except torch.OutOfMemoryError:
        if not str(device).startswith("cuda"):
            raise
        _release_failed_cuda_operation()
        logging.warning(
            "[CWB Merge] CUDA OOM for '%s'; retrying this layer on CPU.",
            operation_label,
        )
        return execute("cpu")


def _merge_lora_pair_to_cpu(
    pair_sources: list[tuple[int, torch.Tensor, torch.Tensor, float]],
    settings: CWBSettings,
    device: str,
    target_dtype: torch.dtype,
    operation_label: str = "LoRA pair",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Own one LoRA pair operation's GPU lifetime and return CPU A/B tensors."""
    max_rank = max(down.shape[0] for _, down, _, _ in pair_sources)
    reference_index = max(
        range(len(pair_sources)),
        key=lambda index: pair_sources[index][1].shape[0],
    )
    def execute(target_device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        downs = [
            _to_compute(_pad_rank_axis(down, 0, max_rank), target_device)
            for _, down, _, _ in pair_sources
        ]
        ups = [
            _to_compute(_pad_rank_axis(up, 1, max_rank), target_device) * scale
            for _, _, up, scale in pair_sources
        ]
        try:
            merged_down, merged_up = merge_cwb_lora_pairs(
                downs,
                ups,
                settings,
                reference_index=reference_index,
            )
            return (
                merged_down.to(target_dtype).cpu().contiguous(),
                merged_up.to(target_dtype).cpu().contiguous(),
                max_rank,
            )
        finally:
            del downs, ups

    try:
        return execute(device)
    except torch.OutOfMemoryError:
        if not str(device).startswith("cuda"):
            raise
        _release_failed_cuda_operation()
        logging.warning(
            "[CWB Merge] CUDA OOM for '%s'; retrying this layer on CPU.",
            operation_label,
        )
        return execute("cpu")


@contextmanager
def _atomic_output_writer(output_path: str, metadata: dict):
    """Publish a completed safetensors file atomically and remove failed partials."""
    directory = os.path.dirname(output_path) or "."
    basename = os.path.basename(output_path)
    temporary_path = os.path.join(directory, f".{basename}.{uuid.uuid4().hex}.tmp")
    try:
        with IncrementalSafetensorsWriter(temporary_path, metadata=metadata) as writer:
            yield writer
        os.replace(temporary_path, output_path)
        logging.info("[CWB Merge] Published '%s'.", output_path)
    except BaseException:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _clear_previous_layer(params: dict) -> None:
    if not params["force_clear_cache"]:
        return
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _requested_dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _is_float_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
    }


def _normalized_lora_core(block_name: str) -> str:
    core = block_name
    changed = True
    while changed:
        changed = False
        for prefix in LORA_PREFIXES:
            if core.startswith(prefix):
                core = core[len(prefix):]
                changed = True
                break
    if core.endswith(".lora"):
        core = core[:-5]
    return core.replace(".", "_")


def _secondary_lora_map(pairs: dict[str, dict[str, str]]) -> dict[str, str]:
    result = {}
    for block_name in pairs:
        result.setdefault(_normalized_lora_core(block_name), block_name)
    return result


def _pad_rank_axis(tensor: torch.Tensor, axis: int, rank: int) -> torch.Tensor:
    if tensor.shape[axis] == rank:
        return tensor
    padding = [0] * (tensor.ndim * 2)
    reverse_axis = tensor.ndim - 1 - axis
    padding[reverse_axis * 2 + 1] = rank - tensor.shape[axis]
    return F.pad(tensor, tuple(padding))


class ConsensusMergerLogic:
    """Shared streaming execution for the dedicated CWB node family."""

    @classmethod
    def execute(
        cls,
        model_names: list[str],
        model_type: str,
        params: dict,
        *,
        embedding_union: bool = False,
        lora_mode: bool = False,
    ) -> str:
        paths = []
        for name in model_names:
            path = folder_paths.get_full_path(model_type, name)
            if not path:
                raise FileNotFoundError(f"{model_type} input '{name}' was not found.")
            paths.append(path)

        total_size = sum(estimate_model_size(path) for path in paths)
        device = params["process_device"]
        if total_size:
            prepare_for_large_operation(total_size * 1.2, torch.device(device))

        handlers = [MemoryEfficientSafeOpen(path, low_memory=params["lazy_load"]) for path in paths]
        try:
            low_bit_sets = [
                inspect_low_bit_input(
                    handler,
                    f"Input {index + 1} ({path})",
                    "CWB Merge",
                )
                for index, (handler, path) in enumerate(zip(handlers, paths))
            ]
            settings = resolve_cwb_settings(**params)
            if lora_mode:
                return cls._merge_loras(handlers, low_bit_sets, model_type, params, settings)
            return cls._merge_generic(
                handlers,
                low_bit_sets,
                model_type,
                params,
                settings,
                embedding_union=embedding_union,
            )
        finally:
            for handler in handlers:
                handler.__exit__(None, None, None)
            cleanup_after_operation()

    @staticmethod
    def _output_path(model_type: str, output_filename: str) -> str:
        output_dir = os.path.join(folder_paths.models_dir, model_type)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{output_filename.strip()}.safetensors")

    @classmethod
    def _merge_generic(
        cls,
        handlers,
        low_bit_sets,
        model_type,
        params,
        settings,
        *,
        embedding_union,
    ):
        primary = handlers[0]
        keys = set(primary.keys())
        if embedding_union:
            for handler in handlers[1:]:
                keys.update(handler.keys())
        keys = sorted(keys)
        output_path = cls._output_path(model_type, params["output_filename"])
        requested_dtype = _requested_dtype(params["save_dtype"])
        mismatch_mode = params["mismatch_mode"]
        glob_mode = params["glob_patterns"]
        exclude = _compile_patterns(params["exclude_patterns"], glob_mode=glob_mode)
        discard = _compile_patterns(params["discard_patterns"], glob_mode=glob_mode)
        pbar = comfy.utils.ProgressBar(len(keys))

        writer_context = _atomic_output_writer(output_path, primary.metadata())
        writer = writer_context.__enter__()
        try:
            with torch.no_grad():
                for key in tqdm(keys, desc="CWB merging tensors", unit="tensors"):
                    _clear_previous_layer(params)
                    if _matches_any_pattern(key, discard, glob_mode=glob_mode):
                        pbar.update(1)
                        continue
                    source_indices = [i for i, handler in enumerate(handlers) if key in handler.keys()]
                    preserve_index = 0 if key in primary.keys() else source_indices[0]
                    guarded = any(key in low_bit_sets[i] for i in source_indices)
                    excluded = _matches_any_pattern(key, exclude, glob_mode=glob_mode)
                    if guarded or excluded:
                        write_preserved_tensor(writer, key, handlers[preserve_index])
                        pbar.update(1)
                        continue

                    source_dtypes = [handlers[i].get_dtype(key) for i in source_indices]
                    if not all(_is_float_dtype(dtype) for dtype in source_dtypes):
                        logging.warning(
                            "[CWB Merge] Preserving non-floating tensor '%s' from input %d.",
                            key,
                            preserve_index + 1,
                        )
                        write_preserved_tensor(writer, key, handlers[preserve_index])
                        pbar.update(1)
                        continue

                    if not embedding_union and len(source_indices) != len(handlers):
                        if mismatch_mode == "error":
                            raise ValueError(f"Tensor '{key}' is missing from a CWB input.")
                        if mismatch_mode == "skip":
                            write_preserved_tensor(writer, key, primary)
                            pbar.update(1)
                            continue

                    raw = {i: handlers[i].get_tensor(key) for i in source_indices}
                    reference_index = 0
                    if embedding_union:
                        reference_source = max(
                            source_indices,
                            key=lambda i: raw[i].shape[0] if raw[i].ndim else 1,
                        )
                        reference_shape = raw[reference_source].shape
                    else:
                        reference_source = 0
                        reference_shape = raw[0].shape

                    tensors = []
                    actual_dtypes = []
                    reference_index = 0
                    for source_index, handler in enumerate(handlers):
                        if source_index not in raw:
                            if mismatch_mode == "error":
                                raise ValueError(f"Tensor '{key}' is missing from input {source_index + 1}.")
                            if mismatch_mode == "zeros":
                                tensors.append(torch.zeros(
                                    reference_shape,
                                    dtype=torch.float32,
                                ))
                            continue
                        aligned = raw[source_index]
                        if not embedding_union:
                            aligned = _copy_to_target_shape(aligned, reference_shape)
                        elif aligned.ndim != len(reference_shape) or (
                            aligned.ndim > 1 and tuple(aligned.shape[1:]) != tuple(reference_shape[1:])
                        ):
                            aligned = None
                        if aligned is None:
                            if mismatch_mode == "error":
                                raise ValueError(f"Tensor shape mismatch for '{key}'.")
                            if mismatch_mode == "zeros":
                                tensors.append(torch.zeros(
                                    reference_shape,
                                    dtype=torch.float32,
                                ))
                            continue
                        if source_index == reference_source:
                            reference_index = len(tensors)
                        tensors.append(aligned)
                        actual_dtypes.append(handler.get_dtype(key))

                    if not tensors:
                        pbar.update(1)
                        continue
                    target_dtype = select_output_dtype(
                        actual_dtypes,
                        requested_dtype,
                        force=params["override_dtype"],
                    )
                    merged = _merge_tensors_to_cpu(
                        tensors,
                        settings,
                        params["process_device"],
                        target_dtype,
                        reference_index=reference_index,
                        allow_similarity_alignment=embedding_union,
                        operation_label=key,
                    )
                    writer.write(key, merged)
                    del merged
                    pbar.update(1)
        except BaseException as exc:
            writer_context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            writer_context.__exit__(None, None, None)
        return os.path.basename(output_path)

    @classmethod
    def _merge_loras(cls, handlers, low_bit_sets, model_type, params, settings):
        parsed = [parse_lora_layers(handler.keys()) for handler in handlers]
        primary_pairs, primary_passthrough = parsed[0]
        validate_canonical_blocks(primary_pairs, "CWB LoRA Merge")
        secondary_maps = [_secondary_lora_map(pairs) for pairs, _ in parsed]
        output_path = cls._output_path(model_type, params["output_filename"])
        requested_dtype = _requested_dtype(params["save_dtype"])
        mismatch_mode = params["mismatch_mode"]
        include_1d = params.get("include_1d_diffs", False)
        glob_mode = params["glob_patterns"]
        exclude = _compile_patterns(params["exclude_patterns"], glob_mode=glob_mode)
        discard = _compile_patterns(params["discard_patterns"], glob_mode=glob_mode)
        written = set()
        preserved_companion_groups = 0
        pbar = comfy.utils.ProgressBar(len(primary_pairs) + len(primary_passthrough))

        def preserve_keys(writer, keys: Iterable[str]):
            for key in keys:
                if key not in written:
                    write_preserved_tensor(writer, key, handlers[0])
                    written.add(key)

        def preserve_roles(writer, block_name: str, keys: dict[str, str], roles):
            recognized_sources = set()
            for role, source_key in layer_tensor_keys(keys).items():
                if role not in roles:
                    continue
                output_key = canonical_lora_key(block_name, role)
                recognized_sources.add(source_key)
                if output_key not in written:
                    write_preserved_tensor(writer, source_key, handlers[0], output_key)
                    written.add(output_key)
            return recognized_sources

        def preserve_layer(writer, block_name: str, keys: dict[str, str]):
            recognized_sources = preserve_roles(
                writer, block_name, keys, layer_tensor_keys(keys)
            )
            preserve_keys(
                writer,
                (
                    key for key in handlers[0].keys()
                    if key.startswith(f"{block_name}.") and key not in recognized_sources
                ),
            )

        writer_context = _atomic_output_writer(output_path, handlers[0].metadata())
        writer = writer_context.__enter__()
        try:
            with torch.no_grad():
                for primary_block, primary_keys in tqdm(
                    primary_pairs.items(), desc="CWB merging LoRA layers", unit="layers"
                ):
                    _clear_previous_layer(params)
                    core = _normalized_lora_core(primary_block)
                    matches = []
                    for index, (pairs, _) in enumerate(parsed):
                        block = primary_block if index == 0 else secondary_maps[index].get(core)
                        matches.append((index, block, pairs.get(block) if block else None))

                    layer_keys = [
                        key for key in handlers[0].keys()
                        if key in primary_keys.values() or key.startswith(f"{primary_block}.")
                    ]
                    if any(_matches_any_pattern(key, discard, glob_mode=glob_mode) for key in layer_keys):
                        pbar.update(1)
                        continue
                    guarded = any(
                        keys is not None and layer_has_low_bit(keys, low_bit_sets[index])
                        for index, _, keys in matches
                    )
                    excluded = any(
                        _matches_any_pattern(key, exclude, glob_mode=glob_mode)
                        for key in layer_keys
                    )
                    companion_bearing = any(
                        keys is not None and layer_has_companions(keys)
                        for _, _, keys in matches
                    )
                    if companion_bearing:
                        preserve_layer(writer, primary_block, primary_keys)
                        preserved_companion_groups += 1
                        pbar.update(1)
                        continue
                    if guarded or excluded:
                        preserve_layer(writer, primary_block, primary_keys)
                        pbar.update(1)
                        continue

                    logical_dtypes = []
                    for index, _, keys in matches:
                        if keys:
                            logical_dtypes.extend(
                                handlers[index].get_dtype(key)
                                for name, key in keys.items()
                                if name in {"down", "up", "alpha", "diff", "diff_b"}
                            )

                    if "down" in primary_keys and "up" in primary_keys:
                        pair_sources = []
                        pair_failed = False
                        primary_down_template = handlers[0].get_tensor(primary_keys["down"])
                        primary_up_template = handlers[0].get_tensor(primary_keys["up"])
                        for index, _, keys in matches:
                            if not keys or "down" not in keys or "up" not in keys:
                                if mismatch_mode == "error":
                                    raise ValueError(f"LoRA pair '{primary_block}' is missing from input {index + 1}.")
                                if mismatch_mode == "skip":
                                    pair_failed = True
                                    break
                                pair_sources.append((
                                    index,
                                    torch.zeros_like(primary_down_template),
                                    torch.zeros_like(primary_up_template),
                                    1.0,
                                ))
                                continue
                            down = handlers[index].get_tensor(keys["down"])
                            up = handlers[index].get_tensor(keys["up"])
                            if down.ndim < 2 or up.ndim < 2 or down.shape[0] != up.shape[1]:
                                if mismatch_mode == "error" or index == 0:
                                    raise ValueError(f"Invalid LoRA rank dimensions for '{primary_block}'.")
                                if mismatch_mode == "skip":
                                    pair_failed = True
                                    break
                                pair_sources.append((
                                    index,
                                    torch.zeros_like(primary_down_template),
                                    torch.zeros_like(primary_up_template),
                                    1.0,
                                ))
                                continue
                            scale = 1.0
                            if "alpha" in keys:
                                alpha = handlers[index].get_tensor(keys["alpha"])
                                if alpha.numel() != 1:
                                    raise ValueError(
                                        f"LoRA alpha for '{primary_block}' must be scalar."
                                    )
                                scale = float(alpha.reshape(-1)[0].item()) / down.shape[0]
                            pair_sources.append((index, down, up, scale))
                        if pair_failed:
                            preserve_roles(writer, primary_block, primary_keys, {"down", "up", "alpha"})
                        elif pair_sources:
                            primary_down = pair_sources[0][1]
                            primary_up = pair_sources[0][2]
                            compatible = []
                            for index, down, up, scale in pair_sources:
                                valid = (
                                    tuple(down.shape[1:]) == tuple(primary_down.shape[1:])
                                    and up.shape[0] == primary_up.shape[0]
                                    and tuple(up.shape[2:]) == tuple(primary_up.shape[2:])
                                )
                                if not valid:
                                    if mismatch_mode == "error":
                                        raise ValueError(f"LoRA pair shape mismatch for '{primary_block}'.")
                                    if mismatch_mode == "skip":
                                        compatible = []
                                        break
                                    compatible.append((
                                        index,
                                        torch.zeros_like(primary_down),
                                        torch.zeros_like(primary_up),
                                        1.0,
                                    ))
                                    continue
                                compatible.append((index, down, up, scale))
                            if not compatible:
                                preserve_roles(writer, primary_block, primary_keys, {"down", "up", "alpha"})
                            else:
                                target_dtype = select_output_dtype(
                                    logical_dtypes,
                                    requested_dtype,
                                    force=params["override_dtype"],
                                )
                                merged_down, merged_up, max_rank = _merge_lora_pair_to_cpu(
                                    compatible,
                                    settings,
                                    params["process_device"],
                                    target_dtype,
                                    operation_label=primary_block,
                                )
                                writer.write(
                                    canonical_lora_key(primary_block, "down"),
                                    merged_down,
                                )
                                writer.write(
                                    canonical_lora_key(primary_block, "up"),
                                    merged_up,
                                )
                                del merged_down, merged_up
                                written.update({
                                    canonical_lora_key(primary_block, "down"),
                                    canonical_lora_key(primary_block, "up"),
                                })
                                if "alpha" in primary_keys:
                                    writer.write(
                                        canonical_lora_key(primary_block, "alpha"),
                                        torch.tensor(float(max_rank), dtype=target_dtype),
                                    )
                                    written.add(canonical_lora_key(primary_block, "alpha"))

                    for kind in ("diff", "diff_b", "w_norm", "b_norm"):
                        if kind not in primary_keys:
                            continue
                        primary_key = primary_keys[kind]
                        output_key = canonical_lora_key(primary_block, kind)
                        primary_tensor = handlers[0].get_tensor(primary_key)
                        if primary_tensor.ndim == 1 and not include_1d:
                            if output_key not in written:
                                write_preserved_tensor(writer, primary_key, handlers[0], output_key)
                                written.add(output_key)
                            continue
                        direct = []
                        direct_dtypes = []
                        failed = False
                        for index, _, keys in matches:
                            if not keys or kind not in keys:
                                if mismatch_mode == "error":
                                    raise ValueError(f"Direct LoRA layer '{primary_block}.{kind}' is missing from input {index + 1}.")
                                if mismatch_mode == "skip":
                                    failed = True
                                    break
                                direct.append(torch.zeros_like(primary_tensor, dtype=torch.float32))
                                continue
                            tensor = handlers[index].get_tensor(keys[kind])
                            if tensor.shape != primary_tensor.shape:
                                if mismatch_mode == "error":
                                    raise ValueError(f"Direct LoRA shape mismatch for '{primary_block}.{kind}'.")
                                if mismatch_mode == "skip":
                                    failed = True
                                    break
                                direct.append(torch.zeros_like(primary_tensor, dtype=torch.float32))
                                continue
                            direct.append(tensor)
                            direct_dtypes.append(handlers[index].get_dtype(keys[kind]))
                        if failed:
                            if output_key not in written:
                                write_preserved_tensor(writer, primary_key, handlers[0], output_key)
                                written.add(output_key)
                            continue
                        target_dtype = select_output_dtype(
                            direct_dtypes,
                            requested_dtype,
                            force=params["override_dtype"],
                            is_1d_diff=primary_tensor.ndim == 1,
                        )
                        merged = _merge_tensors_to_cpu(
                            direct,
                            settings,
                            params["process_device"],
                            target_dtype,
                            allow_similarity_alignment=False,
                            operation_label=output_key,
                        )
                        writer.write(output_key, merged)
                        del merged
                        written.add(output_key)

                    preserve_layer(writer, primary_block, primary_keys)
                    pbar.update(1)

                for key in primary_passthrough:
                    _clear_previous_layer(params)
                    if key in written:
                        continue
                    if _matches_any_pattern(key, discard, glob_mode=glob_mode):
                        pbar.update(1)
                        continue
                    preserve_keys(writer, [key])
                    pbar.update(1)
        except BaseException as exc:
            writer_context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            writer_context.__exit__(None, None, None)
        if preserved_companion_groups:
            logging.warning(
                "[CWB LoRA Merge] Preserved %d companion-bearing Model-A group(s)",
                preserved_companion_groups,
            )
        return os.path.basename(output_path)


def _common_inputs(model_type: str, input_count: int, default_filename: str):
    label = model_type
    inputs = [
        io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"]),
        io.Combo.Input("model_a", options=folder_paths.get_filename_list(label)),
        io.Combo.Input("model_b", options=folder_paths.get_filename_list(label)),
    ]
    if input_count == 3:
        inputs.append(io.Combo.Input("model_c", options=folder_paths.get_filename_list(label)))
    inputs.extend([
        io.Combo.Input("cwb_preset", options=CWB_PRESET_OPTIONS, default="baseline"),
        io.Combo.Input("consensus_type", options=["mean", "median"], default="median"),
        io.Combo.Input("alignment_method", options=["index", "similarity"], default="similarity"),
        io.Float.Input("alignment_threshold", default=0.4, min=0.0, max=1.0, step=0.01),
        io.Float.Input("similarity_threshold", default=0.0, min=-1.0, max=1.0, step=0.01),
        io.Float.Input("power_alpha", default=2.0, min=0.0, max=10.0, step=0.1),
        io.Float.Input("diversity_beta", default=0.0, min=0.0, max=10.0, step=0.1),
        io.Boolean.Input("rescale_norm", default=False),
        io.Float.Input("global_scale", default=1.0, min=0.0, max=10.0, step=0.01),
        io.Boolean.Input("dynamic_similarity_contrast", default=False),
        io.Boolean.Input("soft_comfort_bandpass", default=False),
        io.Float.Input("position_weight", default=0.0, min=0.0, max=1.0, step=0.01),
        io.Boolean.Input("preserve_common_prefix", default=False),
        io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip"),
        io.String.Input("output_filename", default=default_filename),
        io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"]),
        io.Combo.Input("process_device", options=["cuda", "cpu"]),
        io.String.Input("exclude_patterns", default="", multiline=True),
        io.String.Input("discard_patterns", default="", multiline=True),
        io.Boolean.Input("glob_patterns", default=False),
        io.Boolean.Input("lazy_load", default=True),
        io.Boolean.Input("force_clear_cache", default=True),
        io.Boolean.Input(
            "override_dtype",
            default=False,
            tooltip="Force generated tensors to save_dtype; guarded tensors and enabled 1D direct diffs are exempt.",
        ),
    ])
    return inputs


class _CWBMergerNode(io.ComfyNode):
    MODEL_TYPE = "diffusion_models"
    INPUT_COUNT = 2
    NODE_ID = ""
    DISPLAY_NAME = ""
    DEFAULT_FILENAME = "cwb_merged"
    EMBEDDING_UNION = False
    LORA_MODE = False

    @classmethod
    def define_schema(cls):
        inputs = _common_inputs(cls.MODEL_TYPE, cls.INPUT_COUNT, cls.DEFAULT_FILENAME)
        if cls.LORA_MODE:
            inputs.append(io.Boolean.Input(
                "include_1d_diffs",
                default=False,
                tooltip="CWB-merge 1D direct diffs as FP32. Disabled preserves Model A.",
            ))
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="ModelUtils/Merging",
            inputs=inputs,
            outputs=[
                io.String.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        documentation = load_documentation_from_file("consensus_mergers.md")
        if kwargs["execution_mode"] == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", documentation)
        names = [kwargs["model_a"], kwargs["model_b"]]
        if cls.INPUT_COUNT == 3:
            names.append(kwargs["model_c"])
        filename = ConsensusMergerLogic.execute(
            names,
            cls.MODEL_TYPE,
            kwargs,
            embedding_union=cls.EMBEDDING_UNION,
            lora_mode=cls.LORA_MODE,
        )
        return io.NodeOutput(filename, documentation)


class CWBCheckpointTwoMerger(_CWBMergerNode):
    MODEL_TYPE = "checkpoints"
    NODE_ID = "CWBCheckpointTwoMerger"
    DISPLAY_NAME = "CWB Merge Checkpoints (2 Models)"
    DEFAULT_FILENAME = "cwb_merged_2_checkpoint"


class CWBCheckpointThreeMerger(CWBCheckpointTwoMerger):
    INPUT_COUNT = 3
    NODE_ID = "CWBCheckpointThreeMerger"
    DISPLAY_NAME = "CWB Merge Checkpoints (3 Models)"
    DEFAULT_FILENAME = "cwb_merged_3_checkpoint"


class CWBModelTwoMerger(_CWBMergerNode):
    NODE_ID = "CWBModelTwoMerger"
    DISPLAY_NAME = "CWB Merge Models (2 Models)"
    DEFAULT_FILENAME = "cwb_merged_2_model"


class CWBModelThreeMerger(CWBModelTwoMerger):
    INPUT_COUNT = 3
    NODE_ID = "CWBModelThreeMerger"
    DISPLAY_NAME = "CWB Merge Models (3 Models)"
    DEFAULT_FILENAME = "cwb_merged_3_model"


class CWBTextEncoderTwoMerger(_CWBMergerNode):
    MODEL_TYPE = "text_encoders"
    NODE_ID = "CWBTextEncoderTwoMerger"
    DISPLAY_NAME = "CWB Merge Text Encoders (2 Models)"
    DEFAULT_FILENAME = "cwb_merged_2_text_encoder"


class CWBTextEncoderThreeMerger(CWBTextEncoderTwoMerger):
    INPUT_COUNT = 3
    NODE_ID = "CWBTextEncoderThreeMerger"
    DISPLAY_NAME = "CWB Merge Text Encoders (3 Models)"
    DEFAULT_FILENAME = "cwb_merged_3_text_encoder"


class CWBLoRATwoMerger(_CWBMergerNode):
    MODEL_TYPE = "loras"
    NODE_ID = "CWBLoRATwoMerger"
    DISPLAY_NAME = "CWB Merge LoRAs (2 Models)"
    DEFAULT_FILENAME = "cwb_merged_2_lora"
    LORA_MODE = True


class CWBLoRAThreeMerger(CWBLoRATwoMerger):
    INPUT_COUNT = 3
    NODE_ID = "CWBLoRAThreeMerger"
    DISPLAY_NAME = "CWB Merge LoRAs (3 Models)"
    DEFAULT_FILENAME = "cwb_merged_3_lora"


class CWBEmbeddingTwoMerger(_CWBMergerNode):
    MODEL_TYPE = "embeddings"
    NODE_ID = "CWBEmbeddingTwoMerger"
    DISPLAY_NAME = "CWB Merge Embeddings (2 Models)"
    DEFAULT_FILENAME = "cwb_merged_2_embedding"
    EMBEDDING_UNION = True


class CWBEmbeddingThreeMerger(CWBEmbeddingTwoMerger):
    INPUT_COUNT = 3
    NODE_ID = "CWBEmbeddingThreeMerger"
    DISPLAY_NAME = "CWB Merge Embeddings (3 Models)"
    DEFAULT_FILENAME = "cwb_merged_3_embedding"


CWB_MERGER_NODES = [
    CWBCheckpointTwoMerger,
    CWBCheckpointThreeMerger,
    CWBModelTwoMerger,
    CWBModelThreeMerger,
    CWBTextEncoderTwoMerger,
    CWBTextEncoderThreeMerger,
    CWBLoRATwoMerger,
    CWBLoRAThreeMerger,
    CWBEmbeddingTwoMerger,
    CWBEmbeddingThreeMerger,
]
