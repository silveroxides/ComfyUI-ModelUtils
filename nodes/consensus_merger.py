"""Dedicated streaming Consensus-Weighted Blending merger nodes."""

from __future__ import annotations

import json
import logging
import os
from contextlib import closing
from dataclasses import asdict, dataclass, field
from typing import Iterable

import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from tqdm import tqdm
from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned

from .device_utils import (
    cleanup_after_operation,
    estimate_model_size,
    prepare_for_large_operation,
)
from .uel_io import atomic_uel_writer, stream_work_units
from .artifact_paths import canonical_model_artifact_path
from .lora_alpha import normalize_lora_pair
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
    DiffusionQuantization,
    diffusion_key_map,
    inspect_low_bit_input,
    write_preserved_tensor,
)


def _preset(
    *, consensus_type, alignment_method, alignment_threshold,
    similarity_threshold, power_alpha, diversity_beta,
    rescale_norm, global_scale, dynamic_similarity_contrast,
    soft_comfort_bandpass, position_weight, preserve_common_prefix,
):
    return {
        "consensus_type": consensus_type,
        "alignment_method": alignment_method,
        "alignment_threshold": alignment_threshold,
        "similarity_threshold": similarity_threshold,
        "power_alpha": power_alpha,
        "diversity_beta": diversity_beta,
        "rescale_norm": rescale_norm,
        "global_scale": global_scale,
        "dynamic_similarity_contrast": dynamic_similarity_contrast,
        "soft_comfort_bandpass": soft_comfort_bandpass,
        "position_weight": position_weight,
        "preserve_common_prefix": preserve_common_prefix,
    }


DENSE_CWB_PRESETS = {
    "balanced_mean": _preset(
        consensus_type="mean", alignment_method="index", alignment_threshold=0.0,
        similarity_threshold=0.0, power_alpha=1.0, diversity_beta=0.0,
        rescale_norm=False, global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.0,
        preserve_common_prefix=False,
    ),
    "robust_medn": _preset(
        consensus_type="median", alignment_method="index", alignment_threshold=0.0,
        similarity_threshold=0.0, power_alpha=2.0, diversity_beta=0.0,
        rescale_norm=False, global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.0,
        preserve_common_prefix=False,
    ),
    "selective_mean": _preset(
        consensus_type="mean", alignment_method="index", alignment_threshold=0.0,
        similarity_threshold=0.3, power_alpha=3.0, diversity_beta=0.0,
        rescale_norm=False, global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.0,
        preserve_common_prefix=False,
    ),
    "varied_mean_rn_softcb": _preset(
        consensus_type="mean", alignment_method="index", alignment_threshold=0.0,
        similarity_threshold=0.0, power_alpha=1.5, diversity_beta=2.0,
        rescale_norm=True, global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=True, position_weight=0.0,
        preserve_common_prefix=False,
    ),
    "diverse_medn_rn_dsc_softcb": _preset(
        consensus_type="median", alignment_method="index", alignment_threshold=0.0,
        similarity_threshold=0.0, power_alpha=2.0, diversity_beta=4.0,
        rescale_norm=True, global_scale=1.0, dynamic_similarity_contrast=True,
        soft_comfort_bandpass=True, position_weight=0.0,
        preserve_common_prefix=False,
    ),
    "strongdiv_medn_rn_dsc_softcb": _preset(
        consensus_type="median", alignment_method="index", alignment_threshold=0.0,
        similarity_threshold=0.0, power_alpha=2.0, diversity_beta=10.0,
        rescale_norm=True, global_scale=1.0, dynamic_similarity_contrast=True,
        soft_comfort_bandpass=True, position_weight=0.0,
        preserve_common_prefix=False,
    ),
}

EMBEDDING_CWB_PRESETS = {
    "balanced_idx_mean": _preset(
        consensus_type="mean", alignment_method="index", alignment_threshold=0.0,
        similarity_threshold=0.0, power_alpha=1.5, diversity_beta=0.0,
        rescale_norm=False, global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.0,
        preserve_common_prefix=False,
    ),
    "balanced_sim_mean": _preset(
        consensus_type="mean", alignment_method="similarity", alignment_threshold=0.4,
        similarity_threshold=0.0, power_alpha=1.5, diversity_beta=0.0,
        rescale_norm=False, global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.1,
        preserve_common_prefix=False,
    ),
    "robust_idx_medn": _preset(
        consensus_type="median", alignment_method="index", alignment_threshold=0.0,
        similarity_threshold=0.0, power_alpha=2.0, diversity_beta=0.0,
        rescale_norm=False, global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.0,
        preserve_common_prefix=False,
    ),
    "robust_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.4, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.1,
        preserve_common_prefix=False,
    ),
    "varied_sim_mean_rn_softcb": _preset(
        consensus_type="mean", alignment_method="similarity",
        alignment_threshold=0.2, similarity_threshold=0.0,
        power_alpha=1.5, diversity_beta=2.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=True, position_weight=0.1,
        preserve_common_prefix=False,
    ),
    "diverse_sim_medn_rn_dsc_softcb": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.0, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=4.0, rescale_norm=True,
        global_scale=1.0,
        dynamic_similarity_contrast=True, soft_comfort_bandpass=True,
        position_weight=0.1, preserve_common_prefix=False,
    ),
    "focused_strong_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.85, similarity_threshold=0.60,
        power_alpha=1.25, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.20,
        preserve_common_prefix=False,
    ),
    "focused_balance_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.75, similarity_threshold=0.55,
        power_alpha=1.25, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.20,
        preserve_common_prefix=False,
    ),
    "focused_soft_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.55, similarity_threshold=0.50,
        power_alpha=1.25, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.20,
        preserve_common_prefix=False,
    ),
    "focused_weak_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.25, similarity_threshold=0.35,
        power_alpha=1.25, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.20,
        preserve_common_prefix=False,
    ),
}

LORA_CWB_PRESETS = {
    "broad_sim_medn_rn_softcb": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.0, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=4.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=True, position_weight=0.05,
        preserve_common_prefix=False,
    ),
    "moderate_sim_medn_rn_softcb": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.0005, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=4.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=True, position_weight=0.05,
        preserve_common_prefix=False,
    ),
    "conservative_sim_medn_rn_softcb": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.0025, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=4.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=True, position_weight=0.05,
        preserve_common_prefix=False,
    ),
    "direct_idx_medn_rn_softcb": _preset(
        consensus_type="median", alignment_method="index",
        alignment_threshold=0.0, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=4.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=True, position_weight=0.05,
        preserve_common_prefix=False,
    ),
    "broad_sim_mean_rn_softcb": _preset(
        consensus_type="mean", alignment_method="similarity",
        alignment_threshold=0.0, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=4.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=True, position_weight=0.05,
        preserve_common_prefix=False,
    ),
    "neutral_sim_medn_rn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.0, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=0.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.05,
        preserve_common_prefix=False,
    ),
    "focused_sim_medn_rn_dsc_softcb": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.0, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=4.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=True,
        soft_comfort_bandpass=True, position_weight=0.05,
        preserve_common_prefix=False,
    ),
    "strongfocus_sim_medn_rn_dsc_softcb": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.0, similarity_threshold=0.0,
        power_alpha=2.0, diversity_beta=7.0, rescale_norm=True,
        global_scale=1.0, dynamic_similarity_contrast=True,
        soft_comfort_bandpass=True, position_weight=0.05,
        preserve_common_prefix=False,
    ),
    "focused_strong_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.85, similarity_threshold=0.60,
        power_alpha=1.25, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.20,
        preserve_common_prefix=False,
    ),
    "focused_balance_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.75, similarity_threshold=0.55,
        power_alpha=1.25, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.20,
        preserve_common_prefix=False,
    ),
    "focused_soft_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.55, similarity_threshold=0.50,
        power_alpha=1.25, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.20,
        preserve_common_prefix=False,
    ),
    "focused_weak_sim_medn": _preset(
        consensus_type="median", alignment_method="similarity",
        alignment_threshold=0.25, similarity_threshold=0.35,
        power_alpha=1.25, diversity_beta=0.0, rescale_norm=False,
        global_scale=1.0, dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False, position_weight=0.20,
        preserve_common_prefix=False,
    ),
}


def _presets_for(*, embedding_union: bool, lora_mode: bool):
    if lora_mode:
        return LORA_CWB_PRESETS
    if embedding_union:
        return EMBEDDING_CWB_PRESETS
    return DENSE_CWB_PRESETS
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


@dataclass
class EmbeddingCoalesceReport:
    """Per-output accounting for dedicated embedding coalescing nodes."""

    target_vector_count: int
    similarity_threshold: float
    position_window: float
    vision_boundary_embeddings: bool
    legacy_boundary_search: bool
    boundary_similarity_threshold: float
    boundary_reference: str | None
    tensors: list[tuple[str, int, int, int, tuple[float, ...], tuple[tuple[int, int], ...]]] = field(default_factory=list)

    def record(self, key, input_rows, output_rows, trimmed_rows, boundary_scores, pairs):
        self.tensors.append((
            key, input_rows, output_rows, trimmed_rows,
            tuple(float(score) for score in boundary_scores), tuple(pairs),
        ))

    def render(self) -> str:
        lines = [
            "", "CWB EMBEDDING COALESCING",
            f"Target vector count: {self.target_vector_count} (0 means all eligible pairs)",
            "Minimum retained body rows: 1",
            f"Similarity threshold: {self.similarity_threshold:.4f}",
            f"Position window: {self.position_window:.4f} normalized sequence distance",
            f"Vision boundaries: {self.vision_boundary_embeddings}",
            f"Legacy boundary search: {self.legacy_boundary_search}",
            f"Boundary similarity threshold: {self.boundary_similarity_threshold:.4f}",
            f"Boundary reference: {self.boundary_reference or 'none'}",
        ]
        for key, before, after, trimmed, scores, pairs in self.tensors:
            lines.append(
                f"{key}: {before} -> {after} rows; trimmed={trimmed}; "
                f"boundary_scores={[round(score, 6) for score in scores]}; pairs={list(pairs)}"
            )
        return "\n".join(lines)


def _optional_min(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None else min(current, value)


def _optional_max(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None else max(current, value)


COUNTERFACTUAL_SIMILARITY_THRESHOLDS = (0.0, 0.1, 0.25)
COUNTERFACTUAL_POWER_ALPHAS = (1.0, 2.0, 4.0)
COUNTERFACTUAL_DIVERSITY_BETAS = (0.0, 2.0, 4.0, 7.0, 10.0)


@dataclass
class CWBWeightSweepStat:
    groups: int = 0
    contributors: int = 0
    accepted: int = 0
    all_rejected: int = 0
    zero_weight: int = 0
    dominant_sum: float = 0.0
    dominant_max: float = 0.0
    effective_sum: float = 0.0
    effective_min: float | None = None

    def absorb(self, other: "CWBWeightSweepStat") -> None:
        self.groups += other.groups
        self.contributors += other.contributors
        self.accepted += other.accepted
        self.all_rejected += other.all_rejected
        self.zero_weight += other.zero_weight
        self.dominant_sum += other.dominant_sum
        self.dominant_max = max(self.dominant_max, other.dominant_max)
        self.effective_sum += other.effective_sum
        self.effective_min = _optional_min(self.effective_min, other.effective_min)


@dataclass
class CWBWeightSweepDiagnostics:
    stats: dict[tuple, CWBWeightSweepStat] = field(default_factory=dict)

    def record(self, stacked: torch.Tensor) -> None:
        for consensus_type in ("mean", "median"):
            consensus = (
                torch.mean(stacked, dim=0)
                if consensus_type == "mean"
                else torch.median(stacked, dim=0).values
            )
            similarities = torch.mv(
                F.normalize(stacked, p=2, dim=1, eps=1e-8),
                F.normalize(consensus, p=2, dim=0, eps=1e-8),
            )
            self._record_similarities(consensus_type, similarities)

    def _record_similarities(
        self,
        consensus_type: str,
        similarities: torch.Tensor,
    ) -> None:
        values = similarities.detach().float().cpu().tolist()
        minimum = min(values)
        maximum = max(values)
        for dsc in (False, True):
            if dsc and maximum > minimum:
                weighted = [
                    0.7 + 0.3 * (value - minimum) / (maximum - minimum + 1e-8)
                    for value in values
                ]
            else:
                weighted = values
            for threshold in COUNTERFACTUAL_SIMILARITY_THRESHOLDS:
                accepted = [value >= threshold for value in values]
                for alpha in COUNTERFACTUAL_POWER_ALPHAS:
                    for beta in COUNTERFACTUAL_DIVERSITY_BETAS:
                        for softcb in (False, True):
                            key = (
                                consensus_type, threshold, alpha, beta, dsc, softcb
                            )
                            stat = self.stats.setdefault(key, CWBWeightSweepStat())
                            stat.groups += 1
                            stat.contributors += len(values)
                            stat.accepted += sum(accepted)
                            if not any(accepted):
                                stat.all_rejected += 1
                                weights = [1.0 / len(values)] * len(values)
                            else:
                                distance_base = 1.5 if softcb else 1.001
                                weights = []
                                for value, include in zip(weighted, accepted):
                                    if not include:
                                        weights.append(0.0)
                                        continue
                                    safe = min(max(value, 0.0), 1.0)
                                    weight = safe ** alpha
                                    if beta > 0.0:
                                        weight *= max(distance_base - safe, 0.0) ** beta
                                    weights.append(weight)
                                total = sum(weights)
                                if total <= 0.0:
                                    stat.zero_weight += 1
                                    weights = [1.0 / len(values)] * len(values)
                                else:
                                    weights = [weight / total for weight in weights]
                            dominant = max(weights)
                            effective = 1.0 / sum(weight * weight for weight in weights)
                            stat.dominant_sum += dominant
                            stat.dominant_max = max(stat.dominant_max, dominant)
                            stat.effective_sum += effective
                            stat.effective_min = _optional_min(
                                stat.effective_min, effective
                            )

    def absorb(self, other: "CWBWeightSweepDiagnostics") -> None:
        for key, other_stat in other.stats.items():
            self.stats.setdefault(key, CWBWeightSweepStat()).absorb(other_stat)

    def render(self) -> str:
        lines = [
            "",
            "CWB COUNTERFACTUAL WEIGHT SWEEP",
            "consensus | sim_threshold | alpha | beta | dsc | softcb | accepted | fallbacks | dominant_mean/max | effective_mean/min",
        ]
        for key in sorted(self.stats):
            consensus_type, threshold, alpha, beta, dsc, softcb = key
            stat = self.stats[key]
            lines.append(
                f"{consensus_type} | {threshold:.6g} | {alpha:.6g} | {beta:.6g} | "
                f"{str(dsc).lower()} | {str(softcb).lower()} | "
                f"{stat.accepted}/{stat.contributors} | "
                f"{stat.all_rejected + stat.zero_weight} | "
                f"{stat.dominant_sum / stat.groups:.6g}/{stat.dominant_max:.6g} | "
                f"{stat.effective_sum / stat.groups:.6g}/{stat.effective_min:.6g}"
            )
        return "\n".join(lines)


@dataclass
class CWBDiagnostics:
    alignment_candidates: int = 0
    alignment_matches: int = 0
    anchor_only_groups: int = 0
    weighting_groups: int = 0
    all_rejected_fallbacks: int = 0
    zero_weight_fallbacks: int = 0
    weighting_contributors: int = 0
    accepted_weighting_contributors: int = 0
    consensus_similarity_sum: float = 0.0
    consensus_similarity_min: float | None = None
    consensus_similarity_max: float | None = None
    dominant_weight_sum: float = 0.0
    dominant_weight_min: float | None = None
    dominant_weight_max: float | None = None
    effective_contributor_sum: float = 0.0
    effective_contributor_min: float | None = None
    effective_contributor_max: float | None = None
    cpu_fallbacks: int = 0
    lora_groups: int = 0
    mixed_rank_lora_groups: int = 0
    genuine_lora_components: int = 0
    structural_rank_slots_excluded: int = 0
    explicit_zero_contributors: int = 0
    lora_layer_reports: list["LoRALayerReport"] = field(default_factory=list)
    weight_sweep: CWBWeightSweepDiagnostics | None = None

    def absorb_runtime(self, other: "CWBDiagnostics") -> None:
        self.alignment_candidates += other.alignment_candidates
        self.alignment_matches += other.alignment_matches
        self.anchor_only_groups += other.anchor_only_groups
        self.weighting_groups += other.weighting_groups
        self.all_rejected_fallbacks += other.all_rejected_fallbacks
        self.zero_weight_fallbacks += other.zero_weight_fallbacks
        self.weighting_contributors += other.weighting_contributors
        self.accepted_weighting_contributors += other.accepted_weighting_contributors
        self.consensus_similarity_sum += other.consensus_similarity_sum
        self.consensus_similarity_min = _optional_min(
            self.consensus_similarity_min, other.consensus_similarity_min
        )
        self.consensus_similarity_max = _optional_max(
            self.consensus_similarity_max, other.consensus_similarity_max
        )
        self.dominant_weight_sum += other.dominant_weight_sum
        self.dominant_weight_min = _optional_min(
            self.dominant_weight_min, other.dominant_weight_min
        )
        self.dominant_weight_max = _optional_max(
            self.dominant_weight_max, other.dominant_weight_max
        )
        self.effective_contributor_sum += other.effective_contributor_sum
        self.effective_contributor_min = _optional_min(
            self.effective_contributor_min, other.effective_contributor_min
        )
        self.effective_contributor_max = _optional_max(
            self.effective_contributor_max, other.effective_contributor_max
        )
        if other.weight_sweep is not None:
            if self.weight_sweep is None:
                self.weight_sweep = CWBWeightSweepDiagnostics()
            self.weight_sweep.absorb(other.weight_sweep)

    def record_weighting(
        self,
        similarities: torch.Tensor,
        accepted: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        count = similarities.numel()
        self.weighting_contributors += count
        self.accepted_weighting_contributors += int(accepted.sum().item())
        minimum = float(similarities.min().item())
        maximum = float(similarities.max().item())
        self.consensus_similarity_sum += float(similarities.sum().item())
        self.consensus_similarity_min = _optional_min(
            self.consensus_similarity_min, minimum
        )
        self.consensus_similarity_max = _optional_max(
            self.consensus_similarity_max, maximum
        )
        dominant = float(weights.max().item())
        effective = float((1.0 / weights.square().sum().clamp_min(1e-12)).item())
        self.dominant_weight_sum += dominant
        self.dominant_weight_min = _optional_min(self.dominant_weight_min, dominant)
        self.dominant_weight_max = _optional_max(self.dominant_weight_max, dominant)
        self.effective_contributor_sum += effective
        self.effective_contributor_min = _optional_min(
            self.effective_contributor_min, effective
        )
        self.effective_contributor_max = _optional_max(
            self.effective_contributor_max, effective
        )

    def render(self, preset_name: str, input_count: int, custom: bool) -> str:
        source = "connected custom configuration" if custom else preset_name
        lines = [
            "CWB MERGE SUMMARY",
            f"Settings: {source}",
            f"Input contributors: {input_count}",
            f"Alignment candidates: {self.alignment_candidates}",
            f"Alignment matches: {self.alignment_matches}",
            f"Anchor-only groups: {self.anchor_only_groups}",
            f"Weighted vector groups: {self.weighting_groups}",
            f"All-rejected equal fallbacks: {self.all_rejected_fallbacks}",
            f"Zero-weight equal fallbacks: {self.zero_weight_fallbacks}",
            "Accepted weighting contributors: "
            f"{self.accepted_weighting_contributors}/{self.weighting_contributors}",
            "Consensus similarity min/mean/max: "
            f"{_format_metric(self.consensus_similarity_min)} / "
            f"{_format_metric(self.consensus_similarity_sum / self.weighting_contributors if self.weighting_contributors else None)} / "
            f"{_format_metric(self.consensus_similarity_max)}",
            "Dominant normalized weight min/mean/max: "
            f"{_format_metric(self.dominant_weight_min)} / "
            f"{_format_metric(self.dominant_weight_sum / self.weighting_groups if self.weighting_groups else None)} / "
            f"{_format_metric(self.dominant_weight_max)}",
            "Effective contributors min/mean/max: "
            f"{_format_metric(self.effective_contributor_min)} / "
            f"{_format_metric(self.effective_contributor_sum / self.weighting_groups if self.weighting_groups else None)} / "
            f"{_format_metric(self.effective_contributor_max)}",
            f"CUDA OOM CPU fallbacks: {self.cpu_fallbacks}",
        ]
        if self.lora_groups:
            lines.extend([
                f"LoRA groups: {self.lora_groups}",
                f"Mixed-rank LoRA groups: {self.mixed_rank_lora_groups}",
                f"Genuine LoRA components: {self.genuine_lora_components}",
                "Structural rank slots excluded: "
                f"{self.structural_rank_slots_excluded}",
                f"Explicit mismatch-zero contributors: {self.explicit_zero_contributors}",
            ])
        if self.lora_layer_reports:
            lines.extend(_render_lora_dimension_groups(self.lora_layer_reports))
        if self.weight_sweep is not None:
            lines.append(self.weight_sweep.render())
        return "\n".join(lines)


@dataclass
class LoRALayerReport:
    layer: str
    ranks: tuple[int, ...]
    reference_input: int
    reference_rank: int
    down_dimensions: tuple[int, ...]
    up_dimensions: tuple[int, ...]
    alignment_method: str
    structural_slots_excluded: int
    explicit_zero_contributors: int
    reference_components: int = 0
    prefix_components: int = 0
    alignable_source_components: int = 0
    matches: int = 0
    anchor_only_components: int = 0
    candidate_score_count: int = 0
    candidate_score_sum: float = 0.0
    candidate_score_min: float | None = None
    candidate_score_max: float | None = None
    matched_scores: list[float] = field(default_factory=list)
    norm_ratios: list[float] = field(default_factory=list)
    matrix_shapes: list[tuple[int, int]] = field(default_factory=list)

    def record_candidates(self, scores: torch.Tensor) -> None:
        values = scores.detach().float()
        count = values.numel()
        if not count:
            return
        minimum = float(values.min().item())
        maximum = float(values.max().item())
        self.candidate_score_count += count
        self.candidate_score_sum += float(values.sum().item())
        self.candidate_score_min = (
            minimum if self.candidate_score_min is None
            else min(self.candidate_score_min, minimum)
        )
        self.candidate_score_max = (
            maximum if self.candidate_score_max is None
            else max(self.candidate_score_max, maximum)
        )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _render_lora_dimension_groups(
    layers: list[LoRALayerReport],
) -> list[str]:
    grouped = {}
    for layer in layers:
        key = (
            layer.down_dimensions,
            layer.up_dimensions,
            layer.ranks,
            layer.reference_input,
            layer.alignment_method,
        )
        grouped.setdefault(key, []).append(layer)

    lines = ["", "CWB LORA DIMENSION GROUPS"]
    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: (-len(group), group[0].layer),
    )
    for number, group in enumerate(ordered_groups, start=1):
        sample = group[0]
        candidate_count = sum(item.candidate_score_count for item in group)
        candidate_sum = sum(item.candidate_score_sum for item in group)
        candidate_min = min(
            (item.candidate_score_min for item in group if item.candidate_score_min is not None),
            default=None,
        )
        candidate_max = max(
            (item.candidate_score_max for item in group if item.candidate_score_max is not None),
            default=None,
        )
        matched_scores = [score for item in group for score in item.matched_scores]
        norm_ratios = [ratio for item in group for ratio in item.norm_ratios]
        alignable = sum(item.alignable_source_components for item in group)
        matches = sum(item.matches for item in group)
        reference_components = sum(item.reference_components for item in group)
        anchor_only = sum(item.anchor_only_components for item in group)
        matrix_shapes = sorted({shape for item in group for shape in item.matrix_shapes})
        matrix_text = (
            ", ".join(f"{rows}x{columns}" for rows, columns in matrix_shapes)
            if matrix_shapes else "index alignment"
        )
        lines.extend([
            "",
            f"Group {number}: {len(group)} layer(s)",
            f"Down non-rank dimensions: {sample.down_dimensions}",
            f"Up non-rank dimensions: {sample.up_dimensions}",
            f"Input ranks: {sample.ranks}",
            f"Reference: input {sample.reference_input}, rank {sample.reference_rank}",
            f"Alignment: {sample.alignment_method}; matrices: {matrix_text}",
            f"Matched source components: {matches}/{alignable} "
            f"({(100.0 * matches / alignable) if alignable else 100.0:.2f}%)",
            f"Reference-only components: {anchor_only}/{reference_components}",
            "Structural rank slots excluded: "
            f"{sum(item.structural_slots_excluded for item in group)}",
            "Explicit mismatch-zero contributors: "
            f"{sum(item.explicit_zero_contributors for item in group)}",
            f"Candidate similarity count: {candidate_count}",
            "Candidate similarity min/mean/max: "
            f"{_format_metric(candidate_min)} / "
            f"{_format_metric(candidate_sum / candidate_count if candidate_count else None)} / "
            f"{_format_metric(candidate_max)}",
            "Matched similarity p05/median/p95: "
            f"{_format_metric(_percentile(matched_scores, 0.05))} / "
            f"{_format_metric(_percentile(matched_scores, 0.5))} / "
            f"{_format_metric(_percentile(matched_scores, 0.95))}",
            "Rank-one delta norm ratio count/min/mean/max: "
            f"{len(norm_ratios)} / "
            f"{_format_metric(min(norm_ratios) if norm_ratios else None)} / "
            f"{_format_metric(sum(norm_ratios) / len(norm_ratios) if norm_ratios else None)} / "
            f"{_format_metric(max(norm_ratios) if norm_ratios else None)}",
        ])

    def coverage(item: LoRALayerReport) -> float:
        return (
            item.matches / item.alignable_source_components
            if item.alignable_source_components else 1.0
        )

    coverage_exceptions = sorted(layers, key=lambda item: (coverage(item), item.layer))[:5]
    norm_exceptions = sorted(
        (item for item in layers if item.norm_ratios),
        key=lambda item: max(abs(value - 1.0) for value in item.norm_ratios),
        reverse=True,
    )[:5]
    lines.extend(["", "CWB LORA EXCEPTIONS", "Lowest match coverage:"])
    lines.extend(
        f"- {item.layer}: {item.matches}/{item.alignable_source_components} "
        f"({coverage(item) * 100.0:.2f}%)"
        for item in coverage_exceptions
    )
    lines.append("Largest rank-one delta norm change:")
    lines.extend(
        f"- {item.layer}: min={min(item.norm_ratios):.6g}, "
        f"max={max(item.norm_ratios):.6g}"
        for item in norm_exceptions
    )
    return lines


@dataclass
class LoRAPairSource:
    """One genuine normalized pair or one intentional mismatch-zero input."""

    source_index: int
    down: torch.Tensor | None
    up: torch.Tensor | None
    scale: float = 1.0
    synthetic_zero: bool = False

    @classmethod
    def zero(cls, source_index: int) -> "LoRAPairSource":
        return cls(source_index, None, None, synthetic_zero=True)

    @property
    def rank(self) -> int:
        if self.synthetic_zero or self.down is None:
            raise ValueError("A synthetic-zero LoRA source has no genuine rank.")
        return int(self.down.shape[0])


CWB_CONFIG = io.Custom("CWB_CONFIG")


def resolve_cwb_settings(
    preset_name: str,
    presets: dict[str, dict],
    custom_config: CWBSettings | None = None,
) -> CWBSettings:
    """Resolve one complete use-case preset or an explicitly connected config."""
    if custom_config is not None:
        if not isinstance(custom_config, CWBSettings):
            raise TypeError("CWB config input must come from CWB Custom Configuration.")
        return custom_config
    try:
        resolved = presets[preset_name].copy()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported CWB preset '{preset_name}' for this merge type. "
            "Select a current preset or connect CWB Custom Configuration."
        ) from exc
    if not 0.0 <= resolved["position_weight"] <= 1.0:
        raise ValueError("Position weight must be between 0.0 and 1.0.")
    return CWBSettings(**resolved)


def build_custom_cwb_settings(**values) -> CWBSettings:
    settings = CWBSettings(
        consensus_type=values["consensus_type"],
        alignment_method=values["alignment_method"],
        alignment_threshold=float(values["alignment_threshold"]),
        similarity_threshold=float(values["similarity_threshold"]),
        power_alpha=float(values["power_alpha"]),
        diversity_beta=float(values["diversity_beta"]),
        rescale_norm=bool(values["rescale_norm"]),
        global_scale=float(values["global_scale"]),
        dynamic_similarity_contrast=bool(values["dynamic_similarity_contrast"]),
        soft_comfort_bandpass=bool(values["soft_comfort_bandpass"]),
        position_weight=float(values["position_weight"]),
        preserve_common_prefix=bool(values["preserve_common_prefix"]),
    )
    if not 0.0 <= settings.position_weight <= 1.0:
        raise ValueError("Position weight must be between 0.0 and 1.0.")
    return settings


def _cwb_merge_metadata(
    source_models: list[str],
    params: dict,
    settings: CWBSettings,
    *,
    operation: str,
) -> str:
    metadata = {
        "schema_version": 1,
        "operation": operation,
        "source_models": source_models,
        "effective_cwb_settings": asdict(settings),
        "merge_options": {},
    }
    for key in (
        "mismatch_mode",
        "save_dtype",
        "process_device",
        "lazy_load",
        "force_clear_cache",
        "override_dtype",
        "include_1d_diffs",
        "include_mode",
        "counterfactual_weight_sweep",
        "embedding_coalesce",
        "target_vector_count",
        "coalesce_similarity_threshold",
        "coalesce_position_window",
        "vision_boundary_embeddings",
        "legacy_boundary_search",
        "boundary_reference_embedding",
        "boundary_similarity_threshold",
    ):
        if key in params:
            metadata["merge_options"][key] = params[key]
    exclude_patterns = params["exclude_patterns"].split()
    discard_patterns = params["discard_patterns"].split()
    if exclude_patterns or discard_patterns or params.get("include_mode", False):
        metadata["filters"] = {
            "exclude_patterns": exclude_patterns,
            "include_mode": params.get("include_mode", False),
            "discard_patterns": discard_patterns,
            "glob_patterns": params["glob_patterns"],
        }
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


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
    diagnostics: CWBDiagnostics | None = None,
) -> torch.Tensor:
    """Merge one aligned vector group using CWB."""
    if stacked.shape[0] == 1:
        merged = stacked[0].clone()
    else:
        if diagnostics is not None:
            diagnostics.weighting_groups += 1
            if diagnostics.weight_sweep is not None:
                diagnostics.weight_sweep.record(stacked)
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
                if diagnostics is not None:
                    diagnostics.zero_weight_fallbacks += 1
                row_weights.fill_(1.0 / len(similarities))
        else:
            if diagnostics is not None:
                diagnostics.all_rejected_fallbacks += 1
            row_weights.fill_(1.0 / len(similarities))
        if diagnostics is not None:
            diagnostics.record_weighting(similarities, mask, row_weights)
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
    diagnostics: CWBDiagnostics | None = None,
) -> list[int]:
    """Match source rows to reference rows without duplicating the score matrix."""
    scores = _position_biased_scores(similarities, settings.position_weight)
    scores.masked_fill_(similarities < settings.alignment_threshold, -100.0)
    matched = [-1] * similarities.shape[0]
    if diagnostics is not None:
        diagnostics.alignment_candidates += similarities.shape[0]
    for _ in range(min(similarities.shape)):
        flat_index = torch.argmax(scores)
        best = scores.flatten()[flat_index].item()
        if best <= -100.0:
            break
        ref_row = int((flat_index // similarities.shape[1]).item())
        source_row = int((flat_index % similarities.shape[1]).item())
        matched[ref_row] = source_row
        if diagnostics is not None:
            diagnostics.alignment_matches += 1
        scores[ref_row, :] = -100.0
        scores[:, source_row] = -100.0
    return matched


def merge_cwb_tensors(
    tensors: list[torch.Tensor],
    settings: CWBSettings,
    *,
    reference_index: int = 0,
    allow_similarity_alignment: bool = True,
    diagnostics: CWBDiagnostics | None = None,
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
        matched = _greedy_similarity_matches(similarities, settings, diagnostics)
        for ref_row, source_row in enumerate(matched):
            if source_row >= 0:
                groups[ref_row].append(tensor[source_row])

    merged_rows = []
    if diagnostics is not None and len(tensors) > 1:
        diagnostics.anchor_only_groups += sum(len(group) == 1 for group in groups)
    for group in groups:
        stacked = torch.stack([row.reshape(-1) for row in group])
        merged_rows.append(
            merge_consensus_group(
                stacked, settings, diagnostics=diagnostics
            ).reshape(group[0].shape)
        )
    merged = torch.stack(merged_rows)
    if prefix_length:
        merged = torch.cat([preserved_prefix, merged], dim=0)
    return merged


def _common_lora_prefix_length(
    downs: list[torch.Tensor],
    ups: list[torch.Tensor],
    reference_index: int,
    synthetic_zero_count: int = 0,
) -> int:
    reference_down = downs[reference_index]
    reference_up = ups[reference_index]
    limit = min(down.shape[0] for down in downs)
    common = torch.ones(limit, dtype=torch.bool, device=reference_down.device)
    for down, up in zip(downs, ups):
        down_equal = torch.isclose(
            reference_down[:limit], down[:limit], rtol=1e-5, atol=1e-6
        ).reshape(limit, -1).all(dim=1)
        up_equal = torch.isclose(
            reference_up.movedim(1, 0)[:limit],
            up.movedim(1, 0)[:limit],
            rtol=1e-5,
            atol=1e-6,
        ).reshape(limit, -1).all(dim=1)
        common &= down_equal & up_equal
    if synthetic_zero_count:
        common &= torch.isclose(
            reference_down[:limit],
            torch.zeros((), device=reference_down.device, dtype=reference_down.dtype),
            rtol=1e-5,
            atol=1e-6,
        ).reshape(limit, -1).all(dim=1)
        common &= torch.isclose(
            reference_up.movedim(1, 0)[:limit],
            torch.zeros((), device=reference_up.device, dtype=reference_up.dtype),
            rtol=1e-5,
            atol=1e-6,
        ).reshape(limit, -1).all(dim=1)
    mismatch = torch.nonzero(~common, as_tuple=False)
    return limit if mismatch.numel() == 0 else int(mismatch[0].item())


def merge_cwb_lora_pairs(
    downs: list[torch.Tensor],
    ups: list[torch.Tensor],
    settings: CWBSettings,
    *,
    reference_index: int,
    synthetic_zero_count: int = 0,
    diagnostics: CWBDiagnostics | None = None,
    layer_report: LoRALayerReport | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CWB LoRA rank components while keeping A rows paired with B columns."""
    if not downs or len(downs) != len(ups):
        raise ValueError("CWB requires matching LoRA A/B source lists.")
    if not 0 <= reference_index < len(downs):
        raise ValueError("CWB LoRA reference index is out of range.")
    for down, up in zip(downs, ups):
        if down.ndim < 2 or up.ndim < 2 or down.shape[0] <= 0:
            raise ValueError("CWB LoRA sources must contain positive-rank factor pairs.")
        if down.shape[0] != up.shape[1]:
            raise ValueError("CWB LoRA down/up factors must have matching ranks.")
    reference_down = downs[reference_index]
    reference_up = ups[reference_index]
    rank = reference_down.shape[0]
    if any(
        tuple(down.shape[1:]) != tuple(reference_down.shape[1:])
        or up.shape[0] != reference_up.shape[0]
        or tuple(up.shape[2:]) != tuple(reference_up.shape[2:])
        for down, up in zip(downs, ups)
    ):
        raise ValueError("CWB LoRA sources must have matching non-rank dimensions.")

    prefix = (
        _common_lora_prefix_length(
            downs, ups, reference_index, synthetic_zero_count
        )
        if settings.preserve_common_prefix else 0
    )
    if layer_report is not None:
        layer_report.prefix_components = prefix
        layer_report.reference_components = rank - prefix
    reference_up_components = reference_up.movedim(1, 0)
    down_groups = [[reference_down[row]] for row in range(prefix, rank)]
    up_groups = [[reference_up_components[row]] for row in range(prefix, rank)]
    matched_score_tensors = []
    norm_ratio_tensors = []

    ref_down_body = reference_down[prefix:].reshape(rank - prefix, -1)
    ref_up_body = reference_up_components[prefix:].reshape(rank - prefix, -1)
    for index, (down, up) in enumerate(zip(downs, ups)):
        if index == reference_index or not down_groups:
            continue
        source_down_components = down[prefix:]
        source_up_components = up.movedim(1, 0)[prefix:]
        if source_down_components.shape[0] == 0:
            continue
        if layer_report is not None:
            layer_report.alignable_source_components += source_down_components.shape[0]
        source_down = source_down_components.reshape(source_down_components.shape[0], -1)
        source_up = source_up_components.reshape(source_up_components.shape[0], -1)
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
            if layer_report is not None:
                layer_report.matrix_shapes.append(tuple(contribution_similarities.shape))
                layer_report.record_candidates(contribution_similarities)
            matched = _greedy_similarity_matches(
                contribution_similarities, settings, diagnostics
            )

        for ref_row, source_row in enumerate(matched):
            if source_row < 0 or ref_row >= len(down_groups):
                continue
            if layer_report is not None:
                layer_report.matches += 1
                if down_similarities is not None:
                    matched_score_tensors.append(
                        down_similarities[ref_row, source_row]
                        * up_similarities[ref_row, source_row]
                    )
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

    if synthetic_zero_count:
        zero_down = torch.zeros_like(reference_down[0])
        zero_up = torch.zeros_like(reference_up_components[0])
        for down_group, up_group in zip(down_groups, up_groups):
            down_group.extend([zero_down] * synthetic_zero_count)
            up_group.extend([zero_up] * synthetic_zero_count)

    if diagnostics is not None and len(downs) > 1:
        diagnostics.anchor_only_groups += sum(
            len(group) == 1 for group in down_groups
        )
    if layer_report is not None:
        layer_report.anchor_only_components = sum(
            len(group) == 1 for group in down_groups
        )
    merged_down_rows = []
    merged_up_columns = []
    for down_group, up_group in zip(down_groups, up_groups):
        merged_down = merge_consensus_group(
            torch.stack([component.reshape(-1) for component in down_group]),
            settings,
            apply_global_scale=False,
            diagnostics=diagnostics,
        ).reshape(down_group[0].shape)
        merged_up = merge_consensus_group(
            torch.stack([component.reshape(-1) for component in up_group]),
            settings,
            diagnostics=diagnostics,
        ).reshape(up_group[0].shape)
        merged_down_rows.append(merged_down)
        merged_up_columns.append(merged_up)
        if layer_report is not None:
            input_norms = torch.stack([
                torch.linalg.vector_norm(down_component)
                * torch.linalg.vector_norm(up_component)
                for down_component, up_component in zip(down_group, up_group)
            ])
            mean_input_norm = input_norms.mean()
            output_norm = (
                torch.linalg.vector_norm(merged_down)
                * torch.linalg.vector_norm(merged_up)
            )
            norm_ratio_tensors.append(
                torch.where(
                    mean_input_norm > 0.0,
                    output_norm / mean_input_norm.clamp_min(1e-12),
                    torch.full_like(output_norm, torch.nan),
                )
            )
    if layer_report is not None:
        if matched_score_tensors:
            layer_report.matched_scores.extend(
                torch.stack(matched_score_tensors).detach().float().cpu().tolist()
            )
        if norm_ratio_tensors:
            ratios = torch.stack(norm_ratio_tensors).detach().float().cpu().tolist()
            layer_report.norm_ratios.extend(value for value in ratios if value == value)
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


def _row_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1), dim=1)[0]


def _find_legacy_boundary_pair(
    tensor: torch.Tensor,
    start_reference: torch.Tensor,
    end_reference: torch.Tensor,
    threshold: float,
) -> tuple[int, int, tuple[float, float]]:
    """Locate one ordered encoded visual-boundary pair in a legacy tensor."""
    if tensor.ndim != 2 or tensor.shape[0] < 2:
        raise ValueError("Legacy vision boundary search requires a 2D tensor with at least two rows.")
    rows = F.normalize(tensor.float(), p=2, dim=1, eps=1e-8)
    start_scores = torch.mv(rows, F.normalize(start_reference.float(), p=2, dim=0, eps=1e-8))
    end_scores = torch.mv(rows, F.normalize(end_reference.float(), p=2, dim=0, eps=1e-8))
    pair_scores = start_scores[:, None] + end_scores[None, :]
    ordered = torch.ones_like(pair_scores, dtype=torch.bool).triu_(diagonal=1)
    pair_scores.masked_fill_(~ordered, -float("inf"))
    flat_index = torch.argmax(pair_scores)
    start = int((flat_index // tensor.shape[0]).item())
    end = int((flat_index % tensor.shape[0]).item())
    start_score = float(start_scores[start].item())
    end_score = float(end_scores[end].item())
    if start >= end or min(start_score, end_score) < threshold:
        raise ValueError(
            "Legacy vision boundary search found no ordered start/end pair meeting "
            f"the boundary similarity threshold {threshold:.4f}."
        )
    return start, end, (start_score, end_score)


def _prepare_vision_embedding(
    tensor: torch.Tensor,
    *,
    start_reference: torch.Tensor,
    end_reference: torch.Tensor,
    legacy_search: bool,
    threshold: float,
) -> tuple[torch.Tensor, tuple[float, float], int]:
    if tensor.ndim != 2 or tensor.shape[0] < 2:
        raise ValueError("Vision boundary embeddings require a 2D tensor with start and end rows.")
    if tensor.shape[1] != start_reference.numel() or tensor.shape[1] != end_reference.numel():
        raise ValueError("Vision boundary embedding hidden sizes must match the selected boundary reference.")
    if legacy_search:
        start, end, scores = _find_legacy_boundary_pair(
            tensor, start_reference, end_reference, threshold
        )
    else:
        start, end = 0, tensor.shape[0] - 1
        scores = (
            float(_row_cosine(tensor[start], start_reference).item()),
            float(_row_cosine(tensor[end], end_reference).item()),
        )
        if min(scores) < threshold:
            raise ValueError(
                "Vision boundary validation failed: encoded start/end rows do not meet "
                f"the boundary similarity threshold {threshold:.4f}."
            )
    return tensor[start + 1:end], scores, start + (tensor.shape[0] - end - 1)


def coalesce_embedding_rows(
    tensor: torch.Tensor,
    settings: CWBSettings,
    *,
    target_vector_count: int,
    similarity_threshold: float,
    position_window: float,
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
    """Repeatedly replace a mutual-nearest local pair with one CWB vector."""
    if tensor.ndim != 2:
        return tensor, ()
    if target_vector_count < 0:
        raise ValueError("Target vector count must be zero or positive.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("Embedding coalescing similarity threshold must be between 0 and 1.")
    if not 0.0 <= position_window <= 1.0:
        raise ValueError("Embedding coalescing position window must be between 0 and 1.")

    rows = tensor
    merged_pairs = []
    while rows.shape[0] > max(target_vector_count, 1):
        count = rows.shape[0]
        if count < 2:
            break
        normalized = F.normalize(rows.float(), p=2, dim=1, eps=1e-8)
        scores = normalized @ normalized.t()
        scores.fill_diagonal_(-float("inf"))
        positions = torch.linspace(0.0, 1.0, count, device=rows.device)
        scores[(positions[:, None] - positions[None, :]).abs() > position_window] = -float("inf")
        nearest = scores.argmax(dim=1)
        candidate_scores = scores[torch.arange(count, device=rows.device), nearest]
        mutual = torch.arange(count, device=rows.device) == nearest[nearest]
        candidate_scores[~mutual] = -float("inf")
        candidate_scores[candidate_scores < similarity_threshold] = -float("inf")
        first = int(torch.argmax(candidate_scores).item())
        if not torch.isfinite(candidate_scores[first]):
            break
        second = int(nearest[first].item())
        first, second = sorted((first, second))
        merged = merge_consensus_group(torch.stack((rows[first], rows[second])), settings)
        insert_at = (first + second) // 2
        keep = torch.ones(count, dtype=torch.bool, device=rows.device)
        keep[first] = False
        keep[second] = False
        remaining = rows[keep]
        rows = torch.cat((remaining[:insert_at], merged.unsqueeze(0), remaining[insert_at:]), dim=0)
        merged_pairs.append((first, second))
    return rows, tuple(merged_pairs)


def _read_visual_boundary_reference(handler, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and release only one reference tensor work unit through UEL."""
    if key not in handler.keys():
        raise ValueError(f"Boundary reference is missing embedding tensor '{key}'.")
    stream = handler.async_stream([key], batch_size=1, prefetch_batches=1, pin_memory=False)
    try:
        batch = next(stream)
        loaded_key, tensor = batch[0]
        if loaded_key != key or tensor.ndim != 2 or tensor.shape[0] < 2:
            raise ValueError("Boundary reference must contain a 2D embedding tensor with start and end rows.")
        return tensor[0].detach().clone(), tensor[-1].detach().clone()
    finally:
        handler.mark_processed(key)
        close = getattr(stream, "close", None)
        if close is not None:
            close()


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
    diagnostics: CWBDiagnostics | None = None,
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
                diagnostics=diagnostics,
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
        if diagnostics is not None:
            diagnostics.cpu_fallbacks += 1
        return execute("cpu")


def _merge_lora_pair_to_cpu(
    pair_sources: list[LoRAPairSource],
    settings: CWBSettings,
    device: str,
    target_dtype: torch.dtype,
    operation_label: str = "LoRA pair",
    diagnostics: CWBDiagnostics | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Own one LoRA pair operation's GPU lifetime and return CPU A/B tensors."""
    genuine_sources = [source for source in pair_sources if not source.synthetic_zero]
    if not genuine_sources:
        raise ValueError("CWB LoRA merge requires at least one genuine factor pair.")
    max_rank = max(source.rank for source in genuine_sources)
    reference_index = max(
        range(len(genuine_sources)),
        key=lambda index: genuine_sources[index].rank,
    )
    reference_source = genuine_sources[reference_index]
    if reference_source.down is None or reference_source.up is None:
        raise ValueError("CWB LoRA reference source is missing its factors.")
    synthetic_zero_count = len(pair_sources) - len(genuine_sources)
    ranks = [source.rank for source in genuine_sources]
    structural_slots_excluded = sum(max_rank - rank for rank in ranks)
    if diagnostics is not None:
        diagnostics.lora_groups += 1
        diagnostics.mixed_rank_lora_groups += int(len(set(ranks)) > 1)
        diagnostics.genuine_lora_components += sum(ranks)
        diagnostics.structural_rank_slots_excluded += structural_slots_excluded
        diagnostics.explicit_zero_contributors += synthetic_zero_count

    def execute(target_device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        downs = []
        ups = []
        attempt_diagnostics = (
            CWBDiagnostics(
                weight_sweep=(
                    CWBWeightSweepDiagnostics()
                    if diagnostics.weight_sweep is not None else None
                )
            )
            if diagnostics is not None else None
        )
        layer_report = LoRALayerReport(
            layer=operation_label,
            ranks=tuple(ranks),
            reference_input=reference_source.source_index + 1,
            reference_rank=max_rank,
            down_dimensions=tuple(reference_source.down.shape[1:]),
            up_dimensions=(
                int(reference_source.up.shape[0]),
                *tuple(reference_source.up.shape[2:]),
            ),
            alignment_method=settings.alignment_method,
            structural_slots_excluded=structural_slots_excluded,
            explicit_zero_contributors=synthetic_zero_count,
        )
        try:
            for source in genuine_sources:
                if source.down is None or source.up is None:
                    raise ValueError("A genuine LoRA source is missing its factors.")
                downs.append(_to_compute(source.down, target_device))
                ups.append(_to_compute(source.up, target_device) * source.scale)
            merged_down, merged_up = merge_cwb_lora_pairs(
                downs,
                ups,
                settings,
                reference_index=reference_index,
                synthetic_zero_count=synthetic_zero_count,
                diagnostics=attempt_diagnostics,
                layer_report=layer_report,
            )
            output_down = merged_down.to(target_dtype).cpu().contiguous()
            output_up = merged_up.to(target_dtype).cpu().contiguous()
            if diagnostics is not None and attempt_diagnostics is not None:
                diagnostics.absorb_runtime(attempt_diagnostics)
                diagnostics.lora_layer_reports.append(layer_report)
            return output_down, output_up, max_rank
        finally:
            downs.clear()
            ups.clear()

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
        if diagnostics is not None:
            diagnostics.cpu_fallbacks += 1
        return execute("cpu")


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


def _secondary_lora_map(
    pairs: dict[str, dict[str, str]],
    *,
    input_index: int | None = None,
) -> dict[str, str]:
    result = {}
    for block_name in pairs:
        core = _normalized_lora_core(block_name)
        previous = result.get(core)
        if previous is not None and previous != block_name:
            label = f" in input {input_index + 1}" if input_index is not None else ""
            raise ValueError(
                f"[CWB LoRA Merge] LoRA keys '{previous}' and '{block_name}' "
                f"normalize to the same logical layer '{core}'{label}."
            )
        result[core] = block_name
    return result


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
    ) -> str | tuple[str, str]:
        diagnostics = CWBDiagnostics(
            weight_sweep=(
                CWBWeightSweepDiagnostics()
                if params.get("counterfactual_weight_sweep", False) else None
            )
        )
        params = {**params, "_cwb_diagnostics": diagnostics}
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
        boundary_handler = None
        boundary_name = params.get("boundary_reference_embedding")
        if boundary_name and boundary_name != "None":
            boundary_path = folder_paths.get_full_path("embeddings", boundary_name)
            if not boundary_path:
                raise FileNotFoundError(f"Boundary reference embedding '{boundary_name}' was not found.")
            boundary_handler = MemoryEfficientSafeOpen(
                boundary_path, low_memory=params["lazy_load"]
            )
        try:
            diffusion_quantizers = []
            diffusion_maps = []
            if model_type == "diffusion_models":
                diffusion_quantizers = [
                    DiffusionQuantization(handler, path, f"CWB input {index + 1}")
                    for index, (handler, path) in enumerate(zip(handlers, paths))
                ]
                diffusion_maps = [
                    diffusion_key_map(quantizer.data_keys, f"CWB input {index + 1}")
                    for index, quantizer in enumerate(diffusion_quantizers)
                ]
                low_bit_sets = [
                    {logical for logical, physical in mapping.items()
                     if physical in quantizer.isolated_low_bit_keys}
                    for mapping, quantizer in zip(diffusion_maps, diffusion_quantizers)
                ]
            else:
                low_bit_sets = [
                    inspect_low_bit_input(
                        handler,
                        f"Input {index + 1} ({path})",
                        "CWB Merge",
                    )
                    for index, (handler, path) in enumerate(zip(handlers, paths))
                ]
            settings = resolve_cwb_settings(
                params["cwb_preset"],
                _presets_for(
                    embedding_union=embedding_union,
                    lora_mode=lora_mode,
                ),
                params.get("cwb_config"),
            )
            if lora_mode:
                result = cls._merge_loras(
                    handlers, low_bit_sets, model_type, params, settings, model_names
                )
            else:
                result = cls._merge_generic(
                    handlers,
                    low_bit_sets,
                    model_type,
                    params,
                    settings,
                    embedding_union=embedding_union,
                    boundary_handler=boundary_handler,
                    source_models=model_names,
                    diffusion_quantizers=diffusion_quantizers,
                    diffusion_maps=diffusion_maps,
                )
            if params.get("_return_cwb_report"):
                report = diagnostics.render(
                    params["cwb_preset"],
                    len(model_names),
                    params.get("cwb_config") is not None,
                )
                embedding_report = params.get("_embedding_coalesce_report")
                if embedding_report is not None:
                    report += embedding_report.render()
                return result, report
            return result
        finally:
            for handler in handlers:
                handler.__exit__(None, None, None)
            if boundary_handler is not None:
                boundary_handler.__exit__(None, None, None)
            cleanup_after_operation()

    @staticmethod
    def _output_path(model_type: str, output_filename: str) -> str:
        return canonical_model_artifact_path(model_type, output_filename)[0]

    @staticmethod
    def _output_name(model_type: str, output_path: str) -> str:
        output_dir = os.path.abspath(os.path.join(folder_paths.models_dir, model_type))
        return os.path.relpath(output_path, output_dir).replace(os.sep, "/")

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
        boundary_handler=None,
        source_models,
        diffusion_quantizers,
        diffusion_maps,
    ):
        primary = handlers[0]
        keys = set()
        if diffusion_maps:
            for mapping in diffusion_maps:
                keys.update(mapping)
        else:
            for handler in handlers:
                keys.update(handler.keys())
        keys = sorted(keys)
        output_path = cls._output_path(model_type, params["output_filename"])
        requested_dtype = _requested_dtype(params["save_dtype"])
        mismatch_mode = params["mismatch_mode"]
        glob_mode = params["glob_patterns"]
        include_mode = params.get("include_mode", False)
        exclude = _compile_patterns(params["exclude_patterns"], glob_mode=glob_mode)
        discard = _compile_patterns(params["discard_patterns"], glob_mode=glob_mode)
        pbar = comfy.utils.ProgressBar(len(keys))
        secondary_only_copied = 0
        secondary_only_merged = 0

        handler_map = dict(enumerate(handlers))
        def source_indices_for(key):
            return [
                i for i, handler in enumerate(handlers)
                if key in (diffusion_maps[i] if diffusion_maps else handler.keys())
            ]

        def physical_key(index, key):
            return diffusion_maps[index][key] if diffusion_maps else key

        def source_dtype(index, key):
            source_key = physical_key(index, key)
            return (
                diffusion_quantizers[index].logical_dtype(source_key, requested_dtype)
                if diffusion_maps else handlers[index].get_dtype(source_key)
            )

        def matches(key, patterns):
            aliases = [key]
            if diffusion_maps:
                aliases.extend(mapping[key] for mapping in diffusion_maps if key in mapping)
            return any(
                _matches_any_pattern(alias, patterns, glob_mode=glob_mode)
                for alias in aliases
            )

        work_units = []
        for key in keys:
            source_indices = source_indices_for(key)
            discarded = matches(key, discard)
            guarded = any(key in low_bit_sets[i] for i in source_indices)
            if discarded:
                entries = {}
            elif guarded:
                preserve_index = 0 if 0 in source_indices else source_indices[0]
                source_key = physical_key(preserve_index, key)
                entries = (
                    {preserve_index: diffusion_quantizers[preserve_index].required_keys(source_key)}
                    if diffusion_maps and source_key in diffusion_quantizers[preserve_index].quantized_keys
                    else {}
                )
            else:
                entries = {
                    i: (diffusion_quantizers[i].required_keys(physical_key(i, key))
                        if diffusion_maps else [key]) for i in source_indices
                }
            work_units.append((key, entries))

        output_metadata = (
            diffusion_quantizers[0].output_metadata()
            if diffusion_maps else (primary.metadata() or {}).copy()
        )
        output_metadata["cwb.merge"] = _cwb_merge_metadata(
            source_models,
            params,
            settings,
            operation="cwb_embedding_merge" if embedding_union else "cwb_tensor_merge",
        )
        def preserve_tensor(writer, index, key, loaded, *, force_raw=False):
            source_key = physical_key(index, key)
            if diffusion_maps and source_key in diffusion_quantizers[index].quantized_keys:
                writer.write(key, loaded[(index, key)].to(requested_dtype).cpu().contiguous())
            else:
                write_preserved_tensor(
                    writer, source_key, handlers[index], output_key=key,
                    force_raw=force_raw, tensor=loaded.get((index, key)),
                )

        with atomic_uel_writer(output_path, output_metadata) as writer, torch.no_grad(), closing(
            stream_work_units(
                handler_map, work_units,
                pin_memory=str(params["process_device"]).startswith("cuda"),
            )
        ) as streamed:
            for key, loaded in tqdm(streamed, total=len(work_units), desc="CWB merging tensors", unit="tensors"):
                    _clear_previous_layer(params)
                    if matches(key, discard):
                        pbar.update(1)
                        continue
                    source_indices = source_indices_for(key)
                    primary_owned = key in (diffusion_maps[0] if diffusion_maps else primary.keys())
                    secondary_only = not primary_owned
                    preserve_index = 0 if primary_owned else source_indices[0]
                    guarded = any(key in low_bit_sets[i] for i in source_indices)
                    if diffusion_maps:
                        for index in source_indices:
                            source_key = physical_key(index, key)
                            if (index, source_key) not in loaded:
                                continue
                            if source_key in diffusion_quantizers[index].quantized_keys:
                                parts = {
                                    part: loaded[(index, part)]
                                    for part in diffusion_quantizers[index].required_keys(source_key)
                                }
                                loaded[(index, key)] = diffusion_quantizers[index].decode(
                                    source_key, parts, torch.float32, device="cpu",
                                )
                            else:
                                loaded[(index, key)] = loaded[(index, source_key)]
                    matched = matches(key, exclude)
                    excluded = not matched if include_mode else matched
                    if guarded or excluded:
                        preserve_tensor(writer, preserve_index, key, loaded, force_raw=guarded)
                        if secondary_only:
                            secondary_only_copied += 1
                        pbar.update(1)
                        continue

                    if secondary_only and len(source_indices) == 1:
                        preserve_tensor(writer, preserve_index, key, loaded)
                        secondary_only_copied += 1
                        pbar.update(1)
                        continue

                    source_dtypes = [source_dtype(i, key) for i in source_indices]
                    if not all(_is_float_dtype(dtype) for dtype in source_dtypes):
                        logging.warning(
                            "[CWB Merge] Preserving non-floating tensor '%s' from input %d.",
                            key,
                            preserve_index + 1,
                        )
                        preserve_tensor(writer, preserve_index, key, loaded)
                        if secondary_only:
                            secondary_only_copied += 1
                        pbar.update(1)
                        continue

                    if primary_owned and not embedding_union and len(source_indices) != len(handlers):
                        if mismatch_mode == "error":
                            raise ValueError(f"Tensor '{key}' is missing from a CWB input.")
                        if mismatch_mode == "skip":
                            preserve_tensor(writer, 0, key, loaded)
                            pbar.update(1)
                            continue

                    raw = {i: loaded[(i, key)] for i in source_indices}
                    embedding_coalesce = bool(params.get("embedding_coalesce", False))
                    vision_boundaries = bool(params.get("vision_boundary_embeddings", False))
                    boundary_scores = []
                    trimmed_rows = 0
                    boundary_start = boundary_end = None
                    if embedding_coalesce and vision_boundaries:
                        if not all(tensor.ndim == 2 for tensor in raw.values()):
                            raise ValueError(
                                f"Vision boundary mode requires 2D embedding tensor '{key}'."
                            )
                        if params.get("legacy_boundary_search", False):
                            if boundary_handler is None:
                                raise ValueError(
                                    "Legacy vision boundary search requires a boundary reference embedding."
                                )
                            boundary_start, boundary_end = _read_visual_boundary_reference(
                                boundary_handler, key
                            )
                        else:
                            anchor = raw[preserve_index]
                            boundary_start = anchor[0].detach().clone()
                            boundary_end = anchor[-1].detach().clone()
                        prepared_raw = {}
                        for source_index, tensor in raw.items():
                            body, scores, trimmed = _prepare_vision_embedding(
                                tensor,
                                start_reference=boundary_start,
                                end_reference=boundary_end,
                                legacy_search=params.get("legacy_boundary_search", False),
                                threshold=float(params["boundary_similarity_threshold"]),
                            )
                            prepared_raw[source_index] = body
                            boundary_scores.extend(scores)
                            trimmed_rows += trimmed
                        raw = prepared_raw
                    reference_index = 0
                    if embedding_union:
                        reference_source = max(
                            source_indices,
                            key=lambda i: raw[i].shape[0] if raw[i].ndim else 1,
                        )
                        reference_shape = raw[reference_source].shape
                    else:
                        reference_source = preserve_index
                        reference_shape = raw[preserve_index].shape

                    tensors = []
                    actual_dtypes = []
                    reference_index = 0
                    preserve_for_mismatch = False
                    candidate_indices = range(len(handlers)) if primary_owned else source_indices
                    for source_index in candidate_indices:
                        handler = handlers[source_index]
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
                            if mismatch_mode == "skip":
                                preserve_for_mismatch = True
                                break
                            if mismatch_mode == "zeros":
                                tensors.append(torch.zeros(
                                    reference_shape,
                                    dtype=torch.float32,
                                ))
                            continue
                        if source_index == reference_source:
                            reference_index = len(tensors)
                        tensors.append(aligned)
                        actual_dtypes.append(source_dtype(source_index, key))

                    if preserve_for_mismatch:
                        preserve_tensor(writer, preserve_index, key, loaded)
                        if secondary_only:
                            secondary_only_copied += 1
                        pbar.update(1)
                        continue
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
                        diagnostics=params.get("_cwb_diagnostics"),
                    )
                    if embedding_coalesce and merged.ndim == 2:
                        before_rows = merged.shape[0]
                        merged, pairs = coalesce_embedding_rows(
                            merged,
                            settings,
                            target_vector_count=int(params["target_vector_count"]),
                            similarity_threshold=float(params["coalesce_similarity_threshold"]),
                            position_window=float(params["coalesce_position_window"]),
                        )
                        if vision_boundaries:
                            merged = torch.cat((
                                boundary_start.to(dtype=merged.dtype, device=merged.device).unsqueeze(0),
                                merged,
                                boundary_end.to(dtype=merged.dtype, device=merged.device).unsqueeze(0),
                            ), dim=0)
                        report = params.get("_embedding_coalesce_report")
                        if report is not None:
                            report.record(
                                key,
                                before_rows + (2 if vision_boundaries else 0),
                                merged.shape[0],
                                trimmed_rows,
                                boundary_scores,
                                pairs,
                            )
                    writer.write_batch([(key, merged)])
                    del merged
                    if secondary_only:
                        secondary_only_merged += 1
                    pbar.update(1)
        if secondary_only_copied or secondary_only_merged:
            logging.info(
                "[CWB Merge] Secondary-only tensors: %d copied, %d merged.",
                secondary_only_copied,
                secondary_only_merged,
            )
        return cls._output_name(model_type, output_path)

    @classmethod
    def _merge_loras(
        cls, handlers, low_bit_sets, model_type, params, settings, source_loras
    ):
        parsed = [parse_lora_layers(handler.keys()) for handler in handlers]
        logical_maps = []
        logical_cores = []
        seen_cores = set()
        passthrough_keys = set()
        for input_index, (pairs, passthrough) in enumerate(parsed):
            validate_canonical_blocks(pairs, f"CWB LoRA Merge input {input_index + 1}")
            logical_map = _secondary_lora_map(pairs, input_index=input_index)
            logical_maps.append(logical_map)
            passthrough_keys.update(passthrough)
            for core in logical_map:
                if core not in seen_cores:
                    logical_cores.append(core)
                    seen_cores.add(core)
        output_path = cls._output_path(model_type, params["output_filename"])
        requested_dtype = _requested_dtype(params["save_dtype"])
        mismatch_mode = params["mismatch_mode"]
        include_1d = params.get("include_1d_diffs", False)
        glob_mode = params["glob_patterns"]
        include_mode = params.get("include_mode", False)
        exclude = _compile_patterns(params["exclude_patterns"], glob_mode=glob_mode)
        discard = _compile_patterns(params["discard_patterns"], glob_mode=glob_mode)
        written = set()
        preserved_companion_groups = 0
        secondary_only_copied = 0
        secondary_only_merged = 0
        pbar = comfy.utils.ProgressBar(len(logical_cores) + len(passthrough_keys))
        current_loaded = {}

        def preserve_keys(writer, keys: Iterable[str], source_index: int = 0):
            for key in keys:
                if key not in written:
                    guarded = key in low_bit_sets[source_index]
                    write_preserved_tensor(
                        writer,
                        key,
                        handlers[source_index],
                        force_raw=guarded,
                        tensor=current_loaded.get((source_index, key)),
                    )
                    written.add(key)

        def preserve_roles(
            writer,
            block_name: str,
            keys: dict[str, str],
            roles,
            source_index: int = 0,
        ):
            recognized_sources = set()
            for role, source_key in layer_tensor_keys(keys).items():
                if role not in roles:
                    continue
                output_key = canonical_lora_key(block_name, role)
                recognized_sources.add(source_key)
                if role == "alpha":
                    written.add(output_key)
                    continue
                if output_key not in written:
                    tensor = current_loaded.get((source_index, source_key))
                    if role == "up" and "alpha" in keys:
                        alpha_key = keys["alpha"]
                        alpha = current_loaded.get((source_index, alpha_key))
                        down = current_loaded.get((source_index, keys.get("down")))
                        if tensor is None or alpha is None or down is None:
                            raise ValueError(
                                f"Cannot normalize alpha for preserved LoRA layer '{block_name}'."
                            )
                        _, tensor = normalize_lora_pair(
                            down, tensor, alpha, layer=block_name
                        )
                        current_loaded[(source_index, source_key)] = tensor
                    write_preserved_tensor(
                        writer,
                        source_key,
                        handlers[source_index],
                        output_key,
                        force_raw=source_key in low_bit_sets[source_index],
                        tensor=tensor,
                    )
                    written.add(output_key)
            return recognized_sources

        def preserve_layer(
            writer,
            block_name: str,
            keys: dict[str, str],
            source_index: int = 0,
        ):
            recognized_sources = preserve_roles(
                writer,
                block_name,
                keys,
                layer_tensor_keys(keys),
                source_index,
            )
            preserve_keys(
                writer,
                (
                    key for key in handlers[source_index].keys()
                    if key.startswith(f"{block_name}.") and key not in recognized_sources
                ),
                source_index,
            )

        def core_matches(core):
            result = []
            for index, ((pairs, _), logical_map) in enumerate(zip(parsed, logical_maps)):
                block = logical_map.get(core)
                result.append((index, block, pairs.get(block) if block else None))
            return result

        def has_non_alpha_low_bit(keys, low_bit_keys):
            return any(
                key in low_bit_keys
                for role, key in layer_tensor_keys(keys).items()
                if role != "alpha"
            )

        def reject_unnormalizable_alpha(matches):
            for index, block, keys in matches:
                if (
                    keys is not None
                    and "alpha" in keys
                    and has_non_alpha_low_bit(keys, low_bit_sets[index])
                ):
                    raise ValueError(
                        f"Cannot alpha-normalize low-bit LoRA factors for '{block}'."
                    )

        work_units = []
        for core in logical_cores:
            matches = core_matches(core)
            reject_unnormalizable_alpha(matches)
            layer_keys = {
                index: [
                    key for key in handlers[index].keys()
                    if key in keys.values() or key.startswith(f"{block}.")
                ]
                for index, block, keys in matches
                if block is not None and keys is not None
            }
            flat_keys = [key for keys in layer_keys.values() for key in keys]
            discarded = any(
                _matches_any_pattern(key, discard, glob_mode=glob_mode)
                for key in flat_keys
            )
            guarded = any(
                keys is not None and has_non_alpha_low_bit(keys, low_bit_sets[index])
                for index, _, keys in matches
            )
            work_units.append((("core", core), {} if discarded or guarded else layer_keys))
        for key in sorted(passthrough_keys):
            source_index = next(
                index for index, (_, passthrough) in enumerate(parsed) if key in passthrough
            )
            entries = {} if key in low_bit_sets[source_index] else {source_index: [key]}
            work_units.append((("passthrough", key), entries))

        output_metadata = (handlers[0].metadata() or {}).copy()
        output_metadata["cwb.merge"] = _cwb_merge_metadata(
            source_loras, params, settings, operation="cwb_lora_merge"
        )
        output_metadata["alpha_normalized"] = "true"
        output_metadata["alpha_normalization"] = (
            "lora_up := lora_up * (alpha / rank); alpha tensors removed"
        )
        with atomic_uel_writer(output_path, output_metadata) as writer, torch.no_grad(), closing(
            stream_work_units(
                dict(enumerate(handlers)), work_units,
                pin_memory=str(params["process_device"]).startswith("cuda"),
            )
        ) as streamed:
            progress = tqdm(
                streamed,
                total=len(work_units),
                desc="CWB merging LoRA layers",
                unit="layers",
            )
            for (unit_kind, unit_value), current_loaded in progress:
                if unit_kind == "passthrough":
                    key = unit_value
                    _clear_previous_layer(params)
                    if key in written:
                        continue
                    if _matches_any_pattern(key, discard, glob_mode=glob_mode):
                        pbar.update(1)
                        continue
                    source_index = next(
                        index for index, (_, passthrough) in enumerate(parsed)
                        if key in passthrough
                    )
                    preserve_keys(writer, [key], source_index)
                    if source_index > 0:
                        secondary_only_copied += 1
                    pbar.update(1)
                    continue

                core = unit_value
                if unit_kind == "core":
                    _clear_previous_layer(params)
                    pair_sources = []
                    compatible = []
                    direct = []
                    down = up = alpha = tensor = anchor_tensor = None
                    matches = core_matches(core)
                    anchor_index, anchor_block, anchor_keys = next(
                        (index, block, keys)
                        for index, block, keys in matches
                        if block is not None and keys is not None
                    )
                    primary_owned = matches[0][2] is not None
                    secondary_only = not primary_owned

                    layer_keys = [
                        key
                        for index, block, keys in matches
                        if block is not None and keys is not None
                        for key in handlers[index].keys()
                        if key in keys.values() or key.startswith(f"{block}.")
                    ]
                    if any(_matches_any_pattern(key, discard, glob_mode=glob_mode) for key in layer_keys):
                        pbar.update(1)
                        continue
                    guarded = any(
                        keys is not None and has_non_alpha_low_bit(keys, low_bit_sets[index])
                        for index, _, keys in matches
                    )
                    matched = any(
                        _matches_any_pattern(key, exclude, glob_mode=glob_mode)
                        for key in layer_keys
                    )
                    excluded = not matched if include_mode else matched
                    companion_bearing = any(
                        keys is not None and layer_has_companions(keys)
                        for _, _, keys in matches
                    )
                    if companion_bearing:
                        preserve_layer(writer, anchor_block, anchor_keys, anchor_index)
                        preserved_companion_groups += 1
                        if secondary_only:
                            secondary_only_copied += 1
                        pbar.update(1)
                        continue
                    if guarded or excluded:
                        preserve_layer(writer, anchor_block, anchor_keys, anchor_index)
                        if secondary_only:
                            secondary_only_copied += 1
                        pbar.update(1)
                        continue

                    available_matches = [match for match in matches if match[2] is not None]
                    if secondary_only and len(available_matches) == 1:
                        preserve_layer(writer, anchor_block, anchor_keys, anchor_index)
                        secondary_only_copied += 1
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

                    group_merged = False
                    if "down" in anchor_keys and "up" in anchor_keys:
                        pair_sources = []
                        pair_failed = False
                        for index, _, keys in matches:
                            if not keys:
                                if secondary_only:
                                    continue
                                if mismatch_mode == "error":
                                    raise ValueError(f"LoRA pair '{anchor_block}' is missing from input {index + 1}.")
                                if mismatch_mode == "skip":
                                    pair_failed = True
                                    break
                                pair_sources.append(LoRAPairSource.zero(index))
                                continue
                            if "down" not in keys or "up" not in keys:
                                if mismatch_mode == "error":
                                    raise ValueError(f"LoRA pair '{anchor_block}' is incomplete in input {index + 1}.")
                                if mismatch_mode == "skip":
                                    pair_failed = True
                                    break
                                pair_sources.append(LoRAPairSource.zero(index))
                                continue
                            down = current_loaded[(index, keys["down"])]
                            up = current_loaded[(index, keys["up"])]
                            if down.ndim < 2 or up.ndim < 2 or down.shape[0] != up.shape[1]:
                                if mismatch_mode == "error" or index == anchor_index:
                                    raise ValueError(f"Invalid LoRA rank dimensions for '{anchor_block}'.")
                                if mismatch_mode == "skip":
                                    pair_failed = True
                                    break
                                pair_sources.append(LoRAPairSource.zero(index))
                                continue
                            alpha = (
                                current_loaded[(index, keys["alpha"])]
                                if "alpha" in keys else None
                            )
                            down, up = normalize_lora_pair(
                                down, up, alpha, layer=anchor_block
                            )
                            current_loaded[(index, keys["down"])] = down
                            current_loaded[(index, keys["up"])] = up
                            pair_sources.append(
                                LoRAPairSource(index, down, up)
                            )
                        if pair_failed:
                            preserve_roles(
                                writer,
                                anchor_block,
                                anchor_keys,
                                {"down", "up", "alpha"},
                                anchor_index,
                            )
                        elif pair_sources:
                            anchor_source = next(
                                source for source in pair_sources
                                if not source.synthetic_zero
                            )
                            anchor_down = anchor_source.down
                            anchor_up = anchor_source.up
                            if anchor_down is None or anchor_up is None:
                                raise ValueError(
                                    f"LoRA pair '{anchor_block}' has no genuine anchor."
                                )
                            compatible = []
                            for source in pair_sources:
                                if source.synthetic_zero:
                                    compatible.append(source)
                                    continue
                                index = source.source_index
                                down = source.down
                                up = source.up
                                if down is None or up is None:
                                    raise ValueError(
                                        f"LoRA pair '{anchor_block}' has missing factors."
                                    )
                                valid = (
                                    tuple(down.shape[1:]) == tuple(anchor_down.shape[1:])
                                    and up.shape[0] == anchor_up.shape[0]
                                    and tuple(up.shape[2:]) == tuple(anchor_up.shape[2:])
                                )
                                if not valid:
                                    if mismatch_mode == "error":
                                        raise ValueError(f"LoRA pair shape mismatch for '{anchor_block}'.")
                                    if mismatch_mode == "skip":
                                        compatible = []
                                        break
                                    compatible.append(LoRAPairSource.zero(index))
                                    continue
                                compatible.append(source)
                            if not compatible:
                                preserve_roles(
                                    writer,
                                    anchor_block,
                                    anchor_keys,
                                    {"down", "up", "alpha"},
                                    anchor_index,
                                )
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
                                    operation_label=anchor_block,
                                    diagnostics=params.get("_cwb_diagnostics"),
                                )
                                writer.write(
                                    canonical_lora_key(anchor_block, "down"),
                                    merged_down,
                                )
                                writer.write(
                                    canonical_lora_key(anchor_block, "up"),
                                    merged_up,
                                )
                                del merged_down, merged_up
                                written.update({
                                    canonical_lora_key(anchor_block, "down"),
                                    canonical_lora_key(anchor_block, "up"),
                                })
                                if "alpha" in anchor_keys:
                                    written.add(canonical_lora_key(anchor_block, "alpha"))
                                group_merged = True

                    for kind in ("diff", "diff_b", "w_norm", "b_norm"):
                        if kind not in anchor_keys:
                            continue
                        anchor_key = anchor_keys[kind]
                        output_key = canonical_lora_key(anchor_block, kind)
                        anchor_tensor = current_loaded[(anchor_index, anchor_key)]
                        if anchor_tensor.ndim == 1 and not include_1d:
                            if output_key not in written:
                                write_preserved_tensor(
                                    writer,
                                    anchor_key,
                                    handlers[anchor_index],
                                    output_key,
                                )
                                written.add(output_key)
                            continue
                        direct = []
                        direct_dtypes = []
                        failed = False
                        for index, _, keys in matches:
                            if not keys:
                                if secondary_only:
                                    continue
                                if mismatch_mode == "error":
                                    raise ValueError(f"Direct LoRA layer '{anchor_block}.{kind}' is missing from input {index + 1}.")
                                if mismatch_mode == "skip":
                                    failed = True
                                    break
                                direct.append(torch.zeros_like(anchor_tensor, dtype=torch.float32))
                                continue
                            if kind not in keys:
                                if secondary_only:
                                    continue
                                if mismatch_mode == "error":
                                    raise ValueError(f"Direct LoRA layer '{anchor_block}.{kind}' is missing from input {index + 1}.")
                                if mismatch_mode == "skip":
                                    failed = True
                                    break
                                direct.append(torch.zeros_like(anchor_tensor, dtype=torch.float32))
                                continue
                            tensor = current_loaded[(index, keys[kind])]
                            if tensor.shape != anchor_tensor.shape:
                                if mismatch_mode == "error":
                                    raise ValueError(f"Direct LoRA shape mismatch for '{anchor_block}.{kind}'.")
                                if mismatch_mode == "skip":
                                    failed = True
                                    break
                                direct.append(torch.zeros_like(anchor_tensor, dtype=torch.float32))
                                continue
                            direct.append(tensor)
                            direct_dtypes.append(handlers[index].get_dtype(keys[kind]))
                        if failed:
                            if output_key not in written:
                                write_preserved_tensor(
                                    writer,
                                    anchor_key,
                                    handlers[anchor_index],
                                    output_key,
                                )
                                written.add(output_key)
                            continue
                        if secondary_only and len(direct) == 1:
                            if output_key not in written:
                                write_preserved_tensor(
                                    writer,
                                    anchor_key,
                                    handlers[anchor_index],
                                    output_key,
                                )
                                written.add(output_key)
                            continue
                        target_dtype = select_output_dtype(
                            direct_dtypes,
                            requested_dtype,
                            force=params["override_dtype"],
                            is_1d_diff=anchor_tensor.ndim == 1,
                        )
                        merged = _merge_tensors_to_cpu(
                            direct,
                            settings,
                            params["process_device"],
                            target_dtype,
                            allow_similarity_alignment=False,
                            operation_label=output_key,
                            diagnostics=params.get("_cwb_diagnostics"),
                        )
                        writer.write(output_key, merged)
                        del merged
                        written.add(output_key)
                        group_merged = True

                    preserve_layer(writer, anchor_block, anchor_keys, anchor_index)
                    if secondary_only:
                        if group_merged:
                            secondary_only_merged += 1
                        else:
                            secondary_only_copied += 1
                    pair_sources.clear()
                    compatible.clear()
                    direct.clear()
                    current_loaded.clear()
                    down = up = alpha = tensor = anchor_tensor = None
                    pbar.update(1)

        if preserved_companion_groups:
            logging.warning(
                "[CWB LoRA Merge] Preserved %d companion-bearing group(s)",
                preserved_companion_groups,
            )
        if secondary_only_copied or secondary_only_merged:
            logging.info(
                "[CWB LoRA Merge] Secondary-only groups: %d copied, %d merged.",
                secondary_only_copied,
                secondary_only_merged,
            )
        return cls._output_name(model_type, output_path)


def _common_inputs(
    model_type: str,
    input_count: int,
    default_filename: str,
    presets: dict[str, dict],
    default_preset: str,
):
    label = model_type
    inputs = [
        io.Combo.Input(
            "execution_mode",
            options=["MERGE", "DOCUMENTATION ONLY"],
            tooltip="MERGE writes a new safetensors file. DOCUMENTATION ONLY returns the CWB reference without loading or merging inputs.",
        ),
        io.Combo.Input(
            "model_a",
            options=folder_paths.get_filename_list(label),
            tooltip="Primary contributor and preservation anchor. Supplies output metadata and anchors shared tensor names and shapes.",
        ),
        io.Combo.Input(
            "model_b",
            options=folder_paths.get_filename_list(label),
            tooltip="Second equal-prior contributor. CWB derives its effective per-vector influence from consensus similarity.",
        ),
    ]
    if input_count == 3:
        inputs.append(io.Combo.Input(
            "model_c",
            options=folder_paths.get_filename_list(label),
            tooltip="Third equal-prior contributor. CWB derives its effective per-vector influence from consensus similarity.",
        ))
    inputs.extend([
        io.Combo.Input(
            "cwb_preset",
            options=list(presets),
            default=default_preset,
            tooltip="Use-case preset. Name suffixes expose alignment, consensus, norm rescaling, DSC, soft comfort bandpass, and prefix preservation. A connected CWB Config overrides it completely.",
        ),
        CWB_CONFIG.Input(
            "cwb_config",
            optional=True,
            tooltip="Optional settings from CWB Custom Configuration. When connected, it completely overrides the selected preset.",
        ),
        io.Combo.Input(
            "mismatch_mode",
            options=["skip", "zeros", "error"],
            default="skip",
            tooltip="For missing or incompatible anchored inputs: skip preserves the anchor, zeros inserts a zero contribution where possible, and error aborts. A lone secondary-only tensor is copied unchanged.",
        ),
        io.String.Input(
            "output_filename",
            default=default_filename,
            tooltip="Filename without extension. The result is atomically written to this model category under ComfyUI's models directory.",
        ),
        io.Combo.Input(
            "save_dtype",
            options=["fp32", "fp16", "bf16"],
            tooltip="Requested dtype for generated floating tensors. Participating FP32 inputs keep a result FP32 unless Override Dtype is enabled.",
        ),
        io.Combo.Input(
            "process_device",
            options=["cuda", "cpu"],
            tooltip="Device used for per-layer FP32 CWB arithmetic. A CUDA out-of-memory error retries only the affected layer on CPU.",
        ),
        io.String.Input(
            "exclude_patterns",
            default="",
            multiline=True,
            tooltip="One pattern per line. Matching layers are preserved from the anchor instead of merged. Uses regex unless Glob Patterns is enabled.",
        ),
        io.String.Input(
            "discard_patterns",
            default="",
            multiline=True,
            tooltip="One pattern per line. Matching tensors or logical LoRA groups are omitted from the output. Uses regex unless Glob Patterns is enabled.",
        ),
        io.Boolean.Input(
            "glob_patterns",
            default=False,
            tooltip="Interpret exclude and discard entries as shell-style glob patterns instead of regular expressions.",
        ),
        io.Boolean.Input(
            "lazy_load",
            default=True,
            tooltip="Use UEL low-memory loading so tensors are read and released per work unit instead of retaining the complete inputs in RAM.",
        ),
        io.Boolean.Input(
            "force_clear_cache",
            default=True,
            tooltip="Run Python garbage collection and clear the CUDA allocator cache before each layer. Reduces retained memory but can substantially slow merging.",
        ),
        io.Boolean.Input(
            "override_dtype",
            default=False,
            tooltip="Force generated tensors to save_dtype; guarded tensors and enabled 1D direct diffs are exempt.",
        ),
        io.Boolean.Input(
            "include_mode",
            default=False,
            tooltip="Use Exclude Patterns as a whitelist instead. Only matching layers are merged; nonmatching layers are preserved from the anchor.",
        ),
    ])
    return inputs


class CWBCustomConfiguration(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CWBCustomConfiguration",
            display_name="CWB Custom Configuration",
            category="ModelUtils/Merging/Configuration",
            description="Build complete custom Consensus-Weighted Blending settings for any CWB merge node.",
            inputs=[
                io.Combo.Input("consensus_type", options=["mean", "median"], default="median", tooltip="Mean is a symmetric center; median is more robust with three or more contributors."),
                io.Combo.Input("alignment_method", options=["index", "similarity"], default="similarity", tooltip="Pair vectors by position or greedy cosine similarity where the merge type supports alignment."),
                io.Float.Input("alignment_threshold", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Minimum cosine product or score for accepting a similarity-aligned row match."),
                io.Float.Input("similarity_threshold", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Minimum contributor similarity to the group consensus before weighting."),
                io.Float.Input("power_alpha", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="Exponent applied to accepted non-negative consensus similarities."),
                io.Float.Input("diversity_beta", default=10.0, min=0.0, max=10.0, step=0.1, tooltip="Bandpass exponent used to suppress near-consensus dominance; zero disables it."),
                io.Boolean.Input("rescale_norm", default=True, tooltip="Set each merged vector norm to the mean participating norm after weighting."),
                io.Float.Input("global_scale", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Multiply the final contribution once; LoRA applies this through the up factor."),
                io.Boolean.Input("dynamic_similarity_contrast", default=False, tooltip="Remap unequal consensus similarities into 0.7 to 1.0 before alpha and beta weighting."),
                io.Boolean.Input("soft_comfort_bandpass", default=True, tooltip="Use 1.5 minus similarity instead of 1.001 minus similarity for diversity weighting."),
                io.Float.Input("position_weight", default=0.05, min=0.0, max=1.0, step=0.01, tooltip="Blend positional affinity into similarity-based greedy matching."),
                io.Boolean.Input("preserve_common_prefix", default=False, tooltip="Copy a numerically identical leading component span from the anchor."),
            ],
            outputs=[CWB_CONFIG.Output(display_name="cwb_config")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(build_custom_cwb_settings(**kwargs))


class _CWBMergerNode(io.ComfyNode):
    MODEL_TYPE = "diffusion_models"
    INPUT_COUNT = 2
    NODE_ID = ""
    DISPLAY_NAME = ""
    DEFAULT_FILENAME = "cwb_merged"
    EMBEDDING_UNION = False
    LORA_MODE = False
    DEFAULT_PRESET = "balanced_mean"

    @classmethod
    def define_schema(cls):
        presets = _presets_for(
            embedding_union=cls.EMBEDDING_UNION,
            lora_mode=cls.LORA_MODE,
        )
        inputs = _common_inputs(
            cls.MODEL_TYPE,
            cls.INPUT_COUNT,
            cls.DEFAULT_FILENAME,
            presets,
            cls.DEFAULT_PRESET,
        )
        if cls.LORA_MODE:
            include_mode = inputs.pop()
            inputs.append(io.Boolean.Input(
                "include_1d_diffs",
                default=False,
                tooltip="CWB-merge 1D direct diffs as FP32. Disabled preserves Model A.",
            ))
            inputs.append(include_mode)
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="ModelUtils/Merging",
            description="Merge two or three safetensors files with Consensus-Weighted Blending. Use each input tooltip or Documentation Only for exact control scope.",
            inputs=inputs,
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
                io.String.Output(display_name="cwb_report"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        documentation = load_documentation_from_file("consensus_mergers.md")
        if kwargs["execution_mode"] == "DOCUMENTATION ONLY":
            return io.NodeOutput(
                "Documentation mode active. No merge performed.",
                documentation,
                "CWB report unavailable because no merge was performed.",
            )
        names = [kwargs["model_a"], kwargs["model_b"]]
        if cls.INPUT_COUNT == 3:
            names.append(kwargs["model_c"])
        filename, report = ConsensusMergerLogic.execute(
            names,
            cls.MODEL_TYPE,
            {**kwargs, "_return_cwb_report": True},
            embedding_union=cls.EMBEDDING_UNION,
            lora_mode=cls.LORA_MODE,
        )
        return io.NodeOutput(filename, documentation, report)


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
    DEFAULT_PRESET = "broad_sim_medn_rn_softcb"


class CWBLoRAThreeMerger(CWBLoRATwoMerger):
    INPUT_COUNT = 3
    NODE_ID = "CWBLoRAThreeMerger"
    DISPLAY_NAME = "CWB Merge LoRAs (3 Models)"
    DEFAULT_FILENAME = "cwb_merged_3_lora"


class CWBLoRAMultiMerger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        lora_options = folder_paths.get_filename_list("loras")
        optional_loras = ["None", *lora_options]
        return io.Schema(
            node_id="CWBLoRAMultiMerger",
            display_name="CWB LoRA Multi-Merge",
            category="ModelUtils/LoRA/Merge",
            description="Consensus-merge 2 to 8 equal-prior LoRAs with bounded UEL streaming.",
            inputs=[
                io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], default="MERGE", tooltip="MERGE writes an output; DOCUMENTATION ONLY opens no model files."),
                io.Combo.Input("lora_count", options=[str(i) for i in range(2, 9)], default="2", tooltip="Number of consecutive LoRA inputs to include."),
                io.Combo.Input("lora_1", options=lora_options, tooltip="Anchor LoRA and metadata source."),
                io.Combo.Input("lora_2", options=lora_options, tooltip="Second equal-prior LoRA contributor."),
                io.Combo.Input("lora_3", options=optional_loras, default="None", tooltip="Optional third equal-prior contributor."),
                io.Combo.Input("lora_4", options=optional_loras, default="None", tooltip="Optional fourth equal-prior contributor."),
                io.Combo.Input("lora_5", options=optional_loras, default="None", tooltip="Optional fifth equal-prior contributor."),
                io.Combo.Input("lora_6", options=optional_loras, default="None", tooltip="Optional sixth equal-prior contributor."),
                io.Combo.Input("lora_7", options=optional_loras, default="None", tooltip="Optional seventh equal-prior contributor."),
                io.Combo.Input("lora_8", options=optional_loras, default="None", tooltip="Optional eighth equal-prior contributor."),
                io.Combo.Input("cwb_preset", options=list(LORA_CWB_PRESETS), default="broad_sim_medn_rn_softcb", tooltip="LoRA-specific preset. A connected CWB Config overrides it completely."),
                CWB_CONFIG.Input("cwb_config", optional=True, tooltip="Optional complete override from CWB Custom Configuration."),
                io.Combo.Input("mismatch_mode", options=["skip", "zeros", "error"], default="skip", tooltip="Preserve the anchor, insert zeros, or abort for missing and incompatible logical groups."),
                io.String.Input("output_filename", default="cwb_merged_multi_lora", tooltip="Filename without extension under ComfyUI's LoRA directory."),
                io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], default="bf16", tooltip="Requested dtype for generated floating factors."),
                io.Combo.Input("process_device", options=["cuda", "cpu"], default="cuda", tooltip="Per-layer FP32 processing device; CUDA OOM retries the affected layer on CPU."),
                io.String.Input("exclude_patterns", default="", multiline=True, tooltip="Preserve matching logical groups from the anchor."),
                io.String.Input("discard_patterns", default="", multiline=True, tooltip="Omit matching tensors or logical groups from the output."),
                io.Boolean.Input("glob_patterns", default=False, tooltip="Interpret filter entries as shell-style globs instead of regular expressions."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Use bounded UEL work-unit streaming and release each completed input layer."),
                io.Boolean.Input("force_clear_cache", default=True, tooltip="Collect Python and CUDA caches before each layer at a potential speed cost."),
                io.Boolean.Input("override_dtype", default=False, tooltip="Force generated floating factors to the requested save dtype."),
                io.Boolean.Input("include_1d_diffs", default=False, tooltip="CWB-merge 1D direct differences as FP32 instead of preserving the anchor."),
                io.Boolean.Input("counterfactual_weight_sweep", default=False, tooltip="Evaluate alpha, beta, similarity-threshold, DSC, and comfort-bandpass weight combinations from each already-computed consensus similarity vector and append them to the report without additional model loads or saved outputs."),
                io.Boolean.Input("include_mode", default=False, tooltip="Use Exclude Patterns as a whitelist instead. Only matching layers are merged; nonmatching layers are preserved from the anchor."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
                io.String.Output(display_name="cwb_report"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        documentation = load_documentation_from_file("consensus_mergers.md")
        if kwargs["execution_mode"] == "DOCUMENTATION ONLY":
            return io.NodeOutput(
                "Documentation mode active. No merge performed.",
                documentation,
                "CWB report unavailable because no merge was performed.",
            )
        count = int(kwargs["lora_count"])
        names = [kwargs[f"lora_{index}"] for index in range(1, count + 1)]
        missing = [index for index, name in enumerate(names, start=1) if not name or name == "None"]
        if missing:
            joined = ", ".join(str(index) for index in missing)
            raise ValueError(f"LoRA Count includes unselected input(s): {joined}.")
        filename, report = ConsensusMergerLogic.execute(
            names,
            "loras",
            {**kwargs, "_return_cwb_report": True},
            lora_mode=True,
        )
        return io.NodeOutput(filename, documentation, report)


class CWBEmbeddingTwoMerger(_CWBMergerNode):
    MODEL_TYPE = "embeddings"
    NODE_ID = "CWBEmbeddingTwoMerger"
    DISPLAY_NAME = "CWB Merge Embeddings (2 Models)"
    DEFAULT_FILENAME = "cwb_merged_2_embedding"
    EMBEDDING_UNION = True
    DEFAULT_PRESET = "balanced_idx_mean"


class CWBEmbeddingThreeMerger(CWBEmbeddingTwoMerger):
    INPUT_COUNT = 3
    NODE_ID = "CWBEmbeddingThreeMerger"
    DISPLAY_NAME = "CWB Merge Embeddings (3 Models)"
    DEFAULT_FILENAME = "cwb_merged_3_embedding"


def _embedding_coalesce_controls(default_filename: str):
    embedding_options = folder_paths.get_filename_list("embeddings")
    return [
        io.Combo.Input("execution_mode", options=["MERGE", "DOCUMENTATION ONLY"], default="MERGE", tooltip="MERGE writes a new embedding. DOCUMENTATION ONLY opens no files."),
        io.Combo.Input("cwb_preset", options=list(EMBEDDING_CWB_PRESETS), default="balanced_sim_mean", tooltip="Embedding CWB preset used for multi-input alignment and coalesced row pairs."),
        CWB_CONFIG.Input("cwb_config", optional=True, tooltip="Optional complete CWB configuration override."),
        io.Int.Input("target_vector_count", default=0, min=0, max=100000, step=1, tooltip="Target body-row count. Zero coalesces every eligible pair; otherwise stops at this count or when no eligible pair remains."),
        io.Float.Input("coalesce_similarity_threshold", default=0.95, min=0.0, max=1.0, step=0.01, tooltip="Minimum cosine similarity for a mutual-nearest body-row pair to coalesce."),
        io.Float.Input("coalesce_position_window", default=0.10, min=0.0, max=1.0, step=0.01, tooltip="Maximum normalized row-position distance for an eligible pair."),
        io.Boolean.Input("vision_boundary_embeddings", default=False, tooltip="Enable for complete encoded visual blocks. Start/end rows are validated and preserved; False treats the full tensor as open textual inversion."),
        io.Boolean.Input("legacy_boundary_search", default=False, tooltip="Search and trim legacy template rows using Boundary Reference Embedding. Requires vision boundary mode and a selected reference."),
        io.Combo.Input("boundary_reference_embedding", options=["None", *embedding_options], default="None", tooltip="Known-clean visual block used by legacy boundary search. Its first and last encoded rows define vision start/end."),
        io.Float.Input("boundary_similarity_threshold", default=0.95, min=-1.0, max=1.0, step=0.01, tooltip="Minimum encoded cosine similarity required independently for visual start and end validation."),
        io.String.Input("output_filename", default=default_filename, tooltip="Filename without extension under ComfyUI's embeddings category."),
        io.Combo.Input("save_dtype", options=["fp32", "fp16", "bf16"], default="fp32", tooltip="Requested dtype for generated floating embedding tensors."),
        io.Combo.Input("process_device", options=["cuda", "cpu"], default="cuda", tooltip="Per-tensor FP32 CWB arithmetic device; CUDA OOM retries that tensor on CPU."),
        io.Boolean.Input("lazy_load", default=True, tooltip="Use bounded UEL streaming and release each tensor after its work unit."),
        io.Boolean.Input("force_clear_cache", default=True, tooltip="Clear Python and CUDA caches before each tensor at a possible speed cost."),
        io.Boolean.Input("override_dtype", default=False, tooltip="Force generated floating tensors to Save Dtype."),
    ]


def _embedding_coalesce_params(kwargs: dict) -> dict:
    return {
        **kwargs,
        "mismatch_mode": "error",
        "exclude_patterns": "",
        "discard_patterns": "",
        "glob_patterns": False,
        "include_1d_diffs": False,
        "embedding_coalesce": True,
        "_embedding_coalesce_report": EmbeddingCoalesceReport(
            target_vector_count=int(kwargs["target_vector_count"]),
            similarity_threshold=float(kwargs["coalesce_similarity_threshold"]),
            position_window=float(kwargs["coalesce_position_window"]),
            vision_boundary_embeddings=bool(kwargs["vision_boundary_embeddings"]),
            legacy_boundary_search=bool(kwargs["legacy_boundary_search"]),
            boundary_similarity_threshold=float(kwargs["boundary_similarity_threshold"]),
            boundary_reference=None if kwargs["boundary_reference_embedding"] == "None" else kwargs["boundary_reference_embedding"],
        ),
        "_return_cwb_report": True,
    }


class CWBEmbeddingSelfCoalesce(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CWBEmbeddingSelfCoalesce",
            display_name="CWB Embedding Self-Coalesce",
            category="ModelUtils/Embeddings/Merge",
            description="Reduce one embedding's locally similar encoded vectors with CWB while preserving optional visual boundaries.",
            inputs=[
                io.Combo.Input("embedding", options=folder_paths.get_filename_list("embeddings"), tooltip="Embedding to coalesce."),
                *_embedding_coalesce_controls("cwb_coalesced_embedding"),
            ],
            outputs=[io.AnyType.Output(display_name="output_filename"), io.String.Output(display_name="documentation"), io.String.Output(display_name="cwb_report")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        documentation = load_documentation_from_file("consensus_mergers.md")
        if kwargs["execution_mode"] == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", documentation, "CWB report unavailable because no merge was performed.")
        if kwargs["legacy_boundary_search"] and not kwargs["vision_boundary_embeddings"]:
            raise ValueError("Legacy boundary search requires Vision Boundary Embeddings.")
        filename, report = ConsensusMergerLogic.execute(
            [kwargs["embedding"]], "embeddings", _embedding_coalesce_params(kwargs), embedding_union=True
        )
        return io.NodeOutput(filename, documentation, report)


class CWBEmbeddingMultiMerger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        embeddings = folder_paths.get_filename_list("embeddings")
        optional_embeddings = ["None", *embeddings]
        return io.Schema(
            node_id="CWBEmbeddingMultiMerger",
            display_name="CWB Embedding Multi-Merge",
            category="ModelUtils/Embeddings/Merge",
            description="CWB-merge 2 to 8 embeddings, then coalesce locally similar output vectors.",
            inputs=[
                io.Combo.Input("embedding_count", options=[str(index) for index in range(2, 9)], default="2", tooltip="Number of consecutive embedding inputs to merge."),
                io.Combo.Input("embedding_1", options=embeddings, tooltip="Primary embedding and normal visual-boundary anchor."),
                io.Combo.Input("embedding_2", options=embeddings, tooltip="Second equal-prior embedding contributor."),
                *[io.Combo.Input(f"embedding_{index}", options=optional_embeddings, default="None", tooltip="Optional embedding contributor.") for index in range(3, 9)],
                *_embedding_coalesce_controls("cwb_merged_coalesced_embedding"),
            ],
            outputs=[io.AnyType.Output(display_name="output_filename"), io.String.Output(display_name="documentation"), io.String.Output(display_name="cwb_report")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        documentation = load_documentation_from_file("consensus_mergers.md")
        if kwargs["execution_mode"] == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No merge performed.", documentation, "CWB report unavailable because no merge was performed.")
        if kwargs["legacy_boundary_search"] and not kwargs["vision_boundary_embeddings"]:
            raise ValueError("Legacy boundary search requires Vision Boundary Embeddings.")
        count = int(kwargs["embedding_count"])
        names = [kwargs[f"embedding_{index}"] for index in range(1, count + 1)]
        missing = [str(index) for index, name in enumerate(names, start=1) if not name or name == "None"]
        if missing:
            raise ValueError(f"Embedding Count includes unselected input(s): {', '.join(missing)}.")
        filename, report = ConsensusMergerLogic.execute(
            names, "embeddings", _embedding_coalesce_params(kwargs), embedding_union=True
        )
        return io.NodeOutput(filename, documentation, report)


CWB_MERGER_NODES = [
    CWBCustomConfiguration,
    CWBCheckpointTwoMerger,
    CWBCheckpointThreeMerger,
    CWBModelTwoMerger,
    CWBModelThreeMerger,
    CWBTextEncoderTwoMerger,
    CWBTextEncoderThreeMerger,
    CWBLoRATwoMerger,
    CWBLoRAThreeMerger,
    CWBLoRAMultiMerger,
    CWBEmbeddingTwoMerger,
    CWBEmbeddingThreeMerger,
    CWBEmbeddingSelfCoalesce,
    CWBEmbeddingMultiMerger,
]
