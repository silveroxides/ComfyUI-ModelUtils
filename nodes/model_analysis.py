"""Streaming two-model similarity analysis nodes."""

from __future__ import annotations

import gc
import heapq
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from tqdm import tqdm
from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned

from .consensus_merger import (
    _secondary_lora_map,
)
from .device_utils import (
    cleanup_after_operation,
    estimate_model_size,
    prepare_for_large_operation,
)
from .merger import load_documentation_from_file
from .lora_resize import layer_tensor_keys, parse_lora_layers, validate_canonical_blocks
from .quantization_guard import inspect_low_bit_input


FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
PARAMETER_SUFFIXES = (
    ".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight",
    ".lora.down.weight", ".lora.up.weight", ".reshape_weight", ".dora_scale",
    ".set_weight", ".diff_b", ".w_norm", ".b_norm", ".alpha", ".weight",
    ".bias", ".scale", ".diff", ".lin",
)


def _is_float_dtype(dtype: torch.dtype) -> bool:
    return dtype in FLOAT_DTYPES


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.6g}"


def _shape_text(shape: tuple[int, ...]) -> str:
    return "scalar" if not shape else "x".join(str(value) for value in shape)


def _layer_name(key: str) -> str:
    for suffix in PARAMETER_SUFFIXES:
        if key.endswith(suffix):
            return key[:-len(suffix)] or key
    return key


def _block_name(key: str) -> str:
    layer = _layer_name(key)
    numeric = re.search(r"(?:^|[._])\d+(?=$|[._])", layer)
    if numeric:
        return layer[:numeric.end()]
    split_at = max(layer.rfind("."), layer.rfind("_"))
    return layer[:split_at] if split_at > 0 else layer


@dataclass
class RawStats:
    finite: int = 0
    total: int = 0
    sum_abs: float = 0.0
    sum_sq_diff: float = 0.0
    dot: float = 0.0
    sum_a2: float = 0.0
    sum_b2: float = 0.0
    sum_a: float = 0.0
    sum_b: float = 0.0
    exact: int = 0
    sign_equal: int = 0
    max_abs: float = 0.0
    nonfinite_a: int = 0
    nonfinite_b: int = 0

    def add(self, other: "RawStats") -> None:
        for name in (
            "finite", "total", "sum_abs", "sum_sq_diff", "dot", "sum_a2",
            "sum_b2", "sum_a", "sum_b", "exact", "sign_equal",
            "nonfinite_a", "nonfinite_b",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.max_abs = max(self.max_abs, other.max_abs)


def _metrics(raw: RawStats) -> dict[str, float | None]:
    if raw.finite == 0:
        return {name: None for name in (
            "mae", "mse", "rmse", "max", "relative_l2", "cosine", "pearson",
            "exact", "sign", "norm_ratio", "coverage",
        )}
    n = raw.finite
    mse = raw.sum_sq_diff / n
    norm_a = math.sqrt(max(raw.sum_a2, 0.0))
    norm_b = math.sqrt(max(raw.sum_b2, 0.0))
    if norm_a == 0.0 and norm_b == 0.0:
        cosine = 1.0 if raw.sum_sq_diff == 0.0 else None
        norm_ratio = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        cosine = None
        norm_ratio = None if norm_a == 0.0 else 0.0
    else:
        cosine = raw.dot / (norm_a * norm_b)
        norm_ratio = norm_b / norm_a
    centered_a = max(raw.sum_a2 - raw.sum_a * raw.sum_a / n, 0.0)
    centered_b = max(raw.sum_b2 - raw.sum_b * raw.sum_b / n, 0.0)
    pearson_denom = math.sqrt(centered_a * centered_b)
    pearson = (
        (raw.dot - raw.sum_a * raw.sum_b / n) / pearson_denom
        if pearson_denom > 0.0 else None
    )
    average_norm = (norm_a + norm_b) / 2.0
    relative_l2 = math.sqrt(raw.sum_sq_diff) / average_norm if average_norm else 0.0
    return {
        "mae": raw.sum_abs / n,
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)),
        "max": raw.max_abs,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "pearson": pearson,
        "exact": raw.exact / n,
        "sign": raw.sign_equal / n,
        "norm_ratio": norm_ratio,
        "coverage": raw.finite / raw.total if raw.total else None,
    }


@dataclass
class CWBStats:
    rows: int = 0
    affinity_rows: int = 0
    pair_sum: float = 0.0
    pair_min: float = math.inf
    pair_max: float = -math.inf
    mean_aff_a: float = 0.0
    mean_aff_b: float = 0.0
    median_aff_a: float = 0.0
    median_aff_b: float = 0.0
    reference_rows: int = 0
    matched_rows: int = 0
    alignment_rows: int = 0
    alignment_sum: float = 0.0
    alignment_min: float = math.inf
    alignment_max: float = -math.inf
    index_alignment_sum: float = 0.0

    def add(self, other: "CWBStats") -> None:
        if other.rows:
            self.pair_min = min(self.pair_min, other.pair_min)
            self.pair_max = max(self.pair_max, other.pair_max)
        if other.alignment_rows:
            self.alignment_min = min(self.alignment_min, other.alignment_min)
            self.alignment_max = max(self.alignment_max, other.alignment_max)
        for name in (
            "rows", "affinity_rows", "pair_sum", "mean_aff_a", "mean_aff_b",
            "median_aff_a", "median_aff_b", "reference_rows", "matched_rows",
            "alignment_rows", "alignment_sum", "index_alignment_sum",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def values(self) -> dict[str, float | None]:
        n = self.rows
        affinity_n = self.affinity_rows
        alignment_n = self.alignment_rows
        alignment = self.alignment_sum / alignment_n if alignment_n else None
        return {
            "pair_mean": self.pair_sum / n if n else None,
            "pair_min": self.pair_min if n else None,
            "pair_max": self.pair_max if n else None,
            "mean_aff_a": self.mean_aff_a / affinity_n if affinity_n else None,
            "mean_aff_b": self.mean_aff_b / affinity_n if affinity_n else None,
            "median_aff_a": self.median_aff_a / affinity_n if affinity_n else None,
            "median_aff_b": self.median_aff_b / affinity_n if affinity_n else None,
            "matched": self.matched_rows / self.reference_rows if self.reference_rows else None,
            "alignment": alignment,
            "alignment_min": self.alignment_min if alignment_n else None,
            "alignment_max": self.alignment_max if alignment_n else None,
            "alignment_gain": (
                alignment - self.index_alignment_sum / alignment_n
                if alignment_n else None
            ),
        }


@dataclass
class TensorRecord:
    key: str
    shape: tuple[int, ...]
    dtype_a: str
    dtype_b: str
    raw: RawStats
    cwb: CWBStats
    cpu_fallback: bool = False


def _row_cosines(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=1, eps=1e-8) * F.normalize(b, dim=1, eps=1e-8)).sum(1)


def _compute_raw(
    a: torch.Tensor,
    b: torch.Tensor,
    top_count: int,
) -> tuple[RawStats, list[tuple[float, int, float, float]]]:
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    finite_a = torch.isfinite(flat_a)
    finite_b = torch.isfinite(flat_b)
    finite = finite_a & finite_b
    raw = RawStats(
        total=flat_a.numel(),
        nonfinite_a=int((~finite_a).sum().item()),
        nonfinite_b=int((~finite_b).sum().item()),
    )
    if not finite.any():
        return raw, []
    va = flat_a[finite]
    vb = flat_b[finite]
    diff = va - vb
    abs_diff = diff.abs()
    raw.finite = va.numel()
    raw.sum_abs = float(abs_diff.sum().item())
    raw.sum_sq_diff = float((diff * diff).sum().item())
    raw.dot = float((va * vb).sum().item())
    raw.sum_a2 = float((va * va).sum().item())
    raw.sum_b2 = float((vb * vb).sum().item())
    raw.sum_a = float(va.sum().item())
    raw.sum_b = float(vb.sum().item())
    raw.exact = int((va == vb).sum().item())
    raw.sign_equal = int((torch.sign(va) == torch.sign(vb)).sum().item())
    raw.max_abs = float(abs_diff.max().item())
    if top_count <= 0:
        return raw, []
    masked = torch.where(finite, (flat_a - flat_b).abs(), torch.full_like(flat_a, -1.0))
    count = min(top_count, masked.numel())
    values, indices = torch.topk(masked, count)
    entries = []
    for value, index in zip(values.tolist(), indices.tolist()):
        if value < 0.0:
            continue
        entries.append((float(value), int(index), float(flat_a[index].item()), float(flat_b[index].item())))
    return raw, entries


def _greedy_alignment(similarities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return threshold-independent one-to-one matches using CWB's greedy rule."""
    if not similarities.numel():
        empty = torch.empty(0, device=similarities.device, dtype=torch.long)
        return empty, empty
    scores = similarities.clone()
    reference_rows = []
    source_rows = []
    for _ in range(min(scores.shape)):
        flat_index = torch.argmax(scores)
        reference_row = flat_index // scores.shape[1]
        source_row = flat_index % scores.shape[1]
        reference_rows.append(reference_row)
        source_rows.append(source_row)
        scores[reference_row, :] = -torch.inf
        scores[:, source_row] = -torch.inf
    return torch.stack(reference_rows), torch.stack(source_rows)


def _consensus_affinities(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean_consensus = (left + right) / 2.0
    median_consensus = torch.median(torch.stack((left, right)), dim=0).values
    return (
        _row_cosines(left, mean_consensus),
        _row_cosines(right, mean_consensus),
        _row_cosines(left, median_consensus),
        _row_cosines(right, median_consensus),
    )


def _compute_cwb(a: torch.Tensor, b: torch.Tensor, allow_alignment: bool) -> CWBStats:
    if a.ndim < 2:
        rows_a = a.reshape(1, -1)
        rows_b = b.reshape(1, -1)
    else:
        rows_a = a.reshape(a.shape[0], -1)
        rows_b = b.reshape(b.shape[0], -1)
    valid_a = torch.isfinite(rows_a).all(dim=1)
    valid_b = torch.isfinite(rows_b).all(dim=1)
    paired_valid = valid_a & valid_b
    indexed_a = rows_a[paired_valid]
    indexed_b = rows_b[paired_valid]
    if not indexed_a.shape[0]:
        return CWBStats()

    index_scores = _row_cosines(indexed_a, indexed_b)
    stats = CWBStats()
    aligned = allow_alignment and a.ndim >= 2
    if aligned:
        finite_a = rows_a[valid_a]
        finite_b = rows_b[valid_b]
        similarities = torch.mm(
            F.normalize(finite_a, dim=1, eps=1e-8),
            F.normalize(finite_b, dim=1, eps=1e-8).T,
        )
        reference_rows, source_rows = _greedy_alignment(similarities)
        if not reference_rows.numel():
            return stats
        compared_a = finite_a[reference_rows]
        compared_b = finite_b[source_rows]
        pair = similarities[reference_rows, source_rows]
        stats.reference_rows = max(finite_a.shape[0], finite_b.shape[0])
        stats.matched_rows = pair.numel()
        stats.alignment_rows = pair.numel()
        stats.alignment_sum = float(pair.sum().item())
        stats.alignment_min = float(pair.min().item())
        stats.alignment_max = float(pair.max().item())
        stats.index_alignment_sum = float(index_scores.mean().item()) * pair.numel()
    else:
        compared_a = indexed_a
        compared_b = indexed_b
        pair = index_scores

    count = pair.numel()
    if count == 0:
        return stats
    mean_a, mean_b, median_a, median_b = _consensus_affinities(compared_a, compared_b)
    stats.pair_sum = float(pair.sum().item())
    stats.pair_min = float(pair.min().item())
    stats.pair_max = float(pair.max().item())
    stats.mean_aff_a = float(mean_a.sum().item())
    stats.mean_aff_b = float(mean_b.sum().item())
    stats.median_aff_a = float(median_a.sum().item())
    stats.median_aff_b = float(median_b.sum().item())
    stats.rows = count
    stats.affinity_rows = count
    return stats


def _compute_pair_on_device(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    device: str,
    allow_alignment: bool,
    top_count: int,
) -> tuple[RawStats, CWBStats, list[tuple[float, int, float, float]]]:
    if str(device).startswith("cuda"):
        a = transfer_to_gpu_pinned(tensor_a, device=device, dtype=torch.float32)
        b = transfer_to_gpu_pinned(tensor_b, device=device, dtype=torch.float32)
    else:
        a = tensor_a.to(device=device, dtype=torch.float32)
        b = tensor_b.to(device=device, dtype=torch.float32)
    raw, top = _compute_raw(a, b, top_count)
    cwb = _compute_cwb(a, b, allow_alignment)
    return raw, cwb, top


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _release_cuda_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _analyze_with_fallback(
    key: str,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    device: str,
    allow_alignment: bool,
    top_count: int,
):
    try:
        return (*_compute_pair_on_device(
            tensor_a, tensor_b, device, allow_alignment, top_count
        ), False)
    except BaseException as oom:
        if not str(device).startswith("cuda") or not _is_cuda_oom(oom):
            raise
        oom.__traceback__ = None
        _release_cuda_oom()
        try:
            result = _compute_pair_on_device(
                tensor_a, tensor_b, "cpu", allow_alignment, top_count
            )
        except BaseException as cpu_error:
            raise RuntimeError(
                f"CPU fallback failed while analyzing '{key}': {cpu_error}"
            ) from oom
        return (*result, True)


def _compute_lora_pair_on_device(
    down_a: torch.Tensor,
    up_a: torch.Tensor,
    down_b: torch.Tensor,
    up_b: torch.Tensor,
    device: str,
    allow_alignment: bool,
    top_count: int,
) -> tuple[
    RawStats, list[tuple[float, int, float, float]],
    RawStats, list[tuple[float, int, float, float]], CWBStats,
]:
    if str(device).startswith("cuda"):
        da = transfer_to_gpu_pinned(down_a, device=device, dtype=torch.float32)
        ua = transfer_to_gpu_pinned(up_a, device=device, dtype=torch.float32)
        db = transfer_to_gpu_pinned(down_b, device=device, dtype=torch.float32)
        ub = transfer_to_gpu_pinned(up_b, device=device, dtype=torch.float32)
    else:
        da = down_a.to(device=device, dtype=torch.float32)
        ua = up_a.to(device=device, dtype=torch.float32)
        db = down_b.to(device=device, dtype=torch.float32)
        ub = up_b.to(device=device, dtype=torch.float32)
    rank = da.shape[0]
    if db.shape[0] != rank or ua.shape[1] != rank or ub.shape[1] != rank:
        raise ValueError("LoRA down/up tensors do not share one rank.")
    raw_down, top_down = _compute_raw(da, db, top_count)
    raw_up, top_up = _compute_raw(ua, ub, top_count)
    da_rows = da.reshape(rank, -1)
    db_rows = db.reshape(rank, -1)
    ua_rows = ua.movedim(1, 0).reshape(rank, -1)
    ub_rows = ub.movedim(1, 0).reshape(rank, -1)
    valid_a = torch.isfinite(da_rows).all(1) & torch.isfinite(ua_rows).all(1)
    valid_b = torch.isfinite(db_rows).all(1) & torch.isfinite(ub_rows).all(1)
    paired_valid = valid_a & valid_b
    indexed_da = da_rows[paired_valid]
    indexed_db = db_rows[paired_valid]
    indexed_ua = ua_rows[paired_valid]
    indexed_ub = ub_rows[paired_valid]
    stats = CWBStats()
    if not indexed_da.shape[0]:
        return raw_down, top_down, raw_up, top_up, stats

    index_scores = _row_cosines(indexed_da, indexed_db) * _row_cosines(indexed_ua, indexed_ub)
    if allow_alignment:
        finite_da, finite_ua = da_rows[valid_a], ua_rows[valid_a]
        finite_db, finite_ub = db_rows[valid_b], ub_rows[valid_b]
        down_matrix = torch.mm(
            F.normalize(finite_da, dim=1, eps=1e-8),
            F.normalize(finite_db, dim=1, eps=1e-8).T,
        )
        up_matrix = torch.mm(
            F.normalize(finite_ua, dim=1, eps=1e-8),
            F.normalize(finite_ub, dim=1, eps=1e-8).T,
        )
        contribution_matrix = down_matrix * up_matrix
        reference_rows, source_rows = _greedy_alignment(contribution_matrix)
        compared_da = finite_da[reference_rows]
        compared_ua = finite_ua[reference_rows]
        compared_db = finite_db[source_rows]
        compared_ub = finite_ub[source_rows]
        selected_down = down_matrix[reference_rows, source_rows]
        selected_up = up_matrix[reference_rows, source_rows]
        sign_flip = (selected_down < 0) & (selected_up < 0)
        compared_db = torch.where(sign_flip[:, None], -compared_db, compared_db)
        compared_ub = torch.where(sign_flip[:, None], -compared_ub, compared_ub)
        pair = contribution_matrix[reference_rows, source_rows]
        stats.reference_rows = max(finite_da.shape[0], finite_db.shape[0])
        stats.matched_rows = pair.numel()
        stats.alignment_rows = pair.numel()
        stats.alignment_sum = float(pair.sum().item())
        stats.alignment_min = float(pair.min().item())
        stats.alignment_max = float(pair.max().item())
        stats.index_alignment_sum = float(index_scores.mean().item()) * pair.numel()
    else:
        compared_da, compared_db = indexed_da, indexed_db
        compared_ua, compared_ub = indexed_ua, indexed_ub
        pair = index_scores

    stats.rows = pair.numel()
    stats.pair_sum = float(pair.sum().item())
    stats.pair_min = float(pair.min().item())
    stats.pair_max = float(pair.max().item())
    for left, right in ((compared_da, compared_db), (compared_ua, compared_ub)):
        mean_a, mean_b, median_a, median_b = _consensus_affinities(left, right)
        stats.mean_aff_a += float(mean_a.sum().item())
        stats.mean_aff_b += float(mean_b.sum().item())
        stats.median_aff_a += float(median_a.sum().item())
        stats.median_aff_b += float(median_b.sum().item())
        stats.affinity_rows += left.shape[0]
    return raw_down, top_down, raw_up, top_up, stats


def _analyze_lora_with_fallback(
    key: str,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    allow_alignment: bool,
    top_count: int,
):
    try:
        return (*_compute_lora_pair_on_device(
            *tensors, device, allow_alignment, top_count
        ), False)
    except BaseException as oom:
        if not str(device).startswith("cuda") or not _is_cuda_oom(oom):
            raise
        oom.__traceback__ = None
        _release_cuda_oom()
        try:
            result = _compute_lora_pair_on_device(
                *tensors, "cpu", allow_alignment, top_count
            )
        except BaseException as cpu_error:
            raise RuntimeError(
                f"CPU fallback failed while analyzing LoRA layer '{key}': {cpu_error}"
            ) from oom
        return (*result, True)


def _coordinate(shape: tuple[int, ...], flat_index: int) -> str:
    if not shape:
        return "()"
    result = []
    remaining = flat_index
    for size in reversed(shape):
        result.append(remaining % size)
        remaining //= size
    return str(tuple(reversed(result)))


def _load_paired_unit(handlers, units, *, pin_memory: bool):
    """Load one logical A/B analysis unit concurrently, with no cross-unit prefetch."""
    stream_a = handlers[0].async_stream(
        [unit[1] for unit in units],
        batch_size=1,
        prefetch_batches=1,
        pin_memory=pin_memory,
    )
    stream_b = handlers[1].async_stream(
        [unit[2] for unit in units],
        batch_size=1,
        prefetch_batches=1,
        pin_memory=pin_memory,
    )
    loaded = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            for unit in units:
                future_a = executor.submit(next, stream_a)
                future_b = executor.submit(next, stream_b)
                batch_a = future_a.result()
                batch_b = future_b.result()
                if len(batch_a) != 1 or len(batch_b) != 1:
                    raise RuntimeError("Asynchronous model analysis expected one tensor per batch.")
                key_a, tensor_a = batch_a[0]
                key_b, tensor_b = batch_b[0]
                if key_a != unit[1] or key_b != unit[2]:
                    raise RuntimeError(
                        "Asynchronous model analysis returned tensors out of order."
                    )
                loaded.append((*unit, tensor_a, tensor_b))
    finally:
        stream_a.close()
        stream_b.close()
    return loaded


def _average_layer_metrics(layer_stats: dict[str, RawStats]) -> dict[str, float | None]:
    metrics = [_metrics(raw) for raw in layer_stats.values()]
    result = {}
    for name in ("mae", "mse", "rmse", "relative_l2", "cosine", "pearson", "exact", "sign", "norm_ratio"):
        values = [entry[name] for entry in metrics if entry[name] is not None and math.isfinite(entry[name])]
        result[name] = sum(values) / len(values) if values else None
    return result


def _metadata_lines(metadata_a: dict | None, metadata_b: dict | None) -> list[str]:
    a = metadata_a or {}
    b = metadata_b or {}
    keys = sorted(set(a) | set(b))
    differences = [key for key in keys if a.get(key) != b.get(key)]
    lines = [f"Metadata keys: A={len(a)}, B={len(b)}, differing={len(differences)}"]
    lines.extend(f"- {key}" for key in differences)
    return lines


def _render_comparison(
    model_a: str,
    model_b: str,
    model_type: str,
    records: list[TensorRecord],
    global_raw: RawStats,
    layer_stats: dict[str, RawStats],
    block_stats: dict[str, RawStats],
    inventory: dict[str, list[str]],
    top_weights: list[tuple],
    fallback_keys: list[str],
    metadata_a: dict | None,
    metadata_b: dict | None,
) -> str:
    weighted = _metrics(global_raw)
    layer_average = _average_layer_metrics(layer_stats)
    lines = [
        "MODEL COMPARISON SUMMARY",
        f"Type: {model_type}",
        f"Model A: {model_a}",
        f"Model B: {model_b}",
        f"Comparable floating tensors: {len(records)}",
        f"Comparable parameters: {global_raw.total}",
        f"A-only keys: {len(inventory['a_only'])}",
        f"B-only keys: {len(inventory['b_only'])}",
        f"Shape mismatches: {len(inventory['shape'])}",
        f"Non-floating shared tensors: {len(inventory['nonfloat'])}",
        f"Unsupported low-bit tensors: {len(inventory['low_bit'])}",
        f"CUDA OOM CPU fallbacks: {len(fallback_keys)}",
        "",
        "PARAMETER-WEIGHTED GLOBAL METRICS",
    ]
    for name in ("mae", "mse", "rmse", "max", "relative_l2", "cosine", "pearson", "exact", "sign", "norm_ratio", "coverage"):
        lines.append(f"{name}: {_fmt(weighted[name])}")
    lines.extend(["", "EQUAL-LAYER AVERAGES"])
    for name, value in layer_average.items():
        lines.append(f"{name}: {_fmt(value)}")

    metrics_by_record = [(record, _metrics(record.raw)) for record in records]
    lines.extend(["", "MOST DIVERGENT TENSORS", "By relative L2:"])
    for record, metric in sorted(metrics_by_record, key=lambda item: item[1]["relative_l2"] or -1.0, reverse=True)[:10]:
        lines.append(f"- {record.key}: {_fmt(metric['relative_l2'])}")
    lines.append("By MAE:")
    for record, metric in sorted(metrics_by_record, key=lambda item: item[1]["mae"] or -1.0, reverse=True)[:10]:
        lines.append(f"- {record.key}: {_fmt(metric['mae'])}")
    lines.append("By lowest cosine:")
    cosine_records = [(record, metric) for record, metric in metrics_by_record if metric["cosine"] is not None]
    for record, metric in sorted(cosine_records, key=lambda item: item[1]["cosine"])[:10]:
        lines.append(f"- {record.key}: {_fmt(metric['cosine'])}")

    lines.extend(["", "INFERRED BLOCKWISE METRICS", "block | params | MAE | RMSE | relative_L2 | cosine"])
    for block in sorted(block_stats):
        raw = block_stats[block]
        metric = _metrics(raw)
        lines.append(
            f"{block} | {raw.total} | {_fmt(metric['mae'])} | {_fmt(metric['rmse'])} | "
            f"{_fmt(metric['relative_l2'])} | {_fmt(metric['cosine'])}"
        )

    lines.extend(["", "LAYERWISE TENSOR METRICS", "key | shape | dtype A/B | MAE | RMSE | max_abs | relative_L2 | cosine | pearson | exact | sign | norm_B/A"])
    for record, metric in sorted(metrics_by_record, key=lambda item: item[0].key):
        lines.append(
            f"{record.key} | {_shape_text(record.shape)} | {record.dtype_a}/{record.dtype_b} | "
            f"{_fmt(metric['mae'])} | {_fmt(metric['rmse'])} | {_fmt(metric['max'])} | "
            f"{_fmt(metric['relative_l2'])} | {_fmt(metric['cosine'])} | {_fmt(metric['pearson'])} | "
            f"{_fmt(metric['exact'])} | {_fmt(metric['sign'])} | {_fmt(metric['norm_ratio'])}"
        )

    lines.extend(["", "TOP SCALAR WEIGHT DIFFERENCES"])
    for value, key, flat_index, a_value, b_value, shape in sorted(top_weights, reverse=True):
        lines.append(
            f"- {key}{_coordinate(shape, flat_index)}: |delta|={_fmt(value)}, "
            f"A={_fmt(a_value)}, B={_fmt(b_value)}"
        )
    if not top_weights:
        lines.append("- None")

    lines.extend(["", "TOPOLOGY AND DATA FINDINGS"])
    for label, name in (
        ("A ONLY", "a_only"), ("B ONLY", "b_only"), ("SHAPE MISMATCH", "shape"),
        ("NON-FLOATING", "nonfloat"), ("LOW-BIT UNSUPPORTED", "low_bit"),
    ):
        lines.append(f"[{label}] ({len(inventory[name])})")
        lines.extend(f"- {value}" for value in inventory[name])
        if not inventory[name]:
            lines.append("- None")
    lines.append(f"[CUDA OOM -> CPU] ({len(fallback_keys)})")
    lines.extend(f"- {key}" for key in fallback_keys)
    if not fallback_keys:
        lines.append("- None")
    lines.extend(["", "METADATA"])
    lines.extend(_metadata_lines(metadata_a, metadata_b))
    return "\n".join(lines)


def _render_cwb(
    model_a: str,
    model_b: str,
    records: list[TensorRecord],
    block_cwb: dict[str, CWBStats],
    fallback_keys: list[str],
    alignment_scope: str,
) -> str:
    global_cwb = CWBStats()
    for record in records:
        global_cwb.add(record.cwb)
    values = global_cwb.values()
    lines = [
        "CWB-DERIVED SIMILARITY SUMMARY",
        f"Model A: {model_a}",
        f"Model B: {model_b}",
        f"Alignment scope: {alignment_scope}",
        "Boundary: CWB similarity, alignment, and consensus diagnostics only.",
        "Excluded: merge weighting, contribution calculation, vector reconstitution, norm rescaling, and output writing.",
        "",
        "GLOBAL CWB-DERIVED DIAGNOSTICS",
        f"Pairwise row cosine mean/min/max: {_fmt(values['pair_mean'])} / {_fmt(values['pair_min'])} / {_fmt(values['pair_max'])}",
        f"Mean-consensus affinity A/B: {_fmt(values['mean_aff_a'])} / {_fmt(values['mean_aff_b'])}",
        f"Median-consensus affinity A/B: {_fmt(values['median_aff_a'])} / {_fmt(values['median_aff_b'])}",
        f"Greedy alignment coverage: {_fmt(values['matched'])}",
        f"Greedy matched cosine mean/min/max: {_fmt(values['alignment'])} / {_fmt(values['alignment_min'])} / {_fmt(values['alignment_max'])}",
        f"Greedy improvement over index alignment: {_fmt(values['alignment_gain'])}",
        "",
        "INFERRED BLOCKWISE CWB DIAGNOSTICS",
        "block | pair_cos mean/min/max | mean_aff_A/B | median_aff_A/B | matched | alignment | gain",
    ]
    for block in sorted(block_cwb):
        entry = block_cwb[block].values()
        lines.append(
            f"{block} | {_fmt(entry['pair_mean'])}/{_fmt(entry['pair_min'])}/{_fmt(entry['pair_max'])} | "
            f"{_fmt(entry['mean_aff_a'])}/{_fmt(entry['mean_aff_b'])} | "
            f"{_fmt(entry['median_aff_a'])}/{_fmt(entry['median_aff_b'])} | "
            f"{_fmt(entry['matched'])} | {_fmt(entry['alignment'])} | {_fmt(entry['alignment_gain'])}"
        )
    lines.extend([
        "", "LAYERWISE CWB DIAGNOSTICS",
        "key | pair_cos mean/min/max | mean_aff_A/B | median_aff_A/B | matched | alignment | gain",
    ])
    for record in sorted(records, key=lambda value: value.key):
        entry = record.cwb.values()
        lines.append(
            f"{record.key} | {_fmt(entry['pair_mean'])}/{_fmt(entry['pair_min'])}/{_fmt(entry['pair_max'])} | "
            f"{_fmt(entry['mean_aff_a'])}/{_fmt(entry['mean_aff_b'])} | "
            f"{_fmt(entry['median_aff_a'])}/{_fmt(entry['median_aff_b'])} | "
            f"{_fmt(entry['matched'])} | {_fmt(entry['alignment'])} | {_fmt(entry['alignment_gain'])}"
        )
    lines.append("")
    lines.append(f"CUDA OOM CPU fallbacks: {len(fallback_keys)}")
    lines.extend(f"- {key}" for key in fallback_keys)
    return "\n".join(lines)


class ModelAnalysisLogic:
    @classmethod
    def execute(
        cls,
        model_a: str,
        model_b: str,
        model_type: str,
        params: dict,
        *,
        embedding_alignment: bool = False,
        lora_mode: bool = False,
    ) -> tuple[str, str]:
        paths = []
        for name in (model_a, model_b):
            path = folder_paths.get_full_path(model_type, name)
            if not path:
                raise FileNotFoundError(f"{model_type} input '{name}' was not found.")
            paths.append(path)
        total_size = sum(estimate_model_size(path) for path in paths)
        if total_size:
            prepare_for_large_operation(total_size * 1.2, torch.device(params["process_device"]))

        handlers = [MemoryEfficientSafeOpen(path, low_memory=True) for path in paths]
        try:
            low_bit = [
                inspect_low_bit_input(handler, f"Input {index + 1} ({path})", "Model Analysis")
                for index, (handler, path) in enumerate(zip(handlers, paths))
            ]
            similarity_alignment = bool(params.get("cwb_similarity_alignment", False))
            keys_a = set(handlers[0].keys())
            keys_b = set(handlers[1].keys())
            inventory = {
                "a_only": [], "b_only": [],
                "shape": [], "nonfloat": [], "low_bit": [],
            }
            units: list[tuple[str, str, str]] = []
            lora_pairs: list[tuple[str, dict[str, str], dict[str, str]]] = []
            if lora_mode:
                parsed_a, parsed_b = parse_lora_layers(list(keys_a)), parse_lora_layers(list(keys_b))
                pairs_a, passthrough_a = parsed_a
                pairs_b, passthrough_b = parsed_b
                pairs_a = {block: layer_tensor_keys(roles) for block, roles in pairs_a.items()}
                pairs_b = {block: layer_tensor_keys(roles) for block, roles in pairs_b.items()}
                validate_canonical_blocks(pairs_a, "Model Analysis input A")
                validate_canonical_blocks(pairs_b, "Model Analysis input B")
                map_a = _secondary_lora_map(pairs_a, input_index=0)
                map_b = _secondary_lora_map(pairs_b, input_index=1)
                for core in sorted(set(map_a) | set(map_b)):
                    if core not in map_a:
                        inventory["b_only"].extend(
                            f"{core}.{role}: {key}" for role, key in sorted(pairs_b[map_b[core]].items())
                        )
                        continue
                    if core not in map_b:
                        inventory["a_only"].extend(
                            f"{core}.{role}: {key}" for role, key in sorted(pairs_a[map_a[core]].items())
                        )
                        continue
                    roles_a = pairs_a[map_a[core]]
                    roles_b = pairs_b[map_b[core]]
                    complete_pair = all(
                        role in roles_a and role in roles_b for role in ("down", "up")
                    )
                    if complete_pair:
                        down_shape_a = tuple(handlers[0].get_shape(roles_a["down"]))
                        up_shape_a = tuple(handlers[0].get_shape(roles_a["up"]))
                        down_shape_b = tuple(handlers[1].get_shape(roles_b["down"]))
                        up_shape_b = tuple(handlers[1].get_shape(roles_b["up"]))
                        complete_pair = (
                            down_shape_a == down_shape_b
                            and up_shape_a == up_shape_b
                            and len(down_shape_a) >= 2
                            and len(up_shape_a) >= 2
                            and down_shape_a[0] == up_shape_a[1]
                            and down_shape_b[0] == up_shape_b[1]
                            and all(_is_float_dtype(dtype) for dtype in (
                                handlers[0].get_dtype(roles_a["down"]),
                                handlers[0].get_dtype(roles_a["up"]),
                                handlers[1].get_dtype(roles_b["down"]),
                                handlers[1].get_dtype(roles_b["up"]),
                            ))
                        )
                    for role in sorted(set(roles_a) | set(roles_b)):
                        display = f"{core}.{role}"
                        if role not in roles_a:
                            inventory["b_only"].append(f"{display}: {roles_b[role]}")
                        elif role not in roles_b:
                            inventory["a_only"].append(f"{display}: {roles_a[role]}")
                        elif not (complete_pair and role in {"down", "up"}):
                            units.append((display, roles_a[role], roles_b[role]))
                    if complete_pair:
                        lora_pairs.append((core, roles_a, roles_b))
                pass_a = set(passthrough_a)
                pass_b = set(passthrough_b)
                units.extend((key, key, key) for key in sorted(pass_a & pass_b))
                inventory["a_only"].extend(sorted(pass_a - pass_b))
                inventory["b_only"].extend(sorted(pass_b - pass_a))
            else:
                units = [(key, key, key) for key in sorted(keys_a & keys_b)]
                inventory["a_only"] = sorted(keys_a - keys_b)
                inventory["b_only"] = sorted(keys_b - keys_a)
            global_raw = RawStats()
            layer_stats: dict[str, RawStats] = {}
            block_stats: dict[str, RawStats] = {}
            block_cwb: dict[str, CWBStats] = {}
            records: list[TensorRecord] = []
            cwb_records: list[TensorRecord] = []
            fallback_keys: list[str] = []
            top_heap: list[tuple] = []
            top_limit = params["top_weight_differences"]

            def add_standard_record(
                key, shape, dtype_a, dtype_b, raw, cwb, top, fallback,
                *, include_cwb=True,
            ):
                record = TensorRecord(
                    key=key, shape=shape, dtype_a=str(dtype_a), dtype_b=str(dtype_b),
                    raw=raw, cwb=cwb, cpu_fallback=fallback,
                )
                records.append(record)
                global_raw.add(raw)
                layer = _layer_name(key)
                block = _block_name(key)
                layer_stats.setdefault(layer, RawStats()).add(raw)
                block_stats.setdefault(block, RawStats()).add(raw)
                if include_cwb and cwb.rows:
                    cwb_records.append(record)
                    block_cwb.setdefault(block, CWBStats()).add(cwb)
                for value, flat_index, a_value, b_value in top:
                    entry = (value, key, flat_index, a_value, b_value, shape)
                    if len(top_heap) < top_limit:
                        heapq.heappush(top_heap, entry)
                    elif value > top_heap[0][0]:
                        heapq.heapreplace(top_heap, entry)

            def clear_after_unit():
                if params["force_clear_cache"]:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            ready_units = []
            skipped_units = 0
            for key, key_a, key_b in units:
                if key_a in low_bit[0] or key_b in low_bit[1]:
                    inventory["low_bit"].append(f"{key}: A={key_a}, B={key_b}")
                    skipped_units += 1
                    continue
                shape_a = tuple(handlers[0].get_shape(key_a))
                shape_b = tuple(handlers[1].get_shape(key_b))
                if shape_a != shape_b:
                    inventory["shape"].append(f"{key}: A={shape_a}, B={shape_b}")
                    skipped_units += 1
                    continue
                ready_units.append((
                    key, key_a, key_b, shape_a,
                    handlers[0].get_dtype(key_a), handlers[1].get_dtype(key_b),
                ))
            progress = comfy.utils.ProgressBar(len(units) + len(lora_pairs))
            if skipped_units:
                progress.update(skipped_units)

            with torch.no_grad():
                iterator = tqdm(ready_units, desc="Analyzing model tensors", unit="tensors")
                for key, key_a, key_b, shape_a, dtype_a, dtype_b in iterator:
                    loaded = _load_paired_unit(
                        handlers, [(key, key_a, key_b)],
                        pin_memory=str(params["process_device"]).startswith("cuda"),
                    )
                    _, _, _, tensor_a, tensor_b = loaded[0]
                    try:
                        if not _is_float_dtype(dtype_a) or not _is_float_dtype(dtype_b):
                            equal = torch.equal(tensor_a, tensor_b)
                            inventory["nonfloat"].append(
                                f"{key}: A={dtype_a}, B={dtype_b}, exact_equal={equal}"
                            )
                            continue
                        raw, cwb, top, fallback = _analyze_with_fallback(
                            key, tensor_a, tensor_b, params["process_device"],
                            embedding_alignment and similarity_alignment, top_limit,
                        )
                        add_standard_record(
                            key, shape_a, dtype_a, dtype_b, raw, cwb, top, fallback
                        )
                        if fallback:
                            fallback_keys.append(key)
                    finally:
                        del tensor_a, tensor_b
                        loaded.clear()
                        handlers[0].mark_processed(key_a)
                        handlers[1].mark_processed(key_b)
                        clear_after_unit()
                        progress.update(1)

                for core, roles_a, roles_b in tqdm(lora_pairs, desc="Analyzing LoRA pairs", unit="layers"):
                    pair_keys = (roles_a["down"], roles_a["up"], roles_b["down"], roles_b["up"])
                    if pair_keys[0] in low_bit[0] or pair_keys[1] in low_bit[0] or pair_keys[2] in low_bit[1] or pair_keys[3] in low_bit[1]:
                        inventory["low_bit"].append(f"{core}.[down+up]")
                        progress.update(1)
                        continue
                    down_shape_a = tuple(handlers[0].get_shape(pair_keys[0]))
                    up_shape_a = tuple(handlers[0].get_shape(pair_keys[1]))
                    down_shape_b = tuple(handlers[1].get_shape(pair_keys[2]))
                    up_shape_b = tuple(handlers[1].get_shape(pair_keys[3]))
                    pair_units = [
                        ("down", pair_keys[0], pair_keys[2]),
                        ("up", pair_keys[1], pair_keys[3]),
                    ]
                    loaded_items = _load_paired_unit(
                        handlers,
                        pair_units,
                        pin_memory=str(params["process_device"]).startswith("cuda"),
                    )
                    loaded = {
                        role: (tensor_a, tensor_b)
                        for role, _, _, tensor_a, tensor_b in loaded_items
                    }
                    tensors = (
                        loaded["down"][0], loaded["up"][0],
                        loaded["down"][1], loaded["up"][1],
                    )
                    try:
                        raw_down, top_down, raw_up, top_up, cwb, fallback = (
                            _analyze_lora_with_fallback(
                                core, tensors, params["process_device"],
                                similarity_alignment, top_limit,
                            )
                        )
                        add_standard_record(
                            f"{core}.down", down_shape_a, tensors[0].dtype, tensors[2].dtype,
                            raw_down, CWBStats(), top_down, fallback, include_cwb=False,
                        )
                        add_standard_record(
                            f"{core}.up", up_shape_a, tensors[1].dtype, tensors[3].dtype,
                            raw_up, CWBStats(), top_up, fallback, include_cwb=False,
                        )
                        synthetic = TensorRecord(
                            key=f"{core}.[down+up]", shape=down_shape_a,
                            dtype_a=str(tensors[0].dtype), dtype_b=str(tensors[2].dtype),
                            raw=RawStats(), cwb=cwb, cpu_fallback=fallback,
                        )
                        cwb_records.append(synthetic)
                        block_cwb.setdefault(_block_name(core), CWBStats()).add(cwb)
                        if fallback:
                            fallback_keys.append(synthetic.key)
                    finally:
                        del tensors
                        loaded.clear()
                        loaded_items.clear()
                        handlers[0].mark_processed(pair_keys[0])
                        handlers[0].mark_processed(pair_keys[1])
                        handlers[1].mark_processed(pair_keys[2])
                        handlers[1].mark_processed(pair_keys[3])
                        clear_after_unit()
                        progress.update(1)

            comparison = _render_comparison(
                model_a, model_b, model_type, records, global_raw, layer_stats,
                block_stats, inventory, top_heap, fallback_keys,
                handlers[0].metadata(), handlers[1].metadata(),
            )
            cwb_report = _render_cwb(
                model_a, model_b, cwb_records, block_cwb, fallback_keys,
                (
                    "paired LoRA rank components with greedy alignment"
                    if lora_mode and similarity_alignment
                    else "paired LoRA rank components (index only)"
                    if lora_mode
                    else "embedding rows with greedy alignment"
                    if embedding_alignment and similarity_alignment
                    else "embedding rows (index only)"
                    if embedding_alignment else "fixed tensor rows (index only)"
                ),
            )
            return comparison, cwb_report
        finally:
            for handler in handlers:
                handler.__exit__(None, None, None)
            cleanup_after_operation()


def _common_inputs(model_type: str, *, alignment_control: bool):
    inputs = [
        io.Combo.Input("execution_mode", options=["ANALYZE", "DOCUMENTATION ONLY"]),
        io.Combo.Input("model_a", options=folder_paths.get_filename_list(model_type)),
        io.Combo.Input("model_b", options=folder_paths.get_filename_list(model_type)),
    ]
    if alignment_control:
        inputs.append(io.Boolean.Input(
            "cwb_similarity_alignment",
            default=False,
            tooltip=(
                "Run threshold-independent CWB-style greedy one-to-one alignment. "
                "This is quadratic and can be slow for large embeddings."
            ),
        ))
    inputs.extend([
        io.Int.Input("top_weight_differences", default=20, min=0, max=1000),
        io.Combo.Input("process_device", options=["cuda", "cpu"]),
        io.Boolean.Input("force_clear_cache", default=True),
    ])
    return inputs


class _ModelAnalysisNode(io.ComfyNode):
    MODEL_TYPE = "diffusion_models"
    NODE_ID = ""
    DISPLAY_NAME = ""
    EMBEDDING_ALIGNMENT = False
    LORA_MODE = False

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="ModelUtils/Analysis",
            inputs=_common_inputs(
                cls.MODEL_TYPE,
                alignment_control=cls.EMBEDDING_ALIGNMENT or cls.LORA_MODE,
            ),
            outputs=[
                io.String.Output(display_name="comparison_report"),
                io.String.Output(display_name="cwb_report"),
                io.String.Output(display_name="documentation"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        documentation = load_documentation_from_file("model_analysis.md")
        if kwargs["execution_mode"] == "DOCUMENTATION ONLY":
            return io.NodeOutput(
                "Documentation mode active. No comparison performed.",
                "Documentation mode active. No CWB analysis performed.",
                documentation,
            )
        comparison, cwb_report = ModelAnalysisLogic.execute(
            kwargs["model_a"], kwargs["model_b"], cls.MODEL_TYPE, kwargs,
            embedding_alignment=cls.EMBEDDING_ALIGNMENT,
            lora_mode=cls.LORA_MODE,
        )
        return io.NodeOutput(comparison, cwb_report, documentation)


class CheckpointModelAnalysis(_ModelAnalysisNode):
    MODEL_TYPE = "checkpoints"
    NODE_ID = "CheckpointModelAnalysis"
    DISPLAY_NAME = "Analyze Checkpoint Similarity (2 Models)"


class DiffusionModelAnalysis(_ModelAnalysisNode):
    NODE_ID = "DiffusionModelAnalysis"
    DISPLAY_NAME = "Analyze Diffusion Model Similarity (2 Models)"


class TextEncoderModelAnalysis(_ModelAnalysisNode):
    MODEL_TYPE = "text_encoders"
    NODE_ID = "TextEncoderModelAnalysis"
    DISPLAY_NAME = "Analyze Text Encoder Similarity (2 Models)"


class LoRAModelAnalysis(_ModelAnalysisNode):
    MODEL_TYPE = "loras"
    NODE_ID = "LoRAModelAnalysis"
    DISPLAY_NAME = "Analyze LoRA Similarity (2 Models)"
    LORA_MODE = True


class EmbeddingModelAnalysis(_ModelAnalysisNode):
    MODEL_TYPE = "embeddings"
    NODE_ID = "EmbeddingModelAnalysis"
    DISPLAY_NAME = "Analyze Embedding Similarity (2 Models)"
    EMBEDDING_ALIGNMENT = True


MODEL_ANALYSIS_NODES = [
    CheckpointModelAnalysis,
    DiffusionModelAnalysis,
    TextEncoderModelAnalysis,
    LoRAModelAnalysis,
    EmbeddingModelAnalysis,
]
