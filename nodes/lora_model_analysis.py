"""Read-only, bounded analysis of a LoRA's effect on a diffusion model."""

from contextlib import closing
import csv
import gc
import heapq
from io import StringIO
import logging

import folder_paths
import torch
from comfy.weight_adapter import LoRAAdapter
from comfy_api.latest import io
from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned

from .device_utils import cleanup_after_operation
from .lora_alpha import normalize_lora_pair
from .lora_resize import (
    AdapterErrorCapture, build_lora_layer_map, canonical_lora_block_name,
    layer_tensor_keys, parse_lora_layers, validate_canonical_blocks,
)
from .merger import _compile_patterns, _matches_any_pattern, load_documentation_from_file
from .model_analysis import (
    CWBStats, RawStats, TensorRecord, _block_name, _common_inputs,
    _compute_pair_on_device, _fmt, _is_cuda_oom, _is_float_dtype, _layer_name,
    _metrics, _release_cuda_oom, _render_comparison, _render_cwb,
)
from .quantization_guard import inspect_low_bit_input
from .uel_io import stream_work_units


def _plan_application(base, lora, base_path, params):
    pairs, passthrough = parse_lora_layers(lora.keys())
    validate_canonical_blocks(pairs, "LoRA-on-model analysis")
    layer_map = build_lora_layer_map([{"pairs": pairs}], base_path)
    targets = {canonical_lora_block_name(key): key for key in base.keys()}
    patches = {}
    mapped = set()
    unmapped_targets = []
    for core, sources in layer_map.items():
        for _, block in sources:
            roles = pairs[block]
            weight_role = next((role for role in ("set_weight", "diff", "w_norm") if role in roles), None)
            specs = []
            if weight_role is not None:
                target = targets.get(core) if weight_role == "diff" else None
                specs.append((target or targets.get(f"{core}.weight"), weight_role))
            elif "down" in roles and "up" in roles:
                specs.append((targets.get(f"{core}.weight"), "pair"))
            bias_role = next((role for role in ("diff_b", "b_norm") if role in roles), None)
            if bias_role:
                specs.append((targets.get(f"{core}.bias"), bias_role))
            for target, role in specs:
                if target is not None:
                    patches.setdefault(target, []).append((block, roles, role))
                    mapped.add(block)
                else:
                    unmapped_targets.append(f"{block} ({role}: no base target)")
    inventory = {name: [] for name in ("a_only", "b_only", "shape", "nonfloat", "low_bit", "excluded_a", "excluded_b")}
    unmatched = sorted(set(pairs) - mapped) + unmapped_targets
    patterns = _compile_patterns(params.get("exclude_patterns", ""), glob_mode=params.get("glob_patterns", False))
    low_base = inspect_low_bit_input(base, "Base diffusion model", "LoRA-on-model analysis")
    low_lora = inspect_low_bit_input(lora, "LoRA", "LoRA-on-model analysis")
    units = []
    for key in sorted(base.keys()):
        if _matches_any_pattern(key, patterns, glob_mode=params.get("glob_patterns", False)) != params.get("include_mode", False):
            inventory["excluded_a"].append(key)
            inventory["excluded_b"].append(key)
            continue
        specs = patches.get(key, [])
        source_keys = list(dict.fromkeys(
            source_key for _, roles, role in specs
            for source_key in (layer_tensor_keys(roles).values() if role == "pair" else (roles[role],))
        ))
        alpha_keys = {roles["alpha"] for _, roles, role in specs if role == "pair" and "alpha" in roles}
        if key in low_base or any(source in low_lora and source not in alpha_keys for source in source_keys):
            inventory["low_bit"].append(key)
            continue
        if not _is_float_dtype(base.get_dtype(key)):
            inventory["nonfloat"].append(key)
            continue
        entries = {0: [key]}
        if source_keys:
            entries[1] = source_keys
        units.append((key, entries))
    return units, patches, inventory, unmatched, passthrough


def _apply_pair(key, weight, block, roles, loaded, strength):
    tensors = {source_key: loaded[(1, source_key)] for source_key in layer_tensor_keys(roles).values()}
    down = up = adapter = None
    try:
        down, up = normalize_lora_pair(tensors[roles["down"]], tensors[roles["up"]], tensors.get(roles.get("alpha")), layer=block)
        tensors[roles["down"]], tensors[roles["up"]] = down, up
        tensors.pop(roles.get("alpha"), None)
        adapter = LoRAAdapter.load(block, tensors, None, tensors.get(roles.get("dora_scale")), set())
        if adapter is None:
            raise ValueError(f"LoRA-on-model analysis: could not load adapter {block!r} for {key!r}.")
        capture = AdapterErrorCapture(key)
        logger = logging.getLogger()
        logger.addHandler(capture)
        try:
            weight = adapter.calculate_weight(weight, key, strength, 1.0, None, lambda value: value, torch.float32, None)
        finally:
            logger.removeHandler(capture)
        if capture.messages:
            raise RuntimeError("; ".join(capture.messages))
        return weight
    finally:
        tensors.clear()
        del adapter, down, up, weight


def _apply_and_analyze(key, loaded, specs, strength, device, top_count):
    source = loaded[(0, key)]
    original = patched = delta = None
    try:
        original = (
            transfer_to_gpu_pinned(source, device=device, dtype=torch.float32)
            if str(device).startswith("cuda") else source.to(device=device, dtype=torch.float32)
        )
        patched = original.clone()
        for block, roles, role in specs:
            if role == "pair":
                patched = _apply_pair(key, patched, block, roles, loaded, strength)
                continue
            delta = loaded[(1, roles[role])].to(device=device, dtype=torch.float32)
            if delta.shape != patched.shape:
                raise ValueError(f"LoRA-on-model analysis: {roles[role]} shape {tuple(delta.shape)} does not match {key} {tuple(patched.shape)}.")
            if role == "set_weight":
                patched.copy_(delta)
            else:
                patched.add_(delta, alpha=strength)
            delta = None
        if patched.shape != original.shape:
            raise ValueError(f"LoRA-on-model analysis: applying {key!r} changed its shape; it cannot be compared with the base tensor.")
        return _compute_pair_on_device(original, patched, device, False, top_count)
    finally:
        del source, original, patched, delta


def _analyze_unit(key, loaded, specs, strength, device, top_count):
    try:
        return (*_apply_and_analyze(key, loaded, specs, strength, device, top_count), False)
    except Exception as error:
        if not str(device).startswith("cuda") or not _is_cuda_oom(error):
            raise
        error.__traceback__ = None
        _release_cuda_oom()
        try:
            return (*_apply_and_analyze(key, loaded, specs, strength, "cpu", top_count), True)
        except Exception as cpu_error:
            raise RuntimeError(f"LoRA-on-model analysis CPU fallback failed for {key!r}: {cpu_error}") from cpu_error


def _csv(headers, rows):
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()


def analyze_lora_on_model(model_a, model_b, params):
    lora_path = folder_paths.get_full_path_or_raise("loras", model_a)
    base_path = folder_paths.get_full_path_or_raise("diffusion_models", model_b)
    strength = params.get("strength", 1.0)
    records, fallback_keys, top_heap = [], [], []
    global_raw = RawStats()
    layer_stats, block_stats, block_cwb = {}, {}, {}
    try:
        with MemoryEfficientSafeOpen(base_path, low_memory=True) as base, MemoryEfficientSafeOpen(lora_path, low_memory=True) as lora:
            units, patches, inventory, unmatched, passthrough = _plan_application(base, lora, base_path, params)
            device = params["process_device"]
            top_limit = params["top_weight_differences"]
            with closing(stream_work_units({0: base, 1: lora}, units, pin_memory=str(device).startswith("cuda"))) as stream:
                for key, loaded in stream:
                    try:
                        raw, cwb, top, fallback = _analyze_unit(key, loaded, patches.get(key, []), strength, device, top_limit)
                        shape = tuple(base.get_shape(key))
                        records.append(TensorRecord(key, shape, str(base.get_dtype(key)), str(torch.float32), raw, cwb, fallback))
                        global_raw.add(raw)
                        layer_stats.setdefault(_layer_name(key), RawStats()).add(raw)
                        block_stats.setdefault(_block_name(key), RawStats()).add(raw)
                        block_cwb.setdefault(_block_name(key), CWBStats()).add(cwb)
                        if fallback:
                            fallback_keys.append(key)
                        for value, index, a, b in top:
                            entry = (value, key, index, a, b, shape)
                            if len(top_heap) < top_limit:
                                heapq.heappush(top_heap, entry)
                            elif top_limit and value > top_heap[0][0]:
                                heapq.heapreplace(top_heap, entry)
                    finally:
                        loaded.clear()
                        if params["force_clear_cache"]:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
            label = f"{model_b} + {model_a} (strength={strength})"
            comparison = _render_comparison(model_b, label, "LoRA applied to diffusion model", records, global_raw, layer_stats, block_stats, inventory, top_heap, fallback_keys, base.metadata(), base.metadata())
            comparison = (
                "# LoRA-on-model analysis\n\nModel A input is the LoRA; Model B input is the base diffusion model. "
                "The comparison below uses original base as A and patched base as B; delta = patched − original. "
                "Unmodified base tensors are included unless filtered. No model is saved.\n\n"
                + comparison
                + "\n## Unmapped LoRA groups\n\n" + ("\n".join(f"- `{name}`" for name in unmatched) or "None.")
                + "\n\n## Unused LoRA tensors\n\n" + ("\n".join(f"- `{name}`" for name in passthrough) or "None.") + "\n"
            )
            cwb_report = _render_cwb(model_b, label, records, block_cwb, fallback_keys, "original versus LoRA-patched base tensor rows (index only)")
            metric_names = ("mae", "mse", "rmse", "max", "relative_l2", "cosine", "pearson", "exact", "sign", "norm_ratio", "coverage")
            metrics_csv = _csv(["key", "parameters", *metric_names], ([record.key, record.raw.total, *(_fmt(_metrics(record.raw)[name]) for name in metric_names)] for record in records))
            cwb_names = ("pair_mean", "pair_min", "pair_max")
            cwb_csv = _csv(["key", *cwb_names], ([record.key, *(_fmt(record.cwb.values()[name]) for name in cwb_names)] for record in records))
            return comparison, cwb_report, metrics_csv, cwb_csv
    finally:
        cleanup_after_operation()


class LoRAOnModelAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = _common_inputs("diffusion_models", alignment_control=False)
        inputs[1] = io.Combo.Input("model_a", options=folder_paths.get_filename_list("loras"), tooltip="LoRA to apply transiently to Model B before comparing the original and patched base tensors.")
        inputs[2] = io.Combo.Input("model_b", options=folder_paths.get_filename_list("diffusion_models"), tooltip="Base diffusion model. Its original weights are compared against a transient LoRA-patched copy, one tensor at a time.")
        inputs.append(io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="LoRA application strength. Alpha scaling and DoRA reconstruction use ComfyUI's adapter; explicit set_weight patches retain their existing replacement semantics."))
        return io.Schema(
            node_id="LoRAOnModelAnalysis", display_name="Analyze LoRA Effect on Diffusion Model",
            category="ModelUtils/Analysis",
            description="Compare the base diffusion model with its LoRA-patched weights using bounded streaming. No model is written or inferred.",
            inputs=inputs,
            outputs=[io.String.Output(display_name="comparison_report"), io.String.Output(display_name="cwb_report"), io.String.Output(display_name="documentation"), io.String.Output(display_name="layerwise_metrics_csv"), io.String.Output(display_name="layerwise_cwb_csv")],
        )

    @classmethod
    def execute(cls, **kwargs):
        docs = load_documentation_from_file("lora_model_analysis.md")
        if kwargs["execution_mode"] == "DOCUMENTATION ONLY":
            return io.NodeOutput("Documentation mode active. No comparison performed.", "No CWB analysis performed.", docs, "", "")
        comparison, cwb, metrics_csv, cwb_csv = analyze_lora_on_model(kwargs["model_a"], kwargs["model_b"], kwargs)
        return io.NodeOutput(comparison, cwb, docs, metrics_csv, cwb_csv)
