"""Run a resumable one-factor CWB LoRA calibration sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import run_cwb_api_workflow as api
import run_model_analysis_api_workflow as analysis_api


CONFIG_KEYS = (
    "consensus_type",
    "alignment_method",
    "alignment_threshold",
    "similarity_threshold",
    "power_alpha",
    "diversity_beta",
    "rescale_norm",
    "global_scale",
    "dynamic_similarity_contrast",
    "soft_comfort_bandpass",
    "position_weight",
    "preserve_common_prefix",
)

INTEGER_METRICS = {
    "alignment_candidates": "Alignment candidates",
    "alignment_matches": "Alignment matches",
    "anchor_only_groups": "Anchor-only groups",
    "weighted_vector_groups": "Weighted vector groups",
    "all_rejected_fallbacks": "All-rejected equal fallbacks",
    "zero_weight_fallbacks": "Zero-weight equal fallbacks",
    "cuda_oom_fallbacks": "CUDA OOM CPU fallbacks",
    "lora_groups": "LoRA groups",
}

TRIPLE_METRICS = {
    "consensus_similarity": "Consensus similarity min/mean/max",
    "dominant_weight": "Dominant normalized weight min/mean/max",
    "effective_contributors": "Effective contributors min/mean/max",
}

DELTA_METRICS = (
    "alignment_match_fraction",
    "anchor_only_groups",
    "accepted_weight_fraction",
    "consensus_similarity_mean",
    "dominant_weight_mean",
    "dominant_weight_max",
    "effective_contributors_mean",
    "effective_contributors_min",
    "norm_ratio_mean",
    "norm_ratio_min",
    "norm_ratio_max",
    "all_rejected_fallbacks",
    "zero_weight_fallbacks",
    "wall_seconds",
    "analysis_relative_l2",
    "analysis_cosine",
    "analysis_exact",
    "analysis_norm_ratio",
    "analysis_pair_cosine",
    "analysis_mean_affinity_a",
    "analysis_mean_affinity_b",
    "analysis_median_affinity_a",
    "analysis_median_affinity_b",
)


def _single_custom_node(workflow: dict) -> tuple[str, dict]:
    return api._single_node(workflow, api.CUSTOM_CLASS)


def _baseline_config(workflow: dict) -> dict:
    _, node = _single_custom_node(workflow)
    inputs = node.get("inputs", {})
    missing = [key for key in CONFIG_KEYS if key not in inputs]
    if missing:
        raise RuntimeError(
            "Custom configuration is missing required inputs: " + ", ".join(missing)
        )
    return {key: inputs[key] for key in CONFIG_KEYS}


def build_sweep_cases(baseline: dict) -> list[dict]:
    variations = [
        ("alignment_method", "index"),
        *[("alignment_threshold", value) for value in (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01)],
        *[("position_weight", value) for value in (0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01)],
        ("consensus_type", "mean"),
        *[("similarity_threshold", value) for value in (-0.25, 0.1, 0.25, 0.5, 0.75)],
        *[("power_alpha", value) for value in (0.5, 1.0, 3.0, 4.0)],
        *[("diversity_beta", value) for value in (0.0, 1.0, 2.0, 4.0, 7.0)],
        ("rescale_norm", False),
        ("dynamic_similarity_contrast", True),
        ("soft_comfort_bandpass", False),
    ]
    cases = [{"id": "baseline", "factor": "baseline", "value": None, "config": baseline.copy()}]
    for factor, value in variations:
        if baseline[factor] == value:
            continue
        config = baseline.copy()
        config[factor] = value
        value_text = str(value).lower().replace(".", "p").replace("-", "neg")
        cases.append({
            "id": f"{factor}_{value_text}",
            "factor": factor,
            "value": value,
            "config": config,
        })
    return cases


def parse_cwb_report(report: str) -> dict:
    metrics = {}
    for metric, label in INTEGER_METRICS.items():
        match = re.search(rf"^{re.escape(label)}: (\d+)$", report, re.MULTILINE)
        if match:
            metrics[metric] = int(match.group(1))

    accepted = re.search(
        r"^Accepted weighting contributors: (\d+)/(\d+)$",
        report,
        re.MULTILINE,
    )
    if not accepted:
        raise RuntimeError(
            "CWB report lacks contributor-weight diagnostics. Restart ComfyUI "
            "after loading the current consensus_merger.py before running the sweep."
        )
    metrics["accepted_weighting_contributors"] = int(accepted.group(1))
    metrics["weighting_contributors"] = int(accepted.group(2))

    number = r"(?:n/a|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
    for prefix, label in TRIPLE_METRICS.items():
        match = re.search(
            rf"^{re.escape(label)}: ({number}) / ({number}) / ({number})$",
            report,
            re.MULTILINE | re.IGNORECASE,
        )
        if not match:
            raise RuntimeError(f"CWB report is missing metric: {label}")
        for suffix, value in zip(("min", "mean", "max"), match.groups()):
            metrics[f"{prefix}_{suffix}"] = None if value == "n/a" else float(value)

    norm_groups = re.findall(
        rf"^Rank-one delta norm ratio count/min/mean/max: (\d+) / "
        rf"({number}) / ({number}) / ({number})$",
        report,
        re.MULTILINE | re.IGNORECASE,
    )
    numeric_norms = [
        (int(count), *(float(value) for value in (minimum, mean, maximum)))
        for count, minimum, mean, maximum in norm_groups
        if count != "0" and "n/a" not in (minimum, mean, maximum)
    ]
    if numeric_norms:
        norm_count = sum(group[0] for group in numeric_norms)
        metrics["norm_ratio_count"] = norm_count
        metrics["norm_ratio_min"] = min(group[1] for group in numeric_norms)
        metrics["norm_ratio_mean"] = (
            sum(group[0] * group[2] for group in numeric_norms) / norm_count
        )
        metrics["norm_ratio_max"] = max(group[3] for group in numeric_norms)

    candidates = metrics.get("alignment_candidates", 0)
    contributors = metrics["weighting_contributors"]
    metrics["alignment_match_fraction"] = (
        metrics.get("alignment_matches", 0) / candidates if candidates else 0.0
    )
    metrics["accepted_weight_fraction"] = (
        metrics["accepted_weighting_contributors"] / contributors
        if contributors else 0.0
    )
    return metrics


def parse_analysis_reports(comparison: str, cwb_report: str) -> dict:
    global_section = comparison.split("PARAMETER-WEIGHTED GLOBAL METRICS", 1)[1].split(
        "EQUAL-LAYER AVERAGES", 1
    )[0]

    def single(label: str, text: str) -> float:
        match = re.search(rf"^{re.escape(label)}: ([-+0-9.e]+)$", text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Analysis report is missing metric: {label}")
        return float(match.group(1))

    metrics = {
        "analysis_relative_l2": single("relative_l2", global_section),
        "analysis_cosine": single("cosine", global_section),
        "analysis_exact": single("exact", global_section),
        "analysis_norm_ratio": single("norm_ratio", global_section),
    }
    pairwise = re.search(
        r"^Pairwise row cosine mean/min/max: ([-+0-9.e]+) /",
        cwb_report,
        re.MULTILINE,
    )
    mean_affinity = re.search(
        r"^Mean-consensus affinity A/B: ([-+0-9.e]+) / ([-+0-9.e]+)$",
        cwb_report,
        re.MULTILINE,
    )
    median_affinity = re.search(
        r"^Median-consensus affinity A/B: ([-+0-9.e]+) / ([-+0-9.e]+)$",
        cwb_report,
        re.MULTILINE,
    )
    if not pairwise or not mean_affinity or not median_affinity:
        raise RuntimeError("CWB analysis report is missing global affinity metrics")
    metrics.update({
        "analysis_pair_cosine": float(pairwise.group(1)),
        "analysis_mean_affinity_a": float(mean_affinity.group(1)),
        "analysis_mean_affinity_b": float(mean_affinity.group(2)),
        "analysis_median_affinity_a": float(median_affinity.group(1)),
        "analysis_median_affinity_b": float(median_affinity.group(2)),
    })
    return metrics


def _run_case(
    workflow_template: dict,
    case: dict,
    server: str,
    poll_seconds: float,
    full_blocks: bool,
    analysis_workflow: dict | None = None,
) -> dict:
    workflow = deepcopy(workflow_template)
    _, custom = _single_custom_node(workflow)
    custom["inputs"].update(case["config"])
    _, linked = api._prepare_workflow(
        workflow,
        preset=None,
        output_filename=None,
        full_blocks=full_blocks,
    )
    started = time.perf_counter()
    queued = api._json_request(f"{server}/prompt", api._prompt_payload(workflow))
    prompt_id = queued["prompt_id"]
    history = api._wait_for_history(server, prompt_id, poll_seconds)
    wall_seconds = time.perf_counter() - started
    status = history.get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(
            f"Sweep case {case['id']} failed: {json.dumps(status, indent=2)}"
        )
    outputs = history.get("outputs", {})
    output_filename = api._text_output(outputs, linked[0])
    report = api._text_output(outputs, linked[2])
    metrics = parse_cwb_report(report)
    metrics["wall_seconds"] = wall_seconds
    result = {
        **case,
        "prompt_id": prompt_id,
        "output_filename": output_filename,
        "metrics": metrics,
        "cwb_report": report,
    }
    if analysis_workflow is not None:
        analysis_result = analysis_api.run_analysis(
            deepcopy(analysis_workflow), server, poll_seconds
        )
        metrics.update(parse_analysis_reports(
            analysis_result["comparison_report"],
            analysis_result["cwb_report"],
        ))
        result["analysis"] = analysis_result
    return result


def _load_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    completed = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                completed.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid sweep JSONL at line {line_number}: {exc}"
                ) from exc
    return completed


def _append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_influence_csv(path: Path, results: list[dict]) -> None:
    baseline = next((result for result in results if result["id"] == "baseline"), None)
    if baseline is None:
        return
    fields = ["id", "factor", "value", *DELTA_METRICS]
    delta_fields = [f"delta_{metric}" for metric in DELTA_METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, *delta_fields])
        writer.writeheader()
        base_metrics = baseline["metrics"]
        for result in results:
            row = {key: result.get(key) for key in ("id", "factor", "value")}
            for metric in DELTA_METRICS:
                value = result["metrics"].get(metric)
                base = base_metrics.get(metric)
                row[metric] = value
                row[f"delta_{metric}"] = (
                    value - base if value is not None and base is not None else None
                )
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a resumable one-factor CWB LoRA parameter sweep."
    )
    parser.add_argument("--workflow", type=Path, default=api.DEFAULT_WORKFLOW)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("audit/cwb_sweep"))
    parser.add_argument("--full-blocks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run only the named case. Repeat to select multiple cases.",
    )
    parser.add_argument(
        "--analysis-workflow",
        type=Path,
        default=analysis_api.DEFAULT_WORKFLOW,
    )
    parser.add_argument(
        "--analysis-every",
        type=int,
        default=5,
        help="Run candidate-versus-anchor analysis for baseline and every Nth case.",
    )
    args = parser.parse_args()

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    analysis_workflow = json.loads(args.analysis_workflow.read_text(encoding="utf-8"))
    if args.full_blocks:
        _, analysis_node = api._single_node(
            analysis_workflow,
            analysis_api.ANALYSIS_CLASS,
        )
        analysis_node.setdefault("inputs", {})["exclude_patterns"] = ""
    cases = build_sweep_cases(_baseline_config(workflow))
    if args.case_ids:
        known_ids = {case["id"] for case in cases}
        unknown = [case_id for case_id in args.case_ids if case_id not in known_ids]
        if unknown:
            raise RuntimeError("Unknown sweep case(s): " + ", ".join(unknown))
        selected = set(args.case_ids)
        cases = [case for case in cases if case["id"] in selected]
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.dry_run:
        for case in cases:
            sys.stdout.write(f"{case['id']}: {case['factor']}={case['value']}\n")
        sys.stdout.write(f"Cases: {len(cases)}\n")
        return 0

    results_path = args.output_dir / "runs.jsonl"
    completed = _load_completed(results_path)
    completed_ids = {result["id"] for result in completed}
    server = args.server.rstrip("/")
    if args.analysis_every < 1:
        raise RuntimeError("--analysis-every must be at least 1")
    for index, case in enumerate(cases, start=1):
        if case["id"] in completed_ids:
            sys.stdout.write(f"[{index}/{len(cases)}] skip {case['id']}\n")
            continue
        sys.stdout.write(f"[{index}/{len(cases)}] run {case['id']}\n")
        result = _run_case(
            workflow,
            case,
            server,
            args.poll_seconds,
            args.full_blocks,
            analysis_workflow=(
                analysis_workflow
                if index == 1 or index % args.analysis_every == 0
                else None
            ),
        )
        _append_result(results_path, result)
        completed.append(result)
        completed_ids.add(case["id"])
        sys.stdout.write(
            f"  {result['metrics']['wall_seconds']:.1f}s, "
            f"match={result['metrics']['alignment_match_fraction']:.4f}, "
            f"effective={result['metrics']['effective_contributors_mean']:.4f}\n"
        )
        _write_influence_csv(args.output_dir / "influence.csv", completed)

    _write_influence_csv(args.output_dir / "influence.csv", completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
