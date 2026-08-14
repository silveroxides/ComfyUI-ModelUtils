"""Queue the CWB six-LoRA API workflow and print its merge report."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from urllib import error, request


DEFAULT_WORKFLOW = Path(__file__).with_name("CWB6LORATEST_API.json")
MERGER_CLASS = "CWBLoRAMultiMerger"
CUSTOM_CLASS = "CWBCustomConfiguration"
PREVIEW_CLASS = "PreviewAny"
CALIBRATION_EXCLUDE_PATTERN = (
    r"^diffusion_model\.blocks\.(?:[12]|[4-9]|1[0-3]|1[5-9]|2[0-3]|2[56])\."
    "\n"
    r"^diffusion_model\.txtfusion\.layerwise_blocks\.1\."
    "\n"
    r"^diffusion_model\.txtfusion\.refiner_blocks\.0\."
)


def parse_counterfactual_weight_sweep(report: str) -> list[dict]:
    pattern = re.compile(
        r"^(?P<consensus_type>mean|median) \| "
        r"(?P<similarity_threshold>[-+0-9.e]+) \| "
        r"(?P<power_alpha>[-+0-9.e]+) \| (?P<diversity_beta>[-+0-9.e]+) \| "
        r"(?P<dsc>true|false) \| (?P<softcb>true|false) \| "
        r"(?P<accepted>\d+)/(?P<contributors>\d+) \| (?P<fallbacks>\d+) \| "
        r"(?P<dominant_mean>[-+0-9.e]+)/(?P<dominant_max>[-+0-9.e]+) \| "
        r"(?P<effective_mean>[-+0-9.e]+)/(?P<effective_min>[-+0-9.e]+)$",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(report):
        values = match.groupdict()
        rows.append({
            "consensus_type": values["consensus_type"],
            "similarity_threshold": float(values["similarity_threshold"]),
            "power_alpha": float(values["power_alpha"]),
            "diversity_beta": float(values["diversity_beta"]),
            "dynamic_similarity_contrast": values["dsc"] == "true",
            "soft_comfort_bandpass": values["softcb"] == "true",
            "accepted_contributors": int(values["accepted"]),
            "contributors": int(values["contributors"]),
            "fallbacks": int(values["fallbacks"]),
            "dominant_weight_mean": float(values["dominant_mean"]),
            "dominant_weight_max": float(values["dominant_max"]),
            "effective_contributors_mean": float(values["effective_mean"]),
            "effective_contributors_min": float(values["effective_min"]),
        })
    return rows


def write_counterfactual_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("CWB report does not contain a counterfactual weight sweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_request(url: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI request failed ({exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach ComfyUI at {url}: {exc.reason}") from exc


def _single_node(workflow: dict, class_type: str) -> tuple[str, dict]:
    matches = [
        (node_id, node)
        for node_id, node in workflow.items()
        if node.get("class_type") == class_type
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {class_type} node, found {len(matches)}."
        )
    return matches[0]


def _linked_preview_nodes(workflow: dict, merger_id: str) -> dict[int, str]:
    linked = {}
    for node_id, node in workflow.items():
        if node.get("class_type") != PREVIEW_CLASS:
            continue
        source = node.get("inputs", {}).get("source")
        if (
            isinstance(source, list)
            and len(source) == 2
            and str(source[0]) == merger_id
            and isinstance(source[1], int)
        ):
            linked[source[1]] = node_id
    missing = {0, 2} - linked.keys()
    if missing:
        raise RuntimeError(
            "Workflow must connect merger output 0 (filename) and output 2 "
            "(CWB report) to PreviewAny nodes."
        )
    return linked


def _prepare_workflow(
    workflow: dict,
    preset: str | None,
    output_filename: str | None,
    full_blocks: bool = False,
) -> tuple[str, dict[int, str]]:
    merger_id, merger = _single_node(workflow, MERGER_CLASS)
    _single_node(workflow, CUSTOM_CLASS)
    inputs = merger.setdefault("inputs", {})

    if preset is not None:
        inputs["cwb_preset"] = preset
        inputs.pop("cwb_config", None)
    if output_filename is not None:
        inputs["output_filename"] = output_filename
    if not full_blocks:
        existing = inputs.get("exclude_patterns", "").strip()
        inputs["exclude_patterns"] = "\n".join(
            value for value in (existing, CALIBRATION_EXCLUDE_PATTERN) if value
        )

    return merger_id, _linked_preview_nodes(workflow, merger_id)


def _prompt_payload(workflow: dict) -> dict:
    return {"prompt": workflow}


def _wait_for_history(server: str, prompt_id: str, poll_seconds: float):
    while True:
        history = _json_request(f"{server}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(poll_seconds)


def _text_output(history_outputs: dict, node_id: str) -> str:
    output = history_outputs.get(node_id, {})
    candidates = (
        output.get("text"),
        output.get("ui", {}).get("text") if isinstance(output.get("ui"), dict) else None,
    )
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            return str(candidate[0])
        if isinstance(candidate, str):
            return candidate
    raise RuntimeError(f"PreviewAny node {node_id} did not return text in API history.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the CWB six-LoRA API workflow and collect its report."
    )
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--preset",
        help="Disconnect the custom configuration and run this CWB preset.",
    )
    parser.add_argument(
        "--output-filename",
        help="Override the merger output filename without its extension.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optionally write prompt ID, filename, and report as JSON.",
    )
    parser.add_argument(
        "--full-blocks",
        action="store_true",
        help="Process all blocks for final verification instead of the calibration subset.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--counterfactual-csv", type=Path)
    parser.add_argument("--alignment-threshold", type=float)
    parser.add_argument("--alignment-method", choices=["index", "similarity"])
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be greater than zero")
    if not args.workflow.is_file():
        parser.error(f"workflow not found: {args.workflow}")

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    if args.alignment_threshold is not None:
        _, custom = _single_node(workflow, CUSTOM_CLASS)
        custom.setdefault("inputs", {})["alignment_threshold"] = (
            args.alignment_threshold
        )
    if args.alignment_method is not None:
        _, custom = _single_node(workflow, CUSTOM_CLASS)
        custom.setdefault("inputs", {})["alignment_method"] = args.alignment_method
    _, linked_outputs = _prepare_workflow(
        workflow,
        args.preset,
        args.output_filename,
        args.full_blocks,
    )
    server = args.server.rstrip("/")
    queued = _json_request(f"{server}/prompt", _prompt_payload(workflow))
    prompt_id = queued["prompt_id"]
    sys.stdout.write(f"Queued prompt {prompt_id}\n")

    history = _wait_for_history(server, prompt_id, args.poll_seconds)
    status = history.get("status", {})
    if status.get("status_str") == "error":
        sys.stderr.write(json.dumps(status, indent=2) + "\n")
        return 1

    outputs = history.get("outputs", {})
    output_filename = _text_output(outputs, linked_outputs[0])
    cwb_report = _text_output(outputs, linked_outputs[2])
    result = {
        "prompt_id": prompt_id,
        "output_filename": output_filename,
        "cwb_report": cwb_report,
    }
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.counterfactual_csv is not None:
        write_counterfactual_csv(
            args.counterfactual_csv,
            parse_counterfactual_weight_sweep(cwb_report),
        )

    sys.stdout.write(f"Output: {output_filename}\n")
    if not args.summary_only:
        sys.stdout.write(f"\n{cwb_report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
