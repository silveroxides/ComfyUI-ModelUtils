"""Queue a model-analysis API workflow and capture both report outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_cwb_api_workflow as api


DEFAULT_WORKFLOW = Path("ANALYZETESTCANDIDATE_API.json")
ANALYSIS_CLASS = "LoRAModelAnalysis"


def _linked_previews(workflow: dict, analysis_id: str) -> dict[int, str]:
    linked = {}
    for node_id, node in workflow.items():
        if node.get("class_type") != "PreviewAny":
            continue
        source = node.get("inputs", {}).get("source")
        if isinstance(source, list) and len(source) == 2 and str(source[0]) == analysis_id:
            linked[int(source[1])] = node_id
    missing = [index for index in (0, 1) if index not in linked]
    if missing:
        raise RuntimeError(f"Missing PreviewAny capture for analysis output(s): {missing}")
    return linked


def run_analysis(workflow: dict, server: str, poll_seconds: float) -> dict:
    analysis_id, _ = api._single_node(workflow, ANALYSIS_CLASS)
    linked = _linked_previews(workflow, analysis_id)
    queued = api._json_request(
        f"{server.rstrip('/')}/prompt",
        api._prompt_payload(workflow),
    )
    prompt_id = queued["prompt_id"]
    history = api._wait_for_history(server.rstrip("/"), prompt_id, poll_seconds)
    status = history.get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(f"Analysis failed: {json.dumps(status, indent=2)}")
    outputs = history.get("outputs", {})
    return {
        "prompt_id": prompt_id,
        "comparison_report": api._text_output(outputs, linked[0]),
        "cwb_report": api._text_output(outputs, linked[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a model-analysis API workflow.")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    result = run_analysis(workflow, args.server, args.poll_seconds)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    sys.stdout.write(f"Prompt: {result['prompt_id']}\n")
    sys.stdout.write("Analysis reports captured\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
