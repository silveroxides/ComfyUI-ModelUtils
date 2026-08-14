import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "model_analysis_api_runner_test",
        REPO_ROOT / "run_model_analysis_api_workflow.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linked_previews_require_both_reports(runner):
    workflow = {
        "1": {"class_type": "LoRAModelAnalysis", "inputs": {}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
        "3": {"class_type": "PreviewAny", "inputs": {"source": ["1", 1]}},
    }
    assert runner._linked_previews(workflow, "1") == {0: "2", 1: "3"}


def test_linked_previews_reject_missing_report(runner):
    workflow = {
        "1": {"class_type": "LoRAModelAnalysis", "inputs": {}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
    }
    with pytest.raises(RuntimeError, match="output"):
        runner._linked_previews(workflow, "1")


def test_run_analysis_captures_both_reports(monkeypatch, runner):
    workflow = {
        "1": {"class_type": "LoRAModelAnalysis", "inputs": {}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
        "3": {"class_type": "PreviewAny", "inputs": {"source": ["1", 1]}},
    }
    monkeypatch.setattr(runner.api, "_json_request", lambda *args: {"prompt_id": "p"})
    monkeypatch.setattr(
        runner.api,
        "_wait_for_history",
        lambda *args: {"outputs": {"2": {"text": ["standard"]}, "3": {"text": ["cwb"]}}},
    )
    result = runner.run_analysis(workflow, "http://server", 0.1)
    assert result == {
        "prompt_id": "p",
        "comparison_report": "standard",
        "cwb_report": "cwb",
    }
