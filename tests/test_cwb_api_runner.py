import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "cwb_api_runner_test", REPO_ROOT / "run_cwb_api_workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.fixture
def workflow():
    return {
        "10": {
            "class_type": "CWBCustomConfiguration",
            "inputs": {"diversity_beta": 10.0},
        },
        "11": {
            "class_type": "CWBLoRAMultiMerger",
            "inputs": {
                "cwb_config": ["10", 0],
                "cwb_preset": "default",
                "output_filename": "original",
            },
        },
        "14": {"class_type": "PreviewAny", "inputs": {"source": ["11", 0]}},
        "15": {"class_type": "PreviewAny", "inputs": {"source": ["11", 2]}},
    }


def test_custom_export_remains_connected_by_default(runner, workflow):
    merger_id, linked = runner._prepare_workflow(workflow, None, None)
    assert merger_id == "11"
    assert workflow["11"]["inputs"]["cwb_config"] == ["10", 0]
    assert workflow["11"]["inputs"]["exclude_patterns"] == (
        runner.CALIBRATION_EXCLUDE_PATTERN
    )
    assert linked == {0: "14", 2: "15"}


def test_preset_run_disconnects_custom_config_and_overrides_filename(
    runner, workflow
):
    runner._prepare_workflow(workflow, "candidate_sim_mean", "candidate/output")
    inputs = workflow["11"]["inputs"]
    assert "cwb_config" not in inputs
    assert inputs["cwb_preset"] == "candidate_sim_mean"
    assert inputs["output_filename"] == "candidate/output"


def test_full_block_verification_does_not_add_calibration_exclusions(
    runner, workflow
):
    workflow["11"]["inputs"]["exclude_patterns"] = "existing"
    runner._prepare_workflow(workflow, None, None, full_blocks=True)
    assert workflow["11"]["inputs"]["exclude_patterns"] == "existing"


@pytest.mark.parametrize(
    ("block", "excluded"),
    [(block, block not in {0, 3, 14, 24, 27}) for block in range(28)],
)
def test_calibration_block_ranges_are_exact(runner, block, excluded):
    import re

    key = f"diffusion_model.blocks.{block}.attn.wq.lora_A.weight"
    patterns = runner.CALIBRATION_EXCLUDE_PATTERN.splitlines()
    assert any(re.search(pattern, key) for pattern in patterns) is excluded


@pytest.mark.parametrize(
    ("key", "excluded"),
    [
        ("diffusion_model.txtfusion.layerwise_blocks.1.attn.lora_A.weight", True),
        ("diffusion_model.txtfusion.refiner_blocks.0.attn.lora_A.weight", True),
        ("diffusion_model.txtfusion.layerwise_blocks.0.attn.lora_A.weight", False),
    ],
)
def test_calibration_txtfusion_exclusions_are_exact(runner, key, excluded):
    import re

    patterns = runner.CALIBRATION_EXCLUDE_PATTERN.splitlines()
    assert any(re.search(pattern, key) for pattern in patterns) is excluded


def test_requires_filename_and_report_capture(runner, workflow):
    del workflow["15"]
    with pytest.raises(RuntimeError, match="output 2"):
        runner._prepare_workflow(workflow, None, None)


def test_parses_counterfactual_weight_sweep(runner):
    report = """CWB COUNTERFACTUAL WEIGHT SWEEP
median | 0.1 | 2 | 4 | true | false | 12/18 | 3 | 0.6/1 | 2.4/1
"""
    rows = runner.parse_counterfactual_weight_sweep(report)
    assert rows == [{
        "consensus_type": "median",
        "similarity_threshold": 0.1,
        "power_alpha": 2.0,
        "diversity_beta": 4.0,
        "dynamic_similarity_contrast": True,
        "soft_comfort_bandpass": False,
        "accepted_contributors": 12,
        "contributors": 18,
        "fallbacks": 3,
        "dominant_weight_mean": 0.6,
        "dominant_weight_max": 1.0,
        "effective_contributors_mean": 2.4,
        "effective_contributors_min": 1.0,
    }]


def test_prompt_payload_contains_only_api_workflow(runner, workflow):
    payload = runner._prompt_payload(workflow)
    assert payload == {"prompt": workflow}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"text": ["value"]}, "value"),
        ({"text": "value"}, "value"),
        ({"ui": {"text": ["value"]}}, "value"),
    ],
)
def test_extracts_supported_preview_history_shapes(runner, output, expected):
    assert runner._text_output({"14": output}, "14") == expected
