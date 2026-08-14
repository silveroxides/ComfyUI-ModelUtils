import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "cwb_preset_sweep_test", REPO_ROOT / "run_cwb_preset_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep():
    return _load_sweep()


@pytest.fixture
def baseline():
    return {
        "consensus_type": "median",
        "alignment_method": "similarity",
        "alignment_threshold": 0.0,
        "similarity_threshold": 0.0,
        "power_alpha": 2.0,
        "diversity_beta": 10.0,
        "rescale_norm": True,
        "global_scale": 1.0,
        "dynamic_similarity_contrast": False,
        "soft_comfort_bandpass": True,
        "position_weight": 0.05,
        "preserve_common_prefix": False,
    }


def test_cases_are_complete_and_change_exactly_one_factor(sweep, baseline):
    cases = sweep.build_sweep_cases(baseline)
    assert cases[0]["id"] == "baseline"
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert tuple(case["config"]) == sweep.CONFIG_KEYS
        changed = [
            key for key in sweep.CONFIG_KEYS
            if case["config"][key] != baseline[key]
        ]
        assert changed == ([] if case["id"] == "baseline" else [case["factor"]])


def test_sweep_exposes_expected_full_block_finalist_ids(sweep, baseline):
    ids = {case["id"] for case in sweep.build_sweep_cases(baseline)}
    assert {
        "baseline",
        "alignment_method_index",
        "alignment_threshold_0p0005",
        "alignment_threshold_0p0025",
        "consensus_type_mean",
        "diversity_beta_4p0",
        "rescale_norm_false",
        "dynamic_similarity_contrast_true",
        "soft_comfort_bandpass_false",
    } <= ids


def test_report_parser_extracts_influence_metrics(sweep):
    report = """CWB MERGE SUMMARY
Alignment candidates: 100
Alignment matches: 75
Anchor-only groups: 5
Weighted vector groups: 20
All-rejected equal fallbacks: 1
Zero-weight equal fallbacks: 2
Accepted weighting contributors: 50/80
Consensus similarity min/mean/max: -0.5 / 0.25 / 1
Dominant normalized weight min/mean/max: 0.2 / 0.4 / 0.8
Effective contributors min/mean/max: 1.2 / 2.5 / 5
CUDA OOM CPU fallbacks: 0
LoRA groups: 10
Rank-one delta norm ratio count/min/mean/max: 3 / 0.7 / 1.1 / 1.4
Rank-one delta norm ratio count/min/mean/max: 1 / 0.8 / 0.9 / 1.2
"""
    metrics = sweep.parse_cwb_report(report)
    assert metrics["alignment_match_fraction"] == pytest.approx(0.75)
    assert metrics["accepted_weight_fraction"] == pytest.approx(0.625)
    assert metrics["dominant_weight_mean"] == pytest.approx(0.4)
    assert metrics["effective_contributors_mean"] == pytest.approx(2.5)
    assert metrics["norm_ratio_min"] == pytest.approx(0.7)
    assert metrics["norm_ratio_count"] == 4
    assert metrics["norm_ratio_mean"] == pytest.approx(1.05)
    assert metrics["norm_ratio_max"] == pytest.approx(1.4)


def test_report_parser_rejects_pre_diagnostic_server(sweep):
    with pytest.raises(RuntimeError, match="Restart ComfyUI"):
        sweep.parse_cwb_report("CWB MERGE SUMMARY\nAlignment matches: 1")


def test_analysis_parser_extracts_global_metrics(sweep):
    comparison = """MODEL COMPARISON SUMMARY
PARAMETER-WEIGHTED GLOBAL METRICS
relative_l2: 1.04
cosine: 0.46
exact: 0.01
norm_ratio: 0.89
EQUAL-LAYER AVERAGES
"""
    cwb_report = """GLOBAL CWB-DERIVED DIAGNOSTICS
Pairwise row cosine mean/min/max: 0.4 / -0.1 / 1
Mean-consensus affinity A/B: 0.2 / 0.3
Median-consensus affinity A/B: 0.25 / 0.35
"""
    metrics = sweep.parse_analysis_reports(comparison, cwb_report)
    assert metrics["analysis_relative_l2"] == pytest.approx(1.04)
    assert metrics["analysis_pair_cosine"] == pytest.approx(0.4)
    assert metrics["analysis_median_affinity_b"] == pytest.approx(0.35)
