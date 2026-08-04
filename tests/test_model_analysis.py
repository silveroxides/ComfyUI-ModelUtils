import importlib
import math
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_analysis_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def analysis():
    return _load_module("nodes.model_analysis")


def _params(**overrides):
    params = {
        "cwb_similarity_alignment": False,
        "top_weight_differences": 3,
        "process_device": "cpu",
        "force_clear_cache": False,
    }
    params.update(overrides)
    return params


def test_five_node_schema_contract(analysis):
    assert len(analysis.MODEL_ANALYSIS_NODES) == 5
    assert len({node.NODE_ID for node in analysis.MODEL_ANALYSIS_NODES}) == 5
    for node in analysis.MODEL_ANALYSIS_NODES:
        schema = node.define_schema()
        assert schema.category == "ModelUtils/Analysis"
        assert [item.id for item in schema.inputs[:3]] == [
            "execution_mode", "model_a", "model_b",
        ]
        assert schema.inputs[0].options == ["ANALYZE", "DOCUMENTATION ONLY"]
        assert "lazy_load" not in [item.id for item in schema.inputs]
        input_ids = [item.id for item in schema.inputs]
        if node in (analysis.LoRAModelAnalysis, analysis.EmbeddingModelAnalysis):
            assert input_ids[3] == "cwb_similarity_alignment"
        else:
            assert "cwb_similarity_alignment" not in input_ids
        assert not {
            "cwb_preset", "consensus_type", "alignment_threshold",
            "similarity_threshold", "power_alpha", "diversity_beta",
            "dynamic_similarity_contrast", "soft_comfort_bandpass",
            "position_weight", "preserve_common_prefix",
        } & set(input_ids)
        assert [item.display_name for item in schema.outputs] == [
            "comparison_report", "cwb_report", "documentation",
        ]


def test_documentation_mode_has_three_separate_outputs(analysis):
    result = analysis.DiffusionModelAnalysis.execute(execution_mode="DOCUMENTATION ONLY")
    assert len(result.result) == 3
    assert "No comparison performed" in result[0]
    assert "No CWB analysis performed" in result[1]
    assert result[2].startswith("# Two-Model Similarity Analysis")


def test_raw_metrics_have_independent_expected_values(analysis):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[1.0, 0.0], [5.0, 4.0]])
    raw, top = analysis._compute_raw(a, b, 2)
    metrics = analysis._metrics(raw)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(2.0))
    assert metrics["max"] == pytest.approx(2.0)
    assert metrics["exact"] == pytest.approx(0.5)
    assert len(top) == 2
    assert {entry[1] for entry in top} == {1, 2}


def test_cwb_derived_metrics_have_independent_expected_values(analysis):
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = analysis._compute_cwb(a, b, False).values()
    assert values["pair_mean"] == pytest.approx(0.5)
    assert values["pair_min"] == pytest.approx(0.0)
    assert values["pair_max"] == pytest.approx(1.0)
    assert values["mean_aff_a"] == pytest.approx((1.0 + 2 ** -0.5) / 2)
    assert values["mean_aff_b"] == pytest.approx((1.0 + 2 ** -0.5) / 2)
    assert values["median_aff_a"] == pytest.approx(0.5)
    assert values["median_aff_b"] == pytest.approx(0.5)
    assert values["alignment"] is None


def test_zero_constant_and_nonfinite_metrics_are_explicit(analysis):
    raw, _ = analysis._compute_raw(torch.zeros(3), torch.zeros(3), 0)
    metrics = analysis._metrics(raw)
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["pearson"] is None
    assert metrics["norm_ratio"] == pytest.approx(1.0)

    raw, _ = analysis._compute_raw(
        torch.tensor([1.0, float("nan"), float("inf")]),
        torch.tensor([2.0, 4.0, 5.0]),
        0,
    )
    metrics = analysis._metrics(raw)
    assert raw.finite == 1
    assert raw.nonfinite_a == 2
    assert metrics["coverage"] == pytest.approx(1 / 3)


def test_end_to_end_reports_and_topology_classification(monkeypatch, tmp_path, analysis):
    path_a = tmp_path / "a.safetensors"
    path_b = tmp_path / "b.safetensors"
    save_file({
        "model.blocks.0.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "only_a": torch.ones(1),
        "shape": torch.ones(2),
        "integer": torch.tensor([1, 2], dtype=torch.int64),
    }, str(path_a), metadata={"source": "a"})
    save_file({
        "model.blocks.0.weight": torch.tensor([[1.0, 0.0], [5.0, 4.0]]),
        "only_b": torch.ones(1),
        "shape": torch.ones(3),
        "integer": torch.tensor([1, 3], dtype=torch.int64),
    }, str(path_b), metadata={"source": "b"})
    paths = {"a": str(path_a), "b": str(path_b)}
    monkeypatch.setattr(analysis.folder_paths, "get_full_path", lambda _, name: paths[name])
    monkeypatch.setattr(analysis, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, "cleanup_after_operation", lambda: None)
    processed = []
    async_calls = []
    original_mark_processed = analysis.MemoryEfficientSafeOpen.mark_processed
    original_async_stream = analysis.MemoryEfficientSafeOpen.async_stream

    def track_processed(loader, key):
        processed.append((loader.filename, key))
        return original_mark_processed(loader, key)

    def track_async_stream(loader, keys, **kwargs):
        async_calls.append((loader.filename, tuple(keys), kwargs))
        yield from original_async_stream(loader, keys, **kwargs)

    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "mark_processed", track_processed)
    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "async_stream", track_async_stream)

    comparison, cwb_report = analysis.ModelAnalysisLogic.execute(
        "a", "b", "diffusion_models", _params()
    )
    assert "Comparable floating tensors: 1" in comparison
    assert "mae: 1" in comparison
    assert "mse: 2" in comparison
    assert "[A ONLY] (1)" in comparison
    assert "[B ONLY] (1)" in comparison
    assert "[SHAPE MISMATCH] (1)" in comparison
    assert "[NON-FLOATING] (1)" in comparison
    assert "model.blocks.0 |" in comparison
    assert "CWB-DERIVED SIMILARITY SUMMARY" in cwb_report
    assert "merge weighting" in cwb_report
    assert "PRESET-IMPLIED CONTRIBUTIONS" not in cwb_report
    assert "Average contribution" not in cwb_report
    assert len(processed) == 4
    assert {key for _, key in processed} == {"model.blocks.0.weight", "integer"}
    assert len(async_calls) == 4
    assert all(len(call[1]) == 1 for call in async_calls)
    assert all(call[2]["batch_size"] == 1 for call in async_calls)
    assert all(call[2]["prefetch_batches"] == 1 for call in async_calls)
    assert not list(tmp_path.glob("**/*analysis*"))


def test_lora_cross_format_keys_compare_as_one_logical_pair(monkeypatch, tmp_path, analysis):
    path_a = tmp_path / "lora_a.safetensors"
    path_b = tmp_path / "lora_b.safetensors"
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }, str(path_a))
    save_file({
        "base_model.model.diffusion_model.foo.lora_down.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "base_model.model.diffusion_model.foo.lora_up.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }, str(path_b))
    paths = {"a": str(path_a), "b": str(path_b)}
    monkeypatch.setattr(analysis.folder_paths, "get_full_path", lambda _, name: paths[name])
    monkeypatch.setattr(analysis, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, "cleanup_after_operation", lambda: None)
    processed = []
    async_calls = []
    original_mark_processed = analysis.MemoryEfficientSafeOpen.mark_processed
    original_async_stream = analysis.MemoryEfficientSafeOpen.async_stream

    def track_processed(loader, key):
        processed.append((loader.filename, key))
        return original_mark_processed(loader, key)

    def track_async_stream(loader, keys, **kwargs):
        async_calls.append((loader.filename, tuple(keys), kwargs))
        yield from original_async_stream(loader, keys, **kwargs)

    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "mark_processed", track_processed)
    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "async_stream", track_async_stream)

    comparison, cwb_report = analysis.ModelAnalysisLogic.execute(
        "a", "b", "loras", _params(), lora_mode=True,
    )
    assert "Comparable floating tensors: 2" in comparison
    assert "[A ONLY] (0)" in comparison
    assert "[B ONLY] (0)" in comparison
    assert "foo.down |" in comparison
    assert "foo.up |" in comparison
    assert "foo.[down+up] |" in cwb_report
    assert len(processed) == 4
    assert len({entry for entry in processed}) == 4
    assert len(async_calls) == 2
    assert all(len(call[1]) == 2 for call in async_calls)


def test_embedding_similarity_alignment_is_explicit_opt_in(analysis):
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    aligned = analysis._compute_cwb(a, b, True).values()
    indexed = analysis._compute_cwb(a, b, False).values()
    assert aligned["alignment"] == pytest.approx(1.0)
    assert aligned["alignment_gain"] == pytest.approx(1.0)
    assert indexed["pair_mean"] == pytest.approx(0.0)
    assert indexed["alignment"] is None


def test_cuda_oom_retries_once_on_cpu(monkeypatch, analysis):
    calls = []

    def fake_compute(a, b, device, allow_alignment, top_count):
        calls.append(device)
        if device == "cuda":
            raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
        return analysis.RawStats(finite=1, total=1), analysis.CWBStats(), []

    monkeypatch.setattr(analysis, "_compute_pair_on_device", fake_compute)
    monkeypatch.setattr(analysis, "_release_cuda_oom", lambda: None)
    result = analysis._analyze_with_fallback(
        "layer", torch.ones(1), torch.ones(1), "cuda", False, 1,
    )
    assert calls == ["cuda", "cpu"]
    assert result[-1] is True


def test_non_oom_error_is_not_retried(monkeypatch, analysis):
    calls = []

    def fake_compute(*args):
        calls.append(args[2])
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(analysis, "_compute_pair_on_device", fake_compute)
    with pytest.raises(RuntimeError, match="unrelated failure"):
        analysis._analyze_with_fallback(
            "layer", torch.ones(1), torch.ones(1), "cuda", False, 1,
        )
    assert calls == ["cuda"]


def test_failed_cpu_retry_names_layer_and_chains_oom(monkeypatch, analysis):
    def fake_compute(a, b, device, allow_alignment, top_count):
        if device == "cuda":
            raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
        raise ValueError("CPU calculation failed")

    monkeypatch.setattr(analysis, "_compute_pair_on_device", fake_compute)
    monkeypatch.setattr(analysis, "_release_cuda_oom", lambda: None)
    with pytest.raises(RuntimeError, match="CPU fallback failed while analyzing 'named.layer'") as caught:
        analysis._analyze_with_fallback(
            "named.layer", torch.ones(1), torch.ones(1), "cuda", False, 1,
        )
    assert isinstance(caught.value.__cause__, torch.cuda.OutOfMemoryError)
