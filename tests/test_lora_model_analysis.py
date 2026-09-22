import csv
import importlib
from io import StringIO
import json
import logging
from pathlib import Path
import sys
import types

import pytest
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter


@pytest.fixture(scope="module")
def analysis():
    name = "modelutils_lora_on_model_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[name] = package
    return importlib.import_module(f"{name}.nodes.lora_model_analysis")


def write(path, tensors):
    with IncrementalSafetensorsWriter(str(path), max_workers=1) as writer:
        writer.write_batch(list(tensors.items()))


def setup(monkeypatch, tmp_path, analysis, adapter, base=None):
    paths = {"loras": tmp_path / "adapter.safetensors", "diffusion_models": tmp_path / "base.safetensors"}
    write(paths["loras"], adapter)
    write(paths["diffusion_models"], base if base is not None else {
        "model.diffusion_model.layer.weight": torch.ones((2, 2)),
        "model.diffusion_model.untouched.weight": torch.ones((2, 2)),
    })
    monkeypatch.setattr(analysis.folder_paths, "get_full_path_or_raise", lambda category, name: str(paths[category]))
    monkeypatch.setattr(analysis, "cleanup_after_operation", lambda: None)
    return paths


def params(**overrides):
    return {"execution_mode": "ANALYZE", "process_device": "cpu", "top_weight_differences": 3, "force_clear_cache": False, "exclude_patterns": "", "glob_patterns": False, "include_mode": False, "strength": 1.0, **overrides}


def pair(alpha=True):
    result = {
        "diffusion_model.layer.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[2.0], [4.0]]),
    }
    if alpha:
        result["diffusion_model.layer.alpha"] = torch.tensor(0.5)
    return result


@pytest.mark.parametrize("alpha,strength,mae,maximum", [(True, 1.0, 2.25, 4), (False, 1.0, 4.5, 8), (True, 0.5, 1.125, 2), (True, 0.0, 0, 0)])
def test_reports_actual_applied_delta_without_saving(monkeypatch, tmp_path, analysis, alpha, strength, mae, maximum):
    paths = setup(monkeypatch, tmp_path, analysis, pair(alpha))
    before = {path: path.read_bytes() for path in paths.values()}
    comparison, cwb, metrics_csv, _ = analysis.analyze_lora_on_model("adapter", "base", params(strength=strength))
    rows = {row["key"]: row for row in csv.DictReader(StringIO(metrics_csv))}
    row = rows["model.diffusion_model.layer.weight"]
    assert float(row["mae"]) == pytest.approx(mae)
    assert float(row["max"]) == maximum
    assert float(rows["model.diffusion_model.untouched.weight"]["mae"]) == 0
    assert "patched − original" in comparison
    assert "original versus LoRA-patched" in cwb
    assert {path: path.read_bytes() for path in paths.values()} == before
    assert set(tmp_path.iterdir()) == set(paths.values())


def test_quantized_base_is_dequantized_and_sidecars_are_not_analyzed(monkeypatch, tmp_path, analysis):
    prefix = "model.diffusion_model.layer"
    quant = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
    setup(monkeypatch, tmp_path, analysis, pair(), {
        f"{prefix}.weight": torch.ones((2, 2), dtype=torch.int8),
        f"{prefix}.weight_scale": torch.tensor(2.0),
        f"{prefix}.comfy_quant": quant,
    })
    comparison, _, metrics_csv, _ = analysis.analyze_lora_on_model("adapter", "base", params())
    rows = list(csv.DictReader(StringIO(metrics_csv)))
    assert [row["key"] for row in rows] == [f"{prefix}.weight"]
    assert float(rows[0]["mae"]) == pytest.approx(2.25)
    assert "weight_scale" not in comparison


@pytest.mark.parametrize("suffixes", [(".lora_down.weight", ".lora_up.weight"), (".lora_A.default.weight", ".lora_B.default.weight"), (".lora_A", ".lora_B")])
def test_pair_formats_and_flattened_reference_mapping(monkeypatch, tmp_path, analysis, suffixes):
    setup(monkeypatch, tmp_path, analysis, {
        "lora_unet_layer" + suffixes[0]: torch.ones((1, 2)),
        "lora_unet_layer" + suffixes[1]: torch.ones((2, 1)),
    })
    _, _, text, _ = analysis.analyze_lora_on_model("adapter", "base", params())
    rows = list(csv.DictReader(StringIO(text)))
    assert float(rows[0]["mae"]) == 1


def test_direct_weight_bias_replacement_and_unmapped_are_explicit(monkeypatch, tmp_path, analysis):
    setup(monkeypatch, tmp_path, analysis, {
        "diffusion_model.layer.diff": torch.full((2, 2), 2.0),
        "diffusion_model.layer.diff_b": torch.full((2,), 3.0),
        "diffusion_model.replaced.set_weight": torch.full((2, 2), 5.0),
        "diffusion_model.missing.diff": torch.ones((2, 2)),
    }, {
        "diffusion_model.layer.weight": torch.ones((2, 2)),
        "diffusion_model.layer.bias": torch.ones(2),
        "diffusion_model.replaced.weight": torch.ones((2, 2)),
    })
    report, _, text, _ = analysis.analyze_lora_on_model("adapter", "base", params(strength=0.5))
    rows = {row["key"]: row for row in csv.DictReader(StringIO(text))}
    assert float(rows["diffusion_model.layer.weight"]["mae"]) == 1
    assert float(rows["diffusion_model.layer.bias"]["mae"]) == 1.5
    assert float(rows["diffusion_model.replaced.weight"]["mae"]) == 4
    assert "diffusion_model.missing" in report


def test_bounded_stream_release_and_filtering(monkeypatch, tmp_path, analysis):
    setup(monkeypatch, tmp_path, analysis, pair())
    calls, released = [], []
    original_stream = analysis.MemoryEfficientSafeOpen.async_stream
    original_mark = analysis.MemoryEfficientSafeOpen.mark_processed

    def stream(handler, keys, **kwargs):
        calls.append((list(keys), kwargs))
        return original_stream(handler, keys, **kwargs)

    def mark(handler, key):
        released.append(key)
        return original_mark(handler, key)

    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "async_stream", stream)
    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "mark_processed", mark)
    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "get_tensor", lambda *args: pytest.fail("synchronous tensor load"))
    _, _, text, _ = analysis.analyze_lora_on_model("adapter", "base", params(exclude_patterns="*layer.weight", glob_patterns=True, include_mode=True))
    assert len(list(csv.DictReader(StringIO(text)))) == 1
    assert len(calls) == 2
    assert all(kwargs == {"batch_size": 1, "prefetch_batches": 1, "pin_memory": False} for _, kwargs in calls)
    assert sorted(released) == sorted(key for keys, _ in calls for key in keys)
    assert len(released) == 4


def test_logged_adapter_failure_is_not_reported_as_zero_delta(monkeypatch, tmp_path, analysis):
    setup(monkeypatch, tmp_path, analysis, pair())
    marks = []
    original_mark = analysis.MemoryEfficientSafeOpen.mark_processed

    def mark(handler, key):
        marks.append(key)
        return original_mark(handler, key)

    def fail(self, weight, key, *args):
        logging.error("ERROR LoRA %s deliberately failed", key)
        return weight

    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "mark_processed", mark)
    monkeypatch.setattr(analysis.LoRAAdapter, "calculate_weight", fail)
    with pytest.raises(RuntimeError, match="deliberately failed"):
        analysis.analyze_lora_on_model("adapter", "base", params())
    assert len(marks) == 4
    assert not any(isinstance(handler, analysis.AdapterErrorCapture) for handler in logging.getLogger().handlers)


def test_complete_unit_cuda_retry(monkeypatch, analysis):
    devices = []

    def calculate(key, loaded, specs, strength, device, top_count):
        devices.append(device)
        if device == "cuda":
            raise torch.cuda.OutOfMemoryError("test")
        return analysis.RawStats(), analysis.CWBStats(), []

    monkeypatch.setattr(analysis, "_apply_and_analyze", calculate)
    monkeypatch.setattr(analysis, "_release_cuda_oom", lambda: None)
    assert analysis._analyze_unit("layer", {}, [], 1, "cuda", 0)[-1] is True
    assert devices == ["cuda", "cpu"]


def test_dora_uses_reconstructed_weight_not_raw_factor_delta(monkeypatch, tmp_path, analysis):
    setup(monkeypatch, tmp_path, analysis, {
        "diffusion_model.layer.lora_A.weight": torch.tensor([[1.0, 0.0]]),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[1.0], [0.0]]),
        "diffusion_model.layer.alpha": torch.tensor(0.5),
        "diffusion_model.layer.dora_scale": torch.tensor([3.0, 4.0]),
    }, {"diffusion_model.layer.weight": torch.eye(2)})
    _, _, text, _ = analysis.analyze_lora_on_model("adapter", "base", params())
    row = next(csv.DictReader(StringIO(text)))
    # Installed Comfy output-axis DoRA scales by the original base row norms.
    # The resulting diagonal is (4.5, 4), not the raw-factor result (1.5, 1).
    assert float(row["mae"]) == pytest.approx(1.625)
    assert float(row["max"]) == pytest.approx(3.5)


def test_locon_mid_and_scalar_low_bit_alpha(monkeypatch, tmp_path, analysis):
    setup(monkeypatch, tmp_path, analysis, {
        "diffusion_model.layer.lora_down.weight": torch.ones((1, 1, 1, 1)),
        "diffusion_model.layer.lora_up.weight": torch.ones((1, 1, 1, 1)),
        "diffusion_model.layer.lora_mid.weight": torch.ones((1, 1, 3, 3)),
        "diffusion_model.layer.alpha": torch.tensor(1, dtype=torch.uint8),
    }, {"diffusion_model.layer.weight": torch.zeros((1, 1, 3, 3))})
    _, _, text, _ = analysis.analyze_lora_on_model("adapter", "base", params())
    row = next(csv.DictReader(StringIO(text)))
    assert float(row["mae"]) == 1


def test_low_bit_factors_reported_without_materializing(monkeypatch, tmp_path, analysis):
    tensors = pair()
    tensors["diffusion_model.layer.lora_A.weight"] = torch.ones((1, 2), dtype=torch.uint8)
    setup(monkeypatch, tmp_path, analysis, tensors)
    requested = []
    original = analysis.MemoryEfficientSafeOpen.async_stream

    def stream(handler, keys, **kwargs):
        requested.extend(keys)
        return original(handler, keys, **kwargs)

    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "async_stream", stream)
    report, _, text, _ = analysis.analyze_lora_on_model("adapter", "base", params())
    assert requested == ["model.diffusion_model.untouched.weight"]
    assert "model.diffusion_model.layer.weight" in report.replace(r"\_", "_")
    assert [row["key"] for row in csv.DictReader(StringIO(text))] == ["model.diffusion_model.untouched.weight"]


def test_streams_close_after_failure(monkeypatch, tmp_path, analysis):
    setup(monkeypatch, tmp_path, analysis, pair())
    streams_closed = []
    original = analysis.MemoryEfficientSafeOpen.async_stream

    def stream(handler, keys, **kwargs):
        iterator = original(handler, keys, **kwargs)
        try:
            yield from iterator
        finally:
            iterator.close()
            streams_closed.append(True)

    monkeypatch.setattr(analysis.MemoryEfficientSafeOpen, "async_stream", stream)
    monkeypatch.setattr(analysis.LoRAAdapter, "calculate_weight", lambda *args: (_ for _ in ()).throw(ValueError("test failure")))
    with pytest.raises(ValueError, match="test failure"):
        analysis.analyze_lora_on_model("adapter", "base", params())
    assert len(streams_closed) == 2


def test_schema_and_documentation_mode(monkeypatch, analysis):
    monkeypatch.setattr(analysis.folder_paths, "get_filename_list", lambda category: [category])
    schema = analysis.LoRAOnModelAnalysis.define_schema()
    assert schema.inputs[1].id == "model_a" and schema.inputs[1].options == ["loras"]
    assert schema.inputs[2].id == "model_b" and schema.inputs[2].options == ["diffusion_models"]
    assert len(schema.outputs) == 5
    assert schema.inputs[-1].id == "strength" and schema.inputs[-1].default == 1
    monkeypatch.setattr(analysis, "analyze_lora_on_model", lambda *args: pytest.fail("documentation loaded tensors"))
    assert analysis.LoRAOnModelAnalysis.execute(execution_mode="DOCUMENTATION ONLY")[2].startswith("# LoRA-on-model analysis")
