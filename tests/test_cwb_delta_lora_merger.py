import importlib
import re
import sys
import types
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter, MemoryEfficientSafeOpen


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_delta_cwb_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def delta_cwb():
    return _load_module("nodes.cwb_delta_lora_merger")


def _settings(delta_cwb):
    return delta_cwb.CWBSettings(
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.0,
        similarity_threshold=-1.0,
        power_alpha=0.0,
        diversity_beta=0.0,
        rescale_norm=False,
        global_scale=1.0,
        dynamic_similarity_contrast=False,
        soft_comfort_bandpass=False,
        position_weight=0.0,
        preserve_common_prefix=False,
    )


def _write(path, tensors, metadata=None):
    with IncrementalSafetensorsWriter(str(path), metadata=metadata or {}, max_workers=1) as writer:
        writer.write_batch(list(tensors.items()))


def _read(path):
    tensors = {}
    with MemoryEfficientSafeOpen(str(path), low_memory=True) as handler:
        stream = handler.async_stream(
            handler.keys(), batch_size=1, prefetch_batches=1, pin_memory=False
        )
        for batch in stream:
            for key, tensor in batch:
                tensors[key] = tensor.clone()
                handler.mark_processed(key)
    return tensors


def test_linear_and_convolutional_factor_expansion(delta_cwb):
    down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    up = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    torch.testing.assert_close(
        delta_cwb._factor_delta(down, up, "linear"), up @ down
    )

    conv_down = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)
    conv_up = torch.tensor([[[[2.0]], [[-1.0]]]])
    expected = (conv_up.reshape(1, 2) @ conv_down.reshape(2, -1)).reshape(1, 1, 3, 3)
    torch.testing.assert_close(
        delta_cwb._factor_delta(conv_down, conv_up, "conv"), expected
    )


def test_dense_cwb_keeps_corresponding_rows_fixed(delta_cwb):
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reversed_source = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    result = delta_cwb.merge_cwb_tensors(
        [reference, reversed_source],
        _settings(delta_cwb),
        reference_index=0,
        allow_similarity_alignment=False,
    )
    torch.testing.assert_close(result, torch.full((2, 2), 0.5))


def test_layer_units_apply_missing_modes_filters_and_shape_checks(delta_cwb):
    class Handler:
        def __init__(self, shapes):
            self.shapes = shapes

        def get_shape(self, key):
            return self.shapes[key]

        def get_dtype(self, _key):
            return torch.float32

    roles = {"down": "layer.down", "up": "layer.up"}
    infos = [
        {"pairs": {"layer": roles}, "low_bit": set()},
        {"pairs": {}, "low_bit": set()},
    ]
    handlers = [Handler({"layer.down": (1, 2), "layer.up": (2, 1)}), Handler({})]
    layer_map = {"layer": [(0, "layer")]}
    empty = delta_cwb._compile_patterns("", False)

    with pytest.raises(ValueError, match="missing or incompatible"):
        delta_cwb._build_layer_units(
            layer_map, infos, handlers, "error", empty, empty, False
        )
    units, _, _ = delta_cwb._build_layer_units(
        layer_map, infos, handlers, "zeros", empty, empty, False
    )
    assert units[0].zero_contributors == 1
    assert units[0].preserve_anchor is False

    exclude = delta_cwb._compile_patterns(r"\.diff$", False)
    units, _, _ = delta_cwb._build_layer_units(
        layer_map, infos, handlers, "zeros", exclude, empty, False
    )
    assert units[0].preserve_anchor is True
    assert units[0].zero_contributors == 0

    discard = delta_cwb._compile_patterns("*.diff", True)
    units, _, rejected = delta_cwb._build_layer_units(
        layer_map, infos, handlers, "skip", empty, discard, True
    )
    assert units == []
    assert rejected == 1


def test_streamed_merge_normalizes_alpha_once_and_supports_mixed_rank(
    monkeypatch, tmp_path, delta_cwb
):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    _write(first, {
        "diffusion_model.layer.lora_A.weight": torch.tensor([[1.0, 0.0]]),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[2.0], [0.0]]),
        "diffusion_model.layer.alpha": torch.tensor(0.5),
    }, {"anchor": "yes"})
    _write(second, {
        "diffusion_model.layer.lora_A.weight": torch.eye(2),
        "diffusion_model.layer.lora_B.weight": torch.diag(torch.tensor([3.0, 4.0])),
    })
    monkeypatch.setattr(delta_cwb.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(delta_cwb, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(delta_cwb, "cleanup_after_operation", lambda: None)

    filename, report = delta_cwb.DeltaCWBLoRAMergerLogic.execute(
        [str(first), str(second)],
        [first.name, second.name],
        _settings(delta_cwb),
        "balanced_mean",
        False,
        "skip",
        "dense/mixed_rank",
        torch.float32,
        "cpu",
        "",
        "",
        False,
        False,
    )
    output = tmp_path / "loras" / filename
    tensors = _read(output)

    assert filename == "dense/mixed_rank.safetensors"
    assert list(tensors) == ["diffusion_model.layer.diff"]
    torch.testing.assert_close(
        tensors["diffusion_model.layer.diff"],
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
    )
    assert "Merged layers: 1" in report
    with MemoryEfficientSafeOpen(str(output), low_memory=True) as handler:
        metadata = handler.metadata()
    assert metadata["anchor"] == "yes"
    assert metadata["merge_method"] == "delta_cwb"
    assert metadata["alpha_normalized"] == "true"
    assert metadata["output_representation"] == "full_direct_difference"


def test_existing_diff_and_1d_output_remain_fp32(monkeypatch, tmp_path, delta_cwb):
    first = tmp_path / "diff_first.safetensors"
    second = tmp_path / "diff_second.safetensors"
    key = "diffusion_model.bias.diff"
    _write(first, {key: torch.tensor([1.0, 3.0])})
    _write(second, {key: torch.tensor([3.0, 5.0])})
    monkeypatch.setattr(delta_cwb.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(delta_cwb, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(delta_cwb, "cleanup_after_operation", lambda: None)

    filename, _ = delta_cwb.DeltaCWBLoRAMergerLogic.execute(
        [str(first), str(second)], [first.name, second.name], _settings(delta_cwb),
        "balanced_mean", False, "skip", "one_dim", torch.float16, "cpu",
        "", "", False, False,
    )
    output = _read(tmp_path / "loras" / filename)[key]
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.tensor([2.0, 4.0]))


def test_atomic_writer_preserves_destination_on_cwb_failure(
    monkeypatch, tmp_path, delta_cwb
):
    first = tmp_path / "atomic_first.safetensors"
    second = tmp_path / "atomic_second.safetensors"
    tensors = {
        "diffusion_model.layer.lora_A.weight": torch.tensor([[1.0, 0.0]]),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[1.0], [0.0]]),
    }
    _write(first, tensors)
    _write(second, tensors)
    destination = tmp_path / "loras" / "atomic.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing destination")
    monkeypatch.setattr(delta_cwb.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(delta_cwb, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(delta_cwb, "cleanup_after_operation", lambda: None)
    released = []
    closed = []
    opener = delta_cwb.MemoryEfficientSafeOpen

    class TrackedHandler:
        def __init__(self, path, low_memory=True):
            self.handler = opener(path, low_memory=low_memory)

        def __getattr__(self, name):
            return getattr(self.handler, name)

        def mark_processed(self, key):
            released.append(key)
            return self.handler.mark_processed(key)

        def __exit__(self, exc_type, exc, traceback):
            closed.append(True)
            return self.handler.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(delta_cwb, "MemoryEfficientSafeOpen", TrackedHandler)
    monkeypatch.setattr(
        delta_cwb,
        "merge_cwb_tensors",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("planned failure")),
    )

    with pytest.raises(RuntimeError, match="planned failure"):
        delta_cwb.DeltaCWBLoRAMergerLogic.execute(
            [str(first), str(second)], [first.name, second.name], _settings(delta_cwb),
            "balanced_mean", False, "skip", "atomic", torch.float32, "cpu",
            "", "", False, False,
        )
    assert destination.read_bytes() == b"existing destination"
    assert not list(destination.parent.glob(".atomic.safetensors.*.tmp"))
    assert sorted(released) == sorted([*tensors, *tensors])
    assert len(closed) == 2


def test_complete_layer_cuda_oom_retries_on_cpu(monkeypatch, delta_cwb):
    source = delta_cwb.DeltaCWBLoRASource(
        0, "layer", None, None, None, "layer.diff", torch.float32
    )
    unit = delta_cwb.DeltaCWBLayerUnit(
        "layer", "layer.diff", (source,), 0, False
    )
    calls = []

    def process(_unit, _loaded, _settings, _dtype, device, _diagnostics):
        calls.append(device.type)
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return torch.ones(1), torch.float32

    monkeypatch.setattr(delta_cwb, "_process_layer_on_device", process)
    monkeypatch.setattr(delta_cwb, "_release_cuda_oom", lambda: None)
    output, dtype, fallback = delta_cwb._process_layer(
        unit, {}, _settings(delta_cwb), torch.float32, "cuda", delta_cwb.CWBDiagnostics()
    )
    assert calls == ["cuda", "cpu"]
    assert fallback is True
    assert dtype == torch.float32
    assert torch.equal(output, torch.ones(1))


def test_exact_node_schema_contract(delta_cwb):
    expected_ids = [
        "DeltaCWBLoRATwoMerger",
        "DeltaCWBLoRAThreeMerger",
        "DeltaCWBLoRAMultiMerger",
    ]
    schemas = [node.define_schema() for node in delta_cwb.DELTA_CWB_LORA_NODES]
    assert [schema.node_id for schema in schemas] == expected_ids
    assert [schema.display_name for schema in schemas] == [
        "Delta CWB Merge LoRAs (2)",
        "Delta CWB Merge LoRAs (3)",
        "Delta CWB LoRA Multi-Merge",
    ]
    assert all(schema.category == "ModelUtils/LoRA/Merge/Delta CWB" for schema in schemas)
    assert all([output.display_name for output in schema.outputs] == ["output_filename", "cwb_report"] for schema in schemas)
    assert all(schema.outputs[0].get_io_type() == "*" for schema in schemas)

    fixed_ids = [value.id for value in schemas[0].inputs]
    assert fixed_ids == [
        "lora_1", "lora_2", "cwb_preset", "cwb_config", "mismatch_mode",
        "output_filename", "save_dtype", "process_device", "exclude_patterns",
        "discard_patterns", "glob_patterns", "force_clear_cache",
    ]
    multi_ids = [value.id for value in schemas[2].inputs]
    assert multi_ids[:9] == ["lora_count", *[f"lora_{index}" for index in range(1, 9)]]
    assert schemas[0].inputs[2].default == "balanced_mean"
    assert schemas[0].inputs[4].default == "skip"
    assert schemas[0].inputs[6].default == "bf16"
    assert schemas[0].inputs[7].default == "cuda"
    assert schemas[0].inputs[-2].default is False
    assert schemas[0].inputs[-1].default is True
    assert "agreement" in schemas[0].inputs[2].tooltip
    assert "Only a failed CUDA layer retries on CPU" in schemas[0].inputs[7].tooltip


def test_invalid_regex_and_unselected_multi_input_fail(delta_cwb):
    with pytest.raises(re.error):
        delta_cwb._compile_patterns("[", False)
    with pytest.raises(ValueError, match="unselected input.*3"):
        delta_cwb.DeltaCWBLoRAMultiMerger.execute(
            lora_count="3", lora_1="a", lora_2="b", lora_3="None"
        )
