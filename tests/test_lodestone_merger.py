import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter, MemoryEfficientSafeOpen


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_lodestone_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def lodestone():
    return _load_module("nodes.lodestone_merger")


def _write(path, tensors):
    with IncrementalSafetensorsWriter(str(path), max_workers=1) as writer:
        writer.write_batch(list(tensors.items()))


def _read(path):
    result = {}
    with MemoryEfficientSafeOpen(str(path), low_memory=True) as handler:
        stream = handler.async_stream(handler.keys(), batch_size=1, prefetch_batches=1, pin_memory=False)
        for batch in stream:
            for key, tensor in batch:
                result[key] = tensor.clone()
                handler.mark_processed(key)
    return result


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("sum", torch.diag(torch.tensor([2.0, 4.0]))),
        ("mean", torch.diag(torch.tensor([1.0, 2.0]))),
        ("slotnorm", torch.eye(2)),
        ("normmatch", torch.diag(torch.tensor([2.0, 4.0])) * (2.0 / torch.sqrt(torch.tensor(20.0)))),
        ("slotnorm-normmatch", torch.eye(2) * torch.sqrt(torch.tensor(2.0))),
    ],
)
def test_methods_match_supplied_dense_math(lodestone, method, expected):
    deltas = [torch.diag(torch.tensor([2.0, 0.0])), torch.diag(torch.tensor([0.0, 4.0]))]
    torch.testing.assert_close(lodestone.merge_lodestone_deltas(deltas, method), expected)


def test_convolutional_factor_expansion(lodestone):
    down = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)
    up = torch.tensor([[[[2.0]], [[-1.0]]]])
    expected = (up.reshape(1, 2) @ down.reshape(2, -1)).reshape(1, 1, 3, 3)
    torch.testing.assert_close(lodestone.factor_delta(down, up, "conv"), expected)


def test_zero_cancellation_remains_zero(lodestone):
    delta = torch.tensor([[1.0, -2.0]])
    result = lodestone.merge_lodestone_deltas([delta, -delta], "normmatch")
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("count", [3, 8])
def test_three_and_multi_input_dense_means(lodestone, count):
    deltas = [torch.full((2, 2), float(index)) for index in range(1, count + 1)]
    expected = torch.full((2, 2), (count + 1) / 2)
    torch.testing.assert_close(
        lodestone.merge_lodestone_deltas(deltas, "mean"), expected
    )


def test_layer_plan_enforces_missing_filters_and_low_bit(lodestone):
    class Handler:
        def get_dtype(self, _key):
            return torch.float32

    handlers = [Handler(), Handler()]
    roles = {"down": "layer.down", "up": "layer.up"}
    infos = [
        {"pairs": {"layer": roles}, "low_bit": set()},
        {"pairs": {}, "low_bit": set()},
    ]
    layer_map = {"layer": [(0, "layer")]}
    with pytest.raises(ValueError, match="missing"):
        lodestone._build_units(layer_map, infos, handlers, "error", "", "", False)
    units, _ = lodestone._build_units(layer_map, infos, handlers, "zeros", "", "", False)
    assert units[0]["missing"] == 1
    excluded, _ = lodestone._build_units(layer_map, infos, handlers, "zeros", "*.diff", "", True)
    assert excluded[0]["missing"] == 0
    discarded, _ = lodestone._build_units(layer_map, infos, handlers, "skip", "", "*.diff", True)
    assert discarded == []

    infos[0]["low_bit"] = {"layer.down"}
    with pytest.raises(ValueError, match="low-bit"):
        lodestone._build_units(layer_map, infos, handlers, "skip", "", "", False)


def test_streamed_merge_writes_only_alpha_normalized_full_delta(monkeypatch, tmp_path, lodestone):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    _write(first, {
        "diffusion_model.layer.lora_A.weight": torch.tensor([[1.0, 0.0]]),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[2.0], [0.0]]),
        "diffusion_model.layer.alpha": torch.tensor(0.5),
    })
    _write(second, {
        "diffusion_model.layer.lora_A.weight": torch.tensor([[0.0, 1.0]]),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[0.0], [4.0]]),
    })
    monkeypatch.setattr(lodestone.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(lodestone, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(lodestone, "cleanup_after_operation", lambda: None)

    output = lodestone.merge_lodestone_loras(
        [str(first), str(second)], "sum", "cpu", torch.float32, "dense", verbose=False
    )
    tensors = _read(output)

    assert list(tensors) == ["diffusion_model.layer.diff"]
    torch.testing.assert_close(tensors["diffusion_model.layer.diff"], torch.diag(torch.tensor([1.0, 4.0])))


@pytest.mark.parametrize("patterns, glob_patterns, selected", [
    (r"\.selected", False, True),
    ("*.selected*", True, True),
    ("", False, False),
    ("no_match", False, False),
])
def test_include_filter_preserves_nonmatches_and_discards_first(
    monkeypatch, tmp_path, lodestone, patterns, glob_patterns, selected,
):
    paths = [tmp_path / f"{name}.safetensors" for name in ("a", "b")]
    for path, value in zip(paths, (2.0, 6.0)):
        _write(path, {
            f"diffusion_model.{layer}.diff": torch.full((2, 2), value)
            for layer in ("selected", "other", "selected_discard")
        })
    monkeypatch.setattr(lodestone.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(lodestone, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(lodestone, "cleanup_after_operation", lambda: None)
    output = lodestone.merge_lodestone_loras(
        [str(path) for path in paths], "sum", "cpu", torch.float32, "include_filter",
        exclude_patterns=patterns, glob_patterns=glob_patterns, include_mode=True,
        discard_patterns="*discard*" if glob_patterns else "discard", verbose=False,
    )
    tensors = _read(output)
    assert set(tensors) == {"diffusion_model.selected.diff", "diffusion_model.other.diff"}
    torch.testing.assert_close(tensors["diffusion_model.selected.diff"], torch.full((2, 2), 8.0 if selected else 2.0))
    torch.testing.assert_close(tensors["diffusion_model.other.diff"], torch.full((2, 2), 2.0))


def test_dedicated_nodes_do_not_modify_factor_merger_schema(lodestone):
    standard = _load_module("nodes.lora_merger")
    assert [node.define_schema().node_id for node in lodestone.LODESTONE_MERGER_NODES] == [
        "LodestoneLoRATwoMerger", "LodestoneLoRAThreeMerger", "LodestoneLoRAMultiMerger"
    ]
    standard_inputs = {item.id: item for item in standard.LoRAMultiMerge.define_schema().inputs}
    assert standard_inputs["merge_mode"].options == ["concatenate", "weighted_sum"]
    assert "calc_mode" not in standard_inputs


def test_dense_cuda_oom_retries_layer_on_cpu(monkeypatch, lodestone):
    unit = {"dtypes": [torch.float32], "missing": 0, "sources": []}
    calls = []

    def process(_unit, _loaded, _method, _dtype, device):
        calls.append(device.type)
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return torch.ones(1)

    monkeypatch.setattr(lodestone, "_process_on_device", process)
    monkeypatch.setattr(lodestone, "_release_cuda_oom", lambda: None)
    output, fallback = lodestone._process(unit, {}, "sum", torch.float32, "cuda")
    assert fallback is True
    assert calls == ["cuda", "cpu"]
    assert torch.equal(output, torch.ones(1))
