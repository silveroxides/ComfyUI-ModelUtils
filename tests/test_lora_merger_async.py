import importlib
import sys
import types
from collections import Counter
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import (
    IncrementalSafetensorsWriter,
    MemoryEfficientSafeOpen,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_lora_merger_async_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def merger():
    return _load_module("nodes.lora_merger")


def _write_uel(path, tensors):
    with IncrementalSafetensorsWriter(str(path), max_workers=1) as writer:
        writer.write_batch(
            [(key, tensor.cpu().contiguous()) for key, tensor in tensors.items()]
        )


def _read_uel(path, async_method=None, mark_method=None):
    result = {}
    with MemoryEfficientSafeOpen(str(path), low_memory=True) as loader:
        stream = (async_method or MemoryEfficientSafeOpen.async_stream)(
            loader,
            loader.keys(),
            batch_size=1,
            prefetch_batches=1,
            pin_memory=False,
        )
        for batch in stream:
            for key, tensor in batch:
                result[key] = tensor.clone()
                (mark_method or MemoryEfficientSafeOpen.mark_processed)(loader, key)
    return result


def _patch_runtime(monkeypatch, tmp_path, merger):
    monkeypatch.setattr(merger.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(merger, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(merger, "cleanup_after_operation", lambda: None)


def _mixed_sources(tmp_path):
    first = tmp_path / "mixed_first.safetensors"
    second = tmp_path / "mixed_second.safetensors"
    first_tensors = {
        "diffusion_model.small.lora_A.weight": torch.tensor([[1.0, 0.0]]),
        "diffusion_model.small.lora_B.weight": torch.tensor([[1.0], [2.0]]),
        "diffusion_model.small.alpha": torch.tensor(0.5),
        "diffusion_model.large.lora_A.weight": torch.eye(3),
        "diffusion_model.large.lora_B.weight": torch.eye(3),
    }
    second_tensors = {
        "diffusion_model.small.lora_A.weight": torch.eye(2),
        "diffusion_model.small.lora_B.weight": torch.eye(2),
        "diffusion_model.large.lora_A.weight": torch.tensor([[1.0, 1.0, 1.0]]),
        "diffusion_model.large.lora_B.weight": torch.tensor([[1.0], [2.0], [3.0]]),
    }
    _write_uel(first, first_tensors)
    _write_uel(second, second_tensors)
    return first, second, first_tensors, second_tensors


def test_concatenate_uses_per_layer_rank_and_alpha(monkeypatch, tmp_path, merger):
    _patch_runtime(monkeypatch, tmp_path, merger)
    first, second, first_tensors, second_tensors = _mixed_sources(tmp_path)

    output = merger.merge_multi_loras(
        [str(first), str(second)],
        [2.0, -0.5],
        "concatenate",
        "cpu",
        torch.float32,
        "mixed_concat",
        verbose=False,
    )
    tensors = _read_uel(output)

    assert tensors["diffusion_model.small.lora_A.weight"].shape[0] == 3
    assert tensors["diffusion_model.large.lora_A.weight"].shape[0] == 4
    assert not any(key.endswith(".alpha") for key in tensors)

    small_expected = (
        2.0
        * (0.5 / 1.0)
        * first_tensors["diffusion_model.small.lora_B.weight"]
        @ first_tensors["diffusion_model.small.lora_A.weight"]
        - 0.5
        * second_tensors["diffusion_model.small.lora_B.weight"]
        @ second_tensors["diffusion_model.small.lora_A.weight"]
    )
    small_actual = (
        tensors["diffusion_model.small.lora_B.weight"]
        @ tensors["diffusion_model.small.lora_A.weight"]
    )
    torch.testing.assert_close(small_actual, small_expected)


@pytest.mark.parametrize("strategy", ["weighted_sum", "dare", "enhanced_dare"])
def test_nonconcatenate_variants_use_each_layers_max_rank(
    monkeypatch, tmp_path, merger, strategy
):
    _patch_runtime(monkeypatch, tmp_path, merger)
    first, second, _, _ = _mixed_sources(tmp_path)
    common = [str(first), str(second)], [1.0, 1.0]
    if strategy == "weighted_sum":
        output = merger.merge_multi_loras(
            *common,
            "weighted_sum",
            "cpu",
            torch.float32,
            "mixed_weighted",
            verbose=False,
        )
    elif strategy == "dare":
        output = merger.merge_multi_loras_dare(
            *common,
            0.0,
            0.0,
            7,
            "cpu",
            torch.float32,
            "mixed_dare",
            verbose=False,
        )
    else:
        output = merger.merge_multi_loras_dare_enhanced(
            *common,
            1.0,
            1.0,
            0.0,
            0.0,
            7,
            "cpu",
            torch.float32,
            "mixed_enhanced",
            verbose=False,
        )
    tensors = _read_uel(output)
    assert tensors["diffusion_model.small.lora_A.weight"].shape[0] == 2
    assert tensors["diffusion_model.large.lora_A.weight"].shape[0] == 3
    assert not any(key.endswith(".alpha") for key in tensors)


@pytest.mark.parametrize("strategy", ["weighted_sum", "dare", "enhanced_dare"])
def test_nonconcatenate_variants_apply_source_alpha_once(
    monkeypatch, tmp_path, merger, strategy
):
    _patch_runtime(monkeypatch, tmp_path, merger)
    first = tmp_path / f"alpha_first_{strategy}.safetensors"
    second = tmp_path / f"alpha_second_{strategy}.safetensors"
    _write_uel(
        first,
        {
            "diffusion_model.layer.lora_A.weight": torch.ones((1, 1)),
            "diffusion_model.layer.lora_B.weight": torch.full((1, 1), 2.0),
            "diffusion_model.layer.alpha": torch.tensor(0.5),
        },
    )
    _write_uel(
        second,
        {
            "diffusion_model.layer.lora_A.weight": torch.ones((1, 1)),
            "diffusion_model.layer.lora_B.weight": torch.full((1, 1), 2.0),
        },
    )
    common = [str(first), str(second)], [1.0, 1.0]
    if strategy == "weighted_sum":
        output = merger.merge_multi_loras(
            *common,
            "weighted_sum",
            "cpu",
            torch.float32,
            "alpha_weighted",
            verbose=False,
        )
        expected_down = 2.0
        expected_up = 3.0
    elif strategy == "dare":
        output = merger.merge_multi_loras_dare(
            *common,
            0.0,
            0.0,
            9,
            "cpu",
            torch.float32,
            "alpha_dare",
            verbose=False,
        )
        expected_down = 1.0
        expected_up = 1.5
    else:
        output = merger.merge_multi_loras_dare_enhanced(
            *common,
            1.0,
            1.0,
            0.0,
            0.0,
            9,
            "cpu",
            torch.float32,
            "alpha_enhanced",
            verbose=False,
        )
        expected_down = 1.0
        expected_up = 1.5

    tensors = _read_uel(output)
    assert tensors["diffusion_model.layer.lora_A.weight"].item() == pytest.approx(
        expected_down
    )
    assert tensors["diffusion_model.layer.lora_B.weight"].item() == pytest.approx(
        expected_up
    )


def test_all_normal_inputs_use_async_uel_and_release_once(
    monkeypatch, tmp_path, merger
):
    _patch_runtime(monkeypatch, tmp_path, merger)
    first, second, _, _ = _mixed_sources(tmp_path)
    original_async = merger.MemoryEfficientSafeOpen.async_stream
    original_mark = merger.MemoryEfficientSafeOpen.mark_processed
    async_calls = []
    marked = []

    def track_async(self, keys, **kwargs):
        async_calls.append((self.filename, tuple(keys), kwargs, self.low_memory))
        yield from original_async(self, keys, **kwargs)

    def track_mark(self, key):
        marked.append((self.filename, key))
        return original_mark(self, key)

    def reject_get_tensor(*args, **kwargs):
        raise AssertionError("multi-merge used synchronous get_tensor")

    monkeypatch.setattr(merger.MemoryEfficientSafeOpen, "async_stream", track_async)
    monkeypatch.setattr(merger.MemoryEfficientSafeOpen, "mark_processed", track_mark)
    monkeypatch.setattr(merger.MemoryEfficientSafeOpen, "get_tensor", reject_get_tensor)

    output = merger.merge_multi_loras(
        [str(first), str(second)],
        [1.0, 1.0],
        "concatenate",
        "cpu",
        torch.float32,
        "async_only",
        verbose=False,
    )

    assert Path(output).is_file()
    assert len(async_calls) == 2
    planned = Counter(
        (filename, key)
        for filename, keys, _, _ in async_calls
        for key in keys
    )
    assert Counter(marked) == planned
    assert all(count == 1 for count in planned.values())
    for _, _, options, low_memory in async_calls:
        assert low_memory is True
        assert options == {
            "batch_size": 1,
            "prefetch_batches": 1,
            "pin_memory": False,
        }


def test_cuda_oom_retries_only_current_unit_on_cpu(
    monkeypatch, tmp_path, merger
):
    _patch_runtime(monkeypatch, tmp_path, merger)
    first, second, _, _ = _mixed_sources(tmp_path)
    original = merger._process_merge_unit_on_device
    calls = []

    def fail_cuda_then_cpu(*args):
        process_device = args[-2]
        calls.append(process_device.type)
        if process_device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return original(*args)

    monkeypatch.setattr(merger, "_process_merge_unit_on_device", fail_cuda_then_cpu)
    monkeypatch.setattr(merger, "_release_cuda_oom_state", lambda: None)

    output = merger.merge_multi_loras(
        [str(first), str(second)],
        [1.0, 1.0],
        "concatenate",
        "cuda",
        torch.float32,
        "oom_fallback",
        verbose=False,
    )

    assert Path(output).is_file()
    assert calls == ["cuda", "cpu", "cuda", "cpu"]
