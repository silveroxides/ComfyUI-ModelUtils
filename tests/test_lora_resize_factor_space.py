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
PACKAGE_NAME = "modelutils_resize_factor_space_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def resize():
    return _load_module("nodes.lora_resize")


def _write_uel(path, tensors, metadata=None):
    with IncrementalSafetensorsWriter(str(path), metadata=metadata or {}) as writer:
        writer.write_batch(
            [(key, tensor.cpu().contiguous()) for key, tensor in tensors.items()]
        )


def _read_uel(path, async_method=None, mark_method=None):
    result = {}
    with MemoryEfficientSafeOpen(str(path), low_memory=True) as loader:
        keys = loader.keys()
        stream = (async_method or MemoryEfficientSafeOpen.async_stream)(
            loader, keys, batch_size=1, prefetch_batches=1, pin_memory=False
        )
        for batch in stream:
            for key, tensor in batch:
                result[key] = tensor.clone()
                (mark_method or MemoryEfficientSafeOpen.mark_processed)(loader, key)
    return result


def _dense_truncated(up, down, rank):
    dense = up.float() @ down.float()
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank, :]


def test_linear_factor_space_matches_dense_reference_and_bounds_svd(
    monkeypatch, resize
):
    generator = torch.Generator().manual_seed(123)
    down = torch.randn((4, 17), generator=generator)
    up = torch.randn((13, 4), generator=generator)
    expected = _dense_truncated(up, down, 3)
    observed_shapes = []
    original_svd = torch.linalg.svd

    def track_svd(tensor, *args, **kwargs):
        observed_shapes.append(tuple(tensor.shape))
        return original_svd(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", track_svd)
    result = resize._resize_lora_factors(down, up, 3, None, None, 1.0)

    actual = result["lora_up"] @ result["lora_down"]
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert observed_shapes == [(4, 4)]


def test_convolutional_factor_space_matches_dense_reference(resize):
    generator = torch.Generator().manual_seed(456)
    down = torch.randn((3, 2, 3, 2), generator=generator)
    up = torch.randn((5, 3, 1, 1), generator=generator)

    result = resize._resize_lora_factors(down, up, 2, None, None, 1.0)

    actual = result["lora_up"].reshape(5, 2) @ result["lora_down"].reshape(2, -1)
    expected = _dense_truncated(up.reshape(5, 3), down.reshape(3, -1), 2)
    assert result["lora_up"].shape == (5, 2, 1, 1)
    assert result["lora_down"].shape == (2, 2, 3, 2)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("method", "parameter", "maximum", "expected_rank"),
    [
        (None, None, 3, 3),
        ("sv_ratio", 4.0, 4, 2),
        ("sv_fro", 0.95, 4, 1),
        ("sv_cumulative", 0.8, 4, 2),
    ],
)
def test_all_resize_modes_use_the_shared_spectrum(
    resize, method, parameter, maximum, expected_rank
):
    singular_values = torch.tensor([10.0, 3.0, 1.0, 0.1])
    result = resize._resize_lora_factors(
        torch.eye(4),
        torch.diag(singular_values),
        maximum,
        method,
        parameter,
        1.0,
    )
    assert result["new_rank"] == expected_rank


def test_mixed_source_ranks_are_capped_per_layer(resize):
    rank_128 = resize._resize_lora_factors(
        torch.eye(128), torch.eye(128), 384, None, None, 1.0
    )
    rank_384 = resize._resize_lora_factors(
        torch.eye(384), torch.eye(384), 384, None, None, 1.0
    )

    assert rank_128["new_rank"] == 128
    assert rank_384["new_rank"] == 384


def test_resize_file_uses_only_async_uel_and_preserves_per_layer_scale(
    monkeypatch, tmp_path, resize
):
    source = tmp_path / "mixed_rank.safetensors"
    _write_uel(
        source,
        {
            "diffusion_model.r2.lora_A.weight": torch.eye(2),
            "diffusion_model.r2.lora_B.weight": torch.eye(2),
            "diffusion_model.r2.alpha": torch.tensor(1.0),
            "diffusion_model.r3.lora_A.weight": torch.eye(3),
            "diffusion_model.r3.lora_B.weight": torch.eye(3),
        },
    )
    monkeypatch.setattr(resize.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(resize, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)

    original_async = resize.MemoryEfficientSafeOpen.async_stream
    original_mark = resize.MemoryEfficientSafeOpen.mark_processed
    async_calls = []
    marked = []

    def track_async(self, keys, **kwargs):
        async_calls.append((tuple(keys), kwargs, self.low_memory))
        yield from original_async(self, keys, **kwargs)

    def track_mark(self, key):
        marked.append(key)
        return original_mark(self, key)

    def reject_sync_get(*args, **kwargs):
        raise AssertionError("resize used synchronous get_tensor")

    monkeypatch.setattr(resize.MemoryEfficientSafeOpen, "async_stream", track_async)
    monkeypatch.setattr(resize.MemoryEfficientSafeOpen, "mark_processed", track_mark)
    monkeypatch.setattr(resize.MemoryEfficientSafeOpen, "get_tensor", reject_sync_get)

    output = resize.resize_lora_file(
        str(source), 3, None, None, "cpu", torch.float32, "mixed_rank_output",
        verbose=False,
    )

    assert len(async_calls) == 1
    streamed_keys, options, low_memory = async_calls[0]
    assert low_memory is True
    assert options == {
        "batch_size": 1,
        "prefetch_batches": 1,
        "pin_memory": False,
    }
    assert Counter(marked) == Counter(streamed_keys)
    assert all(count == 1 for count in Counter(marked).values())

    output_tensors = _read_uel(output, original_async, original_mark)
    assert output_tensors["diffusion_model.r2.lora_A.weight"].shape[0] == 2
    assert output_tensors["diffusion_model.r3.lora_A.weight"].shape[0] == 3
    assert output_tensors["diffusion_model.r2.alpha"].item() == pytest.approx(1.0)
    assert "diffusion_model.r3.alpha" not in output_tensors


def test_normalize_alpha_uses_bounded_uel_and_removes_alpha(
    monkeypatch, tmp_path, resize
):
    source = tmp_path / "alpha_input.safetensors"
    down = torch.eye(2, dtype=torch.float16)
    up = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float16)
    _write_uel(
        source,
        {
            "diffusion_model.block.lora_A.weight": down,
            "diffusion_model.block.lora_B.weight": up,
            "diffusion_model.block.alpha": torch.tensor(1.0),
            "other.weight": torch.tensor([3.0]),
        },
        metadata={"source": "test"},
    )
    monkeypatch.setattr(resize.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)

    original_async = resize.MemoryEfficientSafeOpen.async_stream
    original_mark = resize.MemoryEfficientSafeOpen.mark_processed
    async_calls = []
    marked = []

    def track_async(self, keys, **kwargs):
        async_calls.append((tuple(keys), kwargs, self.low_memory))
        yield from original_async(self, keys, **kwargs)

    def track_mark(self, key):
        marked.append(key)
        return original_mark(self, key)

    def reject_sync_get(*args, **kwargs):
        raise AssertionError("alpha normalization used synchronous get_tensor")

    monkeypatch.setattr(resize.MemoryEfficientSafeOpen, "async_stream", track_async)
    monkeypatch.setattr(resize.MemoryEfficientSafeOpen, "mark_processed", track_mark)
    monkeypatch.setattr(resize.MemoryEfficientSafeOpen, "get_tensor", reject_sync_get)

    output = resize.normalize_lora_alpha_file(
        str(source), "alpha_normalized", verbose=False
    )

    assert len(async_calls) == 1
    streamed_keys, options, low_memory = async_calls[0]
    assert streamed_keys == (
        "diffusion_model.block.lora_B.weight",
        "diffusion_model.block.alpha",
    )
    assert options == {
        "batch_size": 1,
        "prefetch_batches": 1,
        "pin_memory": False,
    }
    assert low_memory is True
    assert Counter(marked) == Counter(streamed_keys)

    output_tensors = _read_uel(output, original_async, original_mark)
    assert "diffusion_model.block.alpha" not in output_tensors
    assert torch.equal(output_tensors["diffusion_model.block.lora_A.weight"], down)
    assert torch.equal(
        output_tensors["diffusion_model.block.lora_B.weight"], up * 0.5
    )
    assert torch.equal(output_tensors["other.weight"], torch.tensor([3.0]))
    with MemoryEfficientSafeOpen(output, low_memory=True) as loader:
        assert loader.metadata()["alpha_normalized"] == "true"


def test_normalize_alpha_rejects_alpha_free_input(monkeypatch, tmp_path, resize):
    source = tmp_path / "no_alpha.safetensors"
    _write_uel(
        source,
        {
            "diffusion_model.block.lora_A.weight": torch.eye(2),
            "diffusion_model.block.lora_B.weight": torch.eye(2),
        },
    )
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)

    with pytest.raises(ValueError, match="contains no alpha tensors"):
        resize.normalize_lora_alpha_file(
            str(source), "unused", verbose=False
        )


def test_resize_does_not_invent_alpha_for_peft_pair(monkeypatch, tmp_path, resize):
    source = tmp_path / "peft_no_alpha.safetensors"
    _write_uel(
        source,
        {
            "base_model.model.transformer.block.to_q.lora_A.default.weight": (
                torch.eye(2)
            ),
            "base_model.model.transformer.block.to_q.lora_B.default.weight": (
                torch.eye(2)
            ),
        },
    )
    monkeypatch.setattr(resize.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(resize, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)

    output = resize.resize_lora_file(
        str(source), 2, None, None, "cpu", torch.float32, "peft_output",
        verbose=False,
    )

    output_tensors = _read_uel(output)
    assert len(output_tensors) == 2
    assert not any(key.endswith(".alpha") for key in output_tensors)


def test_cuda_oom_retries_only_current_pair_on_cpu(monkeypatch, resize):
    calls = []
    original = resize._resize_factors_on_device

    def fail_cuda_then_run_cpu(*args):
        process_device = args[-1]
        calls.append(process_device)
        if process_device == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return original(*args)

    monkeypatch.setattr(resize, "_resize_factors_on_device", fail_cuda_then_run_cpu)
    monkeypatch.setattr(resize, "_release_cuda_after_oom", lambda: None)

    result = resize._resize_lora_factors(
        torch.eye(2), torch.eye(2), 1, None, None, 1.0,
        process_device="cuda",
    )

    assert calls == ["cuda", "cpu"]
    assert result["cpu_fallback"] is True


def test_low_bit_pair_uses_raw_copy_without_tensor_loading(
    monkeypatch, tmp_path, resize
):
    source = tmp_path / "low_bit_pair.safetensors"
    down = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    up = torch.eye(2)
    _write_uel(
        source,
        {
            "diffusion_model.low.lora_A.weight": down,
            "diffusion_model.low.lora_B.weight": up,
        },
    )
    monkeypatch.setattr(resize.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(resize, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)
    async_calls = []
    original_async = resize.MemoryEfficientSafeOpen.async_stream
    original_mark = resize.MemoryEfficientSafeOpen.mark_processed

    def track_async(self, keys, **kwargs):
        async_calls.append(tuple(keys))
        yield from original_async(self, keys, **kwargs)

    def reject_sync_get(*args, **kwargs):
        raise AssertionError("low-bit raw copy decoded a tensor")

    monkeypatch.setattr(resize.MemoryEfficientSafeOpen, "async_stream", track_async)
    monkeypatch.setattr(resize.MemoryEfficientSafeOpen, "get_tensor", reject_sync_get)

    output = resize.resize_lora_file(
        str(source), 1, None, None, "cpu", torch.float16, "low_bit_output",
        verbose=False,
    )

    assert async_calls == []
    output_tensors = _read_uel(output, original_async, original_mark)
    assert torch.equal(output_tensors["diffusion_model.low.lora_A.weight"], down)
    assert torch.equal(output_tensors["diffusion_model.low.lora_B.weight"], up)


def test_resize_node_interfaces_remove_obsolete_loader_and_iteration_controls(resize):
    schemas = [
        resize.LoRAResizeFixed.define_schema(),
        resize.LoRAResizeRatio.define_schema(),
        resize.LoRAResizeFrobenius.define_schema(),
        resize.LoRAResizeCumulative.define_schema(),
    ]
    input_ids = [[item.id for item in schema.inputs] for schema in schemas]

    assert "svd_niter" not in input_ids[0]
    assert all("lazy_load" not in ids for ids in input_ids)
    assert input_ids[2][:4] == ["lora_name", "max_rank", "min_rank", "target"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_and_cpu_factor_space_results_match(resize):
    generator = torch.Generator().manual_seed(789)
    down = torch.randn((4, 11), generator=generator)
    up = torch.randn((9, 4), generator=generator)

    cpu = resize._resize_lora_factors(down, up, 3, None, None, 1.0, process_device="cpu")
    cuda = resize._resize_lora_factors(down, up, 3, None, None, 1.0, process_device="cuda")

    cpu_update = cpu["lora_up"] @ cpu["lora_down"]
    cuda_update = cuda["lora_up"] @ cuda["lora_down"]
    assert torch.allclose(cpu_update, cuda_update, atol=2e-4, rtol=2e-4)
