import importlib
import sys
import types
from collections import Counter
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter, MemoryEfficientSafeOpen


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_adaptive_extraction_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def adaptive():
    return _load_module("nodes.adaptive_svd")


def _exact_partial(weight, q, niter):
    del niter
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    return U[:, :q], S[:q], Vh[:q, :].T


@pytest.mark.parametrize(
    ("mode", "parameter", "expected"),
    [("ratio", 2.0, 2), ("quantile", 0.7, 2), ("sv_fro", 0.8, 1)],
)
def test_adaptive_partial_rank_matches_reference(monkeypatch, adaptive, mode, parameter, expected):
    monkeypatch.setattr(adaptive.torch, "svd_lowrank", _exact_partial)
    weight = torch.diag(torch.tensor([8.0, 5.0, 2.0, 1.0, 0.5, 0.25]))
    _, _, _, rank = adaptive.adaptive_partial_svd(
        weight, mode, parameter, max_rank=4, probe_offset=1, niter=2
    )
    assert rank == expected


def test_unresolved_ratio_expands_and_caps(monkeypatch, adaptive):
    probes = []

    def tracked(weight, q, niter):
        probes.append(q)
        return _exact_partial(weight, q, niter)

    monkeypatch.setattr(adaptive.torch, "svd_lowrank", tracked)
    weight = torch.diag(torch.linspace(10.0, 8.0, 12))
    _, _, _, rank = adaptive.adaptive_partial_svd(
        weight, "ratio", 100.0, max_rank=4, probe_offset=2, niter=2
    )
    assert probes == [6, 8]
    assert rank == 4


def test_frobenius_uses_energy_outside_output_cap(monkeypatch, adaptive):
    monkeypatch.setattr(adaptive.torch, "svd_lowrank", _exact_partial)
    weight = torch.eye(10)
    _, _, _, rank = adaptive.adaptive_partial_svd(
        weight, "sv_fro", 0.9, max_rank=3, probe_offset=2, niter=2
    )
    assert rank == 3


class _FakeStream:
    def __init__(self, batches):
        self._batches = iter(batches)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._batches)

    def close(self):
        self.closed = True


class _FakeHandler:
    def __init__(self, values):
        self.values = values
        self.calls = []
        self.marked = []
        self.streams = []

    def async_stream(self, keys, **kwargs):
        self.calls.append((tuple(keys), kwargs))
        stream = _FakeStream([[(key, self.values[key])] for key in keys])
        self.streams.append(stream)
        return stream

    def mark_processed(self, key):
        self.marked.append(key)


def test_paired_stream_is_bounded_ordered_and_releases_once():
    streaming = _load_module("nodes.extraction_stream")
    handler_a = _FakeHandler({"a1": torch.tensor(1), "a2": torch.tensor(2)})
    handler_b = _FakeHandler({"b1": torch.tensor(3)})
    units = [("one", "a1", "b1"), ("two", "a2", None)]

    observed = []
    for key, tensor_a, tensor_b in streaming.paired_async_tensors(
        handler_a, handler_b, units, pin_memory=True
    ):
        observed.append((key, tensor_a.item(), None if tensor_b is None else tensor_b.item()))

    assert observed == [("one", 1, 3), ("two", 2, None)]
    assert handler_a.calls == [(('a1', 'a2'), {"batch_size": 1, "prefetch_batches": 1, "pin_memory": True})]
    assert handler_b.calls == [(('b1',), {"batch_size": 1, "prefetch_batches": 1, "pin_memory": True})]
    assert Counter(handler_a.marked) == Counter({"a1": 1, "a2": 1})
    assert Counter(handler_b.marked) == Counter({"b1": 1})
    assert all(stream.closed for stream in handler_a.streams + handler_b.streams)


def test_cuda_oom_compute_retries_on_cpu(monkeypatch):
    streaming = _load_module("nodes.extraction_stream")
    monkeypatch.setattr(streaming, "release_cuda_oom", lambda: None)
    calls = []

    def fail_cuda():
        calls.append("cuda")
        raise RuntimeError("CUDA out of memory")

    def run_cpu():
        calls.append("cpu")
        return 7

    assert streaming.retry_cuda_oom_on_cpu(fail_cuda, run_cpu, "cuda:0") == 7
    assert calls == ["cuda", "cpu"]


def test_adaptive_controls_cover_all_public_families():
    modules_and_nodes = [
        (_load_module("nodes.lora_extract_svd"), prefix)
        for prefix in ("LoRAExtractRatio", "LoRAExtractQuantile", "LoRAExtractFrobenius")
    ]
    modules_and_nodes += [
        (_load_module("nodes.dora_extract_wd"), prefix)
        for prefix in ("DoRAExtractRatio", "DoRAExtractQuantile", "DoRAExtractFrobenius")
    ]
    modules_and_nodes += [
        (_load_module("nodes.dora_learned_wd"), prefix)
        for prefix in ("DoRALearnedExtractRatio", "DoRALearnedExtractQuantile", "DoRALearnedExtractFrobenius")
    ]
    modules_and_nodes += [
        (_load_module("nodes.text_encoder_extract"), prefix)
        for prefix in (
            "TextEncoderLoRAExtractRatio", "TextEncoderLoRAExtractQuantile",
            "TextEncoderLoRAExtractFrobenius", "TextEncoderDoRAExtractRatio",
            "TextEncoderDoRAExtractQuantile", "TextEncoderDoRAExtractFrobenius",
        )
    ]

    for module, node_name in modules_and_nodes:
        names = [item.id for item in getattr(module, node_name).define_schema().inputs]
        probe_index = names.index("probe_offset")
        assert names[probe_index + 1] == "linear_max_rank"


def _write_uel(path, tensors):
    with IncrementalSafetensorsWriter(str(path), max_workers=1) as writer:
        writer.write_dict(tensors)


def test_lora_extraction_uses_only_async_uel_and_releases_sources(monkeypatch, tmp_path):
    extraction = _load_module("nodes.lora_extract_svd")
    model_a = tmp_path / "a.safetensors"
    model_b = tmp_path / "b.safetensors"
    output = tmp_path / "out.safetensors"
    key = "model.diffusion_model.layer.weight"
    _write_uel(model_a, {key: torch.eye(3)})
    _write_uel(model_b, {key: torch.zeros(3, 3)})
    monkeypatch.setattr(extraction, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(extraction, "cleanup_after_operation", lambda: None)
    original_async = extraction.MemoryEfficientSafeOpen.async_stream
    original_mark = extraction.MemoryEfficientSafeOpen.mark_processed
    async_calls = []
    marked = []

    def track_async(self, keys, **kwargs):
        async_calls.append((self.filename, tuple(keys), kwargs))
        yield from original_async(self, keys, **kwargs)

    def track_mark(self, source_key):
        marked.append((self.filename, source_key))
        return original_mark(self, source_key)

    monkeypatch.setattr(extraction.MemoryEfficientSafeOpen, "async_stream", track_async)
    monkeypatch.setattr(extraction.MemoryEfficientSafeOpen, "mark_processed", track_mark)
    monkeypatch.setattr(
        extraction.MemoryEfficientSafeOpen,
        "get_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("synchronous tensor load")),
    )

    extraction.extract_lora_from_files(
        str(model_a), str(model_b), "fixed", 1, 1, "cpu", "fp32", str(output),
        linear_max_rank=1, conv_max_rank=1, force_clear_cache=False,
    )

    assert output.is_file()
    assert len(async_calls) == 2
    assert all(options == {"batch_size": 1, "prefetch_batches": 1, "pin_memory": False} for _, _, options in async_calls)
    assert Counter(marked) == Counter((filename, key) for filename, keys, _ in async_calls for key in keys)
    with MemoryEfficientSafeOpen(str(output), low_memory=True) as result:
        assert sorted(result.keys()) == [
            "diffusion_model.layer.lora_A.weight",
            "diffusion_model.layer.lora_B.weight",
        ]


@pytest.mark.parametrize("module_name, function_name, extra", [
    ("nodes.lora_extract_svd", "extract_lora_from_files", {"include_1d_diffs": False}),
    ("nodes.dora_extract_wd", "extract_dora_from_files", {}),
    ("nodes.dora_learned_wd", "extract_dora_learned_from_files", {"optimize_iters": 0}),
    ("nodes.text_encoder_extract", "extract_te_from_files", {"is_dora": False}),
    ("nodes.text_encoder_extract", "extract_te_from_files", {"is_dora": True}),
])
def test_extraction_include_mode_selects_only_matching_layers(monkeypatch, tmp_path, module_name, function_name, extra):
    module = _load_module(module_name)
    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    match, other = "model.encoder.match.weight", "model.encoder.other.weight"
    _write_uel(first, {match: torch.eye(2), other: torch.eye(2)})
    _write_uel(second, {match: torch.zeros(2, 2), other: torch.zeros(2, 2)})
    monkeypatch.setattr(module, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(module, "cleanup_after_operation", lambda: None)
    common = dict(linear_max_rank=1, conv_max_rank=1, force_clear_cache=False,
                  skip_patterns_str="match", include_mode=True, **extra)
    selected = tmp_path / "selected.safetensors"
    getattr(module, function_name)(str(first), str(second), "fixed", 1, 1, "cpu", "fp32", str(selected), **common)
    with MemoryEfficientSafeOpen(str(selected), low_memory=True) as result:
        keys = list(result.keys())
        assert keys
        assert all("match" in key for key in keys)
    empty = tmp_path / "empty.safetensors"
    common["skip_patterns_str"] = ""
    getattr(module, function_name)(str(first), str(second), "fixed", 1, 1, "cpu", "fp32", str(empty), **common)
    with MemoryEfficientSafeOpen(str(empty), low_memory=True) as result:
        assert list(result.keys()) == []
    globbed = tmp_path / "globbed.safetensors"
    common.update(skip_patterns_str="*.match.weight", glob_skip_patterns=True)
    getattr(module, function_name)(str(first), str(second), "fixed", 1, 1, "cpu", "fp32", str(globbed), **common)
    with MemoryEfficientSafeOpen(str(globbed), low_memory=True) as result:
        assert list(result.keys()) == keys
