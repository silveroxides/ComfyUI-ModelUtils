import importlib
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_knee_partial_svd_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module", params=["nodes.lora_extract_svd", "nodes.dora_extract_wd"])
def extraction(request):
    return _load_module(request.param)


def _fake_factors(weight, q, niter):
    del niter
    return (
        torch.eye(weight.shape[0], q),
        torch.linspace(float(q), 1.0, q),
        torch.eye(weight.shape[1], q),
    )


def test_knee_probe_uses_offset_and_expands_at_boundary(monkeypatch, extraction):
    probes = []
    detected = iter([9, 5])

    def fake_svd(weight, q, niter):
        probes.append(q)
        return _fake_factors(weight, q, niter)

    monkeypatch.setattr(extraction.torch, "svd_lowrank", fake_svd)
    monkeypatch.setattr(
        extraction, "_compute_rank", lambda singular_values, mode, param, cap: next(detected)
    )

    _, _, _, rank = extraction._svd_extract_knee_lowrank(
        torch.eye(20), "sv_knee", max_rank=6, probe_offset=4, niter=2
    )

    assert probes == [10, 12]
    assert rank == 5


def test_knee_probe_stops_when_knee_is_clear(monkeypatch, extraction):
    probes = []

    def fake_svd(weight, q, niter):
        probes.append(q)
        return _fake_factors(weight, q, niter)

    monkeypatch.setattr(extraction.torch, "svd_lowrank", fake_svd)
    monkeypatch.setattr(extraction, "_compute_rank", lambda *args: 3)

    _, _, _, rank = extraction._svd_extract_knee_lowrank(
        torch.eye(20), "sv_knee", max_rank=6, probe_offset=4, niter=2
    )

    assert probes == [10]
    assert rank == 3


def test_lowrank_factor_construction_does_not_build_residual(monkeypatch, extraction):
    monkeypatch.setattr(extraction.torch, "svd_lowrank", _fake_factors)

    result, mode = extraction._svd_extract_linear_lowrank(
        torch.eye(6), rank=3, device="cpu", clamp_quantile=1.0, niter=2
    )
    down, up, residual = result

    assert mode == "low rank"
    assert residual is None
    assert torch.equal(up, torch.eye(6, 3) * torch.tensor([3.0, 2.0, 1.0]))
    assert torch.equal(down, torch.eye(3, 6))


def test_knee_control_is_adjacent_to_method():
    modules_and_nodes = [
        (_load_module("nodes.lora_extract_svd"), "LoRAExtractKnee"),
        (_load_module("nodes.dora_extract_wd"), "DoRAExtractKnee"),
        (_load_module("nodes.dora_learned_wd"), "DoRALearnedExtractKnee"),
        (_load_module("nodes.text_encoder_extract"), "TextEncoderLoRAExtractKnee"),
        (_load_module("nodes.text_encoder_extract"), "TextEncoderDoRAExtractKnee"),
    ]

    for module, node_name in modules_and_nodes:
        names = [item.id for item in getattr(module, node_name).define_schema().inputs]
        method_index = names.index("knee_method")
        assert names[method_index + 1] == "knee_probe_offset"
