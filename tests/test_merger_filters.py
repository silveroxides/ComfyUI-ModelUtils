import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_merger_filter_tests"


def _module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def modules():
    return _module("nodes.merger"), _module("nodes.merger_ops")


def _params(name, *, patterns="", include=False, discard="", glob=False):
    return {
        "mismatch_mode": "skip", "alignment_mode": "pad/crop", "alpha": 0.5,
        "beta": 0.0, "gamma": 0.5, "delta": 2.0, "epsilon": 0.01,
        "zeta": 0.0, "seed": 0, "output_filename": name, "save_dtype": "fp32",
        "override_dtype": True, "device": "cpu", "dtype": torch.float32,
        "exclude_patterns": patterns, "include_mode": include,
        "discard_patterns": discard, "glob_patterns": glob, "lazy_load": True,
        "force_clear_cache": False,
    }


def _run(monkeypatch, tmp_path, merger, operations, params):
    paths = {name: str(tmp_path / f"{name}.safetensors") for name in ("a", "b")}
    save_file({"match.weight": torch.tensor([2.0]), "other.weight": torch.tensor([4.0])}, paths["a"])
    save_file({"match.weight": torch.tensor([6.0]), "other.weight": torch.tensor([8.0])}, paths["b"])
    monkeypatch.setattr(merger.folder_paths, "get_full_path", lambda _kind, name: paths.get(name))
    monkeypatch.setattr(merger.folder_paths, "get_folder_paths", lambda _kind: [str(tmp_path)])
    monkeypatch.setattr(merger.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(merger, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(merger, "cleanup_after_operation", lambda: None)
    params.update({"model_a": "a", "model_b": "b"})
    merger.MergerLogic.execute_merge({"model_a": "a", "model_b": "b"}, "Weight-Sum", operations.TWO_MODEL_MODES, params, "checkpoints")
    return load_file(str(tmp_path / "checkpoints" / f"{params['output_filename']}.safetensors"))


def test_include_filter_controls_preflight_and_streaming(monkeypatch, tmp_path, modules):
    merger, operations = modules
    selected = _run(monkeypatch, tmp_path, merger, operations, _params("selected", patterns="match", include=True))
    assert selected["match.weight"].item() == 4.0
    assert selected["other.weight"].item() == 4.0

    empty = _run(monkeypatch, tmp_path, merger, operations, _params("empty", include=True))
    assert empty["match.weight"].item() == 2.0
    assert empty["other.weight"].item() == 4.0

    discarded = _run(monkeypatch, tmp_path, merger, operations, _params("discarded", patterns="match", include=True, discard="match"))
    assert "match.weight" not in discarded
    assert discarded["other.weight"].item() == 4.0

    globbed = _run(monkeypatch, tmp_path, merger, operations, _params("globbed", patterns="*.weight", include=True, glob=True))
    assert globbed["match.weight"].item() == 4.0
    assert globbed["other.weight"].item() == 6.0
