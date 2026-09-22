import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
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


def test_layer_parameters_apply_per_layer_without_leakage(monkeypatch, tmp_path, modules):
    merger, operations = modules

    def resolve(_payload, operation, layers, defaults, *, node_name):
        assert operation == "merge:Weight-Sum"
        assert node_name == "checkpoints Weight-Sum merger"
        assert set(layers) == {"match.weight", "other.weight"}
        assert defaults["alpha"] == 0.5
        return {"match.weight": {"alpha": 0.0}, "other.weight": {"alpha": 1.0}}

    monkeypatch.setattr(merger, "resolve_layer_parameters", resolve)
    result = _run(monkeypatch, tmp_path, merger, operations, _params("per_layer"))
    assert result["match.weight"].item() == 2.0
    assert result["other.weight"].item() == 8.0


def test_parsed_layer_parameters_change_only_matching_output(monkeypatch, tmp_path, modules):
    merger, operations = modules
    layer_parameters = _module("nodes.layer_parameters")
    params = _params("parsed")
    params["layer_parameters"] = layer_parameters.parse_rules("(match\\.weight) alpha:0\n")
    result = _run(monkeypatch, tmp_path, merger, operations, params)
    assert result["match.weight"].item() == 2.0
    assert result["other.weight"].item() == 6.0


def test_layer_parameter_failure_precedes_writer_and_streaming(monkeypatch, tmp_path, modules):
    merger, operations = modules
    monkeypatch.setattr(
        merger,
        "resolve_layer_parameters",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unmatched rule line 1")),
    )
    params = _params("must_not_exist")
    with pytest.raises(ValueError, match="unmatched rule"):
        _run(monkeypatch, tmp_path, merger, operations, params)
    assert not (tmp_path / "checkpoints" / "must_not_exist.safetensors").exists()


def test_lora_factor_group_uses_one_normalized_layer_name(monkeypatch, tmp_path, modules):
    merger, operations = modules
    paths = {name: str(tmp_path / f"{name}.safetensors") for name in ("a", "b")}
    tensors = {
        "diffusion_model.layer.lora_A.weight": torch.tensor([[1.0, 0.0]]),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[1.0], [2.0]]),
    }
    save_file({**tensors, "diffusion_model.layer.alpha": torch.tensor(1.0)}, paths["a"])
    save_file(tensors, paths["b"])
    monkeypatch.setattr(merger.folder_paths, "get_full_path", lambda _kind, name: paths.get(name))
    monkeypatch.setattr(merger.folder_paths, "get_folder_paths", lambda _kind: [str(tmp_path)])
    monkeypatch.setattr(merger.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(merger, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(merger, "cleanup_after_operation", lambda: None)
    params = _params("lora_group")
    params.update({"model_a": "a", "model_b": "b", "layer_parameters": _module("nodes.layer_parameters").parse_rules("(diffusion_model\\.layer) alpha:0\n")})
    merger.MergerLogic.execute_merge({"model_a": "a", "model_b": "b"}, "Weight-Sum", operations.TWO_MODEL_MODES, params, "loras")
    result = load_file(str(tmp_path / "loras" / "lora_group.safetensors"))
    assert "diffusion_model.layer.lora_A.weight" in result
    assert "diffusion_model.layer.lora_B.weight" in result
    assert not any(key.endswith(".alpha") for key in result)


def test_diffusion_merge_decodes_tensorwise_quantization_and_drops_sidecars(
    monkeypatch, tmp_path, modules,
):
    merger, operations = modules
    paths = {name: str(tmp_path / f"{name}.safetensors") for name in ("a", "b")}
    quant = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
    save_file({
        "model.diffusion_model.quant.weight": torch.tensor([1, 2], dtype=torch.int8),
        "model.diffusion_model.quant.weight_scale": torch.tensor(2.0),
        "model.diffusion_model.quant.comfy_quant": quant,
        "dense.weight": torch.tensor([2.0, 4.0]),
    }, paths["a"])
    save_file({
        "model.diffusion_model.quant.weight": torch.tensor([3, 6], dtype=torch.int8),
        "model.diffusion_model.quant.weight_scale": torch.tensor(2.0),
        "model.diffusion_model.quant.comfy_quant": quant,
        "dense.weight": torch.tensor([6.0, 8.0]),
    }, paths["b"])

    monkeypatch.setattr(merger.folder_paths, "get_full_path", lambda _kind, name: paths.get(name))
    monkeypatch.setattr(merger.folder_paths, "get_folder_paths", lambda _kind: [str(tmp_path)])
    monkeypatch.setattr(merger.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(merger, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(merger, "cleanup_after_operation", lambda: None)
    params = _params("quantized_merge")
    params.update({"model_a": "a", "model_b": "b"})

    merger.MergerLogic.execute_merge(
        {"model_a": "a", "model_b": "b"}, "Weight-Sum",
        operations.TWO_MODEL_MODES, params, "diffusion_models",
    )

    output_path = tmp_path / "diffusion_models" / "quantized_merge.safetensors"
    result = load_file(str(output_path))
    key = "model.diffusion_model.quant.weight"
    assert set(result) == {key, "dense.weight"}
    assert result[key].tolist() == pytest.approx([4.0, 8.0])
    assert result[key].dtype == torch.float32
    assert result["dense.weight"].tolist() == pytest.approx([4.0, 6.0])
    with safe_open(str(output_path), framework="pt", device="cpu") as output:
        metadata = output.metadata()
    assert metadata in (None, {})
