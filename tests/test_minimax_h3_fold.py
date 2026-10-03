import importlib
from pathlib import Path
import sys
import types

import pytest
from safetensors.torch import load_file, save_file
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_minimax_fold_tests"


def _load_module(name: str):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def folder_module():
    return _load_module("nodes.minimax_h3_fold")


def _make_h3_checkpoint(prefix: str = "") -> dict[str, torch.Tensor]:
    return {
        f"{prefix}time_embedder.proj_in.weight": torch.eye(16, dtype=torch.float32),
        f"{prefix}time_embedder.proj_in.bias": torch.zeros(16, dtype=torch.float32),
        f"{prefix}time_embedder.proj_out.weight": torch.ones((2688, 16), dtype=torch.float32) * 0.01,
        f"{prefix}time_embedder.proj_out.bias": torch.zeros(2688, dtype=torch.float32),
        f"{prefix}blocks.0.adaln_proj.linear.weight": torch.ones((128, 2688), dtype=torch.float32),
        f"{prefix}blocks.0.adaln_proj.linear.bias": torch.zeros(128, dtype=torch.float32),
        f"{prefix}blocks.1.adaln_proj.linear.weight": torch.ones((128, 2688), dtype=torch.float32) * 2.0,
        f"{prefix}blocks.1.adaln_proj.linear.bias": torch.zeros(128, dtype=torch.float32),
        f"{prefix}blocks.0.attn.qkv_proj.weight": torch.eye(64, dtype=torch.float32),
    }


def test_fold_h3_diffusion_model_exact_basis(folder_module, tmp_path):
    source = tmp_path / "model_full.safetensors"
    output = tmp_path / "model_folded.safetensors"

    original = _make_h3_checkpoint()
    save_file(original, str(source))

    report = folder_module.fold_minimax_h3_diffusion_model(
        str(source), str(output), process_device="cpu"
    )
    assert "MiniMax H3 AdaLN Fold complete" in report
    assert "AdaLN layers folded: 4" in report

    converted = load_file(str(output))

    assert "adaln_t_table" in converted
    assert converted["adaln_t_table"].shape == (1025, 8)
    assert converted["adaln_t_table"].dtype == torch.float32

    # Folded AdaLN weight and bias
    assert converted["blocks.0.adaln_proj.linear.weight"].shape == (128, 8)
    assert converted["blocks.0.adaln_proj.linear.weight"].dtype == torch.float16
    assert converted["blocks.0.adaln_proj.linear.bias"].dtype == torch.float16

    # time_embedder should be dropped
    assert "time_embedder.proj_in.weight" not in converted
    assert "time_embedder.proj_out.weight" not in converted

    # Unrelated layers preserved
    assert "blocks.0.attn.qkv_proj.weight" in converted
    assert torch.equal(converted["blocks.0.attn.qkv_proj.weight"], original["blocks.0.attn.qkv_proj.weight"])


def test_fold_h3_discard_patterns(folder_module, tmp_path):
    source = tmp_path / "model_discard.safetensors"
    output = tmp_path / "model_discarded.safetensors"

    original = _make_h3_checkpoint()
    save_file(original, str(source))

    folder_module.fold_minimax_h3_diffusion_model(
        str(source),
        str(output),
        discard_patterns="blocks.0.attn",
        process_device="cpu",
    )
    converted = load_file(str(output))

    assert "blocks.0.attn.qkv_proj.weight" not in converted
    assert "blocks.0.adaln_proj.linear.weight" in converted


def test_fold_h3_exclude_patterns_preserves_unfolded_and_retains_time_embedder(folder_module, tmp_path):
    source = tmp_path / "model_partial.safetensors"
    output = tmp_path / "model_partial_folded.safetensors"

    original = _make_h3_checkpoint()
    save_file(original, str(source))

    folder_module.fold_minimax_h3_diffusion_model(
        str(source),
        str(output),
        exclude_patterns="blocks.1",
        include_mode=False,
        process_device="cpu",
    )
    converted = load_file(str(output))

    # Block 0 folded
    assert converted["blocks.0.adaln_proj.linear.weight"].shape == (128, 8)
    # Block 1 kept at full width
    assert converted["blocks.1.adaln_proj.linear.weight"].shape == (128, 2688)

    # time_embedder MUST be retained because block 1 needs it
    assert "time_embedder.proj_in.weight" in converted
    assert "time_embedder.proj_out.weight" in converted


def test_node_schema_and_execution_contract(folder_module, monkeypatch, tmp_path):
    schema = folder_module.MiniMaxH3FoldAdaLN.define_schema()
    assert schema.node_id == "MiniMaxH3FoldAdaLN"
    assert schema.is_output_node is True

    input_ids = [item.id for item in schema.inputs]
    assert "model_name" in input_ids
    assert "output_filename" in input_ids
    assert "exclude_patterns" in input_ids
    assert "include_mode" in input_ids
    assert "discard_patterns" in input_ids
    assert "glob_patterns" in input_ids

    # Every input must have a non-empty static tooltip
    for item in schema.inputs:
        assert hasattr(item, "tooltip") and item.tooltip.strip()

    output_displays = [item.display_name for item in schema.outputs]
    assert output_displays == ["output_path", "report"]

    # Test execution
    models_dir = tmp_path / "models"
    diff_dir = models_dir / "diffusion_models"
    diff_dir.mkdir(parents=True)

    source_file = diff_dir / "test_h3_model.safetensors"
    save_file(_make_h3_checkpoint("diffusion_model."), str(source_file))

    monkeypatch.setattr(folder_module.folder_paths, "models_dir", str(models_dir))
    monkeypatch.setattr(
        folder_module.folder_paths,
        "get_full_path_or_raise",
        lambda folder, name: str(diff_dir / name),
    )

    result = folder_module.MiniMaxH3FoldAdaLN.execute(
        model_name="test_h3_model.safetensors",
        output_filename="sub/folded_h3",
        process_device="cpu",
    )

    assert result.args[0] == "sub/folded_h3.safetensors"
    assert "MiniMax H3 AdaLN Fold complete" in result.args[1]
    assert (diff_dir / "sub" / "folded_h3.safetensors").is_file()
