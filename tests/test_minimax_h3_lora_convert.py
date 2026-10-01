import importlib
from pathlib import Path
import sys
import types

import pytest
from safetensors.torch import load_file, save_file
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_minimax_convert_tests"


def _load_module(name: str):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def converter():
    return _load_module("nodes.minimax_h3_lora_convert")


def _make_source_tensors(prefix: str = "transformer_blocks.0") -> dict[str, torch.Tensor]:
    tensors = {}
    for offset, part in enumerate(("q", "k", "v"), start=1):
        # Down: rank 2, in_dim 3
        tensors[f"{prefix}.attn.to_{part}.lora_A.default.weight"] = torch.tensor(
            [[1.0, float(offset), 0.0], [0.0, 1.0, float(offset)]],
            dtype=torch.float32,
        )
        # Up: out_dim 2, rank 2
        tensors[f"{prefix}.attn.to_{part}.lora_B.default.weight"] = torch.tensor(
            [[float(offset), 0.0], [0.0, float(offset) + 1.0]],
            dtype=torch.float32,
        )

    # out_proj: rank 2, in_dim 3, out_dim 2
    tensors[f"{prefix}.attn.to_out.0.lora_A.default.weight"] = torch.arange(
        6, dtype=torch.float32
    ).reshape(2, 3)
    tensors[f"{prefix}.attn.to_out.0.lora_B.default.weight"] = torch.arange(
        4, dtype=torch.float32
    ).reshape(2, 2)

    # mlp.fc1 (SwiGLU): rank 2, in_dim 3, out_dim 4 (halves of size 2)
    tensors[f"{prefix}.ff.net.0.proj.lora_A.default.weight"] = torch.ones(
        (2, 3), dtype=torch.float32
    )
    # Val half: rows 0..1 (all 1s), Gate half: rows 2..3 (all 2s)
    tensors[f"{prefix}.ff.net.0.proj.lora_B.default.weight"] = torch.cat(
        [torch.full((2, 2), 1.0), torch.full((2, 2), 2.0)], dim=0
    )

    # mlp.fc2: rank 2, in_dim 4, out_dim 3
    tensors[f"{prefix}.ff.net.2.lora_A.default.weight"] = torch.ones(
        (2, 4), dtype=torch.float32
    )
    tensors[f"{prefix}.ff.net.2.lora_B.default.weight"] = torch.ones(
        (3, 2), dtype=torch.float32
    )

    return tensors


def test_full_rank_qkv_conversion_is_exact(converter, tmp_path):
    source = tmp_path / "source_exact.safetensors"
    output = tmp_path / "output_exact.safetensors"

    original = _make_source_tensors("token_refiner.refiner_blocks.0")
    save_file(original, str(source))

    report = converter.convert_minimax_h3_diffusers_lora(str(source), str(output))
    assert "MiniMax H3 Diffusers LoRA conversion complete" in report
    assert "Fused QKV modules: 1" in report

    converted = load_file(str(output))

    down = converted["diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_A.weight"]
    up = converted["diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_B.weight"]

    # Ranks: 2 + 2 + 2 = 6, in_dim: 3, out_dim: 2 + 2 + 2 = 6
    assert down.shape == (6, 3)
    assert up.shape == (6, 6)

    # Mathematical delta equality: up @ down must equal stacked (up_q @ down_q, up_k @ down_k, up_v @ down_v)
    fused_delta = up @ down
    expected_delta = torch.cat(
        [
            original[f"token_refiner.refiner_blocks.0.attn.to_{p}.lora_B.default.weight"]
            @ original[f"token_refiner.refiner_blocks.0.attn.to_{p}.lora_A.default.weight"]
            for p in ("q", "k", "v")
        ],
        dim=0,
    )
    assert torch.allclose(fused_delta, expected_delta, atol=1e-5, rtol=1e-5)


def test_swiglu_mlp_fc1_halves_swapped(converter, tmp_path):
    source = tmp_path / "source_swiglu.safetensors"
    output = tmp_path / "output_swiglu.safetensors"

    original = _make_source_tensors("transformer_blocks.0")
    save_file(original, str(source))

    converter.convert_minimax_h3_diffusers_lora(str(source), str(output))
    converted = load_file(str(output))

    fc1_b = converted["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]
    orig_b = original["transformer_blocks.0.ff.net.0.proj.lora_B.default.weight"]

    # In source: top half (rows 0..1) is 1.0, bottom half (rows 2..3) is 2.0
    # In converted: top half must be 2.0 (gate), bottom half must be 1.0 (value)
    assert torch.equal(fc1_b[:2], orig_b[2:])
    assert torch.equal(fc1_b[2:], orig_b[:2])


def test_rejects_incomplete_qkv_group(converter, tmp_path):
    source = _make_source_tensors()
    del source["transformer_blocks.0.attn.to_v.lora_B.default.weight"]

    path = tmp_path / "incomplete.safetensors"
    save_file(source, str(path))

    with pytest.raises(ValueError, match="Incomplete LoRA A/B pair"):
        converter.convert_minimax_h3_diffusers_lora(
            str(path), str(tmp_path / "unused.safetensors")
        )


def test_rejects_unsupported_module(converter, tmp_path):
    source = {"unknown_layer.lora_A.weight": torch.ones(2, 2), "unknown_layer.lora_B.weight": torch.ones(2, 2)}
    path = tmp_path / "unknown.safetensors"
    save_file(source, str(path))

    with pytest.raises(ValueError, match="Unsupported MiniMax H3 Diffusers module"):
        converter.convert_minimax_h3_diffusers_lora(
            str(path), str(tmp_path / "unused.safetensors")
        )


def test_node_schema_and_execution_contract(converter, monkeypatch, tmp_path):
    schema = converter.MiniMaxH3DiffusersLoRAConvert.define_schema()
    assert schema.node_id == "MiniMaxH3DiffusersLoRAConvert"
    assert schema.is_output_node is True

    input_ids = [item.id for item in schema.inputs]
    assert input_ids == ["lora_name", "output_filename"]

    output_displays = [item.display_name for item in schema.outputs]
    assert output_displays == ["output_path", "conversion_report"]

    # Test execution through node
    models_dir = tmp_path / "models"
    loras_dir = models_dir / "loras"
    loras_dir.mkdir(parents=True)

    source_file = loras_dir / "test_diffusers_lora.safetensors"
    save_file(_make_source_tensors(), str(source_file))

    monkeypatch.setattr(converter.folder_paths, "models_dir", str(models_dir))
    monkeypatch.setattr(
        converter.folder_paths,
        "get_full_path_or_raise",
        lambda folder, name: str(loras_dir / name),
    )

    result = converter.MiniMaxH3DiffusersLoRAConvert.execute(
        lora_name="test_diffusers_lora.safetensors",
        output_filename="sub/converted_h3",
    )

    assert result.args[0] == "sub/converted_h3.safetensors"
    assert "MiniMax H3 Diffusers LoRA conversion complete" in result.args[1]
    assert (loras_dir / "sub" / "converted_h3.safetensors").is_file()
