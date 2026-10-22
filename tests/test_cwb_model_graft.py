import importlib
from pathlib import Path
import sys
import types

import pytest
from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def graft_mod():
    package_name = "modelutils_cwb_graft_tests"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.nodes.cwb_model_graft")


def _patch_io(monkeypatch, graft_mod, tmp_path, paths):
    monkeypatch.setattr(
        graft_mod.folder_paths,
        "get_full_path",
        lambda _kind, name: paths.get(name),
    )
    monkeypatch.setattr(
        graft_mod.folder_paths,
        "get_folder_paths",
        lambda _kind: [str(tmp_path)],
    )
    monkeypatch.setattr(graft_mod.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(graft_mod, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(graft_mod, "cleanup_after_operation", lambda: None)


def test_cwb_model_graft_schema_and_documentation(graft_mod):
    schema = graft_mod.CWBModelGraft.define_schema()
    assert schema.node_id == "CWBModelGraft"
    assert schema.display_name == "CWB Model Graft"
    ids = [inp.id for inp in schema.inputs]
    assert "attention_preset" in ids
    assert "ffn_preset" in ids
    assert "protect_early_blocks" in ids
    assert "attention_block_rule" in ids
    assert "ffn_block_rule" in ids
    assert "attention_subpart" in ids
    assert "ffn_subpart" in ids
    assert "custom_key_map" in ids
    assert ids[-1] == "include_mode"

    doc_out = graft_mod.CWBModelGraft.execute(execution_mode="DOCUMENTATION ONLY")
    filename, docs, report = doc_out.result
    assert filename == "Documentation mode active. No merge performed."
    assert "# CWB Model Graft" in docs
    assert "Complete Input Reference" in docs
    assert "no merge was performed" in report


def test_cwb_model_graft_protected_blocks_subparts_and_mismatched_shapes(
    monkeypatch, tmp_path, graft_mod
):
    base_path = tmp_path / "base.safetensors"
    donor_path = tmp_path / "donor.safetensors"

    base_qkv_0 = torch.ones((6, 4), dtype=torch.float32) * 2.0
    base_qkv_2 = torch.cat(
        [
            torch.ones((2, 4), dtype=torch.float32) * 1.0,
            torch.ones((2, 4), dtype=torch.float32) * 2.0,
            torch.ones((2, 4), dtype=torch.float32) * 3.0,
        ],
        dim=0,
    )
    base_ffn_2 = torch.cat(
        [
            torch.ones((2, 4), dtype=torch.float32) * 4.0,
            torch.ones((2, 4), dtype=torch.float32) * 5.0,
        ],
        dim=0,
    )
    base_custom = torch.ones((2, 4), dtype=torch.float32) * 2.0

    save_file(
        {
            "layers.0.proj.qkv.weight": base_qkv_0,
            "layers.2.proj.qkv.weight": base_qkv_2,
            "layers.2.proj.ffn_up_gate.weight": base_ffn_2,
            "layers.3.proj.qkv.weight": base_qkv_0.clone(),
            "custom_target.weight": base_custom,
        },
        str(base_path),
    )

    save_file(
        {
            "layers.0.proj.qkv.weight": torch.ones((6, 4), dtype=torch.float32) * 99.0,
            "layers.2.proj.q.weight": torch.ones((2, 4), dtype=torch.float32) * 50.0,
            "layers.2.proj.k.weight": torch.ones((2, 4), dtype=torch.float32) * 60.0,
            "layers.2.proj.v.weight": torch.tensor(
                [[9.0, 1.0, 1.0, 1.0], [1.0, 9.0, 1.0, 1.0]], dtype=torch.float32
            ),
            "layers.2.proj.ffn_up_gate.weight": torch.cat(
                [
                    torch.tensor(
                        [[8.0, 1.0, 1.0, 1.0], [1.0, 8.0, 1.0, 1.0]],
                        dtype=torch.float32,
                    ),
                    torch.ones((2, 4), dtype=torch.float32) * 77.0,
                ],
                dim=0,
            ),
            "layers.3.proj.qkv.weight": torch.ones((6, 4), dtype=torch.float32) * 88.0,
            "custom_donor.weight": torch.tensor(
                [
                    [10.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 10.0, 2.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 10.0, 2.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        },
        str(donor_path),
    )

    paths = {"base": str(base_path), "donor": str(donor_path)}
    _patch_io(monkeypatch, graft_mod, tmp_path, paths)

    out = graft_mod.CWBModelGraft.execute(
        execution_mode="MERGE",
        model_a="base",
        model_b="donor",
        attention_preset="balanced_mean",
        attention_config=None,
        ffn_preset="balanced_mean",
        ffn_config=None,
        protect_early_blocks=2,
        attention_block_rule="interleaved_even",
        ffn_block_rule="all_after_protected",
        attention_subpart="value_and_output",
        ffn_subpart="up_only",
        custom_key_map="custom_donor.weight -> custom_target.weight",
        output_filename="grafted",
        save_dtype="fp32",
        process_device="cpu",
        exclude_patterns="",
        discard_patterns="",
        glob_patterns=False,
        lazy_load=True,
        force_clear_cache=False,
        override_dtype=False,
        include_mode=False,
    )

    rel_name, docs, report = out.result
    assert "# CWB Model Graft" in docs
    assert "attn=balanced_mean, ffn=balanced_mean" in report
    assert "Grafted tensors: 3" in report
    tensors = load_file(str(tmp_path / "diffusion_models" / rel_name))

    # 1. Protected block 0 is byte-exact identical to Base
    assert torch.equal(tensors["layers.0.proj.qkv.weight"], base_qkv_0)
    # 2. Odd block 3 attention is untouched under interleaved_even
    assert torch.equal(tensors["layers.3.proj.qkv.weight"], base_qkv_0)
    # 3. Even block 2 qkv: Q and K slices stay Base, only V slice is grafted from split donor
    q_out, k_out, v_out = tensors["layers.2.proj.qkv.weight"].chunk(3, dim=0)
    assert torch.equal(q_out, base_qkv_2[:2])
    assert torch.equal(k_out, base_qkv_2[2:4])
    assert not torch.equal(v_out, base_qkv_2[4:6])
    # 4. Block 2 ffn_up_gate: up half is grafted, gate half stays Base
    up_out, gate_out = tensors["layers.2.proj.ffn_up_gate.weight"].chunk(2, dim=0)
    assert not torch.equal(up_out, base_ffn_2[:2])
    assert torch.equal(gate_out, base_ffn_2[2:])
    # 5. Mismatched-shape custom mapped tensor blends without changing shape or norm
    assert tensors["custom_target.weight"].shape == base_custom.shape
    assert not torch.equal(tensors["custom_target.weight"], base_custom)
    assert torch.linalg.vector_norm(tensors["custom_target.weight"]).item() == pytest.approx(
        torch.linalg.vector_norm(base_custom).item(), rel=1e-5
    )
