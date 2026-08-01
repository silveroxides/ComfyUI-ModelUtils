import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_cwb_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def cwb():
    return _load_module("nodes.consensus_merger")


def _patch_io(monkeypatch, module, tmp_path, paths):
    monkeypatch.setattr(module.folder_paths, "get_full_path", lambda _, name: paths.get(name))
    monkeypatch.setattr(module.folder_paths, "get_folder_paths", lambda _: [str(tmp_path)])
    monkeypatch.setattr(module, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "cleanup_after_operation", lambda: None)


def _params(output_filename, **overrides):
    params = {
        "cwb_preset": "custom",
        "consensus_type": "mean",
        "alignment_method": "index",
        "alignment_threshold": 0.4,
        "similarity_threshold": 0.0,
        "power_alpha": 2.0,
        "diversity_beta": 0.0,
        "rescale_norm": False,
        "global_scale": 1.0,
        "dynamic_similarity_contrast": False,
        "soft_comfort_bandpass": False,
        "position_weight": 0.0,
        "preserve_common_prefix": False,
        "mismatch_mode": "skip",
        "output_filename": output_filename,
        "save_dtype": "fp16",
        "process_device": "cpu",
        "exclude_patterns": "",
        "discard_patterns": "",
        "glob_patterns": False,
        "lazy_load": True,
        "force_clear_cache": False,
        "override_dtype": False,
        "include_1d_diffs": False,
    }
    params.update(overrides)
    return params


def test_preset_resolution_matches_extended_contract(cwb):
    power = cwb.resolve_cwb_settings(cwb_preset="power_blend")
    assert power.consensus_type == "median"
    assert power.alignment_threshold == pytest.approx(0.9)
    assert power.power_alpha == pytest.approx(8.0)
    assert power.dynamic_similarity_contrast is True
    assert power.soft_comfort_bandpass is False

    baseline = cwb.resolve_cwb_settings(
        cwb_preset="baseline",
        alignment_method="index",
        dynamic_similarity_contrast=True,
        soft_comfort_bandpass=True,
    )
    assert baseline.alignment_method == "similarity"
    assert baseline.dynamic_similarity_contrast is False
    assert baseline.soft_comfort_bandpass is False

    varied = cwb.resolve_cwb_settings(
        cwb_preset="varied_merge",
        global_scale=1.25,
        position_weight=0.2,
        preserve_common_prefix=True,
    )
    assert varied.global_scale == pytest.approx(1.25)
    assert varied.position_weight == pytest.approx(0.2)
    assert varied.preserve_common_prefix is True


def test_cwb_math_index_prefix_and_scale(cwb):
    settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        alignment_method="index",
        global_scale=1.0,
    )
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    second = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    torch.testing.assert_close(
        cwb.merge_cwb_tensors([first, second], settings),
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
    )

    prefix_settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        alignment_method="index",
        global_scale=2.0,
        preserve_common_prefix=True,
    )
    prefixed = cwb.merge_cwb_tensors(
        [torch.tensor([[9.0, 9.0], [1.0, 0.0]]),
         torch.tensor([[9.0, 9.0], [3.0, 0.0]])],
        prefix_settings,
    )
    torch.testing.assert_close(prefixed[0], torch.tensor([9.0, 9.0]))
    torch.testing.assert_close(prefixed[1], torch.tensor([4.0, 0.0]))


def test_similarity_alignment_reorders_matching_rows(cwb):
    settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reversed_source = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    torch.testing.assert_close(
        cwb.merge_cwb_tensors([reference, reversed_source], settings),
        torch.tensor([[1.5, 0.0], [0.0, 1.5]]),
    )

    scores = torch.ones((2, 2))
    biased = cwb._position_biased_scores(scores, 1.0)
    assert biased[0, 0] > biased[0, 1]


def test_norm_rescale_and_dsc_bandpass_are_finite(cwb):
    rescale = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        rescale_norm=True,
    )
    merged = cwb.merge_consensus_group(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]), rescale
    )
    assert torch.linalg.vector_norm(merged).item() == pytest.approx(2.0)

    dsc = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="median",
        diversity_beta=1.5,
        dynamic_similarity_contrast=True,
        soft_comfort_bandpass=True,
    )
    result = cwb.merge_consensus_group(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]), dsc
    )
    assert torch.isfinite(result).all()


def test_all_ten_schemas_and_lora_switch_contract(cwb):
    assert len(cwb.CWB_MERGER_NODES) == 10
    assert len({node.NODE_ID for node in cwb.CWB_MERGER_NODES}) == 10
    for node in cwb.CWB_MERGER_NODES:
        schema = node.define_schema()
        ids = [value.id for value in schema.inputs]
        assert ids[:3] == ["execution_mode", "model_a", "model_b"]
        if node.INPUT_COUNT == 3:
            assert ids[3] == "model_c"
        if node.LORA_MODE:
            assert schema.inputs[-1].id == "include_1d_diffs"
            assert schema.inputs[-1].default is False
        else:
            assert "include_1d_diffs" not in ids


def test_model_a_anchored_streaming_dtype_and_nonfloat_preservation(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    save_file({
        "shared": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "only_a": torch.tensor([5.0, 6.0], dtype=torch.float16),
        "metadata_tensor": torch.tensor([1, 2], dtype=torch.int64),
    }, str(a), metadata={"source": "A"})
    save_file({
        "shared": torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float16),
        "extra": torch.ones((2, 2)),
        "metadata_tensor": torch.tensor([9, 9], dtype=torch.int64),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "diffusion_models", _params("anchored")
    )
    tensors = load_file(str(tmp_path / result))
    assert set(tensors) == {"shared", "only_a", "metadata_tensor"}
    torch.testing.assert_close(tensors["shared"], torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    assert tensors["shared"].dtype == torch.float32
    torch.testing.assert_close(tensors["only_a"], torch.tensor([5.0, 6.0], dtype=torch.float16))
    torch.testing.assert_close(tensors["metadata_tensor"], torch.tensor([1, 2]))

    overridden = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "diffusion_models",
        _params("anchored_override", override_dtype=True),
    )
    overridden_tensors = load_file(str(tmp_path / overridden))
    assert overridden_tensors["shared"].dtype == torch.float16
    assert overridden_tensors["metadata_tensor"].dtype == torch.int64


def test_embedding_union_uses_longest_first_dimension(monkeypatch, tmp_path, cwb):
    a = tmp_path / "embed_a.safetensors"
    b = tmp_path / "embed_b.safetensors"
    save_file({"emb": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}, str(a))
    save_file({
        "emb": torch.tensor([[3.0, 0.0], [0.0, 3.0], [4.0, 4.0]]),
        "secondary_only": torch.tensor([[7.0, 0.0]]),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "embeddings",
        _params("embedding_union"),
        embedding_union=True,
    )
    tensors = load_file(str(tmp_path / result))
    assert tensors["emb"].shape == (3, 2)
    torch.testing.assert_close(tensors["emb"][:2], torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    torch.testing.assert_close(tensors["emb"][2], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(tensors["secondary_only"], torch.tensor([[7.0, 0.0]]))


def test_three_input_streaming_merge(monkeypatch, tmp_path, cwb):
    paths = {}
    for name, value in (("a", 1.0), ("b", 3.0), ("c", 5.0)):
        path = tmp_path / f"three_{name}.safetensors"
        save_file({"layer": torch.tensor([[value, 0.0]])}, str(path))
        paths[name] = str(path)
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b", "c"], "diffusion_models", _params("three_inputs")
    )
    tensor = load_file(str(tmp_path / result))["layer"]
    torch.testing.assert_close(tensor, torch.tensor([[3.0, 0.0]]))


def test_lora_cross_format_rank_padding_preserves_model_a_names(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "lora_a.safetensors"
    b = tmp_path / "lora_b.safetensors"
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        "diffusion_model.foo.dora_scale": torch.tensor([1.0, 1.0]),
    }, str(a))
    save_file({
        "lora_unet_foo.lora_down.weight": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        "lora_unet_foo.lora_up.weight": torch.tensor([[3.0, 5.0], [4.0, 6.0]]),
        "lora_unet_secondary.lora_down.weight": torch.ones((1, 2)),
        "lora_unet_secondary.lora_up.weight": torch.ones((2, 1)),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("lora_cross_format"), lora_mode=True
    )
    tensors = load_file(str(tmp_path / result))
    assert set(tensors) == {
        "diffusion_model.foo.lora_A.weight",
        "diffusion_model.foo.lora_B.weight",
        "diffusion_model.foo.dora_scale",
    }
    assert tensors["diffusion_model.foo.lora_A.weight"].shape == (2, 2)
    assert tensors["diffusion_model.foo.lora_B.weight"].shape == (2, 2)
    assert tensors["diffusion_model.foo.lora_A.weight"].dtype == torch.float32
    torch.testing.assert_close(
        tensors["diffusion_model.foo.dora_scale"], torch.tensor([1.0, 1.0])
    )


def test_lora_peft_prefix_matching_and_existing_alpha(monkeypatch, tmp_path, cwb):
    a = tmp_path / "peft_a.safetensors"
    b = tmp_path / "peft_b.safetensors"
    save_file({
        "base_model.model.diffusion_model.foo.lora_A.weight": torch.ones((1, 2)),
        "base_model.model.diffusion_model.foo.lora_B.weight": torch.ones((2, 1)),
        "base_model.model.diffusion_model.foo.alpha": torch.tensor(1.0),
    }, str(a))
    save_file({
        "diffusion_model.foo.lora_down.weight": torch.ones((2, 2)),
        "diffusion_model.foo.lora_up.weight": torch.ones((2, 2)),
        "diffusion_model.foo.alpha": torch.tensor(2.0),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("peft_prefix"), lora_mode=True
    )
    tensors = load_file(str(tmp_path / result))
    assert set(tensors) == {
        "base_model.model.diffusion_model.foo.lora_A.weight",
        "base_model.model.diffusion_model.foo.lora_B.weight",
        "base_model.model.diffusion_model.foo.alpha",
    }
    assert tensors["base_model.model.diffusion_model.foo.alpha"].item() == pytest.approx(2.0)


def test_lora_1d_direct_switch_is_default_fallback_and_fp32(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "direct_a.safetensors"
    b = tmp_path / "direct_b.safetensors"
    save_file({"diffusion_model.norm.diff": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}, str(a))
    save_file({"diffusion_model.norm.diff": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)}, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)

    default = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("direct_default"), lora_mode=True
    )
    default_tensor = load_file(str(tmp_path / default))["diffusion_model.norm.diff"]
    torch.testing.assert_close(default_tensor, torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    assert default_tensor.dtype == torch.bfloat16

    enabled = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "loras",
        _params("direct_enabled", include_1d_diffs=True, override_dtype=True),
        lora_mode=True,
    )
    enabled_tensor = load_file(str(tmp_path / enabled))["diffusion_model.norm.diff"]
    torch.testing.assert_close(enabled_tensor, torch.tensor([2.0, 3.0]))
    assert enabled_tensor.dtype == torch.float32


def test_quantized_marker_errors_before_output(monkeypatch, tmp_path, cwb):
    a = tmp_path / "quant_a.safetensors"
    b = tmp_path / "quant_b.safetensors"
    save_file({
        "layer.weight": torch.ones((2, 2)),
        "layer.comfy_quant": torch.tensor([1], dtype=torch.uint8),
    }, str(a))
    save_file({"layer.weight": torch.ones((2, 2))}, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    with pytest.raises(ValueError, match="ComfyUI quantization metadata"):
        cwb.ConsensusMergerLogic.execute(
            ["a", "b"], "diffusion_models", _params("must_not_exist")
        )
    assert not (tmp_path / "must_not_exist.safetensors").exists()


def test_isolated_low_bit_preserves_anchored_tensor(monkeypatch, tmp_path, cwb):
    a = tmp_path / "guard_a.safetensors"
    b = tmp_path / "guard_b.safetensors"
    save_file({"layer": torch.tensor([1.0, 2.0])}, str(a))
    save_file({"layer": torch.tensor([8, 9], dtype=torch.uint8)}, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "diffusion_models", _params("guarded_generic")
    )
    tensor = load_file(str(tmp_path / result))["layer"]
    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))
    assert tensor.dtype == torch.float32


def test_isolated_low_bit_preserves_complete_lora_layer(monkeypatch, tmp_path, cwb):
    a = tmp_path / "guard_lora_a.safetensors"
    b = tmp_path / "guard_lora_b.safetensors"
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        "diffusion_model.foo.alpha": torch.tensor(1.0),
        "diffusion_model.foo.dora_scale": torch.tensor([5.0, 6.0]),
    }, str(a))
    save_file({
        "diffusion_model.foo.lora_A.weight": torch.tensor([[9.0, 9.0]]),
        "diffusion_model.foo.lora_B.weight": torch.tensor([[9.0], [9.0]]),
        "diffusion_model.foo.alpha": torch.tensor(1, dtype=torch.uint8),
    }, str(b))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)
    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("guarded_lora"), lora_mode=True
    )
    tensors = load_file(str(tmp_path / result))
    torch.testing.assert_close(
        tensors["diffusion_model.foo.lora_A.weight"], torch.tensor([[1.0, 2.0]])
    )
    torch.testing.assert_close(
        tensors["diffusion_model.foo.lora_B.weight"], torch.tensor([[3.0], [4.0]])
    )
    assert tensors["diffusion_model.foo.alpha"].item() == pytest.approx(1.0)
    torch.testing.assert_close(
        tensors["diffusion_model.foo.dora_scale"], torch.tensor([5.0, 6.0])
    )
