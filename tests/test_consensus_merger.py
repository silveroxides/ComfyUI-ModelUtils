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
    monkeypatch.setattr(module.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(module, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "cleanup_after_operation", lambda: None)


def _result_path(tmp_path, category, result):
    return tmp_path / category / result


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


def test_fixed_coordinate_merge_never_reorders_model_rows(cwb):
    settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reversed_source = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    expected = torch.stack([
        cwb.merge_consensus_group(
            torch.stack([reference[row], reversed_source[row]]), settings
        )
        for row in range(reference.shape[0])
    ])
    torch.testing.assert_close(
        cwb.merge_cwb_tensors(
            [reference, reversed_source],
            settings,
            allow_similarity_alignment=False,
        ),
        expected,
    )


def test_lora_similarity_alignment_pairs_a_rows_with_b_columns(cwb):
    settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    reference_down = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reference_up = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    source_down = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    source_up = torch.tensor([[0.0, -2.0], [3.0, 0.0]])

    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [reference_down, source_down],
        [reference_up, source_up],
        settings,
        reference_index=0,
    )
    torch.testing.assert_close(merged_down, reference_down)
    torch.testing.assert_close(merged_up, reference_up)


def test_lora_krea_shape_scores_only_rank_components(monkeypatch, cwb):
    rank = 256
    output_features = 36864
    down = torch.ones((rank, 2))
    up = torch.ones((output_features, rank))
    matrix_shapes = []

    def bounded_mm(left, right):
        matrix_shapes.append((left.shape, right.shape))
        return torch.eye(left.shape[0], right.shape[1], dtype=left.dtype)

    monkeypatch.setattr(cwb.torch, "mm", bounded_mm)
    monkeypatch.setattr(
        cwb,
        "merge_consensus_group",
        lambda stacked, settings, **kwargs: stacked[0].clone(),
    )
    settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        alignment_method="similarity",
        alignment_threshold=0.5,
    )
    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [down, down], [up, up], settings, reference_index=0
    )

    assert merged_down.shape == down.shape
    assert merged_up.shape == up.shape
    assert matrix_shapes
    assert all(left[0] == rank and right[1] == rank for left, right in matrix_shapes)


def test_lora_alpha_and_global_scale_apply_once_to_pair(cwb):
    settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        alignment_method="index",
        global_scale=3.0,
    )
    down = torch.ones((1, 1))
    up = torch.ones((1, 1))
    merged_down, merged_up, rank = cwb._merge_lora_pair_to_cpu(
        [(0, down, up, 2.0), (1, down, up, 2.0)],
        settings,
        "cpu",
        torch.float32,
    )
    assert rank == 1
    torch.testing.assert_close(merged_down, torch.ones((1, 1)))
    torch.testing.assert_close(merged_up, torch.full((1, 1), 6.0))


def test_lora_pair_alignment_supports_convolution_factors(cwb):
    settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        alignment_method="index",
    )
    down = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1)
    up = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1, 1)
    merged_down, merged_up = cwb.merge_cwb_lora_pairs(
        [down, down], [up, up], settings, reference_index=0
    )
    torch.testing.assert_close(merged_down, down)
    torch.testing.assert_close(merged_up, up)


def test_lora_cuda_oom_retries_current_pair_on_cpu(monkeypatch, caplog, cwb):
    settings = cwb.resolve_cwb_settings(
        cwb_preset="custom",
        consensus_type="mean",
        alignment_method="index",
    )
    original_to_compute = cwb._to_compute
    attempted_devices = []

    def fail_cuda(tensor, device):
        attempted_devices.append(device)
        if str(device).startswith("cuda"):
            raise torch.OutOfMemoryError("injected allocation failure")
        return original_to_compute(tensor, device)

    monkeypatch.setattr(cwb, "_to_compute", fail_cuda)
    monkeypatch.setattr(cwb, "_release_failed_cuda_operation", lambda: None)
    down = torch.ones((1, 2))
    up = torch.ones((2, 1))
    with caplog.at_level("WARNING"):
        merged_down, merged_up, rank = cwb._merge_lora_pair_to_cpu(
            [(0, down, up, 1.0), (1, down, up, 1.0)],
            settings,
            "cuda",
            torch.float32,
            operation_label="diffusion_model.foo",
        )

    assert rank == 1
    torch.testing.assert_close(merged_down, down)
    torch.testing.assert_close(merged_up, up)
    assert attempted_devices[0] == "cuda"
    assert "cpu" in attempted_devices
    assert "retrying this layer on CPU" in caplog.text
    assert "diffusion_model.foo" in caplog.text


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
    tensors = load_file(str(_result_path(tmp_path, "diffusion_models", result)))
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
    overridden_tensors = load_file(
        str(_result_path(tmp_path, "diffusion_models", overridden))
    )
    assert overridden_tensors["shared"].dtype == torch.float16
    assert overridden_tensors["metadata_tensor"].dtype == torch.int64


def test_failed_merge_preserves_existing_output_and_removes_temporary_file(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "atomic_a.safetensors"
    b = tmp_path / "atomic_b.safetensors"
    output = tmp_path / "diffusion_models" / "atomic_output.safetensors"
    output.parent.mkdir(parents=True)
    save_file({"layer": torch.ones((2, 2))}, str(a))
    save_file({"layer": torch.full((2, 2), 2.0)}, str(b))
    save_file({"existing": torch.tensor([7.0])}, str(output))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)

    def fail_merge(*args, **kwargs):
        raise RuntimeError("injected CWB failure")

    monkeypatch.setattr(cwb, "merge_cwb_tensors", fail_merge)
    with pytest.raises(RuntimeError, match="injected CWB failure"):
        cwb.ConsensusMergerLogic.execute(
            ["a", "b"], "diffusion_models", _params("atomic_output")
        )

    existing = load_file(str(output))
    assert set(existing) == {"existing"}
    torch.testing.assert_close(existing["existing"], torch.tensor([7.0]))
    assert not list(output.parent.glob(".atomic_output.safetensors.*.tmp"))


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
    tensors = load_file(str(_result_path(tmp_path, "embeddings", result)))
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
    tensor = load_file(str(_result_path(tmp_path, "diffusion_models", result)))["layer"]
    torch.testing.assert_close(tensor, torch.tensor([[3.0, 0.0]]))


def test_lora_cross_format_companion_group_preserves_model_a_canonically(
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
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    assert set(tensors) == {
        "diffusion_model.foo.lora_A.weight",
        "diffusion_model.foo.lora_B.weight",
        "diffusion_model.foo.dora_scale",
    }
    assert tensors["diffusion_model.foo.lora_A.weight"].shape == (1, 2)
    assert tensors["diffusion_model.foo.lora_B.weight"].shape == (2, 1)
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
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
    assert set(tensors) == {
        "diffusion_model.foo.lora_A.weight",
        "diffusion_model.foo.lora_B.weight",
        "diffusion_model.foo.alpha",
    }
    assert tensors["diffusion_model.foo.alpha"].item() == pytest.approx(2.0)


def test_lora_mochi_inputs_emit_preferred_canonical_output(
    monkeypatch, tmp_path, cwb
):
    a = tmp_path / "mochi_a.safetensors"
    b = tmp_path / "mochi_b.safetensors"
    for path, value in ((a, 1.0), (b, 3.0)):
        save_file({
            "diffusion_model.foo.lora_A": torch.full((1, 2), value),
            "diffusion_model.foo.lora_B": torch.full((2, 1), value),
            "diffusion_model.foo.alpha": torch.tensor(1.0),
        }, str(path))
    paths = {"a": str(a), "b": str(b)}
    _patch_io(monkeypatch, cwb, tmp_path, paths)

    result = cwb.ConsensusMergerLogic.execute(
        ["a", "b"], "loras", _params("mochi_canonical"), lora_mode=True
    )
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))

    assert set(tensors) == {
        "diffusion_model.foo.lora_A.weight",
        "diffusion_model.foo.lora_B.weight",
        "diffusion_model.foo.alpha",
    }


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
    default_tensor = load_file(str(_result_path(tmp_path, "loras", default)))[
        "diffusion_model.norm.diff"
    ]
    torch.testing.assert_close(default_tensor, torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    assert default_tensor.dtype == torch.bfloat16

    enabled = cwb.ConsensusMergerLogic.execute(
        ["a", "b"],
        "loras",
        _params("direct_enabled", include_1d_diffs=True, override_dtype=True),
        lora_mode=True,
    )
    enabled_tensor = load_file(str(_result_path(tmp_path, "loras", enabled)))[
        "diffusion_model.norm.diff"
    ]
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
    assert not (tmp_path / "diffusion_models" / "must_not_exist.safetensors").exists()


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
    tensor = load_file(str(_result_path(tmp_path, "diffusion_models", result)))["layer"]
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
    tensors = load_file(str(_result_path(tmp_path, "loras", result)))
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
