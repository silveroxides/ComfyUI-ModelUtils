import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_contract_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def modules():
    return _load_module("nodes.lora_resize"), _load_module("nodes.lora_merger")


@pytest.fixture(scope="module")
def generic_modules():
    return _load_module("nodes.merger"), _load_module("nodes.merger_ops")


def _patch_output(monkeypatch, module, output_dir):
    monkeypatch.setattr(module.folder_paths, "get_folder_paths", lambda _: [str(output_dir)])
    monkeypatch.setattr(module, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "cleanup_after_operation", lambda: None)


def test_mixed_parser_scans_every_key_and_retains_unpaired(modules):
    resize, _ = modules
    filler = [f"metadata.tensor_{index}" for index in range(60)]
    keys = filler + [
        "diffusion_model.linear.lora_down.weight",
        "diffusion_model.linear.lora_up.weight",
        "diffusion_model.linear.alpha",
        "diffusion_model.norm.diff",
        "diffusion_model.block.diff_b",
        "diffusion_model.orphan.lora_down.weight",
        "diffusion_model.linear.dora_scale",
    ]

    info = resize.detect_lora_format(keys)
    layers, passthrough = resize.parse_lora_layers(keys)

    assert info["format"] == "mixed"
    assert layers["diffusion_model.linear"]["down"].endswith("lora_down.weight")
    assert layers["diffusion_model.norm"]["diff"].endswith(".diff")
    assert layers["diffusion_model.block"]["diff_b"].endswith(".diff_b")
    assert "diffusion_model.orphan.lora_down.weight" in passthrough
    assert "diffusion_model.linear.dora_scale" in passthrough


def test_dtype_contract(modules):
    resize, _ = modules
    assert resize.select_output_dtype([torch.bfloat16], torch.float16, is_1d_diff=True) == torch.float32
    assert resize.select_output_dtype([torch.float32, torch.bfloat16], torch.float16) == torch.float32
    assert resize.select_output_dtype([torch.float32], torch.bfloat16, force=True) == torch.bfloat16


def test_include_1d_switches_are_appended_and_default_off(modules, generic_modules):
    resize, multi = modules
    generic, _ = generic_modules
    node_classes = [
        generic.LoRATwoMerger,
        generic.LoRAThreeMerger,
        multi.LoRAMultiMerge,
        multi.LoRAMultiMergeDARE,
        multi.LoRAMultiMergeDAREEnhanced,
        resize.LoRAMergeToModel,
    ]

    for node_class in node_classes:
        final_input = node_class.define_schema().inputs[-1]
        assert final_input.id == "include_1d_diffs"
        assert final_input.default is False


def test_resize_preserves_mixed_direct_auxiliary_and_fp32(monkeypatch, tmp_path, modules):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    source = tmp_path / "mixed.safetensors"
    save_file({
        "diffusion_model.linear.lora_down.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "diffusion_model.linear.lora_up.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "diffusion_model.linear.alpha": torch.tensor(2.0, dtype=torch.float32),
        "diffusion_model.norm.diff": torch.tensor([0.125, -0.25], dtype=torch.bfloat16),
        "diffusion_model.scale.diff": torch.tensor(0.5, dtype=torch.bfloat16),
        "diffusion_model.lin.diff": torch.full((2, 2), 1e-8, dtype=torch.float32),
        "diffusion_model.linear.dora_scale": torch.ones(4, dtype=torch.float32),
        "diffusion_model.orphan.lora_down.weight": torch.ones((1, 2), dtype=torch.bfloat16),
    }, str(source))

    output = resize.resize_lora_file(
        str(source), 1, None, None, "cpu", torch.float16, "resized", verbose=False
    )
    tensors = load_file(output)

    assert tensors["diffusion_model.linear.lora_down.weight"].shape[0] == 1
    assert tensors["diffusion_model.linear.lora_down.weight"].dtype == torch.float32
    assert tensors["diffusion_model.linear.lora_up.weight"].dtype == torch.float32
    assert tensors["diffusion_model.linear.alpha"].dtype == torch.float32
    assert tensors["diffusion_model.norm.diff"].dtype == torch.float32
    assert tensors["diffusion_model.scale.diff"].dtype == torch.float16
    assert tensors["diffusion_model.lin.diff"].dtype == torch.float32
    assert tensors["diffusion_model.linear.dora_scale"].dtype == torch.float32
    assert tensors["diffusion_model.orphan.lora_down.weight"].dtype == torch.float16

    pure_source = tmp_path / "pure_diff.safetensors"
    save_file({
        "diffusion_model.pure.diff": torch.ones((2, 2), dtype=torch.bfloat16),
        "diffusion_model.pure_norm.diff": torch.ones(2, dtype=torch.bfloat16),
    }, str(pure_source))
    pure_output = resize.resize_lora_file(
        str(pure_source), 1, None, None, "cpu", torch.float16, "pure_resized", verbose=False
    )
    pure_tensors = load_file(pure_output)
    assert set(pure_tensors) == {"diffusion_model.pure.diff", "diffusion_model.pure_norm.diff"}
    assert pure_tensors["diffusion_model.pure.diff"].dtype == torch.float16
    assert pure_tensors["diffusion_model.pure_norm.diff"].dtype == torch.float32


def test_standard_multi_merge_emits_weighted_direct_union(monkeypatch, tmp_path, modules):
    _, merger = modules
    _patch_output(monkeypatch, merger, tmp_path)
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    base = tmp_path / "multi_base.safetensors"
    save_file({
        "model.diffusion_model.norm.weight": torch.zeros(2),
        "model.diffusion_model.matrix.weight": torch.zeros((2, 2)),
        "model.diffusion_model.only_first.weight": torch.zeros(()),
        "model.diffusion_model.scale": torch.zeros(()),
        "model.diffusion_model.lowrank.weight": torch.zeros((2, 2)),
    }, str(base))
    save_file({
        "diffusion_model.norm.diff": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "diffusion_model.matrix.diff": torch.ones((2, 2), dtype=torch.float32),
        "diffusion_model.only_first.diff": torch.tensor(4.0, dtype=torch.bfloat16),
        "diffusion_model.scale.diff": torch.tensor(1.0, dtype=torch.bfloat16),
        "diffusion_model.lowrank.lora_down.weight": torch.ones((1, 2), dtype=torch.float32),
        "diffusion_model.lowrank.lora_up.weight": torch.ones((2, 1), dtype=torch.float32),
        "diffusion_model.lowrank.alpha": torch.tensor(1.0, dtype=torch.float32),
    }, str(first))
    save_file({
        "diffusion_model.norm.diff": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        "diffusion_model.matrix.diff": torch.full((2, 2), 2.0, dtype=torch.bfloat16),
        "diffusion_model.lowrank.lora_down.weight": torch.full((1, 2), 2.0, dtype=torch.bfloat16),
        "diffusion_model.lowrank.lora_up.weight": torch.full((2, 1), 2.0, dtype=torch.bfloat16),
        "diffusion_model.lowrank.alpha": torch.tensor(1.0, dtype=torch.bfloat16),
    }, str(second))

    output = merger.merge_multi_loras(
        [str(first), str(second)], [2.0, -1.0], "concatenate", "cpu",
        torch.float16, "merged", base_model_path=str(base), verbose=False,
        include_1d_diffs=True,
    )
    tensors = load_file(output)

    torch.testing.assert_close(tensors["diffusion_model.norm.diff"], torch.tensor([-1.0, 0.0]))
    torch.testing.assert_close(tensors["diffusion_model.matrix.diff"], torch.zeros((2, 2)))
    assert tensors["diffusion_model.norm.diff"].dtype == torch.float32
    assert tensors["diffusion_model.matrix.diff"].dtype == torch.float32
    assert tensors["diffusion_model.only_first.diff"].item() == pytest.approx(8.0)
    assert tensors["diffusion_model.only_first.diff"].dtype == torch.float16
    assert tensors["diffusion_model.scale.diff"].item() == pytest.approx(2.0)
    assert tensors["diffusion_model.lowrank.lora_down.weight"].shape[0] == 2
    assert tensors["diffusion_model.lowrank.lora_down.weight"].dtype == torch.float32
    assert tensors["diffusion_model.lowrank.lora_up.weight"].dtype == torch.float32
    assert tensors["diffusion_model.lowrank.alpha"].dtype == torch.float32

    default_output = merger.merge_multi_loras(
        [str(first), str(second)], [2.0, -1.0], "concatenate", "cpu",
        torch.float16, "merged_default", base_model_path=str(base), verbose=False,
    )
    default_tensors = load_file(default_output)
    assert "diffusion_model.norm.diff" not in default_tensors
    assert "diffusion_model.matrix.diff" in default_tensors
    assert "diffusion_model.scale.diff" in default_tensors
    assert "diffusion_model.lowrank.lora_down.weight" in default_tensors


def test_multi_merge_rejects_direct_shape_mismatch(monkeypatch, tmp_path, modules):
    _, merger = modules
    _patch_output(monkeypatch, merger, tmp_path)
    first = tmp_path / "shape_first.safetensors"
    second = tmp_path / "shape_second.safetensors"
    save_file({"diffusion_model.bad.diff": torch.ones(2)}, str(first))
    save_file({"diffusion_model.bad.diff": torch.ones(3)}, str(second))

    with pytest.raises(ValueError, match="Direct LoRA shape mismatch"):
        merger.merge_multi_loras(
            [str(first), str(second)], [1.0, 1.0], "weighted_sum", "cpu",
            torch.float16, "bad_shapes", verbose=False, include_1d_diffs=True,
        )


@pytest.mark.parametrize("enhanced", [False, True])
def test_dare_variants_emit_direct_layers(monkeypatch, tmp_path, modules, enhanced):
    _, merger = modules
    _patch_output(monkeypatch, merger, tmp_path)
    first = tmp_path / f"dare_first_{enhanced}.safetensors"
    second = tmp_path / f"dare_second_{enhanced}.safetensors"
    save_file({
        "diffusion_model.norm.diff": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "diffusion_model.lowrank.lora_down.weight": torch.ones((1, 2), dtype=torch.float32),
        "diffusion_model.lowrank.lora_up.weight": torch.ones((2, 1), dtype=torch.float32),
    }, str(first))
    save_file({
        "diffusion_model.norm.diff": torch.tensor([3.0, 4.0], dtype=torch.float32),
        "diffusion_model.lowrank.lora_down.weight": torch.ones((1, 2), dtype=torch.bfloat16),
        "diffusion_model.lowrank.lora_up.weight": torch.ones((2, 1), dtype=torch.bfloat16),
    }, str(second))

    if enhanced:
        output = merger.merge_multi_loras_dare_enhanced(
            [str(first), str(second)], [1.0, 1.0], 1.0, 1.0, 0.0, 0.0, 7,
            "cpu", torch.float16, "enhanced", verbose=False, include_1d_diffs=True,
        )
    else:
        output = merger.merge_multi_loras_dare(
            [str(first), str(second)], [1.0, 1.0], 0.0, 0.0, 7,
            "cpu", torch.float16, "dare", verbose=False, include_1d_diffs=True,
        )
    tensors = load_file(output)

    torch.testing.assert_close(tensors["diffusion_model.norm.diff"], torch.tensor([2.0, 3.0]))
    assert tensors["diffusion_model.norm.diff"].dtype == torch.float32
    assert tensors["diffusion_model.lowrank.lora_down.weight"].dtype == torch.float32
    assert tensors["diffusion_model.lowrank.lora_up.weight"].dtype == torch.float32

    if enhanced:
        default_output = merger.merge_multi_loras_dare_enhanced(
            [str(first), str(second)], [1.0, 1.0], 1.0, 1.0, 0.0, 0.0, 7,
            "cpu", torch.float16, "enhanced_default", verbose=False,
        )
    else:
        default_output = merger.merge_multi_loras_dare(
            [str(first), str(second)], [1.0, 1.0], 0.0, 0.0, 7,
            "cpu", torch.float16, "dare_default", verbose=False,
        )
    default_tensors = load_file(default_output)
    assert "diffusion_model.norm.diff" not in default_tensors
    assert "diffusion_model.lowrank.lora_down.weight" in default_tensors


def test_merge_to_model_applies_generic_direct_keys_and_preserves_dtype(
    monkeypatch, tmp_path, modules, capsys
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "base.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    save_file({
        "model.diffusion_model.block.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
        "model.diffusion_model.norm.weight": torch.zeros(2, dtype=torch.bfloat16),
        "model.diffusion_model.block.bias": torch.zeros(2, dtype=torch.bfloat16),
        "model.diffusion_model.scale": torch.tensor(1.0, dtype=torch.bfloat16),
        "model.diffusion_model.lin": torch.zeros((2, 2), dtype=torch.bfloat16),
        "model.diffusion_model.bad.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "model.diffusion_model.untouched": torch.tensor(1e-8, dtype=torch.float32),
    }, str(base))
    save_file({
        "diffusion_model.block.diff": torch.full((2, 2), 1e-8, dtype=torch.float32),
        "diffusion_model.norm.diff": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "diffusion_model.block.diff_b": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        "diffusion_model.scale.diff": torch.tensor(0.5, dtype=torch.bfloat16),
        "diffusion_model.lin.diff": torch.ones((2, 2), dtype=torch.float32),
        "diffusion_model.bad.diff": torch.ones(3, dtype=torch.float32),
    }, str(adapter))

    output = resize.merge_loras_to_model(
        [str(adapter)], [0.5], str(base), "cpu", torch.float16, "merged_model",
        verbose=False, include_1d_diffs=True,
    )
    tensors = load_file(output)

    torch.testing.assert_close(tensors["model.diffusion_model.norm.weight"], torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(tensors["model.diffusion_model.block.bias"], torch.tensor([1.5, 2.0]))
    assert tensors["model.diffusion_model.scale"].item() == pytest.approx(1.25)
    torch.testing.assert_close(tensors["model.diffusion_model.lin"], torch.full((2, 2), 0.5))
    torch.testing.assert_close(tensors["model.diffusion_model.bad.weight"].float(), torch.ones((2, 2)))
    assert "Shape mismatch" in capsys.readouterr().out
    assert tensors["model.diffusion_model.block.weight"].dtype == torch.float32
    assert tensors["model.diffusion_model.norm.weight"].dtype == torch.float32
    assert tensors["model.diffusion_model.block.bias"].dtype == torch.float32
    assert tensors["model.diffusion_model.scale"].dtype == torch.float16
    assert tensors["model.diffusion_model.lin"].dtype == torch.float32
    assert tensors["model.diffusion_model.untouched"].dtype == torch.float32

    default_output = resize.merge_loras_to_model(
        [str(adapter)], [0.5], str(base), "cpu", torch.float16, "merged_model_default",
        verbose=False,
    )
    default_tensors = load_file(default_output)
    torch.testing.assert_close(
        default_tensors["model.diffusion_model.norm.weight"].float(), torch.zeros(2)
    )
    torch.testing.assert_close(
        default_tensors["model.diffusion_model.block.bias"].float(), torch.zeros(2)
    )
    assert default_tensors["model.diffusion_model.scale"].item() == pytest.approx(1.25)
    torch.testing.assert_close(
        default_tensors["model.diffusion_model.lin"], torch.full((2, 2), 0.5)
    )


def _generic_params(output_filename, override_dtype, include_1d_diffs=False):
    return {
        "mismatch_mode": "skip",
        "alignment_mode": "pad/crop",
        "alpha": 0.5,
        "beta": 0.0,
        "gamma": 0.5,
        "delta": 2.0,
        "epsilon": 0.01,
        "zeta": 0.0,
        "seed": 0,
        "output_filename": output_filename,
        "save_dtype": "bf16",
        "override_dtype": override_dtype,
        "device": "cpu",
        "dtype": torch.float32,
        "exclude_patterns": "",
        "discard_patterns": "",
        "glob_patterns": False,
        "lazy_load": True,
        "force_clear_cache": False,
        "include_1d_diffs": include_1d_diffs,
    }


def test_generic_two_and_three_mergers_enforce_direct_dtype_contract(
    monkeypatch, tmp_path, generic_modules
):
    generic, operations = generic_modules
    paths = {}
    for name, tensors in {
        "a": {
            "diffusion_model.norm.diff": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "diffusion_model.matrix.diff": torch.ones((2, 2), dtype=torch.float32),
        },
        "b": {
            "diffusion_model.norm.diff": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            "diffusion_model.matrix.diff": torch.full((2, 2), 3.0, dtype=torch.bfloat16),
            "diffusion_model.secondary_only.diff": torch.ones(1, dtype=torch.float32),
        },
        "c": {
            "diffusion_model.norm.diff": torch.tensor([0.5, 1.0], dtype=torch.float32),
            "diffusion_model.matrix.diff": torch.full((2, 2), 0.5, dtype=torch.bfloat16),
        },
    }.items():
        path = tmp_path / f"{name}.safetensors"
        save_file(tensors, str(path))
        paths[name] = str(path)

    monkeypatch.setattr(generic.folder_paths, "get_full_path", lambda _, name: paths.get(name))
    monkeypatch.setattr(generic.folder_paths, "get_folder_paths", lambda _: [str(tmp_path)])
    monkeypatch.setattr(generic, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(generic, "cleanup_after_operation", lambda: None)

    two_params = _generic_params("generic_two", override_dtype=True)
    two_params.update({"model_a": "a", "model_b": "b"})
    generic.MergerLogic.execute_merge(
        {"model_a": "a", "model_b": "b"},
        "Weight-Sum",
        operations.TWO_MODEL_MODES,
        two_params,
        "loras",
    )
    two = load_file(str(tmp_path / "generic_two.safetensors"))
    torch.testing.assert_close(
        two["diffusion_model.norm.diff"].float(), torch.tensor([1.0, 2.0])
    )
    assert two["diffusion_model.norm.diff"].dtype == torch.bfloat16
    assert two["diffusion_model.matrix.diff"].dtype == torch.bfloat16
    assert "diffusion_model.secondary_only.diff" not in two

    two_enabled_params = _generic_params(
        "generic_two_enabled", override_dtype=True, include_1d_diffs=True
    )
    two_enabled_params.update({"model_a": "a", "model_b": "b"})
    generic.MergerLogic.execute_merge(
        {"model_a": "a", "model_b": "b"},
        "Weight-Sum",
        operations.TWO_MODEL_MODES,
        two_enabled_params,
        "loras",
    )
    two_enabled = load_file(str(tmp_path / "generic_two_enabled.safetensors"))
    torch.testing.assert_close(
        two_enabled["diffusion_model.norm.diff"], torch.tensor([2.0, 3.0])
    )
    assert two_enabled["diffusion_model.norm.diff"].dtype == torch.float32

    three_params = _generic_params("generic_three", override_dtype=False)
    three_params.update({"model_a": "a", "model_b": "b", "model_c": "c"})
    generic.MergerLogic.execute_merge(
        {"model_a": "a", "model_b": "b", "model_c": "c"},
        "Add-Difference",
        operations.THREE_MODEL_MODES,
        three_params,
        "loras",
    )
    three = load_file(str(tmp_path / "generic_three.safetensors"))
    torch.testing.assert_close(
        three["diffusion_model.norm.diff"].float(), torch.tensor([1.0, 2.0])
    )
    assert three["diffusion_model.norm.diff"].dtype == torch.bfloat16
    assert three["diffusion_model.matrix.diff"].dtype == torch.float32

    three_enabled_params = _generic_params(
        "generic_three_enabled", override_dtype=False, include_1d_diffs=True
    )
    three_enabled_params.update({"model_a": "a", "model_b": "b", "model_c": "c"})
    generic.MergerLogic.execute_merge(
        {"model_a": "a", "model_b": "b", "model_c": "c"},
        "Add-Difference",
        operations.THREE_MODEL_MODES,
        three_enabled_params,
        "loras",
    )
    three_enabled = load_file(str(tmp_path / "generic_three_enabled.safetensors"))
    torch.testing.assert_close(
        three_enabled["diffusion_model.norm.diff"], torch.tensor([2.25, 3.5])
    )
    assert three_enabled["diffusion_model.norm.diff"].dtype == torch.float32
