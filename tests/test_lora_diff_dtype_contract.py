import importlib
import logging
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
    monkeypatch.setattr(module.folder_paths, "models_dir", str(output_dir))
    monkeypatch.setattr(module, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "cleanup_after_operation", lambda: None)


COMFY_LORA_PAIR_SUFFIXES = [
    (".lora_down.weight", ".lora_up.weight"),
    ("_lora.down.weight", "_lora.up.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora.down.weight", ".lora.up.weight"),
    (".lora_A", ".lora_B"),
    (".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight"),
    (".lora_A.default.weight", ".lora_B.default.weight"),
]


@pytest.mark.parametrize("down_suffix,up_suffix", COMFY_LORA_PAIR_SUFFIXES)
def test_all_comfy_lora_pair_spellings_parse_to_canonical_keys(
    modules, down_suffix, up_suffix
):
    resize, _ = modules
    block = "diffusion_model.layer"
    layers, passthrough = resize.parse_lora_layers([
        f"{block}{down_suffix}",
        f"{block}{up_suffix}",
        f"{block}.alpha",
    ])

    assert not passthrough
    assert layers[block]["down"] == f"{block}{down_suffix}"
    assert layers[block]["up"] == f"{block}{up_suffix}"
    assert resize.canonical_lora_key(block, "down") == f"{block}.lora_A.weight"
    assert resize.canonical_lora_key(block, "up") == f"{block}.lora_B.weight"


@pytest.mark.parametrize("down_suffix,up_suffix", COMFY_LORA_PAIR_SUFFIXES)
def test_merge_to_model_applies_every_comfy_lora_pair_spelling(
    monkeypatch, tmp_path, modules, down_suffix, up_suffix
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    variant = COMFY_LORA_PAIR_SUFFIXES.index((down_suffix, up_suffix))
    base = tmp_path / f"base_{variant}.safetensors"
    adapter = tmp_path / f"adapter_{variant}.safetensors"
    block = "diffusion_model.layer"
    save_file({
        "model.diffusion_model.layer.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
    }, str(base))
    save_file({
        f"{block}{down_suffix}": torch.tensor([[1.0, 2.0]]),
        f"{block}{up_suffix}": torch.tensor([[3.0], [4.0]]),
        f"{block}.alpha": torch.tensor(0.5),
    }, str(adapter))

    output = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float16,
        f"merged_variant_{variant}", verbose=False,
    )
    result = load_file(output)["model.diffusion_model.layer.weight"]
    torch.testing.assert_close(
        result,
        torch.tensor([[1.5, 3.0], [2.0, 4.0]]),
    )
    assert result.dtype == torch.float32


def test_merge_to_model_applies_comfy_direct_norm_and_set_forms(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "direct_base.safetensors"
    adapter = tmp_path / "direct_adapter.safetensors"
    save_file({
        "model.diffusion_model.norm.weight": torch.zeros(2),
        "model.diffusion_model.norm.bias": torch.zeros(2),
        "model.diffusion_model.replace.weight": torch.zeros((2, 2)),
    }, str(base))
    save_file({
        "diffusion_model.norm.w_norm": torch.tensor([1.0, 2.0]),
        "diffusion_model.norm.b_norm": torch.tensor([3.0, 4.0]),
        "diffusion_model.replace.set_weight": torch.full((2, 2), 7.0),
    }, str(adapter))

    output = resize.merge_loras_to_model(
        [str(adapter)], [0.5], str(base), "cpu", torch.float16,
        "direct_forms", verbose=False, include_1d_diffs=True,
    )
    result = load_file(output)
    torch.testing.assert_close(
        result["model.diffusion_model.norm.weight"], torch.tensor([0.5, 1.0])
    )
    torch.testing.assert_close(
        result["model.diffusion_model.norm.bias"], torch.tensor([1.5, 2.0])
    )
    torch.testing.assert_close(
        result["model.diffusion_model.replace.weight"], torch.full((2, 2), 7.0)
    )


def test_merge_to_model_uses_comfy_reshape_companion(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "reshape_base.safetensors"
    adapter = tmp_path / "reshape_adapter.safetensors"
    save_file({
        "model.diffusion_model.layer.weight": torch.zeros((1, 2)),
    }, str(base))
    save_file({
        "diffusion_model.layer.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        "diffusion_model.layer.reshape_weight": torch.tensor([2, 2]),
    }, str(adapter))

    output = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float16,
        "reshape_companion", verbose=False,
    )
    result = load_file(output)["model.diffusion_model.layer.weight"]
    assert result.shape == (2, 2)
    torch.testing.assert_close(result, torch.tensor([[3.0, 6.0], [4.0, 8.0]]))


def test_merge_to_model_reports_success_and_separates_base_only_biases(
    monkeypatch, tmp_path, modules, capsys
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "summary_base.safetensors"
    adapter = tmp_path / "summary_adapter.safetensors"
    save_file({
        "model.diffusion_model.layer.weight": torch.zeros((2, 2)),
        "model.diffusion_model.layer.bias": torch.zeros(2),
    }, str(base))
    save_file({
        "diffusion_model.layer.lora_A.weight": torch.ones((1, 2)),
        "diffusion_model.layer.lora_B.weight": torch.ones((2, 1)),
    }, str(adapter))

    path, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "summary", verbose=True, return_report=True,
    )
    output = capsys.readouterr().out

    assert Path(path).is_file()
    assert report.startswith("SUCCESS: 1/1 adapter groups fully applied")
    assert "[PATCHED]" not in report
    assert "[APPLIED]" not in report
    assert "L1: summary_adapter.safetensors, strength=1.0" in report
    assert "[BASE-ONLY BIASES RETAINED] (1)" in report
    assert "model.diffusion_model.layer.bias" in report
    assert "SUCCESS: 1/1 groups applied" in output
    assert "1 tensors patched, 1 base tensors not patched, 0 adapter issues" in output
    assert "model.diffusion_model.layer.bias" not in output


def test_success_report_stays_compact_and_console_is_summary_only(
    monkeypatch, tmp_path, modules, capsys
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "many_success_base.safetensors"
    adapter = tmp_path / "many_success_adapter.safetensors"
    save_file({
        f"model.diffusion_model.layer_{index}.weight": torch.zeros((2, 2))
        for index in range(40)
    }, str(base))
    save_file({
        f"diffusion_model.layer_{index}.diff": torch.ones((2, 2))
        for index in range(40)
    }, str(adapter))

    _, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "many_success", verbose=True, return_report=True,
    )
    console = capsys.readouterr().out

    assert report.startswith("SUCCESS: 40/40")
    assert "layer_0.weight" not in report
    assert "ADAPTER GROUP ISSUES\n- None" in report
    assert len(report) < 1000
    assert max(map(len, report.splitlines())) < 160
    assert "layer_0.weight" not in console
    assert len(console.splitlines()) <= 9


def test_merge_to_model_report_identifies_shape_mismatch_and_unresolved_group(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "diagnostic_base.safetensors"
    adapter = tmp_path / "diagnostic_adapter.safetensors"
    save_file({
        "model.diffusion_model.layer.weight": torch.zeros((2, 2)),
    }, str(base))
    save_file({
        "diffusion_model.layer.diff": torch.ones(3),
        "diffusion_model.orphan.lora_A.weight": torch.ones((1, 2)),
        "diffusion_model.orphan.lora_B.weight": torch.ones((2, 1)),
    }, str(adapter))

    _, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "diagnostic", verbose=False, include_1d_diffs=True, return_report=True,
    )

    assert report.startswith("COMPLETED WITH ERRORS: 0/2")
    assert "[MAPPED BUT UNCHANGED] (1)" in report
    assert "[SHAPE MISMATCH] (1)" in report
    assert "source_shape=(3,), target_shape=(2, 2)" in report
    assert "[UNRESOLVED] (1)" in report
    assert "diffusion_model.orphan" in report
    assert "no matching base tensor" in report


def test_merge_to_model_detects_comfy_adapter_logged_failure_atomically(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "logged_failure_base.safetensors"
    adapter = tmp_path / "logged_failure_adapter.safetensors"
    base_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({"model.diffusion_model.layer.weight": base_weight}, str(base))
    save_file({
        "diffusion_model.layer.lora_A.weight": torch.ones((1, 2)),
        "diffusion_model.layer.lora_B.weight": torch.ones((2, 1)),
    }, str(adapter))

    class LoggedFailureAdapter:
        def calculate_weight(self, weight, key, *args, **kwargs):
            weight.add_(99.0)
            logging.error("ERROR lora %s synthetic internal failure", key)
            return weight

    class FakeLoRAAdapter:
        @staticmethod
        def load(*args, **kwargs):
            return LoggedFailureAdapter()

    monkeypatch.setattr(resize, "LoRAAdapter", FakeLoRAAdapter)
    path, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "logged_failure", verbose=False, return_report=True,
    )

    torch.testing.assert_close(
        load_file(path)["model.diffusion_model.layer.weight"], base_weight
    )
    assert report.startswith("COMPLETED WITH ERRORS: 0/1")
    assert "[CALCULATION FAILURE] (1)" in report
    assert "synthetic internal failure" in report
    assert "[MAPPED BUT UNCHANGED] (1)" in report


def test_merge_to_model_report_classifies_disabled_1d_as_exclusion(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "disabled_1d_base.safetensors"
    adapter = tmp_path / "disabled_1d_adapter.safetensors"
    save_file({"model.diffusion_model.norm.weight": torch.zeros(2)}, str(base))
    save_file({"diffusion_model.norm.diff": torch.ones(2)}, str(adapter))

    path, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "disabled_1d", verbose=False, return_report=True,
    )

    torch.testing.assert_close(
        load_file(path)["model.diffusion_model.norm.weight"], torch.zeros(2)
    )
    assert report.startswith("COMPLETED WITH EXCLUSIONS: 0/1")
    assert "[1D DISABLED] (1)" in report
    assert "source=diffusion_model.norm.diff" in report
    assert "include_1d_diffs is disabled" in report


def test_merge_to_model_report_marks_partially_applied_group(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "partial_base.safetensors"
    adapter = tmp_path / "partial_adapter.safetensors"
    save_file({
        "model.diffusion_model.layer.weight": torch.zeros((2, 2)),
        "model.diffusion_model.layer.bias": torch.zeros(2),
    }, str(base))
    save_file({
        "diffusion_model.layer.lora_A.weight": torch.ones((1, 2)),
        "diffusion_model.layer.lora_B.weight": torch.ones((2, 1)),
        "diffusion_model.layer.diff_b": torch.ones(3),
    }, str(adapter))

    path, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "partial", verbose=False, include_1d_diffs=True, return_report=True,
    )
    result = load_file(path)

    torch.testing.assert_close(
        result["model.diffusion_model.layer.weight"], torch.ones((2, 2))
    )
    torch.testing.assert_close(
        result["model.diffusion_model.layer.bias"], torch.zeros(2)
    )
    assert report.startswith("COMPLETED WITH ERRORS: 0/1")
    assert "[PARTIALLY APPLIED] (1)" in report
    assert "source_shape=(3,), target_shape=(2,)" in report


def test_merge_to_model_detects_raised_adapter_failure_atomically(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "raised_failure_base.safetensors"
    adapter = tmp_path / "raised_failure_adapter.safetensors"
    base_weight = torch.eye(2)
    save_file({"model.diffusion_model.layer.weight": base_weight}, str(base))
    save_file({
        "diffusion_model.layer.lora_A.weight": torch.ones((1, 2)),
        "diffusion_model.layer.lora_B.weight": torch.ones((2, 1)),
    }, str(adapter))

    class RaisedFailureAdapter:
        def calculate_weight(self, weight, key, *args, **kwargs):
            weight.zero_()
            raise RuntimeError("synthetic raised failure")

    class FakeLoRAAdapter:
        @staticmethod
        def load(*args, **kwargs):
            return RaisedFailureAdapter()

    monkeypatch.setattr(resize, "LoRAAdapter", FakeLoRAAdapter)
    path, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "raised_failure", verbose=False, return_report=True,
    )

    torch.testing.assert_close(
        load_file(path)["model.diffusion_model.layer.weight"], base_weight
    )
    assert "[CALCULATION FAILURE] (1)" in report
    assert "RuntimeError: synthetic raised failure" in report


def test_merge_to_model_report_lists_skipped_tensor_and_pattern(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "skip_base.safetensors"
    adapter = tmp_path / "skip_adapter.safetensors"
    save_file({
        "model.diffusion_model.layer.weight": torch.zeros((2, 2)),
        "model.diffusion_model.layer.bias": torch.zeros(2),
    }, str(base))
    save_file({
        "diffusion_model.layer.lora_A.weight": torch.ones((1, 2)),
        "diffusion_model.layer.lora_B.weight": torch.ones((2, 1)),
    }, str(adapter))

    path, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "skip", skip_patterns_str=r"layer\.weight$", verbose=False,
        return_report=True,
    )

    assert "model.diffusion_model.layer.weight" not in load_file(path)
    assert report.startswith("COMPLETED WITH EXCLUSIONS: 0/1")
    assert "[OMITTED BY SKIP PATTERN] (1)" in report
    assert r"pattern=layer\.weight$" in report
    assert "[SKIPPED BY PATTERN] (1)" in report


@pytest.mark.parametrize("patterns, selected", [
    (r"selected\.weight$", True),
    ("", False),
    ("no_match", False),
])
def test_merge_to_model_include_filter_limits_stream_and_output(
    monkeypatch, tmp_path, modules, patterns, selected,
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "include_base.safetensors"
    adapter = tmp_path / "include_adapter.safetensors"
    save_file({
        f"model.diffusion_model.{name}.weight": torch.zeros((2, 2))
        for name in ("selected", "other")
    }, str(base))
    save_file({
        f"diffusion_model.{name}.{factor}": torch.ones(shape)
        for name in ("selected", "other")
        for factor, shape in (("lora_A.weight", (1, 2)), ("lora_B.weight", (2, 1)))
    }, str(adapter))
    planned_keys = []
    real_stream = resize.stream_work_units

    def capture_units(handlers, units, **kwargs):
        units = list(units)
        planned_keys.extend(key for _, entries in units for keys in entries.values() for key in keys)
        return real_stream(handlers, units, **kwargs)

    monkeypatch.setattr(resize, "stream_work_units", capture_units)
    path, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "include_result", skip_patterns_str=patterns, verbose=False,
        return_report=True, include_mode=True,
    )
    tensors = load_file(path)
    assert set(tensors) == ({"model.diffusion_model.selected.weight"} if selected else set())
    if selected:
        torch.testing.assert_close(tensors["model.diffusion_model.selected.weight"], torch.ones((2, 2)))
        assert len(planned_keys) == 3
        assert all("selected" in key for key in planned_keys)
    else:
        assert planned_keys == []
    assert "no include pattern matched" in report


@pytest.mark.parametrize("include_mode", [False, True])
def test_merge_to_model_node_returns_path_blank_line_report(
    monkeypatch, modules, include_mode,
):
    resize, _ = modules
    monkeypatch.setattr(
        resize.folder_paths, "get_full_path_or_raise",
        lambda category, name: f"{category}/{name}",
    )
    received = {}

    def merge(**kwargs):
        received.update(kwargs)
        return "saved/model.safetensors", "SUCCESS\nDETAILS"

    monkeypatch.setattr(resize, "merge_loras_to_model", merge)

    result = resize.LoRAMergeToModel.execute(
        "base.safetensors", "1",
        "one.safetensors", 1.0,
        "None", 1.0, "None", 1.0, "None", 1.0,
        "None", 1.0, "None", 1.0, "None", 1.0, "None", 1.0,
        "", "merged", "fp32", "cpu", True, False, False,
        include_mode=include_mode,
    )

    assert result.result == ("merged.safetensors", "SUCCESS\nDETAILS")
    assert received["include_mode"] is include_mode


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
    assert layers["diffusion_model.linear"]["dora_scale"].endswith(".dora_scale")


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
        expected = "include_mode" if node_class in {
            generic.LoRATwoMerger, generic.LoRAThreeMerger, resize.LoRAMergeToModel
        } else "include_1d_diffs"
        assert final_input.id == expected
        assert final_input.default is False

    outputs = resize.LoRAMergeToModel.define_schema().outputs
    assert [output.display_name for output in outputs] == [
        "output_path",
        "merge_report",
    ]


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

    assert tensors["diffusion_model.linear.lora_A.weight"].shape[0] == 2
    assert tensors["diffusion_model.linear.lora_A.weight"].dtype == torch.float32
    assert tensors["diffusion_model.linear.lora_B.weight"].dtype == torch.float32
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


def test_resize_normalizes_mochi_pair_to_preferred_output(
    monkeypatch, tmp_path, modules
):
    resize, _ = modules
    _patch_output(monkeypatch, resize, tmp_path)
    source = tmp_path / "mochi.safetensors"
    save_file({
        "diffusion_model.layer.lora_A": torch.ones((2, 2)),
        "diffusion_model.layer.lora_B": torch.ones((2, 2)),
        "diffusion_model.layer.alpha": torch.tensor(2.0),
    }, str(source))

    output = resize.resize_lora_file(
        str(source), 1, None, None, "cpu", torch.float32,
        "mochi_normalized", verbose=False,
    )
    tensors = load_file(output)
    assert set(tensors) == {
        "diffusion_model.layer.lora_A.weight",
        "diffusion_model.layer.lora_B.weight",
        "diffusion_model.layer.alpha",
    }


def test_standard_multi_merge_normalizes_mochi_pair_to_preferred_output(
    monkeypatch, tmp_path, modules
):
    _, merger = modules
    _patch_output(monkeypatch, merger, tmp_path)
    first = tmp_path / "multi_mochi_a.safetensors"
    second = tmp_path / "multi_mochi_b.safetensors"
    for path, value in ((first, 1.0), (second, 3.0)):
        save_file({
            "diffusion_model.layer.lora_A": torch.full((1, 2), value),
            "diffusion_model.layer.lora_B": torch.full((2, 1), value),
            "diffusion_model.layer.alpha": torch.tensor(1.0),
        }, str(path))

    output = merger.merge_multi_loras(
        [str(first), str(second)], [1.0, 1.0], "weighted_sum", "cpu",
        torch.float32, "multi_mochi_normalized", verbose=False,
    )
    tensors = load_file(output)

    assert set(tensors) == {
        "diffusion_model.layer.lora_A.weight",
        "diffusion_model.layer.lora_B.weight",
    }


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
    assert tensors["diffusion_model.lowrank.lora_A.weight"].shape[0] == 2
    assert tensors["diffusion_model.lowrank.lora_A.weight"].dtype == torch.float32
    assert tensors["diffusion_model.lowrank.lora_B.weight"].dtype == torch.float32
    assert "diffusion_model.lowrank.alpha" not in tensors

    default_output = merger.merge_multi_loras(
        [str(first), str(second)], [2.0, -1.0], "concatenate", "cpu",
        torch.float16, "merged_default", base_model_path=str(base), verbose=False,
    )
    default_tensors = load_file(default_output)
    assert "diffusion_model.norm.diff" not in default_tensors
    assert "diffusion_model.matrix.diff" in default_tensors
    assert "diffusion_model.scale.diff" in default_tensors
    assert "diffusion_model.lowrank.lora_A.weight" in default_tensors


def test_multi_merge_rejects_alpha_bearing_low_bit_factors(
    monkeypatch, tmp_path, modules
):
    _, merger = modules
    _patch_output(monkeypatch, merger, tmp_path)
    source = tmp_path / "alpha_low_bit.safetensors"
    save_file({
        "diffusion_model.foo.lora_down.weight": torch.tensor(
            [[1, 2]], dtype=torch.uint8
        ),
        "diffusion_model.foo.lora_up.weight": torch.tensor([[3.0], [4.0]]),
        "diffusion_model.foo.alpha": torch.tensor(1.0),
    }, str(source))

    with pytest.raises(ValueError, match="Cannot alpha-normalize low-bit LoRA factors"):
        merger.merge_multi_loras(
            [str(source)], [1.0], "concatenate", "cpu", torch.float16,
            "alpha_low_bit", verbose=False,
        )


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
    assert tensors["diffusion_model.lowrank.lora_A.weight"].dtype == torch.float32
    assert tensors["diffusion_model.lowrank.lora_B.weight"].dtype == torch.float32

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
    assert "diffusion_model.lowrank.lora_A.weight" in default_tensors


def test_merge_to_model_applies_generic_direct_keys_and_preserves_dtype(
    monkeypatch, tmp_path, modules, caplog
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
    assert "Shape mismatch" in caplog.text
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
    monkeypatch.setattr(generic.folder_paths, "models_dir", str(tmp_path))
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
    two = load_file(str(tmp_path / "loras" / "generic_two.safetensors"))
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
    two_enabled = load_file(
        str(tmp_path / "loras" / "generic_two_enabled.safetensors")
    )
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
    three = load_file(str(tmp_path / "loras" / "generic_three.safetensors"))
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
    three_enabled = load_file(
        str(tmp_path / "loras" / "generic_three_enabled.safetensors")
    )
    torch.testing.assert_close(
        three_enabled["diffusion_model.norm.diff"], torch.tensor([2.25, 3.5])
    )
    assert three_enabled["diffusion_model.norm.diff"].dtype == torch.float32


def test_generic_lora_merge_normalizes_alpha_before_factor_merge(
    monkeypatch, tmp_path, generic_modules
):
    generic, operations = generic_modules
    paths = {}
    for name, tensors in {
        "a": {
            "diffusion_model.layer.lora_A.weight": torch.ones((1, 2)),
            "diffusion_model.layer.lora_B.weight": torch.full((2, 1), 2.0),
            "diffusion_model.layer.alpha": torch.tensor(0.5),
        },
        "b": {
            "diffusion_model.layer.lora_A.weight": torch.ones((1, 2)),
            "diffusion_model.layer.lora_B.weight": torch.full((2, 1), 3.0),
        },
    }.items():
        path = tmp_path / f"alpha_{name}.safetensors"
        save_file(tensors, str(path))
        paths[name] = str(path)

    monkeypatch.setattr(generic.folder_paths, "get_full_path", lambda _, name: paths.get(name))
    monkeypatch.setattr(generic.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(generic, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(generic, "cleanup_after_operation", lambda: None)

    params = _generic_params("alpha_normalized", override_dtype=False)
    params.update({"model_a": "a", "model_b": "b"})
    generic.MergerLogic.execute_merge(
        {"model_a": "a", "model_b": "b"},
        "Weight-Sum",
        operations.TWO_MODEL_MODES,
        params,
        "loras",
    )

    tensors = load_file(str(tmp_path / "loras" / "alpha_normalized.safetensors"))
    torch.testing.assert_close(
        tensors["diffusion_model.layer.lora_B.weight"],
        torch.full((2, 1), 2.0),
    )
    assert not any(key.endswith(".alpha") for key in tensors)
