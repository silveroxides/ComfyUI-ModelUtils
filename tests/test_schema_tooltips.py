import ast
import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest
from comfy_api.latest import io


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"


def test_every_schema_input_has_a_direct_static_tooltip():
    missing = []
    non_static = []

    for path in sorted(NODE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "Input"
            ):
                continue

            name = (
                call.args[0].value
                if call.args and isinstance(call.args[0], ast.Constant)
                else "<dynamic input name>"
            )
            tooltip = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "tooltip"),
                None,
            )
            location = f"{path.name}:{call.lineno}:{name}"
            if tooltip is None:
                missing.append(location)
            elif not (
                isinstance(tooltip, ast.Constant)
                and isinstance(tooltip.value, str)
                and tooltip.value.strip()
            ):
                if not (path.name == "layer_parameters.py" and name == "layer_parameters"):
                    non_static.append(location)

    assert not missing, "Inputs without tooltips:\n" + "\n".join(missing)
    assert not non_static, "Tooltips must be direct non-empty string literals:\n" + "\n".join(non_static)


FILTER_MODULES = [
    "merger", "consensus_merger", "cwb_delta_lora_merger", "lodestone_merger",
    "lora_resize", "dtype_conversion", "model_analysis",
    "lora_extract_svd", "dora_extract_wd", "dora_learned_wd", "text_encoder_extract",
    "minimax_h3_fold",
]


def test_filter_schema_audit_covers_all_direct_filter_definitions():
    discovered = set()
    for path in NODE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "Input"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value in {"exclude_patterns", "skip_patterns"}
            ):
                discovered.add(path.stem)
    assert discovered <= set(FILTER_MODULES), f"Unaudited filter modules: {discovered - set(FILTER_MODULES)}"


def _load_filter_module(module_name):
    package_name = "modelutils_filter_schema_tests"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(NODE_DIR.parent)]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.nodes.{module_name}")


@pytest.mark.parametrize("module_name", FILTER_MODULES)
def test_layer_filter_schemas_append_disabled_include_toggle(module_name):
    module = _load_filter_module(module_name)
    checked = []
    for name, node in vars(module).items():
        if name.startswith("_") or not isinstance(node, type):
            continue
        if node.__module__ != module.__name__ or not issubclass(node, io.ComfyNode):
            continue
        schema = node.define_schema()
        previous_inputs = [item for item in schema.inputs if item.id != "layer_parameters"]
        ids = [item.id for item in previous_inputs]
        if not {"exclude_patterns", "skip_patterns"}.intersection(ids):
            continue
        assert ids.count("include_mode") == 1, schema.node_id
        assert ids[-1] == "include_mode", schema.node_id
        assert isinstance(previous_inputs[-1], io.Boolean.Input), schema.node_id
        assert previous_inputs[-1].default is False, schema.node_id
        checked.append(schema.node_id)
    assert checked, f"No filter nodes checked in {module_name}"


@pytest.mark.parametrize("module_name, helper_name, expected_count", [
    ("lora_extract_svd", "extract_lora_from_files", 5),
    ("dora_extract_wd", "extract_dora_from_files", 5),
    ("dora_learned_wd", "extract_dora_learned_from_files", 5),
    ("text_encoder_extract", "extract_te_from_files", 10),
])
@pytest.mark.parametrize("include_mode", [False, True])
def test_all_extraction_nodes_forward_include_mode_without_shifting_parameters(
    monkeypatch, module_name, helper_name, expected_count, include_mode,
):
    module = _load_filter_module(module_name)
    signature = inspect.signature(getattr(module, helper_name))
    assert [name for name in signature.parameters if name != "layer_parameters"][-1] == "include_mode"
    captured = []

    def extract(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        captured.append(bound.arguments)

    monkeypatch.setattr(module, helper_name, extract)
    monkeypatch.setattr(module.folder_paths, "get_full_path_or_raise", lambda category, name: name)
    monkeypatch.setattr(module, "_build_lora_output_path", lambda name: (name, name))
    for name, node in vars(module).items():
        if name.startswith("_") or not isinstance(node, type):
            continue
        if node.__module__ != module.__name__ or not issubclass(node, io.ComfyNode):
            continue
        schema = node.define_schema()
        values = {}
        for widget in schema.inputs:
            if widget.id == "layer_parameters":
                continue
            value = getattr(widget, "default", None)
            if value is None:
                options = getattr(widget, "options", [])
                value = options[0] if options else widget.id
            values[widget.id] = value
        values["include_mode"] = include_mode
        values["skip_patterns"] = "selected_layer"
        node.execute(**values)
        assert captured[-1]["include_mode"] is include_mode, schema.node_id
        assert captured[-1]["skip_patterns_str"] == "selected_layer", schema.node_id
        assert captured[-1]["knee_probe_offset"] == values.get("knee_probe_offset", 32), schema.node_id
    assert len(captured) == expected_count
