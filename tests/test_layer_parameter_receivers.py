import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest
from comfy_api.latest import io


RECEIVERS = [
    ("merger", 10), ("lora_extract_svd", 5), ("dora_extract_wd", 5),
    ("dora_learned_wd", 5), ("text_encoder_extract", 10), ("lora_resize", 4),
]


def load(name):
    package_name = "modelutils_layer_receiver_tests"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.nodes.{name}")


def nodes(module):
    return [
        node for name, node in vars(module).items()
        if isinstance(node, type) and not name.startswith("_")
        and node.__module__ == module.__name__ and issubclass(node, io.ComfyNode)
        and (not module.__name__.endswith(".lora_resize") or name.startswith("LoRAResize"))
    ]


@pytest.mark.parametrize("module_name,count", RECEIVERS)
def test_all_39_receivers_append_optional_typed_socket(module_name, count):
    module = load(module_name)
    classes = nodes(module)
    assert len(classes) == count
    config_output = load("layer_parameters").LayerParameterConfiguration.define_schema().outputs[0]
    for node in classes:
        schema = node.define_schema()
        input_ids = [item.id for item in schema.inputs]
        assert input_ids[-2:] == ["include_mode", "layer_parameters"], schema.node_id
        assert input_ids.count("layer_parameters") == 1
        socket = schema.inputs[-1]
        assert socket.optional
        assert socket.Parent.io_type == config_output.Parent.io_type == "MODELUTILS_LAYER_PARAMETERS"
        assert "a=" in socket.tooltip
        assert inspect.signature(node.execute).parameters["layer_parameters"].default is None


@pytest.mark.parametrize("module_name,count", RECEIVERS)
def test_all_39_receivers_forward_configuration(monkeypatch, module_name, count):
    module = load(module_name)
    rules = load("layer_parameters").parse_rules("(layer) a:2")
    captured = []
    helper_names = {
        "lora_extract_svd": "extract_lora_from_files",
        "dora_extract_wd": "extract_dora_from_files",
        "dora_learned_wd": "extract_dora_learned_from_files",
        "text_encoder_extract": "extract_te_from_files",
        "lora_resize": "resize_lora_file",
    }
    if module_name == "merger":
        def merge(model_names, calc_mode, all_modes, params, model_type):
            captured.append(params["layer_parameters"])
            return "result.safetensors"
        monkeypatch.setattr(module.MergerLogic, "execute_merge", merge)
    else:
        name = helper_names[module_name]
        signature = inspect.signature(getattr(module, name))

        def process(*args, **kwargs):
            captured.append(signature.bind(*args, **kwargs).arguments["layer_parameters"])
            return "result.safetensors"

        monkeypatch.setattr(module, name, process)
    monkeypatch.setattr(module.folder_paths, "get_full_path_or_raise", lambda category, name: name)
    if module_name in {"lora_extract_svd", "dora_extract_wd", "dora_learned_wd", "text_encoder_extract"}:
        monkeypatch.setattr(module, "_build_lora_output_path", lambda name: (name, name))
    if module_name == "lora_resize":
        monkeypatch.setattr(module, "canonical_model_artifact_path", lambda category, name: (name, name))
    for node in nodes(module):
        values = {}
        for item in node.define_schema().inputs:
            value = getattr(item, "default", None)
            if value is None:
                options = getattr(item, "options", [])
                value = options[0] if options else item.id
            values[item.id] = value
        values["layer_parameters"] = rules
        if "execution_mode" in values:
            values["execution_mode"] = "MERGE"
        node.execute(**values)
    assert len(captured) == count
    assert all(value is rules for value in captured)


def test_no_expansion_to_other_node_families():
    for module_name in ("consensus_merger", "cwb_delta_lora_merger", "lodestone_merger", "lora_merger", "model_analysis", "dtype_conversion"):
        module = load(module_name)
        for node in nodes(module):
            assert "layer_parameters" not in {item.id for item in node.define_schema().inputs}
    resize = load("lora_resize")
    for node in (resize.LoRAMergeToModel, resize.LoRANormalizeAlpha):
        assert "layer_parameters" not in {item.id for item in node.define_schema().inputs}
