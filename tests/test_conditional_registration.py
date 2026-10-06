import asyncio
import importlib.util
import sys
from pathlib import Path
from comfy_api.latest import io

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_extension_module():
    package_name = "ComfyUI_ModelUtils_init_test"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            REPO_ROOT / "__init__.py",
            submodule_search_locations=[str(REPO_ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    return sys.modules[package_name]


def test_startup_node_registration_when_downloaders_absent(monkeypatch):
    ext_module = _load_extension_module()

    ext = ext_module.ModelUtilsExtension()
    nodes_with_downloader = asyncio.run(ext.get_node_list())
    assert len(nodes_with_downloader) > 0

    # Simulate registry install by setting DOWNLOADER_NODES to empty
    monkeypatch.setattr(ext_module, "DOWNLOADER_NODES", [])
    nodes_registry = asyncio.run(ext.get_node_list())
    assert len(nodes_registry) < len(nodes_with_downloader)
    assert all(issubclass(n, io.ComfyNode) for n in nodes_registry)


def test_model_info_nodes_importable_standalone():
    spec = importlib.util.spec_from_file_location(
        "modelutils_model_info_nodes_test",
        REPO_ROOT / "nodes" / "model_info_nodes.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "get_potential_preview_files")
    assert hasattr(module, "load_image_tensor")
    assert hasattr(module, "get_model_workflows")
    assert hasattr(module, "get_model_metadata_file")
