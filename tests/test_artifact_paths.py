import importlib
import os
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_artifact_path_tests"
MIGRATED_SAVE_MODULES = (
    "consensus_merger.py",
    "cwb_delta_lora_merger.py",
    "merger.py",
    "lora_merger.py",
    "lodestone_merger.py",
    "lora_resize.py",
    "dtype_conversion.py",
    "lora_extract_svd.py",
    "dora_extract_wd.py",
    "dora_learned_wd.py",
    "text_encoder_extract.py",
    "renamekeys.py",
    "prunekeys.py",
    "lora_rename.py",
    "minimax_h3_lora_convert.py",
)


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def artifact_paths():
    return _load_module("nodes.artifact_paths")


@pytest.mark.parametrize(
    "category",
    ["checkpoints", "diffusion_models", "text_encoders", "loras", "embeddings"],
)
def test_canonical_categories_use_models_dir_and_return_relative_name(
    monkeypatch, tmp_path, artifact_paths, category
):
    monkeypatch.setattr(artifact_paths.folder_paths, "models_dir", str(tmp_path))
    output_path, output_name = artifact_paths.canonical_model_artifact_path(
        category, "group/subgroup/model"
    )

    expected = tmp_path / category / "group" / "subgroup" / "model.safetensors"
    assert Path(output_path) == expected
    assert output_name == "group/subgroup/model.safetensors"
    assert expected.parent.is_dir()
    assert not os.path.isabs(output_name)


def test_registered_search_paths_do_not_influence_write_root(
    monkeypatch, tmp_path, artifact_paths
):
    canonical = tmp_path / "cli-model-root"
    legacy = tmp_path / "legacy-unet"
    additional = tmp_path / "additional-models"
    monkeypatch.setattr(artifact_paths.folder_paths, "models_dir", str(canonical))
    monkeypatch.setattr(
        artifact_paths.folder_paths,
        "get_folder_paths",
        lambda _: [str(legacy), str(additional)],
    )

    output_path, output_name = artifact_paths.canonical_model_artifact_path(
        "diffusion_models", "nested/model.safetensors"
    )

    assert Path(output_path) == canonical / "diffusion_models" / "nested" / "model.safetensors"
    assert output_name == "nested/model.safetensors"


@pytest.mark.parametrize(
    "name",
    ["model.safetensors", "model.safetensors.safetensors"],
)
def test_appends_exactly_one_extension(monkeypatch, tmp_path, artifact_paths, name):
    monkeypatch.setattr(artifact_paths.folder_paths, "models_dir", str(tmp_path))
    _, output_name = artifact_paths.canonical_model_artifact_path("loras", name)
    assert output_name == "model.safetensors"


@pytest.mark.parametrize(
    "name",
    ["../escape", "folder/../../escape", "", "   ", "/absolute", ".safetensors"],
)
def test_rejects_empty_or_escaping_names(monkeypatch, tmp_path, artifact_paths, name):
    monkeypatch.setattr(artifact_paths.folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(ValueError):
        artifact_paths.canonical_model_artifact_path("loras", name)


def test_rejects_unrecognized_category(monkeypatch, tmp_path, artifact_paths):
    monkeypatch.setattr(artifact_paths.folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(ValueError, match="Unsupported model artifact category"):
        artifact_paths.canonical_model_artifact_path("unet", "model")


def test_persistent_save_nodes_use_shared_path_contract_and_wildcard_outputs():
    for filename in MIGRATED_SAVE_MODULES:
        source = (REPO_ROOT / "nodes" / filename).read_text(encoding="utf-8")
        assert (
            "canonical_model_artifact_path" in source
            or "_build_lora_output_path" in source
        ), filename
        assert 'io.String.Output(display_name="output_path")' not in source, filename
        assert 'io.String.Output(display_name="output_filename")' not in source, filename
