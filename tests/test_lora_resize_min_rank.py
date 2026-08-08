import importlib
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_resize_min_rank_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def resize():
    return _load_module("nodes.lora_resize")


def test_frobenius_minimum_rank_is_enforced(resize):
    singular_values = torch.tensor([10.0, 1.0, 0.1, 0.01, 0.001])

    rank, _, _ = resize._compute_resize(
        singular_values, 4, "sv_fro", 0.5, 1.0, min_rank=3
    )

    assert rank == 3


def test_default_minimum_rank_remains_one(resize):
    singular_values = torch.tensor([10.0, 1.0, 0.1, 0.01, 0.001])

    rank, _, _ = resize._compute_resize(
        singular_values, 4, "sv_fro", 0.5, 1.0
    )

    assert rank == 1


def test_factor_space_resize_passes_minimum_rank_to_dynamic_selection(resize):
    down = torch.eye(5)
    up = torch.diag(torch.tensor([10.0, 1.0, 0.1, 0.01, 0.001]))

    result = resize._resize_lora_factors(
        down, up, 4, "sv_fro", 0.5, 1.0, min_rank=3
    )

    assert result["new_rank"] == 3
    assert result["lora_down"].shape[0] == 3


def test_minimum_rank_is_capped_by_layer_dimensions(resize):
    singular_values = torch.tensor([10.0, 1.0, 0.1])

    rank, _, _ = resize._compute_resize(
        singular_values, 4, "sv_fro", 0.5, 1.0, min_rank=4
    )

    assert rank == 3


def test_frobenius_schema_places_minimum_beside_maximum(resize):
    schema = resize.LoRAResizeFrobenius.define_schema()

    assert [item.id for item in schema.inputs[:4]] == [
        "lora_name", "max_rank", "min_rank", "target"
    ]
    assert schema.inputs[2].default == 1


def test_resize_rejects_minimum_above_maximum(tmp_path, resize):
    with pytest.raises(ValueError, match="min_rank cannot exceed max rank"):
        resize.resize_lora_file(
            str(tmp_path / "unused.safetensors"),
            4,
            "sv_fro",
            0.999,
            "cpu",
            torch.float32,
            "unused",
            min_rank=5,
        )
