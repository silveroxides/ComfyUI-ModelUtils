import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter, MemoryEfficientSafeOpen


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_layer_parameter_resize_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def resize():
    return _load_module("nodes.lora_resize")


@pytest.fixture(scope="module")
def layer_parameters():
    return _load_module("nodes.layer_parameters")


def _write_uel(path, tensors):
    with IncrementalSafetensorsWriter(str(path), metadata={}) as writer:
        writer.write_batch(
            [(key, tensor.cpu().contiguous()) for key, tensor in tensors.items()]
        )


def _read_uel(path):
    result = {}
    with MemoryEfficientSafeOpen(str(path), low_memory=True) as loader:
        stream = loader.async_stream(
            loader.keys(), batch_size=1, prefetch_batches=1, pin_memory=False
        )
        for batch in stream:
            for key, tensor in batch:
                result[key] = tensor.clone()
                loader.mark_processed(key)
    return result


@pytest.mark.parametrize(
    ("node_name", "extra"),
    [
        ("LoRAResizeFixed", {"new_rank": 1}),
        ("LoRAResizeRatio", {"max_rank": 1, "ratio": 2.0}),
        (
            "LoRAResizeFrobenius",
            {"max_rank": 1, "min_rank": 1, "target": 0.9},
        ),
        (
            "LoRAResizeCumulative",
            {"max_rank": 1, "target": 0.9},
        ),
    ],
)
def test_resize_nodes_append_and_forward_layer_parameters(
    monkeypatch, tmp_path, resize, node_name, extra
):
    node = getattr(resize, node_name)
    schema = node.define_schema()
    assert schema.inputs[-1].id == "layer_parameters"

    calls = []
    layer_rules = object()
    monkeypatch.setattr(
        resize.folder_paths, "get_full_path_or_raise", lambda *args: "input"
    )
    monkeypatch.setattr(resize.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(
        resize, "resize_lora_file", lambda *args, **kwargs: calls.append(kwargs)
    )
    node.execute(
        lora_name="input",
        output_filename="out",
        save_dtype="fp16",
        device="cpu",
        force_clear_cache=False,
        layer_parameters=layer_rules,
        **extra,
    )
    assert calls[0]["layer_parameters"] is layer_rules


def test_fixed_resize_rule_applies_only_to_matched_canonical_layer(
    monkeypatch, tmp_path, resize, layer_parameters
):
    source = tmp_path / "source.safetensors"
    tensors = {}
    for name in ("matched", "unmatched"):
        tensors[f"diffusion_model.{name}.lora_A.weight"] = torch.eye(3)
        tensors[f"diffusion_model.{name}.lora_B.weight"] = torch.eye(3)
    _write_uel(source, tensors)
    monkeypatch.setattr(resize.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(resize, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)

    output = resize.resize_lora_file(
        str(source),
        1,
        None,
        None,
        "cpu",
        torch.float32,
        "resized",
        verbose=False,
        layer_parameters=layer_parameters.parse_rules(r"(diffusion_model\.matched) a:2"),
    )

    actual = _read_uel(output)
    assert actual["diffusion_model.matched.lora_A.weight"].shape[0] == 2
    assert actual["diffusion_model.unmatched.lora_A.weight"].shape[0] == 1


@pytest.mark.parametrize(
    ("method", "parameter", "max_rank", "min_rank", "rule", "expected_rank"),
    [
        (None, None, 1, 1, "a:2", 2),
        ("sv_ratio", 2.0, 1, 1, "a:3 b:100", 3),
        ("sv_fro", 0.1, 1, 1, "a:3 b:2 c:0.1", 2),
        ("sv_cumulative", 0.1, 1, 1, "a:3 b:0.9", 2),
    ],
)
def test_resize_rule_overrides_each_profile_numerically(
    monkeypatch,
    tmp_path,
    resize,
    layer_parameters,
    method,
    parameter,
    max_rank,
    min_rank,
    rule,
    expected_rank,
):
    source = tmp_path / f"{method or 'fixed'}.safetensors"
    _write_uel(
        source,
        {
            "diffusion_model.target.lora_A.weight": torch.eye(3),
            "diffusion_model.target.lora_B.weight": torch.diag(
                torch.tensor([10.0, 3.0, 1.0])
            ),
        },
    )
    monkeypatch.setattr(resize.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(resize, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)

    output = resize.resize_lora_file(
        str(source),
        max_rank,
        method,
        parameter,
        "cpu",
        torch.float32,
        f"{method or 'fixed'}_output",
        verbose=False,
        min_rank=min_rank,
        layer_parameters=layer_parameters.parse_rules(
            rf"(diffusion_model\.target) {rule}"
        ),
    )

    assert _read_uel(output)["diffusion_model.target.lora_A.weight"].shape[0] == expected_rank


def test_resize_rules_respect_include_discard_and_low_bit_precedence(
    monkeypatch, tmp_path, resize, layer_parameters
):
    source = tmp_path / "precedence.safetensors"
    low_down = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    _write_uel(
        source,
        {
            "diffusion_model.low.lora_A.weight": low_down,
            "diffusion_model.low.lora_B.weight": torch.eye(2),
            "diffusion_model.normal.lora_A.weight": torch.eye(3),
            "diffusion_model.normal.lora_B.weight": torch.eye(3),
            "diffusion_model.discard.lora_A.weight": torch.eye(3),
            "diffusion_model.discard.lora_B.weight": torch.eye(3),
        },
    )
    monkeypatch.setattr(resize.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(resize, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)

    output = resize.resize_lora_file(
        str(source),
        2,
        None,
        None,
        "cpu",
        torch.float32,
        "precedence_output",
        verbose=False,
        exclude_patterns=r"diffusion_model\.(low|normal|discard)",
        discard_patterns=r"diffusion_model\.discard",
        include_mode=True,
        layer_parameters=layer_parameters.parse_rules(
            "\n".join(
                [
                    r"(diffusion_model\.low) a:1",
                    r"(diffusion_model\.normal) a:1",
                    r"(diffusion_model\.discard) a:1",
                ]
            )
        ),
    )

    actual = _read_uel(output)
    assert torch.equal(actual["diffusion_model.low.lora_A.weight"], low_down)
    assert actual["diffusion_model.normal.lora_A.weight"].shape[0] == 1
    assert not any(".discard." in key for key in actual)


def test_frobenius_rule_validation_precedes_writer(
    monkeypatch, tmp_path, resize, layer_parameters
):
    source = tmp_path / "source.safetensors"
    _write_uel(
        source,
        {
            "diffusion_model.target.lora_A.weight": torch.eye(3),
            "diffusion_model.target.lora_B.weight": torch.eye(3),
        },
    )
    monkeypatch.setattr(resize, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(resize, "cleanup_after_operation", lambda: None)
    monkeypatch.setattr(
        resize,
        "atomic_uel_writer",
        lambda *args, **kwargs: pytest.fail("Writer must not open for invalid rule"),
    )

    with pytest.raises(ValueError, match="min_rank=3 exceeds max_rank=2"):
        resize.resize_lora_file(
            str(source),
            4,
            "sv_fro",
            0.9,
            "cpu",
            torch.float32,
            "unused",
            verbose=False,
            layer_parameters=layer_parameters.parse_rules(r"(diffusion_model\.target) a:2 b:3"),
        )
