import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import (
    IncrementalSafetensorsWriter,
    MemoryEfficientSafeOpen,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_dtype_conversion_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def conversion():
    return _load_module("nodes.dtype_conversion")


def _write_uel(path, tensors, metadata=None):
    with IncrementalSafetensorsWriter(
        str(path), metadata=metadata or {}, max_workers=1
    ) as writer:
        writer.write_batch(list(tensors.items()))


def _read_uel(path):
    result = {}
    with MemoryEfficientSafeOpen(str(path), low_memory=True) as loader:
        metadata = loader.metadata()
        for batch in loader.async_stream(
            loader.keys(), batch_size=1, prefetch_batches=1, pin_memory=False
        ):
            key, tensor = batch[0]
            result[key] = tensor.clone()
            loader.mark_processed(key)
    return result, metadata


def _patch_runtime(monkeypatch, tmp_path, conversion, source):
    monkeypatch.setattr(conversion.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(
        conversion.folder_paths,
        "get_full_path_or_raise",
        lambda model_type, name: str(source),
    )
    monkeypatch.setattr(conversion, "cleanup_after_operation", lambda: None)


def test_schema_is_diffusion_model_scoped(conversion):
    schema = conversion.DiffusionModelDtypeConversion.define_schema()
    assert schema.node_id == "DiffusionModelDtypeConversion"
    assert schema.category == "ModelUtils/Conversion"
    assert [item.id for item in schema.inputs] == [
        "model_name",
        "target_dtype",
        "reference_model",
        "exclude_patterns",
        "output_filename",
        "include_mode",
    ]
    assert schema.inputs[-1].default is False
    assert schema.inputs[1].options == ["fp32", "fp16", "bf16"]
    assert [item.display_name for item in schema.outputs] == [
        "output_path",
        "conversion_report",
    ]


def test_reference_model_dtypes_override_target_without_tensor_loading(
    monkeypatch, tmp_path, conversion
):
    source = tmp_path / "source.safetensors"
    reference = tmp_path / "reference.safetensors"
    _write_uel(source, {
        "mapped_bf16": torch.ones(2, dtype=torch.float32),
        "mapped_fp32": torch.ones(2, dtype=torch.float16),
        "fallback": torch.ones(2, dtype=torch.float32),
        "nonfloating": torch.tensor([1], dtype=torch.int64),
    })
    _write_uel(reference, {
        "mapped_bf16": torch.ones(2, dtype=torch.bfloat16),
        "mapped_fp32": torch.ones(2, dtype=torch.float32),
        "nonfloating": torch.tensor([1], dtype=torch.int64),
    })
    monkeypatch.setattr(conversion.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(
        conversion.folder_paths,
        "get_full_path_or_raise",
        lambda _, name: {"source": str(source), "reference": str(reference)}[name],
    )
    monkeypatch.setattr(conversion, "cleanup_after_operation", lambda: None)
    monkeypatch.setattr(
        conversion.MemoryEfficientSafeOpen,
        "get_tensor",
        lambda *args, **kwargs: pytest.fail("synchronous reference loading was used"),
    )

    output_path, report = conversion.convert_diffusion_model_dtype(
        "source", "fp16", "", "reference_mapped", "reference"
    )
    result, _ = _read_uel(tmp_path / "diffusion_models" / output_path)
    assert result["mapped_bf16"].dtype == torch.bfloat16
    assert result["mapped_fp32"].dtype == torch.float32
    assert result["fallback"].dtype == torch.float16
    assert result["nonfloating"].dtype == torch.int64
    assert "Reference-mapped floating tensors: 2" in report
    assert "Reference-missing fallback tensors: 1" in report


@pytest.mark.parametrize(
    ("target_name", "expected_dtype"),
    [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_converts_floating_and_preserves_excluded_and_nonfloating(
    monkeypatch, tmp_path, conversion, target_name, expected_dtype
):
    source = tmp_path / f"source_{target_name}.safetensors"
    tensors = {
        "blocks.0.weight": torch.arange(4, dtype=torch.float32),
        "blocks.0.norm.weight": torch.ones(3, dtype=torch.float32),
        "position_ids": torch.tensor([1, 2], dtype=torch.int64),
    }
    _write_uel(source, tensors, metadata={"source": "test"})
    _patch_runtime(monkeypatch, tmp_path, conversion, source)

    output_path, report = conversion.convert_diffusion_model_dtype(
        source.name,
        target_name,
        r"\.norm\.weight$",
        f"converted_{target_name}",
    )
    assert output_path == f"converted_{target_name}.safetensors"
    result, metadata = _read_uel(
        tmp_path / "diffusion_models" / output_path
    )

    assert result["blocks.0.weight"].dtype == expected_dtype
    assert result["blocks.0.norm.weight"].dtype == torch.float32
    assert result["position_ids"].dtype == torch.int64
    assert metadata == {"source": "test"}
    assert "Excluded tensors preserved: 1" in report
    assert "Non-floating tensors preserved: 1" in report


@pytest.mark.parametrize("patterns, selected", [
    (r"^blocks\.0\.weight$", {"blocks.0.weight"}),
    (r"^blocks\.0\.weight$" + "\n" + r"^blocks\.0\.norm\.weight$", {"blocks.0.weight", "blocks.0.norm.weight"}),
    ("", set()),
    ("no_such_layer", set()),
])
@pytest.mark.parametrize("use_reference", [False, True])
def test_include_filter_only_converts_matches(
    monkeypatch, tmp_path, conversion, patterns, selected, use_reference,
):
    source = tmp_path / "source.safetensors"
    reference = tmp_path / "reference.safetensors"
    tensors = {
        "blocks.0.weight": torch.arange(4, dtype=torch.float32),
        "blocks.0.norm.weight": torch.ones(3, dtype=torch.float32),
        "position_ids": torch.tensor([1, 2], dtype=torch.int64),
    }
    _write_uel(source, tensors)
    _write_uel(reference, {"blocks.0.weight": tensors["blocks.0.weight"].to(torch.bfloat16)})
    _patch_runtime(monkeypatch, tmp_path, conversion, source)
    monkeypatch.setattr(
        conversion.folder_paths, "get_full_path_or_raise",
        lambda _, name: str(reference if name == "reference" else source),
    )

    output_name, _ = conversion.DiffusionModelDtypeConversion.execute(
        source.name, "fp16", "reference" if use_reference else "None",
        patterns, "include_converted", include_mode=True,
    )
    result, _ = _read_uel(tmp_path / "diffusion_models" / output_name)
    for key, tensor in tensors.items():
        expected_dtype = tensor.dtype
        if key in selected:
            expected_dtype = torch.bfloat16 if use_reference and key == "blocks.0.weight" else torch.float16
        assert result[key].dtype == expected_dtype
        torch.testing.assert_close(result[key].to(tensor.dtype), tensor)


def test_uses_bounded_async_stream_and_releases_every_tensor(
    monkeypatch, tmp_path, conversion
):
    source = tmp_path / "source_async.safetensors"
    _write_uel(
        source,
        {
            "a": torch.ones(2, dtype=torch.float32),
            "b": torch.ones(2, dtype=torch.float16),
            "c": torch.ones(2, dtype=torch.int32),
        },
    )
    _patch_runtime(monkeypatch, tmp_path, conversion, source)
    async_calls = []
    processed = []
    original_async = conversion.MemoryEfficientSafeOpen.async_stream
    original_mark = conversion.MemoryEfficientSafeOpen.mark_processed

    def track_async(loader, keys, **kwargs):
        async_calls.append((tuple(keys), kwargs))
        yield from original_async(loader, keys, **kwargs)

    def track_mark(loader, key):
        processed.append(key)
        return original_mark(loader, key)

    monkeypatch.setattr(
        conversion.MemoryEfficientSafeOpen, "async_stream", track_async
    )
    monkeypatch.setattr(
        conversion.MemoryEfficientSafeOpen, "mark_processed", track_mark
    )
    monkeypatch.setattr(
        conversion.MemoryEfficientSafeOpen,
        "get_tensor",
        lambda *args, **kwargs: pytest.fail("synchronous loading was used"),
    )

    conversion.convert_diffusion_model_dtype(
        source.name, "bf16", "^b$", "converted_async"
    )

    assert len(async_calls) == 1
    assert async_calls[0][1] == {
        "batch_size": 1,
        "prefetch_batches": 1,
        "pin_memory": False,
    }
    assert processed == ["a", "b", "c"]


def test_invalid_regex_does_not_create_output(monkeypatch, tmp_path, conversion):
    source = tmp_path / "source_invalid.safetensors"
    _write_uel(source, {"weight": torch.ones(1)})
    _patch_runtime(monkeypatch, tmp_path, conversion, source)

    with pytest.raises(ValueError, match="Invalid exclusion regex"):
        conversion.convert_diffusion_model_dtype(
            source.name, "fp16", "[", "must_not_exist"
        )

    assert not (tmp_path / "diffusion_models" / "must_not_exist.safetensors").exists()


def test_rejects_input_output_collision(monkeypatch, tmp_path, conversion):
    model_dir = tmp_path / "diffusion_models"
    model_dir.mkdir()
    source = model_dir / "same.safetensors"
    _write_uel(source, {"weight": torch.ones(1)})
    _patch_runtime(monkeypatch, tmp_path, conversion, source)

    with pytest.raises(ValueError, match="must differ"):
        conversion.convert_diffusion_model_dtype(
            source.name, "fp16", "", "same"
        )
