import importlib
import json
import logging
import struct
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from unifiedefficientloader import IncrementalSafetensorsWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_quant_guard_tests"


def _load_module(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


@pytest.fixture(scope="module")
def modules():
    return {
        "guard": _load_module("nodes.quantization_guard"),
        "extract": _load_module("nodes.lora_extract_svd"),
        "resize": _load_module("nodes.lora_resize"),
        "multi": _load_module("nodes.lora_merger"),
        "generic": _load_module("nodes.merger"),
        "operations": _load_module("nodes.merger_ops"),
    }


def _patch_output(monkeypatch, module, output_dir):
    monkeypatch.setattr(module.folder_paths, "get_folder_paths", lambda _: [str(output_dir)])
    monkeypatch.setattr(module.folder_paths, "models_dir", str(output_dir))
    monkeypatch.setattr(module, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "cleanup_after_operation", lambda: None)


def _inspect_file(path, modules, label="input"):
    handler = modules["resize"].MemoryEfficientSafeOpen(str(path), low_memory=True)
    try:
        return modules["guard"].inspect_low_bit_input(handler, label, "Guard Test")
    finally:
        handler.__exit__(None, None, None)


def _round_trip_preserved(path, output, modules):
    handler = modules["resize"].MemoryEfficientSafeOpen(str(path), low_memory=True)
    try:
        with IncrementalSafetensorsWriter(str(output)) as writer:
            modules["guard"].write_preserved_tensor(writer, "low", handler)
    finally:
        handler.__exit__(None, None, None)
    return load_file(str(output))["low"]


def _tensor_payload(path, key="low"):
    with open(path, "rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size).decode("utf-8"))
        start, end = header[key]["data_offsets"]
        source.seek(8 + header_size + start)
        return source.read(end - start)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_int8_uint8_are_detected_and_warn_once(tmp_path, modules, caplog, dtype):
    path = tmp_path / f"{dtype}.safetensors"
    save_file({"low": torch.arange(4, dtype=dtype), "normal": torch.ones(1)}, str(path))

    with caplog.at_level(logging.WARNING):
        detected = _inspect_file(path, modules, "threshold input")

    assert detected == {"low"}
    records = [record for record in caplog.records if "threshold input" in record.message]
    assert len(records) == 1
    assert "1 isolated low-bit tensor" in records[0].message
    assert "low" in records[0].message


def test_exact_two_tensor_threshold(tmp_path, modules):
    for count in range(4):
        path = tmp_path / f"threshold_{count}.safetensors"
        tensors = {f"low_{i}": torch.ones(2, dtype=torch.uint8) for i in range(count)}
        tensors["normal"] = torch.ones(1)
        save_file(tensors, str(path))
        if count <= 2:
            assert len(_inspect_file(path, modules)) == count
        else:
            with pytest.raises(ValueError, match="3 low-bit tensors.*casted or quantized"):
                _inspect_file(path, modules)


@pytest.mark.parametrize(
    ("tensors", "metadata"),
    [
        ({"layer.comfy_quant": torch.tensor([123], dtype=torch.uint8)}, None),
        ({"normal": torch.ones(1)}, {"_quantization_metadata": "{}"}),
        ({"scaled_fp8": torch.ones(2)}, None),
    ],
)
def test_recognized_comfy_quantization_hard_errors(tmp_path, modules, tensors, metadata):
    path = tmp_path / f"quant_{len(list(tmp_path.iterdir()))}.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="ComfyUI quantization metadata.*unsupported"):
        _inspect_file(path, modules, "recognized input")


def test_supported_fp8_storage_round_trip(tmp_path, modules):
    candidates = [
        dtype
        for name in (
            "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
            "float8_e8m0fnu",
        )
        if (dtype := getattr(torch, name, None)) is not None
    ]
    supported = []
    for index, dtype in enumerate(candidates):
        path = tmp_path / f"low_float_{index}.safetensors"
        try:
            save_file({"low": torch.zeros(4, dtype=dtype)}, str(path))
            loaded = load_file(str(path))["low"]
        except (KeyError, RuntimeError, TypeError, ValueError):
            continue
        supported.append(dtype)
        assert loaded.dtype == dtype
        assert _inspect_file(path, modules) == {"low"}
        preserved = _round_trip_preserved(path, tmp_path / f"preserved_{index}.safetensors", modules)
        assert preserved.dtype == dtype
        torch.testing.assert_close(preserved.float(), loaded.float())

    if candidates and not supported:
        pytest.skip("Installed safetensors cannot write any installed torch FP8 dtype")


def test_fp4_storage_when_supported(tmp_path, modules):
    dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if dtype is None:
        pytest.skip("Installed PyTorch has no torch.float4_e2m1fn_x2")
    path = tmp_path / "fp4.safetensors"
    try:
        save_file({"low": torch.zeros(4, dtype=dtype)}, str(path))
        loaded = load_file(str(path))["low"]
    except (KeyError, RuntimeError, TypeError, ValueError):
        pytest.skip("Installed safetensors cannot write torch.float4_e2m1fn_x2")
    assert loaded.dtype == dtype
    assert _inspect_file(path, modules) == {"low"}
    preserved = _round_trip_preserved(path, tmp_path / "preserved_fp4.safetensors", modules)
    assert preserved.dtype == dtype
    assert _tensor_payload(path) == _tensor_payload(tmp_path / "preserved_fp4.safetensors")


def test_realistic_packed_uint8_comfy_sidecar_is_rejected(tmp_path, modules):
    path = tmp_path / "packed_comfy.safetensors"
    save_file({
        "layer.weight": torch.arange(8, dtype=torch.uint8).reshape(2, 4),
        "layer.comfy_quant": torch.tensor(list(b'{"format":"int8"}'), dtype=torch.uint8),
        "layer.weight_scale": torch.ones(2),
    }, str(path))
    with pytest.raises(ValueError, match="ComfyUI quantization metadata"):
        _inspect_file(path, modules)


def test_resize_preserves_complete_low_bit_layer_and_resizes_normal(
    monkeypatch, tmp_path, modules
):
    resize = modules["resize"]
    _patch_output(monkeypatch, resize, tmp_path)
    source = tmp_path / "resize_quant.safetensors"
    guarded_down = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    save_file({
        "diffusion_model.guarded.lora_down.weight": guarded_down,
        "diffusion_model.guarded.lora_up.weight": torch.ones((2, 2)),
        "diffusion_model.guarded.alpha": torch.tensor(2.0),
        "diffusion_model.guarded.dora_scale": torch.tensor([0.125, 0.25]),
        "diffusion_model.normal.lora_down.weight": torch.ones((2, 2)),
        "diffusion_model.normal.lora_up.weight": torch.ones((2, 2)),
        "isolated_aux": torch.tensor([7], dtype=torch.int8),
    }, str(source))

    output = resize.resize_lora_file(
        str(source), 1, None, None, "cpu", torch.float16, "resize_guarded", verbose=False
    )
    result = load_file(output)
    torch.testing.assert_close(result["diffusion_model.guarded.lora_A.weight"], guarded_down)
    assert result["diffusion_model.guarded.lora_A.weight"].dtype == torch.uint8
    assert result["diffusion_model.guarded.lora_B.weight"].dtype == torch.float32
    assert result["diffusion_model.guarded.dora_scale"].dtype == torch.float32
    assert result["diffusion_model.normal.lora_A.weight"].shape[0] == 1
    assert result["isolated_aux"].dtype == torch.int8


def test_extract_excludes_low_bit_candidate_and_extracts_float_layer(
    monkeypatch, tmp_path, modules
):
    extract = modules["extract"]
    monkeypatch.setattr(extract, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(extract, "cleanup_after_operation", lambda: None)
    model_a = tmp_path / "extract_a.safetensors"
    model_b = tmp_path / "extract_b.safetensors"
    output = tmp_path / "extracted.safetensors"
    save_file({
        "model.diffusion_model.normal.weight": torch.eye(2),
        "model.diffusion_model.quant.weight": torch.ones((2, 2), dtype=torch.uint8),
    }, str(model_a))
    save_file({
        "model.diffusion_model.normal.weight": torch.zeros((2, 2)),
        "model.diffusion_model.quant.weight": torch.zeros((2, 2), dtype=torch.uint8),
    }, str(model_b))

    extract.extract_lora_from_files(
        str(model_a), str(model_b), "fixed", 1, 1, "cpu", "fp32", str(output),
        force_clear_cache=False,
    )
    result = load_file(str(output))
    assert any("normal" in key for key in result)
    assert not any("quant" in key for key in result)


def test_extract_bias_diff_round_trips_through_merge_to_model(
    monkeypatch, tmp_path, modules
):
    extract = modules["extract"]
    resize = modules["resize"]
    monkeypatch.setattr(extract, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(extract, "cleanup_after_operation", lambda: None)
    _patch_output(monkeypatch, resize, tmp_path)

    finetuned = tmp_path / "bias_finetuned.safetensors"
    base = tmp_path / "bias_base.safetensors"
    adapter = tmp_path / "bias_adapter.safetensors"
    save_file({
        "model.diffusion_model.layer.bias": torch.tensor([0.25, -0.5]),
    }, str(finetuned))
    save_file({
        "model.diffusion_model.layer.bias": torch.tensor([0.0, 0.5]),
    }, str(base))

    extract.extract_lora_from_files(
        str(finetuned), str(base), "fixed", 1, 1, "cpu", "fp16", str(adapter),
        force_clear_cache=False, include_1d_diffs=True,
    )
    extracted = load_file(str(adapter))
    assert set(extracted) == {"diffusion_model.layer.diff_b"}
    assert extracted["diffusion_model.layer.diff_b"].dtype == torch.float32

    merged_path = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float32,
        "bias_round_trip", verbose=False, include_1d_diffs=True,
    )
    merged = load_file(merged_path)
    torch.testing.assert_close(
        merged["model.diffusion_model.layer.bias"],
        load_file(str(finetuned))["model.diffusion_model.layer.bias"],
    )


def _run_multi_variant(module, variant, paths, output_name):
    common = (paths, [1.0, 1.0])
    if variant == "standard":
        return module.merge_multi_loras(
            *common, "weighted_sum", "cpu", torch.float16, output_name,
            verbose=False, include_1d_diffs=True,
        )
    if variant == "dare":
        return module.merge_multi_loras_dare(
            *common, 0.0, 0.0, 11, "cpu", torch.float16, output_name,
            verbose=False, include_1d_diffs=True,
        )
    return module.merge_multi_loras_dare_enhanced(
        *common, 1.0, 1.0, 0.0, 0.0, 11, "cpu", torch.float16, output_name,
        verbose=False, include_1d_diffs=True,
    )


@pytest.mark.parametrize("variant", ["standard", "dare", "enhanced"])
def test_multi_variants_preserve_earliest_complete_affected_layer(
    monkeypatch, tmp_path, modules, variant
):
    multi = modules["multi"]
    _patch_output(monkeypatch, multi, tmp_path)
    first = tmp_path / f"{variant}_first.safetensors"
    second = tmp_path / f"{variant}_second.safetensors"
    first_down = torch.tensor([[1.0, 2.0]])
    save_file({
        "diffusion_model.guarded.lora_down.weight": first_down,
        "diffusion_model.guarded.lora_up.weight": torch.tensor([[3.0], [4.0]]),
        "diffusion_model.guarded.dora_scale": torch.tensor([0.125, 0.25]),
        "diffusion_model.normal.diff": torch.ones((2, 2)),
    }, str(first))
    save_file({
        "diffusion_model.guarded.lora_down.weight": torch.tensor([[1, 2]], dtype=torch.uint8),
        "diffusion_model.guarded.lora_up.weight": torch.tensor([[9.0], [9.0]]),
        "diffusion_model.guarded.dora_scale": torch.tensor([9.0, 9.0]),
        "diffusion_model.normal.diff": torch.full((2, 2), 3.0),
    }, str(second))

    output = _run_multi_variant(multi, variant, [str(first), str(second)], f"guard_{variant}")
    result = load_file(output)
    torch.testing.assert_close(result["diffusion_model.guarded.lora_A.weight"], first_down)
    torch.testing.assert_close(
        result["diffusion_model.guarded.lora_B.weight"], torch.tensor([[3.0], [4.0]])
    )
    torch.testing.assert_close(
        result["diffusion_model.guarded.dora_scale"], torch.tensor([0.125, 0.25])
    )
    assert "diffusion_model.normal.diff" in result


def test_hard_error_happens_before_resize_output_creation(monkeypatch, tmp_path, modules):
    resize = modules["resize"]
    _patch_output(monkeypatch, resize, tmp_path)
    source = tmp_path / "three_low.safetensors"
    save_file({f"low_{i}": torch.ones(1, dtype=torch.uint8) for i in range(3)}, str(source))
    output = tmp_path / "must_not_exist.safetensors"

    with pytest.raises(ValueError, match="3 low-bit tensors"):
        resize.resize_lora_file(
            str(source), 1, None, None, "cpu", torch.float16, output.stem, verbose=False
        )
    assert not output.exists()


def test_recognized_quantization_creates_no_resize_output(monkeypatch, tmp_path, modules):
    resize = modules["resize"]
    _patch_output(monkeypatch, resize, tmp_path)
    source = tmp_path / "recognized_quant.safetensors"
    save_file({
        "layer.weight": torch.ones((2, 2)),
        "layer.comfy_quant": torch.tensor(list(b'{"format":"int8"}'), dtype=torch.uint8),
    }, str(source))
    output = tmp_path / "recognized_must_not_exist.safetensors"

    with pytest.raises(ValueError, match="ComfyUI quantization metadata"):
        resize.resize_lora_file(
            str(source), 1, None, None, "cpu", torch.float16, output.stem, verbose=False
        )
    assert not output.exists()


def _generic_params(output_filename):
    return {
        "mismatch_mode": "skip", "alignment_mode": "pad/crop", "alpha": 0.5,
        "beta": 0.0, "gamma": 0.5, "delta": 2.0, "epsilon": 0.01, "zeta": 0.0,
        "seed": 0, "output_filename": output_filename, "save_dtype": "bf16",
        "override_dtype": True, "device": "cpu", "dtype": torch.float32,
        "exclude_patterns": "", "discard_patterns": "", "glob_patterns": False,
        "lazy_load": True, "force_clear_cache": False, "include_1d_diffs": True,
    }


def test_generic_merge_preserves_model_a_when_either_source_key_is_low_bit(
    monkeypatch, tmp_path, modules
):
    generic = modules["generic"]
    operations = modules["operations"]
    model_a = tmp_path / "generic_guard_a.safetensors"
    model_b = tmp_path / "generic_guard_b.safetensors"
    a_tensors = {
        "a_low": torch.tensor([[1, 2]], dtype=torch.uint8),
        "b_low": torch.tensor([[0.125, 0.25]], dtype=torch.float32),
        "normal": torch.ones((1, 2)),
    }
    save_file(a_tensors, str(model_a))
    save_file({
        "a_low": torch.tensor([[9.0, 9.0]]),
        "b_low": torch.tensor([[3, 4]], dtype=torch.int8),
        "normal": torch.full((1, 2), 3.0),
    }, str(model_b))
    paths = {"a": str(model_a), "b": str(model_b)}
    monkeypatch.setattr(generic.folder_paths, "get_full_path", lambda _, name: paths[name])
    monkeypatch.setattr(generic.folder_paths, "get_folder_paths", lambda _: [str(tmp_path)])
    monkeypatch.setattr(generic.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(generic, "prepare_for_large_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(generic, "cleanup_after_operation", lambda: None)
    params = _generic_params("generic_guarded")
    params.update({"model_a": "a", "model_b": "b"})

    generic.MergerLogic.execute_merge(
        {"model_a": "a", "model_b": "b"}, "Weight-Sum",
        operations.TWO_MODEL_MODES, params, "loras",
    )
    result = load_file(
        str(tmp_path / "loras" / "generic_guarded.safetensors")
    )
    torch.testing.assert_close(result["a_low"], a_tensors["a_low"])
    torch.testing.assert_close(result["b_low"], a_tensors["b_low"])
    assert result["a_low"].dtype == torch.uint8
    assert result["b_low"].dtype == torch.float32
    torch.testing.assert_close(result["normal"].float(), torch.full((1, 2), 2.0))


def test_merge_to_model_preserves_low_bit_base_and_skips_low_bit_adapter_layer(
    monkeypatch, tmp_path, modules
):
    resize = modules["resize"]
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "merge_guard_base.safetensors"
    adapter = tmp_path / "merge_guard_adapter.safetensors"
    base_quant = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    save_file({
        "model.diffusion_model.base_quant.weight": base_quant,
        "model.diffusion_model.guarded.weight": torch.zeros((2, 2)),
        "model.diffusion_model.normal.weight": torch.zeros((2, 2)),
    }, str(base))
    save_file({
        "diffusion_model.guarded.lora_down.weight": torch.tensor([[1, 1]], dtype=torch.int8),
        "diffusion_model.guarded.lora_up.weight": torch.ones((2, 1)),
        "diffusion_model.guarded.diff": torch.full((2, 2), 5.0),
        "diffusion_model.normal.diff": torch.ones((2, 2)),
    }, str(adapter))

    output, report = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float16, "merge_guarded",
        verbose=False, include_1d_diffs=True, return_report=True,
    )
    result = load_file(output)
    torch.testing.assert_close(result["model.diffusion_model.base_quant.weight"], base_quant)
    assert result["model.diffusion_model.base_quant.weight"].dtype == torch.uint8
    torch.testing.assert_close(
        result["model.diffusion_model.guarded.weight"].float(), torch.zeros((2, 2))
    )
    torch.testing.assert_close(
        result["model.diffusion_model.normal.weight"].float(), torch.ones((2, 2))
    )
    assert report.startswith("COMPLETED WITH EXCLUSIONS: 1/2")
    assert "[GUARDED] (1)" in report
    assert "diffusion_model.guarded.lora_down.weight" in report
    assert "torch.int8" in report
    assert "model.diffusion_model.base_quant.weight" in report


def test_merge_to_model_dequantizes_base_and_omits_sidecars(monkeypatch, tmp_path, modules):
    resize = modules["resize"]
    _patch_output(monkeypatch, resize, tmp_path)
    base = tmp_path / "quantized_base.safetensors"
    adapter = tmp_path / "quantized_adapter.safetensors"
    quant = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
    prefix = "model.diffusion_model.layer"
    save_file({
        f"{prefix}.weight": torch.ones((2, 2), dtype=torch.int8),
        f"{prefix}.weight_scale": torch.tensor(2.0),
        f"{prefix}.comfy_quant": quant,
    }, str(base))
    save_file({"diffusion_model.layer.diff": torch.ones((2, 2))}, str(adapter))

    output = resize.merge_loras_to_model(
        [str(adapter)], [1.0], str(base), "cpu", torch.float16,
        "dequantized_merge", verbose=False,
    )
    result = load_file(output)
    assert list(result) == [f"{prefix}.weight"]
    torch.testing.assert_close(result[f"{prefix}.weight"].float(), torch.full((2, 2), 3.0))


def test_diffusion_decoder_matches_core_quantized_tensor(tmp_path, modules):
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class

    path = tmp_path / "core_int8.safetensors"
    key = "layer.weight"
    qdata = torch.tensor([[1, -2], [3, 4]], dtype=torch.int8)
    scale = torch.tensor(0.5)
    save_file({
        key: qdata,
        "layer.weight_scale": scale,
        "layer.comfy_quant": torch.tensor(
            list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8
        ),
    }, str(path))
    handler = modules["resize"].MemoryEfficientSafeOpen(str(path), low_memory=True)
    try:
        decoder = modules["guard"].DiffusionQuantization(handler, str(path), "Core comparison")
        keys = decoder.required_keys(key)
        assert set(keys) == {key, "layer.weight_scale", "layer.comfy_quant"}
        assert decoder.data_keys == (key,)
        stream = handler.async_stream(keys, batch_size=1, prefetch_batches=1, pin_memory=False)
        tensors = {}
        try:
            for expected in keys:
                batch = next(stream)
                assert len(batch) == 1 and batch[0][0] == expected
                tensors[expected] = batch[0][1]
            actual = decoder.decode(key, tensors, torch.float32)
        finally:
            for source_key in tensors:
                handler.mark_processed(source_key)
            stream.close()

        layout_name = QUANT_ALGOS["int8_tensorwise"]["comfy_tensor_layout"]
        layout = get_layout_class(layout_name)
        expected = QuantizedTensor(
            qdata, layout_name,
            layout.Params(scale=scale, orig_dtype=torch.float32, orig_shape=(2, 2)),
        ).dequantize()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, qdata.float() * scale)
    finally:
        handler.__exit__(None, None, None)


@pytest.mark.parametrize("prefix", ["model.", "model.diffusion_model."])
def test_padded_shape_comes_from_core_meta_architecture(monkeypatch, modules, prefix):
    import comfy.model_detection

    stem = f"{prefix}layer"

    class HeaderOnly:
        def keys(self):
            return [f"{stem}.weight", f"{stem}.weight_scale", f"{stem}.weight_scale_2"]

        def metadata(self):
            return {"_quantization_metadata": json.dumps({"layers": {stem: {"format": "nvfp4"}}})}

        def get_shape(self, key):
            return {f"{stem}.weight": (16, 8), f"{stem}.weight_scale": (16, 1), f"{stem}.weight_scale_2": ()}[key]

        def get_dtype(self, key):
            return torch.uint8 if key == f"{stem}.weight" else torch.float32

    class Config:
        def get_model(self, state, prefix, device):
            assert device == torch.device("meta")
            assert all(tensor.device.type == "meta" for tensor in state.values())
            model = torch.nn.Module()
            model.diffusion_model = torch.nn.Module()
            model.diffusion_model.layer = torch.nn.Module()
            model.diffusion_model.layer._orig_shape = (3, 5)
            return model

    monkeypatch.setattr(comfy.model_detection, "model_config_from_unet", lambda *args, **kwargs: Config())
    decoder = modules["guard"].DiffusionQuantization(HeaderOnly(), "header-only", "Shape test")
    assert decoder.logical_shape(f"{stem}.weight") == (3, 5)


def test_nvfp4_stream_decode_matches_core_full_dequantization(monkeypatch, tmp_path, modules):
    import comfy.model_detection
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class

    layout_name = QUANT_ALGOS["nvfp4"]["comfy_tensor_layout"]
    layout = get_layout_class(layout_name)
    original = torch.arange(15, dtype=torch.float16).reshape(3, 5) / 10
    qdata, params = layout.quantize(original)
    path = tmp_path / "nvfp4.safetensors"
    metadata = {"_quantization_metadata": json.dumps({"layers": {"model.layer": {"format": "nvfp4"}}})}
    save_file({
        "model.layer.weight": qdata,
        "model.layer.weight_scale": params.block_scale,
        "model.layer.weight_scale_2": params.scale,
    }, str(path), metadata=metadata)

    class Config:
        def get_model(self, state, prefix, device):
            model = torch.nn.Module()
            model.diffusion_model = torch.nn.Module()
            model.diffusion_model.layer = torch.nn.Module()
            model.diffusion_model.layer._orig_shape = tuple(original.shape)
            return model

    monkeypatch.setattr(comfy.model_detection, "model_config_from_unet", lambda *args, **kwargs: Config())
    handler = modules["resize"].MemoryEfficientSafeOpen(str(path), low_memory=True)
    try:
        decoder = modules["guard"].DiffusionQuantization(handler, str(path), "NVFP4 test")
        key = "model.layer.weight"
        assert decoder.logical_shape(key) == tuple(original.shape)
        keys = decoder.required_keys(key)
        stream = handler.async_stream(keys, batch_size=1, prefetch_batches=1, pin_memory=False)
        tensors = {}
        try:
            for source_key in keys:
                batch = next(stream)
                assert len(batch) == 1 and batch[0][0] == source_key
                tensors[source_key] = batch[0][1]
            actual = decoder.decode(key, tensors, torch.float16)
        finally:
            for source_key in tensors:
                handler.mark_processed(source_key)
            stream.close()
        expected = QuantizedTensor(qdata, layout_name, params).dequantize()
        torch.testing.assert_close(actual, expected)
        assert actual.shape == original.shape
    finally:
        handler.__exit__(None, None, None)
