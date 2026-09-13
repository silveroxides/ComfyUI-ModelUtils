import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter, MemoryEfficientSafeOpen


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "modelutils_layer_parameter_extraction_tests"


def _load(name):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


def _write(path, tensors):
    with IncrementalSafetensorsWriter(str(path), max_workers=1) as writer:
        writer.write_dict(tensors)


def test_fixed_extraction_rule_overrides_only_matching_layer(monkeypatch, tmp_path):
    extraction = _load("nodes.lora_extract_svd")
    parameters = _load("nodes.layer_parameters")
    first, second, output = (tmp_path / name for name in ("first.safetensors", "second.safetensors", "out.safetensors"))
    match = "blocks.45.attn.qkv_proj.weight"
    other = "blocks.46.attn.qkv_proj.weight"
    _write(first, {match: torch.eye(3), other: torch.eye(3)})
    _write(second, {match: torch.zeros(3, 3), other: torch.zeros(3, 3)})
    monkeypatch.setattr(extraction, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(extraction, "cleanup_after_operation", lambda: None)
    rules = parameters.parse_rules(r"(blocks\.45\.attn\.qkv_proj) linear_dim:2 clamp_quantile:1")

    extraction.extract_lora_from_files(
        str(first), str(second), "fixed", 1, 1, "cpu", "fp32", str(output),
        linear_max_rank=1, conv_max_rank=1, force_clear_cache=False,
        layer_parameters=rules,
    )

    with MemoryEfficientSafeOpen(str(output), low_memory=True) as result:
        assert result.get_shape("diffusion_model.blocks.45.attn.qkv_proj.lora_A.weight")[0] == 2
        assert result.get_shape("diffusion_model.blocks.46.attn.qkv_proj.lora_A.weight")[0] == 1


def test_unmatched_fixed_rule_preserves_existing_rank_cap(monkeypatch, tmp_path):
    extraction = _load("nodes.lora_extract_svd")
    first, second, output = (tmp_path / name for name in ("first.safetensors", "second.safetensors", "out.safetensors"))
    key = "blocks.45.attn.qkv_proj.weight"
    _write(first, {key: torch.eye(3)})
    _write(second, {key: torch.zeros(3, 3)})
    monkeypatch.setattr(extraction, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(extraction, "cleanup_after_operation", lambda: None)
    caps = []

    def extract_linear(weight, mode, parameter, device, max_rank, clamp, niter, probe):
        caps.append((parameter, max_rank))
        return (torch.ones((1, weight.shape[1])), torch.ones((weight.shape[0], 1)), 1), "lora"

    monkeypatch.setattr(extraction, "_svd_extract_linear", extract_linear)
    extraction.extract_lora_from_files(
        str(first), str(second), "fixed", 2, 2, "cpu", "fp32", str(output),
        linear_max_rank=1, conv_max_rank=1, force_clear_cache=False,
    )
    assert caps == [(2, 1)]


def test_adaptive_rule_and_cpu_retry_use_resolved_values(monkeypatch, tmp_path):
    extraction = _load("nodes.lora_extract_svd")
    parameters = _load("nodes.layer_parameters")
    first, second, output = (tmp_path / name for name in ("first.safetensors", "second.safetensors", "out.safetensors"))
    key = "blocks.45.attn.qkv_proj.weight"
    _write(first, {key: torch.eye(3)})
    _write(second, {key: torch.zeros(3, 3)})
    monkeypatch.setattr(extraction, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(extraction, "cleanup_after_operation", lambda: None)
    calls = []

    def retry(cuda_call, cpu_call, _device):
        return cpu_call()

    def extract_linear(weight, mode, parameter, device, max_rank, clamp, niter, probe):
        calls.append((mode, parameter, device, max_rank, clamp, niter, probe))
        return (torch.ones((1, weight.shape[1])), torch.ones((weight.shape[0], 1)), 1), "lora"

    monkeypatch.setattr(extraction, "retry_cuda_oom_on_cpu", retry)
    monkeypatch.setattr(extraction, "_svd_extract_linear", extract_linear)
    rules = parameters.parse_rules(r"(blocks\.45) linear_ratio:3 linear_max_rank:2 clamp_quantile:1")
    extraction.extract_lora_from_files(
        str(first), str(second), "ratio", 2, 2, "cuda", "fp32", str(output),
        linear_max_rank=1, conv_max_rank=1, force_clear_cache=False,
        layer_parameters=rules,
    )
    assert calls == [("ratio", 3, "cpu", 2, 1, 2, 32)]


def test_rule_min_diff_gates_matching_layer_before_svd(monkeypatch, tmp_path):
    extraction = _load("nodes.lora_extract_svd")
    parameters = _load("nodes.layer_parameters")
    first, second, output = (tmp_path / name for name in ("first.safetensors", "second.safetensors", "out.safetensors"))
    key = "blocks.45.attn.qkv_proj.weight"
    _write(first, {key: torch.eye(2) * 0.5})
    _write(second, {key: torch.zeros(2, 2)})
    monkeypatch.setattr(extraction, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(extraction, "cleanup_after_operation", lambda: None)
    calls = []
    monkeypatch.setattr(extraction, "_svd_extract_linear", lambda *args: calls.append(args))
    rules = parameters.parse_rules(r"(blocks\.45) min_diff:1")
    extraction.extract_lora_from_files(
        str(first), str(second), "fixed", 1, 1, "cpu", "fp32", str(output),
        linear_max_rank=1, conv_max_rank=1, force_clear_cache=False,
        layer_parameters=rules,
    )
    with MemoryEfficientSafeOpen(str(output), low_memory=True) as result:
        assert list(result.keys()) == []
    assert calls == []


def test_chunked_fallback_receives_resolved_clamp(monkeypatch, tmp_path):
    extraction = _load("nodes.lora_extract_svd")
    parameters = _load("nodes.layer_parameters")
    first, second, output = (tmp_path / name for name in ("first.safetensors", "second.safetensors", "out.safetensors"))
    key = "blocks.45.attn.qkv_proj.weight"
    _write(first, {key: torch.eye(6, 2)})
    _write(second, {key: torch.zeros(6, 2)})
    monkeypatch.setattr(extraction, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(extraction, "cleanup_after_operation", lambda: None)
    monkeypatch.setattr(extraction, "_svd_extract_linear", lambda *args: (_ for _ in ()).throw(RuntimeError("force chunk")))
    monkeypatch.setattr(extraction, "_detect_fused_layer", lambda *_: 3)
    observed = []

    def chunk(weight, chunks, mode, parameter, device, rank, probe, clamp):
        observed.append((mode, parameter, rank, probe, clamp))
        return torch.ones((weight.shape[0], 1)), torch.ones((1, weight.shape[1])), 1

    monkeypatch.setattr(extraction, "_extract_chunked_layer", chunk)
    rules = parameters.parse_rules(r"(blocks\.45) linear_dim:2 clamp_quantile:1")
    extraction.extract_lora_from_files(
        str(first), str(second), "fixed", 1, 1, "cpu", "fp32", str(output),
        linear_max_rank=1, conv_max_rank=1, force_clear_cache=False,
        layer_parameters=rules,
    )
    assert observed == [("fixed", 2, 2, 32, 1)]


@pytest.mark.parametrize(
    ("module_name", "function_name", "extra"),
    [
        ("nodes.dora_extract_wd", "extract_dora_from_files", {}),
        ("nodes.dora_learned_wd", "extract_dora_learned_from_files", {"optimize_iters": 0}),
        ("nodes.text_encoder_extract", "extract_te_from_files", {"is_dora": False}),
        ("nodes.text_encoder_extract", "extract_te_from_files", {"is_dora": True}),
    ],
)
def test_fixed_rule_changes_effective_rank_in_every_other_extraction_family(
    monkeypatch, tmp_path, module_name, function_name, extra
):
    module = _load(module_name)
    parameters = _load("nodes.layer_parameters")
    first, second, output = (tmp_path / name for name in ("first.safetensors", "second.safetensors", "out.safetensors"))
    key = "blocks.45.attn.qkv_proj.weight"
    _write(first, {key: torch.eye(3)})
    _write(second, {key: torch.zeros(3, 3)})
    monkeypatch.setattr(module, "prepare_for_large_operation", lambda *args: None)
    monkeypatch.setattr(module, "cleanup_after_operation", lambda: None)
    rules = parameters.parse_rules(r"(blocks\.45\.attn\.qkv_proj) a:2")
    kwargs = dict(linear_max_rank=1, conv_max_rank=1, force_clear_cache=False,
                  layer_parameters=rules, **extra)
    getattr(module, function_name)(
        str(first), str(second), "fixed", 1, 1, "cpu", "fp32", str(output), **kwargs
    )
    with MemoryEfficientSafeOpen(str(output), low_memory=True) as result:
        factor_key = next(key for key in result.keys() if key.endswith((".lora_A.weight", ".lora_down.weight")))
        assert result.get_shape(factor_key)[0] == 2


@pytest.mark.parametrize(
    ("module_name", "nodes"),
    [
        ("nodes.lora_extract_svd", ("LoRAExtractFixed", "LoRAExtractRatio", "LoRAExtractQuantile", "LoRAExtractKnee", "LoRAExtractFrobenius")),
        ("nodes.dora_extract_wd", ("DoRAExtractFixed", "DoRAExtractRatio", "DoRAExtractQuantile", "DoRAExtractKnee", "DoRAExtractFrobenius")),
        ("nodes.dora_learned_wd", ("DoRALearnedExtractFixed", "DoRALearnedExtractRatio", "DoRALearnedExtractQuantile", "DoRALearnedExtractKnee", "DoRALearnedExtractFrobenius")),
        ("nodes.text_encoder_extract", ("TextEncoderLoRAExtractFixed", "TextEncoderLoRAExtractRatio", "TextEncoderLoRAExtractQuantile", "TextEncoderLoRAExtractKnee", "TextEncoderLoRAExtractFrobenius", "TextEncoderDoRAExtractFixed", "TextEncoderDoRAExtractRatio", "TextEncoderDoRAExtractQuantile", "TextEncoderDoRAExtractKnee", "TextEncoderDoRAExtractFrobenius")),
    ],
)
def test_all_extraction_schemas_append_layer_parameter_socket(module_name, nodes):
    module = _load(module_name)
    for node_name in nodes:
        assert module.__dict__[node_name].define_schema().inputs[-1].id == "layer_parameters"
