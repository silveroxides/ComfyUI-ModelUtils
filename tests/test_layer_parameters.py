import importlib
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def rules_module():
    name = "modelutils_layer_rule_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[name] = package
    return importlib.import_module(f"{name}.nodes.layer_parameters")


def resolve(module, text, *, profile="extract:fixed", names=("blocks.45.attn.qkv_proj.weight",), defaults=None, glob=False):
    return module.resolve_layer_parameters(
        module.parse_rules(text, glob), profile, names,
        defaults or {"linear_dim": 8, "conv_dim": 4, "clamp_quantile": 0.99, "min_diff": 0},
        node_name="Example Node",
    )


@pytest.mark.parametrize("assignments", [
    "a:64 b:32 c:0.99 d:0.0",
    "a : 64, b:32 c:0.99, d : 0.0",
    "linear_dim:64 b:3.2e1 clamp_quantile:+.99 d:0e0",
])
def test_aliases_separators_and_inheritance(rules_module, assignments):
    module = rules_module
    result = resolve(module, rf"(blocks\.4[589]\.attn\.qkv_proj) {assignments}")
    assert result == {"blocks.45.attn.qkv_proj.weight": {
        "linear_dim": 64, "conv_dim": 32, "clamp_quantile": 0.99, "min_diff": 0,
    }}
    partial = resolve(module, "(qkv_proj) a:64", names=("other", "blocks.45.attn.qkv_proj.weight"))
    assert "other" not in partial
    assert partial["blocks.45.attn.qkv_proj.weight"]["conv_dim"] == 4


def test_patterns_preserve_groups_spaces_escapes_and_lines(rules_module):
    text = " # explanation\r\n\r\n" + r"((?:blocks\.45|blocks\.49)\.my layer\.(?:qkv_proj)) a:16"
    parsed = rules_module.parse_rules(text)
    assert parsed.rules[0].line == 3
    assert parsed.rules[0].pattern == r"(?:blocks\.45|blocks\.49)\.my layer\.(?:qkv_proj)"
    result = rules_module.resolve_layer_parameters(parsed, "extract:fixed", ["blocks.49.my layer.qkv_proj"], {}, node_name="Fixed")
    assert result["blocks.49.my layer.qkv_proj"]["linear_dim"] == 16


@pytest.mark.parametrize("text", [
    "() a:1", "( ) a:1", "(layer)", "layer a:1", "(layer)a:1",
    "([invalid) a:1", "(layer) a:", "(layer) a:nan", "(layer) a:inf",
    "(layer) a:1e309", "(layer) a:1a:2", "(layer) a:1,", "(layer) a:1,,b:2",
    "(layer) a:1 # comment", "(layer) a:1 a:2", "(layer) a:1 rubbish",
])
def test_invalid_syntax_errors_at_configurator(rules_module, text):
    with pytest.raises(ValueError, match="line 1"):
        rules_module.LayerParameterConfiguration.execute(text)


@pytest.mark.parametrize("assignments, message", [
    ("a:1 alpha:2", "unsupported"),
    ("a:1 linear_dim:1", "duplicate aliases"),
    ("a:1.5", "integral"),
    ("a:0", "linear_dim"),
    ("a:16385", "linear_dim"),
    ("c:0.49", "clamp_quantile"),
    ("d:1.01", "min_diff"),
    ("seed:1", "unsupported"),
])
def test_receiver_validates_parameter_domains(rules_module, assignments, message):
    with pytest.raises(ValueError, match=message) as error:
        resolve(rules_module, f"(qkv_proj) {assignments}")
    assert "Example Node" in str(error.value)
    assert "line 1" in str(error.value)


def test_strict_overlap_even_for_disjoint_assignments(rules_module):
    with pytest.raises(ValueError, match="conflicting") as error:
        resolve(rules_module, "(blocks) a:32\n\n(qkv_proj) b:16")
    assert "blocks.45.attn.qkv_proj.weight" in str(error.value)
    assert "line 1" in str(error.value) and "line 3" in str(error.value)


def test_zero_match_and_repeated_logical_names(rules_module):
    with pytest.raises(ValueError, match="line 2"):
        resolve(rules_module, "(blocks) a:32\n(missing) b:16")
    assert resolve(rules_module, "(qkv_proj) a:32", names=["qkv_proj", "qkv_proj"]) == {
        "qkv_proj": {"linear_dim": 32, "conv_dim": 4, "clamp_quantile": 0.99, "min_diff": 0},
    }


def test_glob_is_case_sensitive_substring(rules_module):
    result = resolve(rules_module, "(blocks.4[589].attn.qkv_*) a:32", glob=True)
    assert len(result) == 1
    with pytest.raises(ValueError, match="no target layer"):
        resolve(rules_module, "(BLOCKS*) a:32", glob=True)


def test_frobenius_validates_inherited_bounds(rules_module):
    defaults = {"max_rank": 128, "min_rank": 32, "target": 0.9}
    with pytest.raises(ValueError, match="min_rank=32 exceeds max_rank=16"):
        resolve(rules_module, "(qkv_proj) a:16", profile="resize:frobenius", defaults=defaults)
    result = resolve(rules_module, "(qkv_proj) a:16 b:1", profile="resize:frobenius", defaults=defaults)
    assert next(iter(result.values())) == {"max_rank": 16, "min_rank": 1, "target": 0.9}
    assert defaults["max_rank"] == 128


def test_shared_rules_do_not_leak_receiver_state(rules_module):
    rules = rules_module.parse_rules("(layer) a:2")
    with pytest.raises(FrozenInstanceError):
        rules.glob_patterns = True
    first = rules_module.resolve_layer_parameters(rules, "extract:fixed", ["layer"], {"conv_dim": 4}, node_name="A")
    second = rules_module.resolve_layer_parameters(rules, "resize:fixed", ["layer"], {"new_rank": 8}, node_name="B")
    first["layer"]["linear_dim"] = 64
    assert second == {"layer": {"new_rank": 2}}
    assert rules.rules[0].assignments == (("a", 2),)
    assert rules_module.resolve_layer_parameters(None, "anything", [], {}, node_name="A") == {}
    assert rules_module.resolve_layer_parameters(rules_module.parse_rules("\n#empty"), "anything", [], {}, node_name="A") == {}
    with pytest.raises(TypeError, match="Layer Parameter Configuration"):
        rules_module.resolve_layer_parameters("not a config", "extract:fixed", [], {}, node_name="A")


def test_fixed_dimensions_update_only_explicit_rank_caps(rules_module):
    defaults = {"linear_dim": 8, "conv_dim": 4, "linear_max_rank": 2, "conv_max_rank": 1, "clamp_quantile": 0.99, "min_diff": 0}
    result = resolve(rules_module, "(qkv_proj) a:8", defaults=defaults)
    values = next(iter(result.values()))
    assert values["linear_max_rank"] == 8
    assert values["conv_max_rank"] == 1
    values = next(iter(resolve(rules_module, "(qkv_proj) c:.9", defaults=defaults).values()))
    assert values["linear_max_rank"] == 2
    assert values["conv_max_rank"] == 1


def test_merge_aliases_mode_restrictions_and_quantiles(rules_module):
    result = resolve(rules_module, "(qkv_proj) a:.5 b:.25 g:.75 d:1", profile="merge:Extract-Features", defaults={})
    assert next(iter(result.values()))["gamma"] == 0.75
    with pytest.raises(ValueError, match="duplicate aliases"):
        resolve(rules_module, "(qkv_proj) c:.5 g:.5", profile="merge:Extract-Features")
    with pytest.raises(ValueError, match="unsupported parameter"):
        resolve(rules_module, "(qkv_proj) b:.5", profile="merge:Weight-Sum")
    with pytest.raises(ValueError, match="unsupported parameter"):
        resolve(rules_module, "(qkv_proj) a:.5", profile="merge:Power-Up Enhanced (DARE+TIES)")
    with pytest.raises(ValueError, match="must be in"):
        resolve(rules_module, "(qkv_proj) b:1.1", profile="merge:Power-Up (DARE+TIES)")


def test_configurator_documentation_and_socket_types(rules_module):
    module = rules_module
    schema = module.LayerParameterConfiguration.define_schema()
    assert [item.id for item in schema.inputs] == ["rules", "glob_patterns"]
    config, docs = module.LayerParameterConfiguration.execute("(layer) a:32")
    assert isinstance(config, module.LayerParameterRules)
    assert docs.startswith("# Per-layer parameter rules")
    assert len(schema.outputs) == 2
    profiles = [*(f"merge:{mode}" for mode in module.MERGE_MODES), *(f"extract:{mode}" for mode in module.EXTRACTION_BINDINGS), *(f"resize:{mode}" for mode in module.RESIZE_BINDINGS)]
    for profile in profiles:
        assert profile in docs
        widget = module.parameter_input("merge" if profile.startswith("merge:") else profile)
        assert widget.optional
        for parameter in module.parameter_bindings(profile):
            assert parameter.name in widget.tooltip
            assert parameter.name in docs
