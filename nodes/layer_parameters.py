"""Parsed per-layer numeric overrides shared by merge, extraction and resize."""

from dataclasses import dataclass
import fnmatch
import math
import re

from comfy_api.latest import io


LAYER_PARAMETERS = io.Custom("MODELUTILS_LAYER_PARAMETERS")


@dataclass(frozen=True)
class LayerParameterRule:
    line: int
    pattern: str
    matcher: re.Pattern
    assignments: tuple[tuple[str, int | float], ...]


@dataclass(frozen=True)
class LayerParameterRules:
    rules: tuple[LayerParameterRule, ...]
    glob_patterns: bool = False


@dataclass(frozen=True)
class Parameter:
    name: str
    aliases: tuple[str, ...]
    minimum: int | float
    maximum: int | float
    integral: bool = False


MERGE_COEFFICIENTS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
MERGE_MODES = {
    "Weight-Sum": ("alpha",),
    "Train-Difference": ("alpha",),
    "Add-Difference": ("alpha", "beta"),
    "Power-Up (DARE)": ("alpha", "beta"),
    "Comparative-Interpolation": ("alpha", "beta", "gamma"),
    "Add-Dissimilarities": ("alpha", "beta", "gamma"),
    "SVD LoRA Extraction": ("alpha", "beta", "gamma"),
    "Enhanced Auto Interp": ("alpha", "beta", "gamma"),
    "Weight-Sum Cutoff": ("alpha", "beta", "gamma"),
    "Power-Up (DARE+TIES)": ("alpha", "beta", "gamma"),
    "Extract-Features": ("alpha", "beta", "gamma", "delta"),
    "Enhanced Man Interp": ("alpha", "beta", "gamma", "delta"),
    "Power-Up Enhanced (DARE)": ("alpha", "beta", "gamma", "delta"),
    "Power-Up Enhanced (DARE+TIES)": ("beta", "gamma", "delta", "epsilon", "zeta"),
}
EXTRACTION_BINDINGS = {
    "fixed": ("linear_dim", "conv_dim", "clamp_quantile", "min_diff"),
    "ratio": ("linear_ratio", "conv_ratio", "clamp_quantile", "min_diff", "linear_max_rank", "conv_max_rank"),
    "quantile": ("linear_quantile", "conv_quantile", "clamp_quantile", "min_diff", "linear_max_rank", "conv_max_rank"),
    "frobenius": ("linear_target", "conv_target", "clamp_quantile", "min_diff", "linear_max_rank", "conv_max_rank"),
    "knee": ("linear_max_rank", "conv_max_rank", "clamp_quantile", "min_diff"),
}
RESIZE_BINDINGS = {
    "fixed": ("new_rank",),
    "ratio": ("max_rank", "ratio"),
    "frobenius": ("max_rank", "min_rank", "target"),
    "cumulative": ("max_rank", "target"),
}
_ASSIGNMENT = re.compile(r"([a-z][a-z0-9_]*)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
_SEPARATOR = re.compile(r"(?:\s*,\s*|\s+)")


def parse_rules(text: str, glob_patterns: bool = False) -> LayerParameterRules:
    rules = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        row = re.fullmatch(r"\((.*)\)\s+(.+)", line)
        if row is None or not row[1].strip():
            raise ValueError(f"Layer parameters line {line_number}: expected (pattern) name:value assignments.")
        pattern, values = row.groups()
        try:
            matcher = re.compile(fnmatch.translate(f"*{pattern}*") if glob_patterns else pattern)
        except re.error as exc:
            raise ValueError(f"Layer parameters line {line_number}, pattern {pattern!r}: invalid regex: {exc}") from exc
        assignments = []
        seen = set()
        position = 0
        while position < len(values):
            match = _ASSIGNMENT.match(values, position)
            if match is None:
                raise ValueError(f"Layer parameters line {line_number}, pattern {pattern!r}: invalid assignment near {values[position:]!r}.")
            name, literal = match.groups()
            if name in seen:
                raise ValueError(f"Layer parameters line {line_number}, pattern {pattern!r}: duplicate parameter {name!r}.")
            try:
                value = float(literal) if any(c in literal for c in ".eE") else int(literal)
                finite = math.isfinite(value)
            except (ValueError, OverflowError):
                finite = False
            if not finite:
                raise ValueError(f"Layer parameters line {line_number}, pattern {pattern!r}: {name} must be finite.")
            assignments.append((name, value))
            seen.add(name)
            position = match.end()
            if position == len(values):
                break
            separator = _SEPARATOR.match(values, position)
            if separator is None or separator.end() == len(values):
                raise ValueError(f"Layer parameters line {line_number}, pattern {pattern!r}: expected another assignment after a comma or whitespace.")
            position = separator.end()
        rules.append(LayerParameterRule(line_number, pattern, matcher, tuple(assignments)))
    return LayerParameterRules(tuple(rules), glob_patterns)


def parameter_bindings(profile: str) -> tuple[Parameter, ...]:
    family, _, mode = profile.partition(":")
    if family == "merge" and mode in MERGE_MODES:
        parameters = []
        probabilities = {
            "Power-Up (DARE)": {"alpha"},
            "Power-Up (DARE+TIES)": {"alpha", "beta"},
            "Power-Up Enhanced (DARE)": {"beta", "gamma"},
            "Power-Up Enhanced (DARE+TIES)": {"beta", "epsilon", "zeta"},
            "SVD LoRA Extraction": {"gamma"},
        }.get(mode, set())
        for name in MERGE_MODES[mode]:
            aliases = ("abcdef"[MERGE_COEFFICIENTS.index(name)],)
            if name == "gamma":
                aliases += ("g",)
            if name == "zeta":
                aliases += ("z",)
            integral = mode == "SVD LoRA Extraction" and name in {"alpha", "beta"}
            minimum, maximum = (0, 1) if name in probabilities else (-10, 10)
            if integral:
                minimum = 1
            parameters.append(Parameter(name, aliases, minimum, maximum, integral))
        return tuple(parameters)
    if family == "extract" and mode in EXTRACTION_BINDINGS:
        names = EXTRACTION_BINDINGS[mode]
    elif family == "resize" and mode in RESIZE_BINDINGS:
        names = RESIZE_BINDINGS[mode]
    else:
        raise ValueError(f"Unsupported layer parameter bindings: {profile!r}")
    parameters = []
    for alias, name in zip("abcdef", names):
        integral = "rank" in name or name.endswith("_dim")
        if integral:
            minimum, maximum = 1, 16384 if family == "extract" else 3072
        elif "ratio" in name:
            minimum, maximum = 1, 100
        elif name == "clamp_quantile":
            minimum, maximum = 0.5, 1
        else:
            minimum, maximum = (0.1 if family == "resize" else 0), 1
        parameters.append(Parameter(name, (alias,), minimum, maximum, integral))
    return tuple(parameters)


def _validated_value(parameter, value, context):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{context}: {parameter.name} must be a finite number, got {value!r}.")
    if parameter.integral and int(value) != value:
        raise ValueError(f"{context}: {parameter.name} must be integral, got {value}.")
    if not parameter.minimum <= value <= parameter.maximum:
        raise ValueError(f"{context}: {parameter.name} must be in [{parameter.minimum}, {parameter.maximum}], got {value}.")
    return int(value) if parameter.integral else value


def resolve_layer_parameters(layer_parameters, profile, layers, defaults, *, node_name):
    """Validate against header-only target names and return matched scalar overrides."""
    if layer_parameters is None:
        return {}
    if not isinstance(layer_parameters, LayerParameterRules):
        raise TypeError(f"{node_name}: layer_parameters must come from Layer Parameter Configuration.")
    if not layer_parameters.rules:
        return {}
    bindings = parameter_bindings(profile)
    aliases = {alias: parameter for parameter in bindings for alias in (*parameter.aliases, parameter.name)}
    assignments = []
    for rule in layer_parameters.rules:
        resolved = {}
        for name, value in rule.assignments:
            parameter = aliases.get(name)
            context = f"{node_name}: line {rule.line}, pattern {rule.pattern!r}"
            if parameter is None:
                allowed = ", ".join(p.name for p in bindings)
                raise ValueError(f"{context}: unsupported parameter {name!r} for {profile}; allowed: {allowed}.")
            if parameter.name in resolved:
                raise ValueError(f"{context}: duplicate aliases for {parameter.name}.")
            resolved[parameter.name] = _validated_value(parameter, value, context)
        assignments.append(resolved)
    hits = set()
    result = {}
    for layer in sorted(set(layers)):
        matched = [i for i, rule in enumerate(layer_parameters.rules) if rule.matcher.search(layer)]
        if len(matched) > 1:
            lines = "; ".join(f"line {layer_parameters.rules[i].line} ({layer_parameters.rules[i].pattern!r})" for i in matched)
            raise ValueError(f"{node_name}: layer {layer!r} matches conflicting rules: {lines}.")
        if not matched:
            continue
        index = matched[0]
        values = dict(defaults)
        values.update(assignments[index])
        if profile == "extract:fixed":
            for dimension, cap in (("linear_dim", "linear_max_rank"), ("conv_dim", "conv_max_rank")):
                if dimension in assignments[index] and cap in defaults:
                    values[cap] = values[dimension]
        rule = layer_parameters.rules[index]
        context = f"{node_name}: layer {layer!r}, line {rule.line}, pattern {rule.pattern!r}"
        for parameter in bindings:
            if parameter.name in values:
                values[parameter.name] = _validated_value(parameter, values[parameter.name], context)
        if profile == "resize:frobenius" and values["min_rank"] > values["max_rank"]:
            raise ValueError(f"{node_name}: layer {layer!r}, line {rule.line}, pattern {rule.pattern!r}: min_rank={values['min_rank']} exceeds max_rank={values['max_rank']}.")
        result[layer] = values
        hits.add(index)
    unmatched = [rule for i, rule in enumerate(layer_parameters.rules) if i not in hits]
    if unmatched:
        details = "; ".join(f"line {rule.line}, pattern {rule.pattern!r}" for rule in unmatched)
        raise ValueError(f"{node_name}: no target layer matches {details}.")
    return result


def _binding_description(profile):
    return "; ".join(f"{'/'.join(p.aliases)}={p.name}" for p in parameter_bindings(profile))


def parameter_input(profile):
    if profile == "merge":
        parameters = {p.name: p for mode in MERGE_MODES for p in parameter_bindings(f"merge:{mode}")}
        description = "; ".join(f"{'/'.join(parameters[name].aliases)}={name}" for name in MERGE_COEFFICIENTS)
        description += ". Only coefficients used by Calc Mode are accepted."
    else:
        description = _binding_description(profile)
    return LAYER_PARAMETERS.Input(
        "layer_parameters", optional=True,
        tooltip="Optional Layer Parameter Configuration. " + description + " Full names are also accepted. Unassigned values use this node's settings; existing filters still apply.",
    )


def documentation():
    lines = [
        "# Per-layer parameter rules", "",
        "Connect Layer Parameter Configuration to the optional layer_parameters socket on standard two/three-input mergers, all 25 extraction variants, or the four LoRA resizers.", "",
        "Write (pattern) a:value b:value, or use full parameter names. Commas and whitespace may separate assignments; whitespace around colons is allowed. Signed decimal/scientific numbers are accepted. Blank lines and whole-line # comments are ignored. Inline comments are not supported.", "",
        "Patterns preserve backslashes, nested groups, spaces and character classes. Regex uses substring search; Glob Patterns uses case-sensitive glob substring matching. These settings do not change the main node's filters.", "",
        "Unconnected or empty configuration changes nothing. Unmatched layers and omitted parameters use main-node settings. Existing exclusions, inclusion filters, discards, mismatch rules and safety guards still apply.", "",
        "Invalid syntax/values, duplicate assignments (including aliases), unsupported parameters, overlapping rule lines and rules matching no target layer are errors. Matching and validation happen before tensor streaming or output writing. Name matches are counted before exclusions.", "",
        "LoRA rules target normalized logical block names, not factor/alpha suffixes. Existing dotted normalization and reference mapping are retained, including underscores inside names such as qkv_proj. Unresolved flattened names retain the existing fallback; patterns are not silently rewritten.", "",
        "## Bindings and bounds", "", "| Receiver/method | Short aliases and full names | Bounds |", "|---|---|---|",
    ]
    profiles = [*(f"merge:{name}" for name in MERGE_MODES), *(f"extract:{name}" for name in EXTRACTION_BINDINGS), *(f"resize:{name}" for name in RESIZE_BINDINGS)]
    for profile in profiles:
        bounds = "; ".join(f"{p.name}: {'integer ' if p.integral else ''}[{p.minimum}, {p.maximum}]" for p in parameter_bindings(profile))
        lines.append(f"| {profile} | {_binding_description(profile)} | {bounds} |")
    lines.extend([
        "", "Frobenius resize requires resolved min_rank <= max_rank. Existing tensor-dimension rank caps remain. Fixed extraction dimension overrides also update their corresponding rank caps. Probe offsets, modes, iterations, optimizer settings, seeds, files, dtype and device remain main-node settings.",
        "", "## Examples", "", "Standard Extract-Features merge:", "", "```text",
        r"(blocks\.4[589]\.attn\.qkv_proj) a:0.5 b:0.25 g:0.75 d:1.0", "```", "",
        "Fixed extraction:", "", "```text",
        r"(blocks\.4[589]\.attn\.qkv_proj) a:64, b:32, c:0.99, d:0.0", "```", "",
        "Frobenius resize:", "", "```text",
        r"(blocks\.4[589]\.attn\.qkv_proj) max_rank:128 min_rank:1 target:0.9", "```",
    ])
    return "\n".join(lines)


class LayerParameterConfiguration(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LayerParameterConfiguration", display_name="Layer Parameter Configuration",
            category="ModelUtils/Configuration",
            description="Configure per-layer numeric overrides for standard merging, extraction and LoRA resizing.",
            inputs=[
                io.String.Input("rules", default="", multiline=True, tooltip="One (pattern) name:value rule per line. Short aliases or full names are accepted; separate assignments with whitespace or commas. See Documentation for receiver-specific bindings."),
                io.Boolean.Input("glob_patterns", default=False, tooltip="Use case-sensitive glob substring matching instead of regex substring search. Independent of receiving nodes' exclusion-filter syntax."),
            ],
            outputs=[LAYER_PARAMETERS.Output(display_name="layer_parameters"), io.String.Output(display_name="documentation")],
        )

    @classmethod
    def execute(cls, rules="", glob_patterns=False):
        return io.NodeOutput(parse_rules(rules, glob_patterns), documentation())
