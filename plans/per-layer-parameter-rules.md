# Per-layer parameter rules for merge, extraction, and resize

## 1. Summary and initial scope

Add one multiline configurator node that supplies optional per-layer numeric overrides to existing nodes. Matching layers receive the explicitly assigned values; unassigned parameters and unmatched layers retain the main node’s settings. Existing exclusions, inclusion filters, discard rules, mismatch handling, and safety guards remain authoritative.

The initial implementation covers these 39 nodes:

| Family | Included variants | Count |
|---|---|---:|
| Standard mergers | Two- and three-input checkpoint, diffusion-model, text-encoder, LoRA, and embedding mergers | 10 |
| Extraction | Fixed, Ratio, Quantile, Knee, and Frobenius for LoRA, DoRA, learned DoRA, text-encoder LoRA, and text-encoder DoRA | 25 |
| LoRA resize | Fixed, Ratio, Frobenius, and Cumulative | 4 |

Do not expand this implementation to CWB configuration, weighted multi-LoRA merging, LoRA application, Lodestone, optimizer settings, iteration counts, seeds, device selection, dtype selection, or new normalization behavior. These remain post-implementation considerations.

No changes are made during planning. After approval and execution is enabled, the first implementation action is to save this entire approved plan verbatim to `plans/per-layer-parameter-rules.md`. Commit that planning document before editing implementation files. Commit the completed feature after verification.

Preserve the pre-existing analysis/report edits currently outside the filter commit.

## 2. Public interface, syntax, and bindings

### Configurator and connection

Implement `LayerParameterConfiguration`, displayed as **Layer Parameter Configuration**, in a new `nodes/layer_parameters.py` module.

Its inputs are:

| Input | Type | Default | Meaning |
|---|---|---|---|
| `rules` | Multiline string | Empty | One pattern-and-assignment rule per line |
| `glob_patterns` | Boolean | False | False uses regex substring search; True uses glob substring matching |

Its outputs are:

| Output | Type | Meaning |
|---|---|---|
| `layer_parameters` | `io.Custom("MODELUTILS_LAYER_PARAMETERS")` | Immutable, parsed rules |
| `documentation` | String | Syntax, bindings, supported nodes, fallback behavior, errors, and examples |

This follows the existing `io.Custom(...)` configurator/socket mechanism already used by CWB. No frontend extension or dynamic socket-type changes are needed.

Append an optional `layer_parameters` input to each supported receiver. Append `layer_parameters=None` to its execution interface and forward it by keyword. Preserve every existing input identifier, widget order, output, and default.

An unconnected input or an empty rule set leaves the operation unchanged.

### Rule grammar

Accept the following forms:

```text
(blocks\.4[589]\.attn\.qkv_proj) a:0.5 b:0.25 c:0.75 d:1.0
(blocks\.4[589]\.attn\.qkv_proj) alpha:0.5, beta:0.25, gamma:0.75, delta:1.0
(blocks\.4[589]\.attn\.qkv_proj) a:64, b:32, c:0.99, d:0.0
(blocks\.4[589]\.attn\.qkv_proj) max_rank:128 min_rank:1 target:0.9
```

Rules:

- Require a parenthesized pattern followed by at least one `name:value` assignment.
- Accept whitespace, commas, or a mixture between assignments, including whitespace around colons.
- Preserve the pattern’s backslashes, internal whitespace, nested groups, and character classes.
- Accept signed decimal and scientific-notation numeric values. Reject nonfinite values.
- Ignore blank lines and whole-line comments whose first nonwhitespace character is `#`. Do not support inline comments.
- Reject empty patterns, missing values, trailing garbage, duplicate assignments, and invalid regex.
- Use `re.search` for regex matching. For glob matching, use case-sensitive `fnmatchcase` with surrounding wildcards to provide substring matching.
- Keep configurator pattern syntax independent of the main node’s existing exclusion-filter syntax.

Short aliases and full parameter names may be mixed. Two aliases assigning the same parameter on one line are an error, even when their values agree.

### Binding tables

The shared binding definitions must generate both receiver tooltips and configurator documentation, preventing the two from drifting.

#### Standard merge coefficients

Bindings remain stable across calculation modes:

| Short name | Full name |
|---|---|
| `a` | `alpha` |
| `b` | `beta` |
| `c` or `g` | `gamma` |
| `d` | `delta` |
| `e` | `epsilon` |
| `f` or `z` | `zeta` |

Accept only coefficients actually used by the selected calculation mode:

| Calculation mode | Supported coefficients |
|---|---|
| Weight-Sum; Train-Difference | alpha |
| Add-Difference; Power-Up (DARE) | alpha, beta |
| Comparative-Interpolation; Add-Dissimilarities; SVD LoRA Extraction; Enhanced Auto Interp; Weight-Sum Cutoff; Power-Up (DARE+TIES) | alpha, beta, gamma |
| Extract-Features; Enhanced Man Interp; Power-Up Enhanced (DARE) | alpha, beta, gamma, delta |
| Power-Up Enhanced (DARE+TIES) | beta, gamma, delta, epsilon, zeta |

Reject assignments to ignored coefficients rather than pretend they affect the selected mode. Seeds remain main-node settings.

#### Extraction

Use the same bindings across LoRA, DoRA, learned DoRA, and both text-encoder extraction families:

| Method | `a` | `b` | `c` | `d` | `e` | `f` |
|---|---|---|---|---|---|---|
| Fixed | linear_dim | conv_dim | clamp_quantile | min_diff | — | — |
| Ratio | linear_ratio | conv_ratio | clamp_quantile | min_diff | linear_max_rank | conv_max_rank |
| Quantile | linear_quantile | conv_quantile | clamp_quantile | min_diff | linear_max_rank | conv_max_rank |
| Frobenius | linear_target | conv_target | clamp_quantile | min_diff | linear_max_rank | conv_max_rank |
| Knee | linear_max_rank | conv_max_rank | clamp_quantile | min_diff | — | — |

Knee method selection, probe offsets, SVD iterations, chunking controls, and learned-DoRA optimization settings remain on the receiving node.

#### LoRA resize

| Method | `a` | `b` | `c` |
|---|---|---|---|
| Fixed | new_rank | — | — |
| Ratio | max_rank | ratio | — |
| Frobenius | max_rank | min_rank | target |
| Cumulative | max_rank | target | — |

### Value validation

Inherit values from the receiving node—not hardcoded configurator defaults.

For supplied overrides:

- Extraction dimensions and rank caps are integral values in `[1, 16384]`; ratios are `[1, 100]`; quantile/Frobenius targets are `[0, 1]`; `clamp_quantile` is `[0.5, 1]`; `min_diff` is `[0, 1]`.
- Resize ranks are integral values in `[1, 3072]`; ratios are `[1, 100]`; targets are `[0.1, 1]`.
- Validate resolved Frobenius resize `min_rank <= max_rank`, including values inherited from the main node.
- Standard merge coefficients retain existing widget bounds. Where a coefficient represents a quantile or probability, validate `[0, 1]`; generic SVD rank coefficients must also be positive integral values.
- Do not silently truncate fractional ranks or repair invalid assignments. Retain existing mathematical rank caps imposed by tensor dimensions.

## 3. Matching and implementation data flow

### Matching contract

For ordinary model merging and extraction, match the receiving operation’s target tensor names.

For LoRA inputs, match the normalized logical layer name after the existing parsing and normalization path. Do not match factor or alpha suffixes independently.

Use the established dotted naming produced by extraction and existing reference mapping. Preserve internal identifiers such as `qkv_proj`. Do not introduce underscore normalization as the new public convention, rewrite regex punctuation, or redesign model-format conversion.

Retain the existing unresolved flattened-name fallback when no reference mapping exists. Documentation must distinguish this fallback from the normal dotted path. A dotted rule that cannot match the resulting normalized input names produces the ordinary unmatched-rule error; do not silently reinterpret it.

A single rule matching multiple keys belonging to one logical LoRA layer counts as one match. That layer receives one parameter set shared by its factors.

### Strict errors and fallback

Validate all rules before opening the output writer or beginning tensor streaming:

1. Parse syntax and compile patterns once in the configurator.
2. At each receiver, resolve aliases and reject unsupported parameters.
3. Use header/key information to identify target layers and their existing normalized names.
4. Detect matching rules per logical layer.
5. Reject multiple matching rule lines for any one layer—even if their assignments concern different parameters.
6. Reject any rule line matching no target layer.
7. Combine the single matching rule with main-node values and validate the resulting parameters.

Errors identify the receiving node, configuration line number, pattern, and affected layer or parameter. Overlap errors include every conflicting line number.

Determine name matches before existing exclusions so an explicitly excluded layer is not falsely reported as nonexistent. Overrides never reactivate an excluded, discarded, guarded, or otherwise unprocessable layer.

### Shared implementation

Keep parsing, immutable rule records, binding definitions, and matching/error handling in the new shared module.

Store only lightweight per-layer assignments and resolved scalar parameters. Do not store tensors, mutate the shared configuration payload, or modify global/main-node parameter dictionaries between layers.

Register the configurator through the existing extension node list. Do not add dependencies.

### Standard merger integration

In `MergerLogic.execute_merge`:

- Prepare the rule-to-layer mapping alongside existing header-based work planning.
- For LoRA mode, use parsed logical groups to assign the same override to their factor keys; retain existing merge and alpha-normalization behavior.
- Immediately before `calc_mode_class.create_recipe`, construct a layer-local parameter dictionary containing the base settings plus that layer’s overrides.
- Keep tensor references and preloaded-source data local to the current work unit.
- Do not mutate `recipe_params` with overrides that could affect later layers.

Do not change recipe formulas, calculation-mode selection, or source-loading behavior.

### Extraction integration

Thread the optional rules through all 25 node entry points into the four extraction backends.

Prepare resolved parameters before streaming, then use them in each backend’s `_process_layer` before `min_diff` testing and SVD selection.

Specific requirements:

- Fixed-rank overrides update both the requested dimension and corresponding maximum-rank argument; otherwise the original cap would negate an increased rank.
- Adaptive overrides update the existing linear/conv method parameter and rank cap without altering method selection or probe settings.
- Normal execution, CPU retry, and chunked fallback use the same resolved layer values.
- Extend the existing chunked extraction helpers with a trailing `clamp_quantile` argument and pass the effective value through. These helpers currently omit it when calling `_svd_extract_linear`; text-encoder extraction shares the SVD helper.
- Perform configuration validation outside broad extraction exception handlers so a configuration error cannot become a silently skipped layer.

Do not alter SVD, chunk recombination, learned optimization, or tensor-lifecycle algorithms.

### Resize integration

Attach resolved rank/target values to the existing logical-layer work units.

When calling `_resize_lora_factors`, translate the public bindings into the existing arguments:

- Fixed `new_rank` → `new_rank`.
- Adaptive `max_rank` → `new_rank`.
- Ratio/target → `dynamic_param`.
- Frobenius `min_rank` → `min_rank`.

Leave `dynamic_method` on the main node. Preserve whole-layer factor handling, alpha behavior, passthrough data, low-bit handling, dtype selection, and CPU fallback.

## 4. Verification and delivery

Extend the repository-owned test selector with focused rule-parser and integration coverage. Map the new production module explicitly; do not rely solely on the broad repository-contract group.

Required tests:

- Both delimiter styles, mixed short/full names, scientific notation, nested regex groups, CRLF, blank lines, and comments.
- Invalid syntax, invalid regex, unsupported parameters, duplicate aliases, nonfinite values, fractional ranks, and invalid resolved rank bounds.
- Overlapping rules fail with layer and line numbers; unmatched rules fail before streaming or writing.
- Partial assignments inherit remaining main-node values. Unmatched layers retain main-node behavior.
- Existing include/exclude/discard and safety paths remain authoritative.
- LoRA matching uses normalized logical names, preserves internal underscores, and assigns one configuration to paired factors.
- All 39 receiver schemas append the optional socket without moving existing controls; every execution wrapper forwards it correctly.
- Standard merge outputs reflect different coefficients on different layers, without leaking overrides to later layers.
- Every extraction family uses the requested per-layer rank, clamp, and `min_diff`; cover adaptive methods, CPU retry, and forced chunk fallback.
- All four resize variants use their resolved rank/target values, including inherited `min_rank` validation.
- Absent or empty configuration preserves existing behavior.
- Configuration failures preserve any existing destination file.

Use small deterministic tensors and existing I/O fixtures. Do not run model inference or add unrelated performance, architecture, or normalization test projects.

Run focused groups during development, `--changed` after accumulated changes, and the required full gate before the feature commit. Run changed-file lint, compilation, and `git diff --check`; distinguish pre-existing lint findings from new ones.

Deliver the saved approved plan, implementation, generated binding documentation/tooltips, regression tests, and a verified feature commit. Leave unrelated analysis/report edits untouched.
