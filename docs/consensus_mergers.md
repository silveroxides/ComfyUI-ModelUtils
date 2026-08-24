# Consensus-Weighted Blending Mergers

The dedicated CWB nodes merge two or three safetensors files using
Consensus-Weighted Blending. They calculate an element-wise mean or median
consensus, compare each aligned source vector to that consensus, and normalize
similarity-derived weights before producing the output vector.

The broad mathematical and integration reference is maintained in
[`references/Spec-CWB`](../references/Spec-CWB/README.md). This document
describes the behavior actually exposed by this repository's offline merger
nodes. The specification covers additional contexts, including online
conditioning and spatial fusion, which are not controls of these nodes.

## Node families

Two-input and three-input variants are provided for checkpoints, standalone
diffusion models, text encoders, LoRAs, and embeddings. LoRA also provides a
Multi-Merge variant for 2 to 8 equal-prior inputs.

## Embedding coalescing

`CWB Embedding Self-Coalesce` reduces one embedding by replacing mutually
nearest body-vector pairs that meet both cosine and normalized-position limits.
`CWB Embedding Multi-Merge` accepts 2 to 8 embeddings, performs normal CWB
alignment first, then applies the same reduction. A target row count of zero
continues until no eligible pair remains.

Vision Boundary Embeddings preserves encoded vision-start and vision-end rows
unchanged and applies CWB only to their interior. Normal vision inputs validate
their outer rows before writing. Legacy Boundary Search trims accidental outer
template rows by matching a selected known-clean visual embedding; an absent,
ambiguous, or low-similarity boundary pair is an error rather than a guessed
crop. Disable Vision Boundary Embeddings for open textual-inversion tensors.

Model A supplies output metadata and anchors shared-layer shape, naming, and
mismatch behavior. The output key set is the union of all inputs. A valid
tensor found only in a secondary input is copied unchanged. When Model A lacks
a tensor that is present in multiple secondary inputs, those available sources
are CWB-merged without inserting a zero contribution for Model A. Embedding
mergers additionally use the longest compatible first dimension as their
alignment reference.

## Presets and custom controls

Presets are separated by merge type. Dense tensors, embeddings, and LoRA
factor pairs never share one menu. Every named preset supplies every CWB
setting and cannot inherit hidden widget state.

Manual controls live in the separate CWB Custom Configuration node. Connecting
its `CWB_CONFIG` output completely overrides the selected preset. Leaving it
unconnected uses the named preset.

Preset names expose categorical and boolean behavior. `_idx` and `_sim`
identify alignment, `_medn` and `_mean` identify consensus, and `_rn`, `_dsc`,
`_softcb`, and `_pcp` identify enabled boolean transformations. Dense presets
omit alignment tokens because dense model coordinates are fixed.

Similarity alignment greedily matches source rows to the reference by cosine
similarity. Index alignment keeps absolute row positions. Position weight can
bias similarity matches toward nearby normalized positions. Preserve Common
Prefix copies the longest numerically identical leading row span directly from
the first input.

Inputs have equal prior weight. The similarity calculation supplies the
effective contribution for each aligned vector. If every candidate is rejected
or receives zero weight, CWB falls back to equal weighting.

### Weighting sequence

For each aligned vector group, the merger performs these operations in order:

1. Compute an element-wise mean or median consensus.
2. Measure each vector's cosine similarity to that consensus.
3. Reject similarities below `similarity_threshold`.
4. Optionally remap unequal accepted similarities with Dynamic Similarity
   Contrast.
5. Apply `similarity^power_alpha` and, when enabled, the diversity bandpass.
6. Normalize the resulting contributor weights and calculate their weighted
   sum. If no usable weight remains, use equal contributor weights.
7. Optionally apply `rescale_norm`.
8. Apply `global_scale`.

### Complete control reference

- **Execution Mode**: `MERGE` writes an output. `DOCUMENTATION ONLY` returns
  this document without opening model inputs.
- **Model A**: primary contributor and preservation anchor. It supplies output
  metadata and anchors shared names and shapes. The output key set remains the
  union of all inputs.
- **Model B / Model C**: additional equal-prior contributors. CWB derives their
  effective per-vector influence from consensus similarity; these nodes do not
  expose model-level weights.
- **CWB Preset**: complete use-case settings selected from the node's own dense,
  embedding, or LoRA registry. A connected CWB Config overrides it completely.
- **Consensus Type**: `mean` uses the arithmetic center; `median` uses the
  element-wise median and is less sensitive to coordinate outliers.
- **Alignment Method**: `index` pairs first-axis vectors by position;
  `similarity` uses greedy one-to-one cosine matching. Row reordering is
  enabled only for embeddings and paired LoRA rank components. Ordinary
  checkpoint, diffusion-model, and text-encoder tensors retain fixed
  coordinates.
- **Alignment Threshold**: minimum original cosine score for accepting a
  greedy row match. It does not filter already-paired contributors during CWB
  weighting; that is `similarity_threshold`.
- **Similarity Threshold**: minimum cosine similarity between a participating
  vector and the group consensus. Rejected vectors receive zero weight. If all
  are rejected, the implementation falls back to equal weights.
- **Power Alpha**: exponent applied to accepted non-negative similarities.
  Values above 1 increasingly favor contributors closest to consensus. Zero
  makes every accepted contributor's similarity term equal to one.
- **Diversity Beta**: exponent for the diversity bandpass. Values above zero
  suppress vectors extremely close to consensus relative to more varied
  accepted vectors. Zero disables this term.
- **Rescale Norm**: after blending, preserve the merged direction but replace
  its L2 magnitude with the mean L2 magnitude of the participating vectors:

  `merged = merged / ||merged|| × mean(||source_i||)`

  A zero merged vector remains unchanged. This is performed per aligned row,
  not once per complete tensor. For LoRA, it is performed separately for each
  paired A row and B column, so enabling it can materially change the effective
  low-rank delta magnitude.
- **Global Scale**: multiplies the merged result after CWB. For LoRA it is
  applied once through B/up so the represented delta is scaled once rather
  than once per factor.
- **Dynamic Similarity Contrast**: when consensus similarities differ, remaps
  them into the range 0.7–1.0 before alpha/beta weighting. It does not alter
  row-alignment scores.
- **Soft Comfort Bandpass**: when beta is positive, uses
  `(1.5 - similarity)^beta` instead of the narrower
  `(1.001 - similarity)^beta`. It has no effect when beta is zero.
- **Position Weight**: blends normalized positional affinity into greedy row
  selection. Zero uses cosine only and one uses positional affinity only. It
  affects only similarity-aligned embeddings and LoRA rank components; the
  original cosine score is still checked against the alignment threshold.
- **Preserve Common Prefix**: copies Model A's longest numerically identical
  leading first-axis span before CWB. The current implementation applies this
  to all tensors with at least two dimensions, including fixed-coordinate
  model tensors.
- **Mismatch Mode**: for anchored tensors that are missing or incompatible,
  `skip` preserves the anchor, `zeros` supplies a compatible zero contribution
  where possible, and `error` aborts. A valid tensor supplied by only one
  secondary input is copied unchanged.
- **Output Filename**: filename without extension, written atomically beneath
  the matching category in ComfyUI's models directory.
- **Save Dtype**: requested dtype for generated floating tensors. Unless
  Override Dtype is enabled, any participating FP32 source keeps the generated
  logical result FP32.
- **Process Device**: device for per-layer FP32 arithmetic. A CUDA OOM releases
  the failed working set and retries only that layer on CPU.
- **Exclude Patterns**: matching layers are preserved from their anchor instead
  of merged.
- **Discard Patterns**: matching tensors or logical LoRA groups are absent from
  the output.
- **Glob Patterns**: switches exclude/discard syntax from regular expressions
  to shell-style globs. Pattern fields accept one entry per line.
- **Lazy Load**: enables UEL low-memory loading so tensors are streamed by work
  unit and explicitly released.
- **Force Clear Cache**: runs garbage collection and clears the CUDA allocator
  cache before each layer. This can reduce retained memory at a substantial
  speed cost.
- **Override Dtype**: forces generated floating tensors to Save Dtype. Guarded
  low-bit tensors and enabled 1D direct differences are exempt.
- **Include 1D Diffs** (LoRA nodes): enables CWB merging of recognized 1D
  direct-difference tensors in FP32. When disabled, shared tensors preserve
  Model A and secondary-only tensors preserve their earliest provider.

Scalars and 1D generic tensors use direct mean/median handling. Alignment,
similarity weighting, alpha/beta, DSC, and norm rescaling apply to vector groups
in tensors with at least two dimensions, not to those scalar/1D paths.

### Presets implemented by these nodes

Dense nodes offer `balanced_mean`, `robust_medn`, `selective_mean`,
`varied_mean_rn_softcb`, `diverse_medn_rn_dsc_softcb`, and
`strongdiv_medn_rn_dsc_softcb`.

Embedding nodes offer `balanced_idx_mean`, `balanced_sim_mean`,
`robust_idx_medn`, `robust_sim_medn`, `varied_sim_mean_rn_softcb`, and
`diverse_sim_medn_rn_dsc_softcb`, `focused_strong_sim_medn`,
`focused_balance_sim_medn`, `focused_soft_sim_medn`, and
`focused_weak_sim_medn`.

LoRA nodes offer `broad_sim_medn_rn_softcb`,
`moderate_sim_medn_rn_softcb`, `conservative_sim_medn_rn_softcb`,
`direct_idx_medn_rn_softcb`, `broad_sim_mean_rn_softcb`,
`neutral_sim_medn_rn`, `focused_sim_medn_rn_dsc_softcb`, and
`strongfocus_sim_medn_rn_dsc_softcb`, `focused_strong_sim_medn`,
`focused_balance_sim_medn`, `focused_soft_sim_medn`, and
`focused_weak_sim_medn`.

The LoRA default is `broad_sim_medn_rn_softcb`: median consensus,
similarity alignment, zero similarity threshold, alpha 2, beta 4, norm
rescaling, scale 1, DSC disabled, soft comfort bandpass enabled, position
weight 0.05, and common-prefix preservation disabled. Broad, moderate, and
conservative use alignment thresholds 0, 0.0005, and 0.0025. They represent
increasing anchor preservation in automated mixed-rank LoRA tests.

Every LoRA preset fixes all 12 controls explicitly:

| preset | consensus | alignment | align threshold | sim threshold | alpha | beta | RN | scale | DSC | soft CB | position | prefix | intent |
|---|---|---|---:|---:|---:|---:|---|---:|---|---|---:|---|---|
| `broad_sim_medn_rn_softcb` | median | similarity | 0 | 0 | 2 | 4 | on | 1 | off | on | 0.05 | off | Broad row matching; default |
| `moderate_sim_medn_rn_softcb` | median | similarity | 0.0005 | 0 | 2 | 4 | on | 1 | off | on | 0.05 | off | Moderate anchor preservation |
| `conservative_sim_medn_rn_softcb` | median | similarity | 0.0025 | 0 | 2 | 4 | on | 1 | off | on | 0.05 | off | Strong anchor preservation |
| `direct_idx_medn_rn_softcb` | median | index | 0 | 0 | 2 | 4 | on | 1 | off | on | 0.05 | off | Fixed rank-row correspondence |
| `broad_sim_mean_rn_softcb` | mean | similarity | 0 | 0 | 2 | 4 | on | 1 | off | on | 0.05 | off | Mean-consensus broad merge |
| `neutral_sim_medn_rn` | median | similarity | 0 | 0 | 2 | 0 | on | 1 | off | off | 0.05 | off | Similarity weighting without diversity modulation |
| `focused_sim_medn_rn_dsc_softcb` | median | similarity | 0 | 0 | 2 | 4 | on | 1 | on | on | 0.05 | off | Moderate DSC concentration |
| `strongfocus_sim_medn_rn_dsc_softcb` | median | similarity | 0 | 0 | 2 | 7 | on | 1 | on | on | 0.05 | off | Strong DSC concentration |
| `focused_strong_sim_medn` | median | similarity | 0.85 | 0.60 | 1.25 | 0 | off | 1 | off | off | 0.20 | off | Strongly focused matching |
| `focused_balance_sim_medn` | median | similarity | 0.75 | 0.55 | 1.25 | 0 | off | 1 | off | off | 0.20 | off | Balanced focused matching |
| `focused_soft_sim_medn` | median | similarity | 0.55 | 0.50 | 1.25 | 0 | off | 1 | off | off | 0.20 | off | Soft focused matching |
| `focused_weak_sim_medn` | median | similarity | 0.25 | 0.35 | 1.25 | 0 | off | 1 | off | off | 0.20 | off | Weak focused matching |

The four focused median similarity presets have identical settings in the
embedding and LoRA registries.

The LoRA Multi-Merge counterfactual weight sweep reuses each consensus vector
already computed by the active merge. It reports mean and median consensus
results across configured alpha, beta, similarity-threshold, DSC, and comfort
bandpass values without additional model loads or output files.

## Missing tensors and filtering

- `skip` preserves Model A when an anchored input is missing or incompatible.
  In embedding-union mode it omits that source for the affected key.
- `zeros` supplies a compatible zero tensor for the missing source.
- `error` stops the merge.
- Exclude patterns preserve the anchored source tensor.
- Discard patterns remove the matching output tensor or logical LoRA layer.

The missing-input modes above apply to Model-A-owned layers. Model A being
absent from a secondary-only layer is not itself a mismatch. A shape conflict
between multiple available secondary providers does use the selected mismatch
mode, with the earliest available input as its preservation anchor.

Patterns use regular expressions by default. Enable Glob Patterns to use glob
syntax instead.

## LoRA behavior

LoRA inputs use the same parser as the resize and multi-merge nodes. It accepts
all seven low-rank A/B spellings recognized by ComfyUI's `LoRAAdapter`, plus
direct `.diff`, `.diff_b`, `.w_norm`, `.b_norm`, and `.set_weight` layers.

Recognized diffusion-model outputs use the canonical
`diffusion_model.<layer>.lora_A.weight` / `.lora_B.weight` convention.
Secondary formats are matched by normalized logical layer name. Alpha is
absorbed mathematically into each B/up factor using its original rank before
alignment, weighting, or merge. Genuine factor ranks remain variable during
CWB. The first maximum-rank input supplies the output component space, so the
output rank remains the largest genuine input rank. Merged outputs never
contain or invent alpha tensors.

Every CWB merge node writes a `cwb.merge` safetensors metadata record. It
contains selected source model names, fully resolved CWB settings, and merge
options. Presets are stored as effective settings rather than by preset name.
Non-empty exclude/discard filters and their glob mode are included. CWB reports
and diagnostics are not saved in metadata.
Similarity alignment operates on paired latent-rank components. A rows and B
columns share one mapping; fixed input/output feature axes are never reordered.
For a rank-R reference and rank-r source it evaluates an R-by-r similarity
matrix, reduced only by a preserved common prefix. Every genuine source
component can therefore match any genuine reference component. Index alignment
maps source component i to reference component i. In either mode, reference
components without a genuine contributor remain singleton groups and are not
weighted or norm-rescaled against nonexistent rank padding.

Structural rank absence is different from `mismatch_mode=zeros`. The latter is
an explicit request for a semantic zero contributor and continues to apply to
every output-rank component where the mismatch policy is active. Numerical zero
values inside a genuine factor do not identify padding; validity comes from the
factor's original rank.

Global scale is applied once to the resulting pair rather than once per factor.
Incomplete pairs and unrecognized tensors are copied from the earliest
available source unchanged.
Companion-bearing groups (`lora_mid`, reshape, DoRA, or set-weight) are kept
atomically from their earliest available source rather than partially
transformed. Secondary-only logical groups are included automatically: a group
from one source is copied unchanged, while compatible groups from multiple
secondary sources are merged using only those sources. Recognized roles are
written with the same canonical output convention as shared layers.

`include_1d_diffs` is disabled by default. Disabled shared 1D direct layers
preserve Model A, while secondary-only layers preserve their earliest source.
Enabled 1D direct layers participate as complete tensors and are always saved
in FP32.

## Precision and quantized inputs

CWB computes in FP32. Unless Override Dtype is enabled, any participating FP32
source keeps the generated logical layer in FP32. Guarded low-bit tensors and
enabled 1D direct diffs are never downcast by the override.

ComfyUI/comfy-kitchen quantized inputs are rejected before an output writer is
opened. Files with three or more bare INT8, UINT8, FP8, or FP4 tensors are also
rejected. One or two isolated low-bit tensors produce one warning per input and
are preserved without CWB arithmetic; an affected LoRA causes its complete
logical layer to be preserved.

## CWB report output

Every CWB merge node provides a separate `cwb_report` string output after the
existing filename and documentation outputs. Documentation-only execution
returns an explicit message that no report exists because no tensors were
processed.

The filename output uses ComfyUI's wildcard socket type and returns the path
relative to the matching ComfyUI model category. A filename such as
`groupfolder/modelname` therefore returns
`groupfolder/modelname.safetensors`. It can connect directly to compatible
loader Combo inputs without exposing the absolute models-directory path.

The report begins with model-wide counters. LoRA arithmetic then adds grouped
mathematical evidence without retaining tensors or dumping complete similarity
matrices. Layers are grouped only when they share:

- down non-rank dimensions;
- up non-rank dimensions;
- ordered genuine input ranks;
- selected reference input and rank;
- alignment method.

Each dimension group reports layer count, effective matrix dimensions, genuine
source-component match coverage, reference-only components, structural rank
slots excluded from arithmetic, explicit mismatch-zero contributors, candidate
similarity min/mean/max, accepted-match similarity percentiles, and rank-one
delta norm ratios. A compact exception section identifies the lowest match
coverage and largest norm changes by layer name.

For a rank-384 reference and rank-128 source, the report identifies a `384x128`
matrix, at most 128 genuine matches, and the 256 structural rank slots that were
excluded rather than treated as zero-valued contributors.

## Streaming and failure safety

Model parameters and direct patches use fixed-coordinate blending; similarity
row matching is limited to embeddings and LoRA rank components. Each logical
layer is loaded, transferred to the processing device, completed, copied back
to CPU, queued for writing, and released before the next layer. Outputs are
written to a temporary sibling and atomically published only after every layer
and asynchronous write succeeds. Failed merges do not replace or finalize the
requested output. A CUDA OOM retries only the affected layer on CPU after
releasing the failed GPU working set; failure of that retry aborts atomically.
Generated files are published under the matching category inside
`folder_paths.models_dir`, never ComfyUI's general output directory.
