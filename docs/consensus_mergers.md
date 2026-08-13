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
diffusion models, text encoders, LoRAs, and embeddings.

Model A supplies output metadata and anchors shared-layer shape, naming, and
mismatch behavior. The output key set is the union of all inputs. A valid
tensor found only in a secondary input is copied unchanged. When Model A lacks
a tensor that is present in multiple secondary inputs, those available sources
are CWB-merged without inserting a zero contribution for Model A. Embedding
mergers additionally use the longest compatible first dimension as their
alignment reference.

## Presets and custom controls

`baseline` is the default. The standard clarity, smooth, varied, and diversity
presets are joined by `power_blend` and the DSC presets demonstrated by the CWB
conditioning implementation. Select `custom` to make every manual CWB control
authoritative. Every named merger preset keeps `global_scale` at `1.0`; scaling
persisted model or LoRA weights is only performed when explicitly selected in
the custom controls.

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
- **CWB Preset**: `custom` makes all manual CWB controls authoritative. A named
  preset sets consensus, similarity cutoff, alpha/beta, norm rescaling, and its
  DSC fields. Named presets force similarity alignment. Only `power_blend`
  supplies an alignment threshold (0.9); other presets retain the manual
  alignment threshold. `position_weight` and `preserve_common_prefix` remain
  explicit extensions. A manually selected `global_scale` other than 1.0 also
  remains active.
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

All named file-merger presets use a neutral `global_scale` of 1.0. This is an
intentional repository safety difference from the broad specification's 0.7
calibration for some conditioning-oriented presets: persisted model and LoRA
strength is not reduced unless the user explicitly changes Global Scale.

| Preset | Consensus | Alpha | Similarity cutoff | Beta | Norm rescale | DSC | Soft comfort | Alignment cutoff |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `baseline` | median | 2.0 | 0.0 | 0.0 | no | no | no | manual value |
| `power_blend` | median | 8.0 | 0.75 | 0.0 | yes | yes | no | 0.9 |
| `high_clarity` | median | 3.0 | 0.3 | 0.0 | no | no | no | manual value |
| `smooth` | mean | 1.5 | 0.0 | 0.0 | no | no | no | manual value |
| `varied_merge` | median | 2.0 | 0.0 | 0.0 | yes | no | no | manual value |
| `diverse_concept` | median | 2.0 | 0.0 | 1.0 | yes | no | no | manual value |
| `high_diversity_concept` | median | 2.0 | 0.0 | 2.0 | yes | no | no | manual value |
| `dsc_baseline` | median | 2.0 | 0.0 | 0.0 | no | yes | yes | manual value |
| `dsc_high_clarity` | median | 4.0 | 0.3 | 0.0 | no | yes | yes | manual value |
| `dsc_smooth` | mean | 1.0 | 0.0 | 0.0 | no | yes | yes | manual value |
| `dsc_varied_merge` | median | 2.5 | 0.0 | 0.0 | yes | yes | yes | manual value |
| `dsc_diverse_concept` | median | 2.0 | 0.0 | 1.5 | yes | yes | yes | manual value |
| `dsc_high_diversity_concept` | median | 2.0 | 0.0 | 3.0 | yes | yes | yes | manual value |

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
Secondary formats are matched by normalized logical layer name. Differing
ranks are zero-padded like the DARE/TIES mergers: A/down tensors on dimension 0
and B/up tensors on dimension 1. An existing anchor-source alpha key is updated
to the resulting maximum rank; no alpha key is invented when the anchor has
none.
Similarity alignment operates on paired latent-rank components. A rows and B
columns share one mapping; fixed input/output feature axes are never reordered.
Input alpha scaling is absorbed into the B factor before blending, and global
scale is applied once to the resulting pair rather than once per factor.
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
