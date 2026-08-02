# Consensus-Weighted Blending Mergers

The dedicated CWB nodes merge two or three safetensors files using
Consensus-Weighted Blending. They calculate an element-wise mean or median
consensus, compare each aligned source vector to that consensus, and normalize
similarity-derived weights before producing the output vector.

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
authoritative.

Similarity alignment greedily matches source rows to the reference by cosine
similarity. Index alignment keeps absolute row positions. Position weight can
bias similarity matches toward nearby normalized positions. Preserve Common
Prefix copies the longest numerically identical leading row span directly from
the first input.

Inputs have equal prior weight. The similarity calculation supplies the
effective contribution for each aligned vector. If every candidate is rejected
or receives zero weight, CWB falls back to equal weighting.

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
