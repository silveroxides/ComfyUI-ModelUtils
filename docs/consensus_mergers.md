# Consensus-Weighted Blending Mergers

The dedicated CWB nodes merge two or three safetensors files using
Consensus-Weighted Blending. They calculate an element-wise mean or median
consensus, compare each aligned source vector to that consensus, and normalize
similarity-derived weights before producing the output vector.

## Node families

Two-input and three-input variants are provided for checkpoints, standalone
diffusion models, text encoders, LoRAs, and embeddings.

Checkpoint, diffusion-model, text-encoder, and LoRA outputs are anchored to
Model A. Model A supplies the output key set, metadata, and naming. Keys found
only in later inputs are ignored. Embedding mergers instead use the union of
all keys and use the longest compatible first dimension as the alignment
reference.

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

Patterns use regular expressions by default. Enable Glob Patterns to use glob
syntax instead.

## LoRA behavior

LoRA inputs use the same parser as the resize and multi-merge nodes. This
includes the repository's plain `.lora_A.weight` / `.lora_B.weight` extraction
output, already-supported alternative A/B or down/up suffixes, and direct
`.diff` / `.diff_b` layers.

Model A's exact keys and suffix convention are retained. Secondary formats are
matched by normalized logical layer name. Differing ranks are zero-padded like
the DARE/TIES mergers: A/down tensors on dimension 0 and B/up tensors on
dimension 1. An existing Model A alpha key is updated to the resulting maximum
rank; no alpha key is invented when Model A has none. Incomplete pairs,
auxiliary tensors, and unrecognized Model A tensors are copied unchanged.

`include_1d_diffs` is disabled by default. Disabled 1D direct layers preserve
Model A. Enabled 1D direct layers participate as complete tensors and are
always saved in FP32.

## Precision and quantized inputs

CWB computes in FP32. Unless Override Dtype is enabled, any participating FP32
source keeps the generated logical layer in FP32. Guarded low-bit tensors and
enabled 1D direct diffs are never downcast by the override.

ComfyUI/comfy-kitchen quantized inputs are rejected before an output writer is
opened. Files with three or more bare INT8, UINT8, FP8, or FP4 tensors are also
rejected. One or two isolated low-bit tensors produce one warning per input and
are preserved without CWB arithmetic; an affected LoRA causes its complete
logical layer to be preserved.
