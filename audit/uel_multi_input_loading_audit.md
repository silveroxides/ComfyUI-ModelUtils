# Initial UEL Loading Audit

Multi-input loading is inconsistent across the repository. Model analysis and
LoRA multi-merge use bounded UEL asynchronous streams with pinned-memory
support, while LoRA extraction, generic model merging, CWB merging, and parts
of LoRA-to-model application still call `get_tensor()` sequentially for each
input.

LoRA Knee Detection is especially affected because synchronous A/B reads are
combined with an exact full FP32 SVD per eligible layer. Its `max_rank` limit is
applied only after the full decomposition. The extractor also builds a full
residual reconstruction that its caller discards, clears caches after every
layer by default, and attempts fused-layer chunking only after the initial SVD
fails.

Initial recommendation: audit all multi-input operations against the bounded
UEL work-unit pattern already used by model analysis and LoRA multi-merge, then
migrate the shared execution paths rather than patching Knee Detection alone.

## Knee SVD follow-up

Knee extraction now uses a configurable partial-spectrum probe beginning at
`max_rank + knee_probe_offset`. A knee detected in the probe tail triggers one
expansion up to `2 * max_rank`; saved rank remains capped by `max_rank`. This is
shared by LoRA, DoRA, learned DoRA, and both text-encoder knee node families.
The same change removes discarded full residual reconstruction and replaces
diagonal-matrix multiplication with direct singular-value column scaling.

The asynchronous multi-input loading migration remains pending and should be
handled separately across all affected execution engines.
