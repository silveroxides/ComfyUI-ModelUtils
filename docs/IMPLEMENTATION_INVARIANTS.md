# Implementation Invariants

This file is the authoritative engineering contract for model and LoRA tensor I/O and merge behavior in ComfyUI-ModelUtils. New implementations and changes to existing implementations MUST follow these rules. Historical code, audit notes, and examples do not override this contract.

## UnifiedEfficientLoader I/O

1. Production node code MUST use `unifiedefficientloader` for safetensors tensor I/O. Direct safetensors imports and synchronous `get_tensor()` calls are prohibited.
2. Header-only operations MUST use UEL header, key, shape, dtype, and metadata APIs without materializing tensors.
3. Tensor processing MUST use `async_stream` with bounded buffering. The repository default is `batch_size=1` and `prefetch_batches=1` unless a measured benchmark and a documented memory bound justify another value.
4. Multi-input operations MUST maintain an async cursor for each input and construct logical work units. A work unit contains only the tensors required to process one output tensor, LoRA factor pair, or other indivisible operation.
5. Pinned-memory transfer MUST be enabled when the selected processing device benefits from it. CPU-only processing MUST avoid unnecessary pinned copies.
6. Implementations MUST NOT accumulate a complete model, LoRA, or unbounded output batch in RAM. The next work unit may be prefetched while the current unit is processed, but the prefetch bound MUST remain explicit.
7. Every yielded source tensor occurrence MUST be marked processed after its logical work unit completes or aborts. All strong references to source, normalized, padded, device, and output intermediates MUST be removed before advancing to the next work unit.
8. Every async stream and handler MUST close in `finally` or through an equivalent deterministic context manager. Failure paths are subject to the same release rules as successful paths.
9. Saving MUST use UEL incremental writing. Outputs MUST be submitted as they are completed rather than retained in an in-memory state dictionary.
10. File replacement MUST be atomic: write to a temporary sibling file, finalize successfully, then replace the destination. A failure MUST preserve any existing destination and remove the incomplete temporary file.
11. Unsupported low-bit tensors MUST use the shared quantization-preservation path when byte-exact copying is valid. Code MUST NOT silently cast, decode, merge, or relabel unsupported low-bit storage.

The shared implementations in `nodes/uel_io.py`, `nodes/extraction_stream.py`, and `nodes/quantization_guard.py` are the required starting points. New node-local stream or writer abstractions require a demonstrated capability that the shared implementation cannot provide.

## Persistent Artifact Paths and Outputs

1. Persistent model artifacts MUST use `nodes/artifact_paths.py` to construct
   their write path from `folder_paths.models_dir` and an explicit canonical
   category: `checkpoints`, `diffusion_models`, `text_encoders`, `loras`, or
   `embeddings`.
2. Registered search paths, additional paths, and legacy aliases such as
   `unet` and `clip` MUST NOT become write roots based on their list position.
3. Nested output names MUST be preserved, exactly one `.safetensors` extension
   MUST be applied, and path traversal outside the category root MUST fail.
4. Public filename outputs MUST return the category-relative name with forward
   slashes. They MUST NOT return an absolute filesystem path.
5. Loader-compatible filename outputs MUST use `io.AnyType`. Reports and
   documentation MUST remain separate string outputs.
6. Persistent model artifacts MUST NOT be written beneath ComfyUI's general
   output directory.

## LoRA Alpha Normalization for Merge Operations

1. Every operation that combines a LoRA with another LoRA or applies a LoRA to a model MUST account for per-layer alpha before weighting, padding, concatenation, consensus processing, DARE/TIES processing, or delta application.
2. For a valid factor pair with rank `r`, normalization is performed mathematically as:

   `normalized_up = up * (alpha / r)`

   `normalized_down = down`

   Scaling one factor preserves the represented delta while avoiding unnecessary work and memory.
3. A layer without alpha has an implicit scale of `1.0`. Inputs with and without alpha may be merged only after alpha-bearing pairs have been normalized to this common representation.
4. Normalized merge outputs MUST NOT contain alpha tensors. Alpha MUST NOT be invented for inputs or output formats that do not provide it.
5. Alpha is a scalar control value. Its dtype MUST NOT determine factor computation dtype, merged tensor dtype, or save dtype.
6. Alpha association MUST come from the repository's LoRA pair parser, not from an assumed key spelling. PEFT, Diffusers, ComfyUI, and supported mixed naming forms are governed by the parsed logical pair.
7. Normalization MUST occur before rank padding or other shape alignment. Padding an unnormalized pair and applying its original alpha against the padded rank changes the represented delta.
8. The unscaled factor reference MUST be replaced immediately when normalization creates a new tensor. Both source factors, the alpha scalar, normalized factors, padded factors, and device copies MUST be released when the logical layer completes.
9. A low-bit alpha scalar is valid when both factors use supported floating storage. If an alpha-bearing down or up factor uses unsupported low-bit storage, the operation MUST fail with an actionable error because byte-exact preservation cannot also produce an alpha-free normalized result.
10. Format metadata may describe that normalization occurred, but metadata MUST NOT be used as a substitute for applying the mathematical transformation.

### CWB Mixed-Rank LoRA Alignment

1. Genuine LoRA rank validity MUST come from each factor pair's original shape
   or explicit source metadata. Tensor values MUST NOT be inspected to decide
   whether a component is padding because a genuine component may be zero.
2. Structural rank padding MUST NOT participate in CWB alignment, consensus,
   similarity weighting, fallback behavior, or norm rescaling.
3. Similarity alignment between rank `R` and rank `r` genuine pairs MUST compare
   the full rectangular `R x r` component space. A smaller source component may
   match any component in the maximum-rank reference.
4. Index alignment MUST pair only shared genuine indices. Higher reference
   indices remain singleton groups unless another genuine source contributes.
5. An explicit zero inserted by `mismatch_mode=zeros` is a semantic contributor
   and MUST remain distinct from structural rank absence.
6. Down rows and corresponding up columns MUST share validity, alignment, sign
   correction, contributor membership, and output position.
7. Output rank MUST remain the largest genuine input rank. Equal maximum-rank
   ties retain source order.

The shared implementation in `nodes/lora_alpha.py` is mandatory for merge paths. A format-specific implementation may only replace it when the represented delta differs mathematically and that difference is documented and tested.

## Required Verification

Any implementation that reads, writes, merges, resizes, extracts, or applies model or LoRA tensors MUST include tests appropriate to its scope. Merge changes involving LoRA factors MUST cover:

- alpha present and absent in the same operation;
- non-neutral alpha where `alpha / rank` is not `1.0`;
- normalization before padding or concatenation;
- alpha-free output;
- mixed factor ranks and dtypes;
- full rectangular CWB matching without structural-padding contributors;
- genuine zero-valued components versus explicit mismatch-zero contributors;
- low-bit scalar alpha acceptance and unsupported low-bit factor rejection;
- stream closure, `mark_processed`, and reference release on success and failure;
- atomic destination preservation on writer failure.

Repository guards MUST continue to reject production `get_tensor()` calls and direct safetensors imports. The complete test suite, changed-file lint, compilation, and `git diff --check` are required before commit.
