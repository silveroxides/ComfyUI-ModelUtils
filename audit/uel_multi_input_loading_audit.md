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

## Recovery contract

This is a working-tree checkpoint, not proof that the implementation is
correct. On recovery, verify each item against the named code path and rerun
its focused test before relying on it. Passing tests prove only their
assertions. Do not infer lifecycle, release behavior, alpha mathematics, or
repository-wide coverage from an import, helper name, or checkbox.

## Repository-wide migration progress

- [x] Audit production `get_tensor()` and direct safetensors call sites.
- [ ] Verify the shared bounded UEL stream and atomic writer implementation.
- [ ] Verify generic and CWB merging has no hidden synchronous tensor I/O.
- [ ] Verify resize-with-base and resize use ordered UEL work units.
- [ ] Verify key utilities, metadata, passthrough, and PT conversion output.
- [ ] Verify the repository guard covers every production tensor-I/O path.
- [ ] Verify alpha normalization precedes every LoRA rank operation.
- [ ] Finish atomic-writer migration and remove obsolete imports/helpers.
- [ ] Add repository guards and complete lifecycle/failure tests.
- [ ] Run focused tests, full validation, and representative benchmarks.

## Unverified working-tree changes

The working tree intends `nodes/uel_io.py` to provide ordered streams, logical
multi-input work units, per-yield release, stream closure, and atomic output.
Generic recipes were changed to receive preloaded mappings. CWB, LoRA-to-model,
key utilities, metadata, and PT conversion were changed toward UEL. These are
implementation claims awaiting focused lifecycle verification.

The intended alpha contract is `up * (alpha / rank)` before padding, alignment,
weighting, CWB, DARE/TIES, or model application. Callers were changed to replace
the original work-unit reference and omit alpha outputs. Verify every caller,
including preservation and early-return branches. The standalone normalization
script should remain untracked and unchanged; confirm this before staging.

Evidence recorded so far: focused CWB and alpha tests passed. The first full
run reached 152 passing tests
with two failures caused only by tests reaching an implementation module's
removed `IncrementalSafetensorsWriter` import. Update those tests to import the
UEL writer directly, then rerun the full suite. A bounded search found no
production `get_tensor()` or direct safetensors imports at that point; rerun the
guard after every later edit.

## Remaining execution checklist

1. Model analysis was rechecked after an incorrect intermediate concern. The
   caller near line 910 passes exactly one A/B tensor pair. The LoRA caller near
   line 953 passes exactly two pairs for one down/up logical group, which must
   coexist for pair analysis. It does not accumulate the model inventory. Both
   callers release and mark their bounded unit before advancing. No change is
   required unless a later test contradicts this inspection.
2. Re-run the new failure lifecycle test after explicit caller-side stream
   closure changes in generic merge, CWB, and LoRA-to-model.
3. Verify extraction and resize atomic-writer conversions and remove obsolete
   imports. Dtype conversion was switched to the shared atomic helper.
4. The quantization test fixture now imports the UEL writer directly. The next
   full run reached 157 passing tests and one newly added lifecycle test failure;
   that failure exposed missing explicit consumer-side generator closure and
   code was changed afterward. Rerun the full suite.
5. The new repository guard and atomic failure test are in
   `tests/test_uel_repository_contract.py`; verify they still pass.
6. Run Ruff F checks, compilation, `git diff --check`, full pytest, then a
   temporary synthetic benchmark. Remove only task-created temp directories.
7. Reinspect every `async_stream()` caller for accumulation, release timing,
   exceptional closure, and duplicate key occurrences. Imports and method names
   are not evidence of a correct implementation.

## Latest verification evidence

- Full configured-environment suite: 158 tests passed.
- Focused alpha and UEL contract rerun after the full suite: 7 tests passed.
- Changed Python files compile with the configured ComfyUI interpreter.
- Ruff F checks pass for all changed implementation and test files. Unrelated
  pre-existing F findings remain in untouched modules outside this change.
- `git diff --check` passes.
- The production guard finds no `.get_tensor()` calls or direct safetensors
  imports under `nodes/*.py`.
- No em dash occurs in the changed audit, implementation, or focused test files.

These results are evidence for this exact working-tree revision only. Rerun
them after any edit, merge, formatting pass, or conflict resolution.
## Compaction checkpoint: alpha normalization edge cases

Verified on 2026-08-14 by bounded source inspection:

- `nodes/merger.py` now records `(alpha_key, rank, down_key)` for each alpha-bearing LoRA pair and rejects alpha normalization when either factor is an unsupported low-bit tensor. Its runtime scales the loaded up factor before merge processing.
- The remaining work is to add the same explicit rejection to LoRA multi-merge and CWB LoRA guarded low-bit paths. Scalar low-bit alpha alone must remain loadable; only unsupported low-bit factor tensors make mathematical normalization impossible.
- `write_preserved_tensor` still needs failure-path lifecycle verification so every yielded tensor is marked processed and its stream is closed if writer submission raises.
- After those edits, rerun focused low-bit/alpha tests, the complete test suite in the configured ComfyUI virtual environment, changed-file compile and Ruff checks, `git diff --check`, and the production guard against direct `get_tensor()` or direct safetensors imports.
- Do not stage this work unless explicitly requested. Keep all unrelated untracked files listed by `git status` untracked.

## Fresh validation evidence after recovery

The following is command evidence from 2026-08-14, not a substitute for inspecting the current tree after later edits:

- The configured ComfyUI Python completed the full suite with `160 passed in 5.52s`.
- Focused regressions for CWB and LoRA multi-merge rejection of alpha-bearing unsupported low-bit factors both passed.
- Changed implementation and test files passed Ruff `F` checks through the installed standalone Ruff executable.
- `python -m compileall -q nodes tests` completed successfully.
- `git diff --check` completed with no diff errors; Git emitted only line-ending conversion warnings.
- A scoped production search found no direct `.get_tensor(` calls and no direct safetensors imports in `nodes/*.py`.
- A scoped search found no em dash in the UEL audit, shared helpers, affected merger files, or UEL contract test.

Implementation consequences verified by the focused source inspection and tests above:

- Alpha-bearing LoRA factors are normalized before merge and before padding when their factor tensors are supported floating storage.
- A low-bit scalar alpha is accepted when both factors are supported; unsupported low-bit down/up factors with alpha are rejected instead of being copied into an output that falsely claims normalization.
- Preserved tensors yielded by the one-key UEL stream are marked processed in `finally`, including writer-failure paths, and the stream is closed.
