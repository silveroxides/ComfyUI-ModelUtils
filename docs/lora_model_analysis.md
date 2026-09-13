# LoRA-on-model analysis

**Analyze LoRA Effect on Diffusion Model** takes a LoRA as `model_a` and a base
diffusion model as `model_b`. It applies the LoRA transiently and compares the
original base weights against the patched weights. The measured update is
`patched − original`; no merged checkpoint, delta file, or inference is produced.

The comparison and CWB reports use original base as A and patched base as B.
The CSV outputs contain per-tensor difference metrics and row cosine summaries.
Unchanged floating base tensors remain in the analysis, so global metrics reflect
the impact on the selected base model, not just the layers touched by the LoRA.
Application and statistics use FP32, matching the existing analysis arithmetic;
this does not simulate rounding to a separately saved model dtype.

LoRA strength defaults to 1. Alpha/rank normalization, DoRA, LoCon, and reshape
handling use the existing application path and ComfyUI adapter. Direct deltas,
including bias deltas, are applied; explicit `set_weight` patches replace weights
regardless of strength, matching the existing application behavior.

Exclude Patterns and Include Mode filter **base tensor keys** before loading.
Unmapped LoRA groups and unused source tensors are listed separately. Unsupported
low-bit work units and non-floating base tensors are reported and omitted from
numeric metrics; they are not decoded or represented as successfully analyzed.
Application errors fail the operation rather than report a failed patch as zero
impact. A CUDA out-of-memory error retries the complete affected unit on CPU.

Loading uses the shared UEL work-unit streamer: one base tensor and its required
adapter tensors, batch size 1 and prefetch depth 1. Sources are marked processed
and streams closed on success or failure. Only scalar statistics and bounded top
differences survive each unit. No whole model or LoRA tensor dictionary is built.
