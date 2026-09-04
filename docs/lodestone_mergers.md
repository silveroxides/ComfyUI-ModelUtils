# Lodestone LoRA Mergers

The dedicated Lodestone nodes merge full logical LoRA deltas rather than
blending down and up factors independently. Stored alpha is normalized before
each factor pair is expanded. Results are saved as canonical full direct-difference
tensors and contain no factor or alpha tensors.

The 2-, 3-, and Multi-Merge nodes treat every selected LoRA as one equal-prior
slot, matching the supplied Lodestone calculation.

## Calculation Modes

- `sum`: sum the source deltas.
- `mean`: divide the summed delta by the number of active contributors.
- `slotnorm`: scale each source delta to the median source Frobenius norm, then
  average.
- `normmatch`: retain the summed direction and set its Frobenius norm to the
  median source norm.
- `slotnorm-normmatch`: equalize source norms, sum, then set the final norm to
  the original median source norm.

A zero target or merged norm remains zero.

## Memory Behavior

UEL streams one logical layer at a time and writes each completed full-delta
tensor incrementally through the atomic writer. Dense tensors used for the
merge and Frobenius norms exist only within the active layer. A CUDA
out-of-memory error releases CUDA state and retries that logical layer on CPU.

## Fixed-Input Controls

LoRA 1 anchors output layers and filtering behavior. `skip` preserves LoRA 1
when another input lacks a required layer, `zeros` supplies an explicit zero
slot, and `error` aborts. Source factors may have different ranks when their
expanded dense shapes agree. Exclude and discard patterns apply to complete
logical layers.
