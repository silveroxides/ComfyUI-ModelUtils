# Delta CWB LoRA Mergers

Delta CWB expands each selected LoRA into full per-layer weight changes and
applies CWB to corresponding dense rows. It preserves model coordinates; it
does not align, reorder, or compress rank components.

LoRA 1 supplies metadata and defines the output layer set. The preset controls
how agreement, disagreement, and row magnitude affect the saved changes. A
connected CWB configuration completely replaces the preset. Inputs have equal
prior weight.

Missing-layer handling can preserve LoRA 1, include explicit zero contributors,
or abort. Excluded layers preserve LoRA 1 and discarded layers are omitted.
Outputs contain only canonical `.diff` tensors with alpha already normalized.

Processing streams one logical layer from each input at a time and writes each
completed layer incrementally. CUDA out-of-memory retries only the failed layer
on CPU. The destination is replaced only after the complete output is written.
