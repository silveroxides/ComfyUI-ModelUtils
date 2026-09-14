# Two-Model Similarity Analysis

These nodes compare two safetensors models without merging them or writing an
output file. Separate nodes are provided for checkpoints, diffusion models,
text encoders, LoRAs, and embeddings.

`exclude_patterns` accepts one regex per line, or glob patterns when
`glob_patterns` is enabled. By default, matches are excluded from metrics and
topology counts. Enable `include_mode` to analyze only matches instead; an empty
include filter selects no tensors. The same field and matching syntax are used
in both modes.

## Outputs

`comparison_report` contains conventional numerical differences and topology
findings. Both reports use Markdown headings and tables. `cwb_report` separately contains diagnostics derived from the
similarity, alignment, and consensus stages of Consensus-Weighted Blending.
`documentation` contains this reference.

Two CSV-text outputs follow the existing three outputs, preserving their socket
positions: `layerwise_metrics_csv` and `layerwise_cwb_csv`. Connect them to a
text-saving node and use a `.csv` filename; analysis itself writes no files.
The CSV rows follow the same key order and displayed numeric precision as the
layerwise Markdown tables. Dtypes A/B, cosine mean/min/max, and A/B affinities
have separate columns. Undefined values remain `N/A`; an empty analysis produces
headers without data rows. Documentation-only mode returns empty CSV strings.

The reports begin with a global summary, followed by inferred blockwise values
and one entry for every comparable tensor. Block names are approximated from
the tensor-key hierarchy through its first numeric path component; they are not
an architecture-specific interpretation.

## Standard metrics

- MAE, MSE, and RMSE describe absolute error at the original weight scale.
- Maximum absolute difference identifies the largest individual change.
- Relative L2 is `||A-B|| / ((||A||+||B||)/2)` and is scale-aware.
- Cosine similarity compares direction; Pearson correlation compares centered
  variation.
- Exact equality, sign agreement, norm ratio, and finite-value coverage expose
  differences hidden by averages.

The global section reports both parameter-weighted metrics and equal-layer
averages. No arbitrary combined similarity score is assigned. Undefined
zero-norm or constant-tensor statistics are shown as `N/A`.

Only shared, same-shape floating tensors enter numerical aggregates. Missing
keys and shape mismatches are listed separately and are never padded, cropped,
or replaced with zeros. Non-floating tensors receive an exact-equality check.
Quantized models carrying ComfyUI quantization metadata are rejected because
encoded storage values do not represent directly comparable model weights.

## CWB diagnostics

The CWB report provides pairwise row-cosine statistics and each input's affinity
to both mean and median consensus. It stops before CWB's merge-weighting stage:
no thresholds, contribution weights, diversity weighting, vector
reconstitution, norm rescaling, or output writer are used.

Checkpoint, diffusion-model, and text-encoder tensors always retain their fixed
coordinate alignment. Embedding and LoRA nodes expose one optional
`cwb_similarity_alignment` control. When enabled, it performs
threshold-independent greedy one-to-one matching derived from CWB and reports
coverage, matched-score statistics, and improvement over index alignment. This
operation is quadratic and may be slow for large embeddings.

LoRA down/up tensors are treated as one logical layer. Rank-component matching
uses the product of down and up cosine similarity, including CWB's paired-sign
equivalence for comparison. No source tensor is modified.

## Memory and failure handling

Inputs always use Unified Efficient Loader's low-memory streaming mode. Matching
A/B tensors are loaded concurrently through bounded `async_stream()` pipelines
with `batch_size=1` and `prefetch_batches=1`, but no subsequent logical layer is
requested early. A complete LoRA down/up pair is loaded once and used for both
reports. CUDA inputs use pinned transfer while retaining the current CPU source
tensors for fallback.

Calculations use FP32. When CUDA processing runs out of memory, CUDA memory is
cleared and the current logical layer is retried once on CPU. After success or
failure, all tensor references for that layer are released and every loaded key
is marked processed exactly once before advancing. Both reports list every CPU
fallback. A failed CPU retry or any non-OOM error stops analysis with the layer
name in the error.

`top_weight_differences` bounds the scalar-coordinate list; every comparable
tensor is still present in the layerwise report. `DOCUMENTATION ONLY` loads no
model data.
