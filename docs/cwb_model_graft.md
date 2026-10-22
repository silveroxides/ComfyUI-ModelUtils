# CWB Model Graft

`CWB Model Graft` (`CWBModelGraft`) is a standalone training-free diffusion model grafting node that transfers selected blocks, operator families, projection sub-slices, or explicitly mapped layers from a donor diffusion model (`model_b`) into a base diffusion model (`model_a`) using Consensus-Weighted Blending (CWB).

`model_a` defines the output architecture, tensor key set, output shapes, and safetensors metadata. Tensors that are protected, excluded by block or sub-part rules, or absent from `model_b` are preserved from `model_a` without modification.

---

## End-to-End Execution Pipeline

For every logical tensor key in `model_a`, `CWB Model Graft` executes the following sequence:

1. **Quantization & Low-Bit Inspection**
   - Both `model_a` and `model_b` are opened with `MemoryEfficientSafeOpen` and inspected via `DiffusionQuantization` and `diffusion_key_map`.
   - Full-model ComfyUI/comfy-kitchen quantized files or files containing 3 or more bare low-bit (`INT8`, `UINT8`, `FP8`, `FP4`) tensors are rejected before opening an output writer.
   - Any layer with 1–2 isolated low-bit tensors in either `model_a` or the resolved `model_b` source keys is preserved byte-for-byte from `model_a` (`force_raw=True`).

2. **Discard & Exclude / Include Pattern Filtering**
   - `discard_patterns`: If a `model_a` key (or its physical key) matches `discard_patterns`, the tensor is omitted from the output file.
   - `exclude_patterns` with `include_mode = False` (default): Matching keys are preserved unchanged from `model_a`.
   - `exclude_patterns` with `include_mode = True`: Acts as a whitelist. Non-matching keys are preserved unchanged from `model_a`. Explicitly matched whitelist keys are eligible for grafting even if they lie outside standard block numbering (such as `text_embedder` or modulation layers).

3. **Donor Key Resolution & Automatic QKV Packing / Splitting**
   - **Custom Key Map (`custom_key_map`)**: Checked first. If `base_key` has an explicit mapping `donor_key -> base_key` and `donor_key` exists in `model_b`, `donor_key` is used as the donor source.
   - **Direct Match**: If `base_key` exists directly in `model_b`, it is paired 1-to-1.
   - **Split-to-Fused QKV (`fuse_qkv`)**: If `model_a` has a fused `<prefix>.proj.qkv.weight` tensor and `model_b` instead stores split `<prefix>.proj.q.weight`, `<prefix>.proj.k.weight`, and `<prefix>.proj.v.weight`, all three donor tensors are streamed in the same work unit and concatenated along `dim=0` (`torch.cat([q, k, v], dim=0)`) before sub-part selection and CWB blending.
   - **Fused-to-Split QKV (`split_qkv`)**: If `model_a` has split `<prefix>.proj.{q,k,v}.weight` and `model_b` stores fused `<prefix>.proj.qkv.weight`, the fused donor tensor is streamed and split into 3 equal chunks along `dim=0` (`chunk(3, dim=0)`), selecting index `0` for `q`, `1` for `k`, or `2` for `v`.
   - **Unmatched Keys**: If no donor source exists for `base_key`, `model_a`'s tensor is preserved unchanged.

4. **Block Index Detection & Early-Block Protection**
   - Main transformer block indices are parsed from keys matching `(^|\.)(layers|blocks|double_blocks|single_blocks|transformer_blocks)\.(\d+)\.`.
   - Auxiliary sub-networks (`text_embedder.*`, `token_refiner.*`) do not count toward main transformer block numbering and are preserved from `model_a` unless targeted via `custom_key_map` or `include_mode=True` with `exclude_patterns`.
   - `max_block_index` is computed as the highest main block index in `model_a`.
   - If a key's `block_idx < protect_early_blocks`, the tensor is preserved 100% from `model_a`.

5. **Operator Classification & Block Scheduling**
   - Each main-block key is classified by its token path:
     - **Attention (`attention`)**: `.qkv.`, `.qkv_proj.`, `.proj.q.`, `.proj.k.`, `.proj.v.`, `.proj.o.`, `.attn.`, `.self_attn.`, `.q_norm.`, `.k_norm.`, `.to_q.`, `.to_k.`, `.to_v.`, `.to_out.`.
     - **Feed-Forward (`ffn`)**: `.ffn_up_gate.`, `.ffn_up.`, `.ffn_gate.`, `.ffn_down.`, `.mlp.`, `.ff.`, `.fc1.`, `.fc2.`, `.w1.`, `.w2.`, `.w3.`.
   - Keys inside main blocks that are neither attention nor feed-forward (such as block-level `norm1` and `norm2`) are preserved from `model_a` unless explicitly targeted via `custom_key_map` or `include_mode=True`.
   - For classified keys with `block_idx >= protect_early_blocks`, the corresponding rule (`attention_block_rule` or `ffn_block_rule`) determines whether the block is grafted:
     - `all_after_protected`: Every block with `block_idx >= protect_early_blocks`.
     - `interleaved_even`: Even-numbered blocks (`block_idx % 2 == 0`) with `block_idx >= protect_early_blocks`.
     - `interleaved_odd`: Odd-numbered blocks (`block_idx % 2 == 1`) with `block_idx >= protect_early_blocks`.
     - `back_half_only`: Blocks in the second half of the network (`block_idx >= max(protect_early_blocks, (max_block_index + 1) // 2)`).
     - `none`: No blocks of this operator family are grafted.

6. **Sub-Part Slicing (`attention_subpart` and `ffn_subpart`)**
   - **Attention Sub-Parts (`attention_subpart`)**:
     - `all`: All attention tensors in selected blocks (`qkv`, `q`, `k`, `v`, `o`, `q_norm`, `k_norm`) are grafted.
     - `value_and_output`:
       - Split query/key projections and norms (`.proj.q.`, `.proj.k.`, `.to_q.`, `.to_k.`, `.q_norm.`, `.k_norm.`) are preserved 100% from `model_a`.
       - Fused QKV weights (`.qkv.`, `.qkv_proj.`) whose first dimension is divisible by 3 are split into `q_a, k_a, v_a = base.chunk(3, dim=0)` and `q_b, k_b, v_b = donor.chunk(3, dim=0)`. `q_a` and `k_a` are kept byte-exact from `model_a`, `v_a` and `v_b` are CWB-blended into `v_merged`, and the tensor is reassembled as `torch.cat([q_a, k_a, v_merged], dim=0)`.
       - Split value projections (`.proj.v.`, `.to_v.`) and output projections (`.proj.o.`, `.out_proj.`, `.to_out.`) are CWB-blended in full.
     - `output_only`: Only attention output projections (`.proj.o.`, `.out_proj.`, `.to_out.`) are CWB-blended; all Q/K/V and norm tensors are preserved 100% from `model_a`.
   - **Feed-Forward Sub-Parts (`ffn_subpart`)**:
     - `all`: All feed-forward tensors in selected blocks (`ffn_up_gate`, `ffn_up`, `ffn_gate`, `ffn_down`, `fc1`, `fc2`, `w1`, `w2`, `w3`) are grafted.
     - `up_only`:
       - Down and gate projections (`.ffn_down.`, `.fc2.`, `.w2.`, `.ff.net.2.`, `.ffn_gate.`, `.w3.`) are preserved 100% from `model_a`.
       - Fused up+gate weights (`.ffn_up_gate.`) whose first dimension is divisible by 2 are split into `up_a, gate_a = base.chunk(2, dim=0)` and `up_b, gate_b = donor.chunk(2, dim=0)`. Only `up_a` and `up_b` are CWB-blended into `up_merged`, `gate_a` is preserved 100% from `model_a`, and the tensor is reassembled as `torch.cat([up_merged, gate_a], dim=0)`.
       - Split up projections (`.ffn_up.`, `.fc1.`, `.w1.`) are CWB-blended in full.
     - `down_only`: Only down projections (`.ffn_down.`, `.fc2.`, `.w2.`, `.ff.net.2.`) are CWB-blended; all up/gate tensors are preserved 100% from `model_a`.

7. **Cross-Shape Alignment, Row Similarity Matching & Norm Locking**
   - **Trailing-Dimension Alignment**: If `donor_tensor` and `base_tensor` differ in shape:
     - Same element count with different rank: `donor_tensor` is reshaped to `base_tensor.shape`.
     - Different rank and different element count: `donor_tensor` is linearly interpolated in FP32 to `base_tensor.numel()` and reshaped to `base_tensor.shape`.
     - Same rank (`ndim >= 2`) with different trailing dimensions (`donor_tensor.shape[1:] != base_tensor.shape[1:]`): Each donor row is linearly interpolated in FP32 along its flattened trailing features to match `prod(base_tensor.shape[1:])` and reshaped to `(donor_rows, *base_tensor.shape[1:])`.
   - **Row Matching & CWB Blending**: `merge_cwb_tensors` runs in FP32 with `allow_similarity_alignment=True` and `reference_index=0` (`model_a` as reference). When the active CWB settings use `alignment_method="similarity"` (or when row counts differ), donor rows are greedily paired to base rows by cosine similarity (subject to `alignment_threshold` and `position_weight`). Unmatched base rows remain anchored to `model_a`.
   - **Base Magnitude Locking**: Whenever `donor_tensor` required shape alignment or `rescale_norm` is enabled in the active CWB settings, the blended tensor's overall Frobenius norm is locked to `model_a`'s tensor norm (`merged * (||base|| / ||merged||)`), preventing activation scale drift across downstream blocks.

8. **Atomic Streaming Output & Metadata**
   - Tensors are streamed one work unit at a time through UEL (`batch_size=1`, `prefetch_batches=1`), written incrementally via `atomic_uel_writer`, and released before advancing to the next layer.
   - If a CUDA out-of-memory error occurs on a layer, the failed GPU working set is released and that layer retries on CPU.
   - Output metadata preserves `model_a`'s metadata and records a JSON summary under `cwb.graft` containing `base_model`, `donor_model`, `attention_preset`, `ffn_preset`, `protect_early_blocks`, `attention_block_rule`, `ffn_block_rule`, `attention_subpart`, and `ffn_subpart`.

---

## Complete Input Reference

- **`execution_mode`** (`MERGE` | `DOCUMENTATION ONLY`, default `MERGE`):
  - `MERGE`: Executes the streaming graft and writes the output `.safetensors` file under `models/diffusion_models/`.
  - `DOCUMENTATION ONLY`: Returns this documentation string without opening any model files.
- **`model_a`** (Combo, `diffusion_models`):
  - Base diffusion model that anchors architecture, output tensor keys, shapes, metadata, and untouched/protected layers.
- **`model_b`** (Combo, `diffusion_models`):
  - Donor diffusion model from which selected blocks, sub-projections, or mapped layers are grafted into `model_a`.
- **`attention_preset`** (Combo, default `robust_medn`):
  - Dense CWB preset applied to grafted attention tensors (`balanced_mean`, `robust_medn`, `selective_mean`, `varied_mean_rn_softcb`, `diverse_medn_rn_dsc_softcb`, `strongdiv_medn_rn_dsc_softcb`). Overridden completely if `attention_config` is connected.
- **`attention_config`** (`CWB_CONFIG`, optional):
  - Optional custom CWB configuration from `CWB Custom Configuration` for attention layers. Enabling `alignment_method = "similarity"` on this config activates greedy cosine row matching across attention rows.
- **`ffn_preset`** (Combo, default `balanced_mean`):
  - Dense CWB preset applied to grafted feed-forward (MLP) tensors and custom-mapped non-attention tensors. Overridden completely if `ffn_config` is connected.
- **`ffn_config`** (`CWB_CONFIG`, optional):
  - Optional custom CWB configuration from `CWB Custom Configuration` for feed-forward (MLP) layers.
- **`protect_early_blocks`** (Int, default `2`, range `0..256`):
  - Number of initial main transformer blocks (`0 .. protect_early_blocks - 1`) kept 100% identical to `model_a` so global image composition and layout remain stable.
- **`attention_block_rule`** (`interleaved_even` | `interleaved_odd` | `back_half_only` | `all_after_protected` | `none`, default `interleaved_even`):
  - Block selection schedule for attention layers at index `>= protect_early_blocks`.
- **`ffn_block_rule`** (`all_after_protected` | `interleaved_even` | `interleaved_odd` | `back_half_only` | `none`, default `all_after_protected`):
  - Block selection schedule for feed-forward (MLP) layers at index `>= protect_early_blocks`.
- **`attention_subpart`** (`all` | `value_and_output` | `output_only`, default `all`):
  - Restricts attention grafting to all attention weights, only the Value (`V`) slice/projection and Output (`O`) projection, or only the Output (`O`) projection.
- **`ffn_subpart`** (`all` | `up_only` | `down_only`, default `all`):
  - Restricts feed-forward grafting to all FFN weights, only the Up projection (including the `up` half of packed `ffn_up_gate`), or only the Down projection.
- **`custom_key_map`** (String, multiline, default `""`):
  - Optional explicit key mappings, one per line, formatted as `donor_key -> base_key` or `donor_key = base_key` (lines starting with `#` are ignored). Mapped keys bypass block-number filtering so arbitrary donor tensors can be grafted into specific base tensors.
- **`output_filename`** (String, default `cwb_grafted_model`):
  - Output filename relative to `models/diffusion_models/` (`.safetensors` extension added automatically).
- **`save_dtype`** (`fp32` | `fp16` | `bf16`, default `bf16`):
  - Target floating-point precision when `override_dtype` is enabled or when decoding quantized tensors.
- **`process_device`** (`cuda` | `cpu`, default `cuda`):
  - Execution device for per-tensor FP32 CWB math; automatically falls back to `cpu` per tensor on CUDA OOM.
- **`exclude_patterns`** (String, multiline, default `""`):
  - Regex (or glob when `glob_patterns=True`) patterns. Preserves matching tensors from `model_a` when `include_mode=False`, or whitelists matching tensors for grafting when `include_mode=True`.
- **`discard_patterns`** (String, multiline, default `""`):
  - Regex (or glob) patterns of tensors to omit completely from the saved model.
- **`glob_patterns`** (Boolean, default `False`):
  - Interpret `exclude_patterns` and `discard_patterns` as shell-style globs (`*`, `?`) instead of regular expressions.
- **`lazy_load`** (Boolean, default `True`):
  - Enable low-memory UEL streaming so only the current work unit resides in memory.
- **`force_clear_cache`** (Boolean, default `False`):
  - Run `gc.collect()` and `torch.cuda.empty_cache()` before each tensor work unit.
- **`override_dtype`** (Boolean, default `False`):
  - Force grafted floating tensors to `save_dtype`; when `False`, each grafted tensor preserves `model_a`'s floating dtype.
- **`include_mode`** (Boolean, default `False`):
  - Invert `exclude_patterns` into a whitelist so only matching tensors are eligible for grafting.

---

## Outputs

1. **`output_filename`** (`*` / `io.AnyType`): Category-relative filename (`<output_filename>.safetensors`) ready to connect directly to diffusion model loader nodes.
2. **`documentation`** (`STRING`): Complete markdown documentation for `CWB Model Graft`.
3. **`cwb_report`** (`STRING`): Diagnostic report detailing active attention/FFN presets, candidate/accepted alignment matches, consensus similarity statistics, CPU fallbacks, total grafted tensor count, and total preserved base tensor count.
