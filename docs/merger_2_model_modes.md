# 2-Model Merger Documentation

---

## Weight-Sum
> `A * (1 - α) + B * α`. A simple linear interpolation between two models.

**Models Used:** A, B
**Parameters:**
- **Alpha:** Interpolation weight. `0.0` is 100% Model A, `1.0` is 100% Model B.

---

## Comparative-Interpolation
> Interpolates between A and B based on the relative differences in their tensor values. Creates a unique blend by deciding how much of each model to use for each individual weight.

**Models Used:** A, B
**Parameters:**
- **Alpha:** Controls the interpolation curve's strength and shape.
- **Beta:** Switches the interpolation style. `0.0` focuses on similarity, `1.0` focuses on difference.
- **Gamma:** Mixes between a randomized (binomial) interpolation at `0.0` and a smooth (linear) one at `1.0`.

---

## Power-Up (DARE)
> Adds the unique capabilities of Model B to Model A using the Drop and Rescale (DARE) technique. This implementation handles shape mismatches between models by padding and uses a randomized dropout mask.

**Models Used:** A, B
**Parameters:**
- **Alpha:** The dropout rate ($p$). This is the proportion of delta parameters from Model B that are randomly set to zero.
- **Beta:** A final multiplier for the rescaled difference before it's added to Model A.
- **Rescaling Logic:** Remaining weights are automatically rescaled by $1/(1-p)$ as per the DARE paper to approximate the original embeddings.

---

## Power-Up (DARE+TIES)
> Combines DARE (Drop and Rescale) with TIES (Trim, Elect Sign) — the current best-practice algorithm for merging fine-tuned models, particularly LoRA-tuned ones. DARE sparsifies the task vector via random dropout and rescaling; TIES then removes residual low-magnitude noise and enforces sign consistency. The result is a cleaner, more targeted capability transfer from B to A with less interference.

**Models Used:** A, B
**Parameters:**
- **Alpha:** DARE drop rate. Fraction of delta parameters randomly zeroed. Higher values produce a sparser, more targeted delta. Typical range: `0.5`–`0.9`.
- **Beta:** TIES trim quantile. Fraction of the smallest-magnitude delta parameters zeroed after DARE, removing residual noise. `0.0` disables trimming. Typical range: `0.1`–`0.3`.
- **Gamma:** Lambda scale. Final multiplier applied to the merged delta before adding to Model A. Equivalent to the task arithmetic scaling coefficient. Typical range: `0.5`–`1.5`.
- **Seed:** Random seed for the DARE dropout mask. Controls reproducibility.

**Algorithm:**
1. Compute task vector `δ = B − A`
2. **DARE:** apply random binary mask with keep probability `1 − alpha`; rescale survivors by `1/(1−alpha)`
3. **TIES trim:** zero out parameters below the `beta`-quantile magnitude threshold
4. **TIES elect:** determine dominant sign per position; zero out disagreeing parameters
5. Return `A + gamma × δ_filtered`

**Guidance:**
- Start with `alpha=0.9, beta=0.2, gamma=1.0` for LoRA-derived models
- Lower `alpha` (e.g. `0.5`) for full fine-tunes where more delta mass should be retained
- Increase `gamma` beyond `1.0` to amplify B's influence; decrease below `1.0` to soften it

---

## Enhanced Man Interp
> Sophisticated interpolation between values from A and B depending on their difference relative to other values, with manual threshold control.

**Models Used:** A, B
**Parameters:**
- **Alpha:** Interpolation strength.
- **Beta:** Lower mean threshold for filtering differences.
- **Gamma:** Upper mean threshold for filtering differences.
- **Delta:** Smoothness factor (mix between randomized mask and powered differences).

---

## Enhanced Auto Interp
> Automated version of the enhanced interpolation mode that dynamically calculates thresholds based on mean differences.

**Models Used:** A, B
**Parameters:**
- **Alpha:** Interpolation strength.
- **Beta:** Threshold adjustment factor.
- **Gamma:** Smoothness factor.

---

## Weight-Sum Cutoff
> A linear interpolation mode that only applies the merge to weights whose differences fall within a specific threshold range.

**Models Used:** A, B
**Parameters:**
- **Alpha:** Interpolation weight (multiplier for the difference).
- **Beta:** Upper threshold for the difference cutoff.
- **Gamma:** Lower threshold for the difference cutoff.

---

## SVD LoRA Extraction
> Extracts the difference between Model B and Model A and saves it as a new LoRA file. **This mode outputs a LoRA, not a full model.** The output file will be saved to your `loras` directory.

**Models Used:** A (Tuned Model), B (Base Model)
**Parameters:**
- **Alpha:** The Rank (dimension) for the LoRA's standard layers.
- **Beta:** The Rank (dimension) specifically for 3x3 convolution layers.
- **Gamma:** The clamp quantile for weight values. `0.99` is a typical value to prevent outlier weights from dominating.

---

## Layer Mismatch Handling

When merging models with different layer structures (e.g., fine-tuned models with added/removed layers, or LoRAs with partial layer coverage), the `mismatch_mode` parameter controls behavior:

| Mode | Behavior |
|------|----------|
| `skip` | Missing layers in B use A's values **(default)** |
| `zeros` | Missing layers in B are treated as zeros |
| `error` | Fail if any layer is missing |

**Common scenarios:**
- Merging a fine-tuned model with added/removed layers
- Combining LoRA-extracted differences with base models
- Cross-architecture experiments

**Note:** Extra layers in Model B that don't exist in Model A are currently ignored. The output will always have the same layer structure as Model A.

---

## Layer Filtering (Regex Patterns)

Use regex patterns to control which layers are merged:

### `exclude_patterns`
Layers matching any pattern will **keep Model A's values only** (no merge).

### `discard_patterns`
Layers matching any pattern will be **removed entirely** from the output.

**Pattern format:**
- Whitespace-separated regex patterns
- Patterns use **substring matching** (not full match)
- Example: `text_model lora` matches any key containing "text_model" OR "lora"
- Example: `layer\.[0-5]\.` matches layers 0-5 using regex syntax

**Pattern Examples:**
| Pattern | Matches |
|---------|---------|
| `text_model` | All text encoder layers |
| `\.norm` | All normalization layers |
| `attn\.(q\|k\|v)` | Query, key, value attention weights |
| `block\.[0-9]\.` | Blocks 0-9 |
