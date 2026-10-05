# 2-Model Merger Documentation

---

### Core Convention Across All 2-Model Modes
- **Model A is ALWAYS the base / anchor model.**
- **Model B is ALWAYS the donor / incoming model being merged into Model A.**
- When interpolating or blending between weights:
  $$\text{Output} = A \cdot (1 - \text{mask}) + B \cdot \text{mask}$$
- A blend factor / weight of `0.0` yields **100% Model A** (pure Model A).
- A blend factor / weight of `1.0` yields **100% Model B** (pure Model B).
- Intermediate values control how much of Model B replaces or alters Model A.

---

## Weight-Sum
> $\text{Output} = A \cdot (1 - \alpha) + B \cdot \alpha$. Direct linear interpolation between Model A and Model B.

**Models Used:** A (Base), B (Incoming)

**Parameters:**
- **Alpha:** The blend ratio determining what fraction of **Model B** replaces **Model A**.
  - `0.0`: **100% Model A, 0% Model B** (exact Model A; Model B is completely ignored).
  - `0.25`: **75% Model A, 25% Model B** (mostly Model A, with 25% of Model B blended in).
  - `0.50`: **50% Model A, 50% Model B** (equal 50/50 mix of both models).
  - `0.75`: **25% Model A, 75% Model B** (mostly Model B, retaining 25% of Model A).
  - `1.0`: **0% Model A, 100% Model B** (exact Model B; Model A is completely replaced).

---

## Comparative-Interpolation
> Interpolates between A and B based on the relative differences in their tensor values, creating a selective blend rather than a flat uniform mix.

**Models Used:** A (Base), B (Incoming)

**Parameters:**
- **Alpha (Blend Intensity):** Controls the overall pull of **Model B into Model A**.
  - `0.0`: Heavily favors Model A (virtually no Model B incorporated).
  - `0.5`: Balanced curve where weights with moderate differences blend equally.
  - `1.0`: Strongly pulls in Model B wherever weights differ.
- **Beta (Feature Focus):** Controls whether the merge targets similar or divergent weights.
  - `0.0`: Focuses blending on weights where Model A and Model B are already similar.
  - `1.0`: Focuses blending on weights where Model A and Model B differ most.
- **Gamma (Transition Style):** Mixes between discrete selection and smooth gradation.
  - `0.0`: Stochastic / binomial selection (individual weights are chosen probabilistically as either pure A or pure B).
  - `1.0`: Smooth linear interpolation across all weights.

---

## Power-Up (DARE)
> Adds the unique capabilities of Model B onto Model A using the Drop and Rescale (DARE) technique. Computes the task delta $(B - A)$, randomly drops a fraction of the delta parameters, rescales the survivors to maintain expected magnitude, and adds them to Model A.

**Models Used:** A (Base), B (Donor)

**Parameters:**
- **Alpha (Dropout Rate $p$):** Proportion of Model B's delta parameters to randomly zero out.
  - Range: `0.0` to `1.0`.
  - Typical: `0.7` to `0.9`. For example, `0.85` means 85% of Model B's delta values are discarded, keeping only 15% of the most distinct changes.
- **Beta (Scale Multiplier):** Final strength multiplier for Model B's rescaled delta before adding to Model A.
  - `1.0`: Standard 100% strength addition of Model B's surviving features.
  - `0.5`: Half-strength addition of Model B's features.
  - `1.2`: Amplified addition of Model B's features.
- **Rescaling Logic:** Surviving parameters are automatically scaled by $1 / (1 - p)$ per the DARE paper to preserve the expected total update magnitude.

---

## Power-Up (DARE+TIES)
> Combines DARE (Drop and Rescale) with TIES (Trim, Elect Sign) to merge fine-tuned models cleanly. DARE sparsifies the delta; TIES trims low-magnitude background noise and enforces sign consensus.

**Models Used:** A (Base), B (Donor)

**Parameters:**
- **Alpha (DARE Drop Rate):** Fraction of Model B's delta parameters randomly dropped (`0.5`–`0.9`). Higher values produce a sparser capability transfer.
- **Beta (TIES Trim Quantile):** Fraction of the smallest-magnitude delta parameters eliminated after DARE to remove noise. `0.0` disables trimming; `0.2` trims the lowest 20% by absolute magnitude. Typical: `0.1`–`0.3`.
- **Gamma (Lambda Scale / Strength):** Multiplier for the filtered delta added to Model A.
  - `1.0`: Standard 100% strength of Model B's filtered capability added to Model A.
  - `0.25`: Gentle 25% addition of Model B's capabilities.
  - `1.5`: Amplified capability transfer.
- **Seed:** Random seed for reproducible DARE dropout masks.

**Algorithm:**
1. Compute task vector $\delta = B - A$
2. **DARE:** Apply random binary mask with keep probability $1 - \alpha$; rescale survivors by $1 / (1 - \alpha)$
3. **TIES trim:** Zero out parameters below the $\beta$-quantile magnitude threshold
4. **TIES elect:** Determine dominant sign direction per position; zero out disagreeing parameters
5. Return $A + \gamma \cdot \delta_{\text{filtered}}$

---

## Enhanced Man Interp (Enhanced Manual Interpolation)
> Selective interpolation between Model A and Model B based on normalized pairwise differences, constrained within a user-defined threshold band.

**Models Used:** A (Base), B (Incoming)

**Formula:**
$$\text{diff} = \frac{\max(|A - B|) - |A - B|}{\max(|A - B|)}$$
$$\text{mask}_{\text{threshold}} = (\beta < \text{mean}(\text{diff}) < \gamma)$$
$$\text{blend\_strength} = \text{diff}^{(1/\alpha - 1)} \cdot \text{mask}_{\text{threshold}}$$
$$\text{Output} = A \cdot (1 - \text{interpolated\_mask}) + B \cdot \text{interpolated\_mask}$$

**Parameters:**
- **Alpha (Model B Pull Strength):** Controls how strongly Model B is pulled into Model A.
  - Mathematically, the exponent applied to similarity is $(1/\alpha - 1)$.
  - **`Alpha = 0.25`**: Exponent is $(1/0.25 - 1) = 3$. This aggressively suppresses Model B's contribution toward zero. Only weights that are virtually identical between both models get even a minor blend of Model B; the final result remains overwhelmingly **Model A** (>90% Model A).
  - **`Alpha = 0.50`**: Exponent is $(1/0.50 - 1) = 1$ (linear). Weights with high similarity incorporate Model B proportionally up to 50%.
  - **`Alpha = 0.75`**: Exponent is $0.33$. Flattens similarity, pulling in substantially more of Model B wherever differences meet the threshold.
  - **`Alpha = 1.0`**: Exponent is $0$. Full maximum pull of Model B across all qualifying weights.
- **Beta (Lower Mean Threshold):** Minimum normalized similarity required before Model B can be blended into Model A. Layers/weights with similarity below $\beta$ remain 100% Model A.
- **Gamma (Upper Mean Threshold):** Maximum normalized similarity allowed for blending. Layers/weights with similarity above $\gamma$ remain 100% Model A. Together, $[\beta, \gamma]$ acts as a bandpass filter targeting only weights with moderate differences.
- **Delta (Smoothness Factor):** Blends between stochastic selection (`0.0`, a Bernoulli coin-flip choosing either pure A or pure B per weight) and smooth continuous blending (`1.0`, smooth linear weighting between A and B).

---

## Enhanced Auto Interp (Enhanced Automatic Interpolation)
> Automated version of Enhanced Interpolation that dynamically computes threshold boundaries centered on the layer's mean difference.

**Models Used:** A (Base), B (Incoming)

**Formula:**
$$\text{threshold\_band} = [\text{mean}(\text{diff}) \cdot (1 - \beta), \;\; \text{mean}(\text{diff}) \cdot (1 + \beta)]$$
$$\text{Output} = A \cdot (1 - \text{interpolated\_mask}) + B \cdot \text{interpolated\_mask}$$

**Parameters:**
- **Alpha (Model B Pull Strength):** Controls how much of Model B replaces Model A inside the active band.
  - `0.25`: Low Model B contribution; output remains mostly Model A.
  - `0.50`: Moderate, balanced blend of Model B into Model A.
  - `1.0`: Maximum Model B influence within the threshold band.
- **Beta (Threshold Width Around Mean):** Defines how wide the window of affected weights is around the layer's average difference.
  - `0.1`: Narrow window ($\pm 10\%$ around mean difference).
  - `0.5`: Broad window ($\pm 50\%$ around mean difference).
- **Gamma (Smoothness Factor):** Blends between stochastic Bernoulli selection (`0.0`) and smooth continuous linear interpolation (`1.0`).

---

## Weight-Sum Cutoff
> Linear interpolation that only blends Model B into Model A for weights whose relative difference falls within an explicit cutoff window.

**Models Used:** A (Base), B (Incoming)

**Formula:**
$$\text{Output} = A \cdot (1 - \alpha \cdot \text{mask}) + B \cdot (\alpha \cdot \text{mask})$$

**Parameters:**
- **Alpha (Model B Mix Ratio):** The percentage of Model B blended into Model A for weights inside the window.
  - `0.0`: 0% Model B (100% Model A everywhere).
  - `0.25`: 25% Model B, 75% Model A for weights inside the cutoff window; weights outside remain 100% Model A.
  - `0.50`: 50% Model B, 50% Model A inside the window.
  - `1.0`: 100% Model B inside the window (complete replacement of qualifying weights).
- **Beta (Upper Difference Threshold):** Upper cutoff for normalized difference.
- **Gamma (Lower Difference Threshold):** Lower cutoff for normalized difference. Weights with difference between $\gamma$ and $\beta$ receive the blend; all other weights stay 100% Model A.

---

## SVD LoRA Extraction
> Computes the difference between Model A and Model B and extracts it as a new LoRA file using Singular Value Decomposition (SVD). **Outputs a LoRA safetensors file into your `loras` directory, not a full model.**

**Models Used:** A (Tuned / Modified Model), B (Base Model)
- Computes delta: $\Delta W = A - B$
- Low-rank approximation: $\Delta W \approx U S V^T \implies \text{lora\_B} \cdot \text{lora\_A}$

**Parameters:**
- **Alpha (Linear Rank):** Target rank (dimension) for 2D dense/linear layer LoRA factors (e.g. `32`, `64`, `128`).
- **Beta (Conv Rank):** Target rank specifically for 3x3 convolution layer LoRA factors (e.g. `16`, `32`).
- **Gamma (Clamp Quantile):** Outlier clamping quantile (e.g. `0.99` clips the top 1% extreme values before SVD to prevent outlier distortion).

---

## Layer Mismatch Handling (`mismatch_mode`)

When Model A and Model B have differing layer structures or when merging partial LoRAs:

| Mode | Behavior |
|------|----------|
| `skip` | Missing layers in Model B use Model A's values unchanged **(default)** |
| `zeros` | Missing layers in Model B are treated as zeros |
| `error` | Raise an error and stop execution if any layer is missing in Model B |

**Note:** Extra layers in Model B that do not exist in Model A are ignored. Output topology always mirrors Model A.

---

## Dtype Preservation & Override

By default, the merger preserves the highest precision `dtype` for each key found across active source models and the requested `save_dtype`:
- **Preserve Higher Precision (default, `override_dtype = False`)**: If a key is `float32` in Model A or Model B, but `save_dtype="bf16"` is selected, that key remains `float32` to protect sensitive normalization layers.
- **Explicit Override (`override_dtype = True`)**: Forces all floating tensors strictly to `save_dtype`.

---

## Layer Filtering

- **`exclude_patterns`**: Newline-separated patterns. Matching layers are **not merged** and keep Model A's values unchanged (unless `include_mode` is enabled).
- **`include_mode`**: Inverts `exclude_patterns` into a whitelist: only matching layers are merged; all non-matching layers keep Model A's values unchanged.
- **`discard_patterns`**: Newline-separated patterns. Matching layers are **removed completely** from the saved output.
- **`glob_patterns`**: When enabled, uses shell wildcards (`*`, `?`). When disabled, uses regex substring matching.
