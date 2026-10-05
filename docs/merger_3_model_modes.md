# 3-Model Merger Documentation

---

### Core Convention Across All 3-Model Modes
- **Model A is ALWAYS the base / receiving model** that receives the merge modifications.
- **Model B is the donor / target fine-tune model.**
- **Model C is the reference base model of Model B.**
- In difference merging, $(B - C)$ calculates the isolated delta (the specific training / fine-tune changes introduced in Model B relative to its base Model C).
- That delta is then added onto Model A with scaling $\alpha$:
  $$\text{Output} = A + (B - C) \cdot \alpha$$

---

## Add-Difference
> $\text{Output} = A + (B - C) \cdot \alpha$. Extracts the isolated fine-tune / style delta from $(B - C)$ and applies it directly to Model A.

**Models Used:** A (Receiving Base), B (Donor Fine-Tune), C (Donor's Base Reference)

**Parameters:**
- **Alpha (Delta Multiplier / Strength):** Scaling factor for how strongly the $(B - C)$ difference is added to Model A.
  - `0.0`: **0% addition** (Output is pure Model A; no changes added).
  - `0.25`: **25% strength addition** (gentle, subtle application of Model B's style/features onto Model A).
  - `0.50`: **50% strength addition** (moderate application).
  - `1.0`: **100% full-strength addition** (applies the exact full delta of B relative to C onto Model A).
  - `1.5`: **150% over-amplified addition** (exaggerates the difference; can introduce visual artifacts if pushed too high).
- **Beta (Smoothing Toggle):**
  - `0`: Off (raw mathematical delta added directly).
  - `1`: On (applies median and Gaussian smoothing to the delta before adding, reducing harsh pixel noise and high-frequency artifacts).

---

## Train-Difference
> A variation of Add-Difference that calculates an adaptive scaling factor based on the relative vector distances between Model A, Model B, and Model C.

**Models Used:** A (Receiving Base), B (Donor Fine-Tune), C (Donor's Base Reference)

**Parameters:**
- **Alpha (Difference Multiplier):** Multiplier for the calculated distance-weighted difference added to Model A.
  - `0.0`: Pure Model A.
  - `1.0`: Standard 100% strength addition of the adaptive training delta.
  - `< 1.0`: Softened application.
  - `> 1.0`: Amplified application.

---

## Extract-Features
> Analyzes feature vectors present in both $(B - A)$ and $(C - A)$, using cosine similarity to isolate shared features and blend them into Model A.

**Models Used:** A (Base), B (Donor 1), C (Donor 2)

**Parameters:**
- **Alpha (B vs C Contribution Balance):** Weights the source contribution between Model B and Model C.
  - `0.0`: 100% Model B's delta features.
  - `0.5`: Equal mix of features from both Model B and Model C.
  - `1.0`: 100% Model C's delta features.
- **Beta (Similarity vs Dissimilarity):**
  - `0.0`: Focuses exclusively on features where B and C agree / are similar relative to A.
  - `1.0`: Focuses on features where B and C diverge / are dissimiliar relative to A.
- **Gamma (Similarity Bias Exponent):** Exponential bias curve shaping how sharply similar features are separated from dissimilar features.
- **Delta (Feature Multiplier / Strength):** Final scaling factor for the extracted features before adding them onto Model A.
  - `0.0`: No features added (pure Model A).
  - `1.0`: Standard strength addition of extracted features onto Model A.

---

## Add-Dissimilarities
> Identifies features that are dissimilar between Model B and Model C (relative to Model A) and transfers them into Model A. Ideal for combining non-overlapping, unique aspects from two different fine-tunes without conflicting duplicate features.

**Models Used:** A (Base), B (Donor 1), C (Donor 2)

**Parameters:**
- **Alpha (B vs C Balance):** Balance weighting between Model B and Model C when extracting differences.
  - `0.0`: 100% weight to Model B's dissimilarities.
  - `0.5`: Balanced extraction from both models.
  - `1.0`: 100% weight to Model C's dissimilarities.
- **Beta (Feature Multiplier / Strength):** Final scaling multiplier for the dissimilar features added to Model A.
  - `0.0`: No features added (pure Model A).
  - `1.0`: Standard strength addition onto Model A.
- **Gamma (Similarity Bias Exponent):** Exponent controlling the cutoff sensitivity between similar and dissimilar directions.

---

## Layer Mismatch Handling (`mismatch_mode`)

When merging models with different layer topologies:

| Mode | Behavior |
|------|----------|
| `skip` | Missing layers in B or C use Model A's values unchanged **(default)** |
| `zeros` | Missing layers in B or C are treated as zero tensors |
| `error` | Raise an error and abort if any layer is missing across models |

**Special rule for $(B - C)$ difference operations (Add-Difference, Train-Difference):**
- If **both** B and C have the layer: Normal difference $(B - C)$ is computed and applied to A.
- If **either** B or C is missing the layer: The difference is treated as zero, and Model A's value is preserved untouched.

---

## Dtype Preservation & Override

- **Preserve Higher Precision (default, `override_dtype = False`)**: If a key is `float32` in any source model (A, B, or C), it is saved as `float32` regardless of `save_dtype`.
- **Explicit Override (`override_dtype = True`)**: Forces all floating tensors strictly to `save_dtype`.

---

## Layer Filtering

- **`exclude_patterns`**: Newline-separated patterns. Matching layers keep Model A's values untouched.
- **`include_mode`**: Inverts `exclude_patterns` into a whitelist: only matching layers receive the merge; all other layers keep Model A's values.
- **`discard_patterns`**: Newline-separated patterns. Matching layers are omitted completely from the output model.
- **`glob_patterns`**: Enables shell wildcards (`*`, `?`) when checked; uses regex substring matching when unchecked.
