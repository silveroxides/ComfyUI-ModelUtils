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
> Adds the unique capabilities of Model B to Model A using the Drop and Rescale (DARE) technique, which often preserves the knowledge of the base model better than simple additions.

**Models Used:** A, B
**Parameters:**
- **Alpha:** The dropout rate. This is the proportion of unique weights from Model B that are *dropped* before merging. A higher value means less of B is merged.
- **Beta:** A final multiplier for the rescaled difference before it's added to Model A.

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