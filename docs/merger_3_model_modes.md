# 3-Model Merger Documentation

---

## Add-Difference
> `A + (B - C) * α`. This classic technique applies the "difference" between models B and C to model A. For example, it can be used to add a style (B) to a base model (A) while subtracting the style's original base (C).

**Models Used:** A, B, C
**Parameters:**
- **Alpha:** A multiplier for the `(B - C)` difference vector before it's added to A.
- **Beta:** A smoothing toggle. `0` for Off, `1` for On. When On, applies a median and gaussian filter to the difference, which can result in a smoother merge.

---

## Train-Difference
> A variation of Add-Difference from the SuperMerger extension that uses a more complex scaling factor based on the relative distances between the tensors of A, B, and C.

**Models Used:** A, B, C
**Parameters:**
- **Alpha:** A multiplier for the final calculated "training" difference.

---

## Extract-Features
> A powerful mode that identifies features present in both `(B - A)` and `(C - A)` and adds them to A. Allows for fine-grained control over combining aspects based on their similarity.

**Models Used:** A, B, C
**Parameters:**
- **Alpha:** Weights the merge between Model B (`0.0`) and Model C (`1.0`).
- **Beta:** Controls the focus on similarity (`0.0`) versus dissimilarity (`1.0`).
- **Gamma:** A bias exponent for similarity. Higher values increase the bias.
- **Delta:** A final multiplier for the extracted features before they are added to Model A.

---

## Add-Dissimilarities
> Identifies features that are dissimilar between Model B and Model C and adds them to Model A. Useful for combining unique aspects of two different models.

**Models Used:** A, B, C
**Parameters:**
- **Alpha:** Weights the comparison between Model B (`0.0`) and Model C (`1.0`).
- **Beta:** A multiplier for the final extracted dissimilar features.
- **Gamma:** A bias exponent for the similarity calculation.

---

## Layer Mismatch Handling

When merging models with different layer structures, the `mismatch_mode` parameter controls behavior:

| Mode | Behavior |
|------|----------|
| `skip` | Missing layers in B/C use A's values **(default)** |
| `zeros` | Missing layers in B/C are treated as zeros |
| `error` | Fail if any layer is missing |

**Special case for B - C subtraction (Add-Difference, Train-Difference):**
- If **both** B and C have the layer: Normal subtraction is performed
- If **either** B or C is missing the layer: The difference is treated as `None` (no contribution from this layer), and Model A's value is used

**Common scenarios:**
- Merging a fine-tuned model with added/removed layers
- Extracting style differences between models with slightly different architectures
- Combining partial LoRAs or adapters

**Note:** Extra layers in Models B/C that don't exist in Model A are currently ignored. The output will always have the same layer structure as Model A.

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