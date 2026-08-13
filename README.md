# ComfyUI-ModelUtils

A collection of ComfyUI custom nodes for inspecting, modifying, merging, and creating model files. Supports Models, TextEncoders, LoRAs, Checkpoints, and Embeddings.

## Features

- **MetaKeys** – Inspect and display metadata and tensor keys from model files
- **RenameKeys** – Batch rename tensor keys using pattern matching
- **PruneKeys** – Remove unwanted layers/keys from models
- **Mergers** – Combine 2 or 3 models with configurable blend modes and ratios
- **LoRA Extraction** – Extract LoRA adapters from model pairs using various SVD rank selection methods (Fixed, Ratio, Quantile, Knee-detection, Frobenius-norm)
- **Diffusion Model Dtype Conversion** – Stream models to fp32, fp16, or bf16 while preserving excluded tensor dtypes

## Example Workflows

<p align="center">
  <img src="assets/GetMetaAndkeys.png" width="400" alt="Get Meta and Keys">
  <br>
  <a href="example_workflows/GetMetaAndkeys.json">📥 GetMetaAndkeys.json</a>
</p>

---

<p align="center">
  <img src="assets/LoRA_Extract_nodes.png" width="400" alt="LoRA Extraction Nodes">
  <br>
  <a href="example_workflows/LoRA_Extract_nodes.json">📥 LoRA_Extract_nodes.json</a>
</p>

---

<p align="center">
  <img src="assets/Merging_Examples.png" width="400" alt="Merging Examples">
  <br>
  <a href="example_workflows/Merging_Examples.json">📥 Merging_Examples.json</a>
</p>

---

<p align="center">
  <img src="assets/RenameKeysInModel.png" width="400" alt="Rename Keys in Model">
  <br>
  <a href="example_workflows/RenameKeysInModel.json">📥 RenameKeysInModel.json</a>
</p>

## Acknowledgements

The LoRA extraction functionality was inspired by and references the excellent work from:

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) – Training scripts for Stable Diffusion
- [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) – Advanced LoRA techniques
- [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss) – Windows-friendly GUI for sd-scripts

## License

See [LICENSE](LICENSE) for details.
