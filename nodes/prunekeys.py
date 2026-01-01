import os
import re
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from safetensors.torch import save_file
from safetensors import safe_open
from .utils import convert_pt_to_safetensors
from .merger_utils import MemoryEfficientSafeOpen
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation


def _prune_keys(model_name: str, model_type: str, keys_to_prune_str: str,
                use_regex: bool, output_filename: str) -> str:
    """Shared logic for all PruneKeys nodes."""
    model_path = folder_paths.get_full_path_or_raise(model_type, model_name)

    if model_path.endswith(('.pt', '.pth', '.bin', '.ckpt')):
        temp_safe_path = model_path + ".safetensors"
        if not os.path.exists(temp_safe_path):
            conversion_successful, error_message = convert_pt_to_safetensors(model_path, temp_safe_path)
            if not conversion_successful:
                raise Exception(f"Conversion failed: {error_message}")
        model_path_to_load = temp_safe_path
    else:
        model_path_to_load = model_path

    # Prepare memory before operation
    model_size_gb = estimate_model_size(model_path_to_load)
    prepare_for_large_operation(model_size_gb * 1.2)

    # Get metadata from safe_open (header only, no tensor loading)
    with safe_open(model_path_to_load, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    patterns = [p.strip() for p in keys_to_prune_str.strip().split('\n') if p.strip()]

    if not patterns:
        raise ValueError("No keys/patterns provided to prune.")

    # Stream tensors and filter on the fly
    pruned_tensors = {}
    with MemoryEfficientSafeOpen(model_path_to_load) as handler:
        all_keys = handler.keys()
        pbar = comfy.utils.ProgressBar(len(all_keys))
        for key in tqdm(all_keys, desc="Pruning keys", unit="keys"):
            is_match = False
            if use_regex:
                if any(re.search(pattern, key) for pattern in patterns):
                    is_match = True
            else:
                if any(pattern in key for pattern in patterns):
                    is_match = True

            if not is_match:
                pruned_tensors[key] = handler.get_tensor(key)
            pbar.update(1)

    # Use [-1] for diffusion_models to get the actual diffusion_models folder, not legacy unet
    model_dir = folder_paths.get_folder_paths(model_type)[-1]
    output_path = os.path.join(model_dir, f"{output_filename.strip()}.safetensors")

    save_file(pruned_tensors, output_path, metadata)

    # Cleanup after operation
    del pruned_tensors
    cleanup_after_operation()

    return output_path


class ModelPruneKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelPruneKeys",
            display_name="Prune Diffusion Model Keys",
            category="ModelUtils/Keys",
            description="Loads a diffusion model, removes specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "diffusionmodel_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                ),
                io.String.Input("keys_to_prune", multiline=True, default=""),
                io.Boolean.Input("use_regex", default=False),
                io.String.Input("output_filename", default="pruned_model"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, diffusionmodel_name: str, keys_to_prune: str,
                use_regex: bool, output_filename: str) -> io.NodeOutput:
        path = _prune_keys(diffusionmodel_name, "diffusion_models",
                          keys_to_prune, use_regex, output_filename)
        return io.NodeOutput(path)


class TextEncoderPruneKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderPruneKeys",
            display_name="Prune Text Encoder Keys",
            category="ModelUtils/Keys",
            description="Loads a text encoder, removes specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "textencoder_name",
                    options=folder_paths.get_filename_list("text_encoders"),
                ),
                io.String.Input("keys_to_prune", multiline=True, default=""),
                io.Boolean.Input("use_regex", default=False),
                io.String.Input("output_filename", default="pruned_textencoder"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, textencoder_name: str, keys_to_prune: str,
                use_regex: bool, output_filename: str) -> io.NodeOutput:
        path = _prune_keys(textencoder_name, "text_encoders",
                          keys_to_prune, use_regex, output_filename)
        return io.NodeOutput(path)


class LoRAPruneKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAPruneKeys",
            display_name="Prune LoRA Keys",
            category="ModelUtils/Keys",
            description="Loads a LoRA, removes specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                ),
                io.String.Input("keys_to_prune", multiline=True, default=""),
                io.Boolean.Input("use_regex", default=False),
                io.String.Input("output_filename", default="pruned_lora"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, lora_name: str, keys_to_prune: str,
                use_regex: bool, output_filename: str) -> io.NodeOutput:
        path = _prune_keys(lora_name, "loras",
                          keys_to_prune, use_regex, output_filename)
        return io.NodeOutput(path)


class CheckpointPruneKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CheckpointPruneKeys",
            display_name="Prune Checkpoint Keys",
            category="ModelUtils/Keys",
            description="Loads a checkpoint, removes specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "ckpt_name",
                    options=folder_paths.get_filename_list("checkpoints"),
                ),
                io.String.Input("keys_to_prune", multiline=True, default=""),
                io.Boolean.Input("use_regex", default=False),
                io.String.Input("output_filename", default="pruned_checkpoint"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name: str, keys_to_prune: str,
                use_regex: bool, output_filename: str) -> io.NodeOutput:
        path = _prune_keys(ckpt_name, "checkpoints",
                          keys_to_prune, use_regex, output_filename)
        return io.NodeOutput(path)


class EmbeddingPruneKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmbeddingPruneKeys",
            display_name="Prune Embedding Keys",
            category="ModelUtils/Keys",
            description="Loads an embedding, removes specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "embedding",
                    options=folder_paths.get_filename_list("embeddings"),
                ),
                io.String.Input("keys_to_prune", multiline=True, default=""),
                io.Boolean.Input("use_regex", default=False),
                io.String.Input("output_filename", default="pruned_embedding"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, embedding: str, keys_to_prune: str,
                use_regex: bool, output_filename: str) -> io.NodeOutput:
        path = _prune_keys(embedding, "embeddings",
                          keys_to_prune, use_regex, output_filename)
        return io.NodeOutput(path)