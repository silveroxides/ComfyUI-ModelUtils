import os
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from safetensors.torch import save_file
from safetensors import safe_open
from .utils import convert_pt_to_safetensors
from .merger_utils import MemoryEfficientSafeOpen
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation


def _rename_keys(model_name: str, model_type: str, old_keys_str: str, 
                 new_keys_str: str, output_filename: str) -> str:
    """Shared logic for all RenameKeys nodes."""
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

    old_keys = [key.strip() for key in old_keys_str.strip().split('\n') if key.strip()]
    new_keys = [key.strip() for key in new_keys_str.strip().split('\n') if key.strip()]

    if len(old_keys) != len(new_keys):
        raise ValueError("The number of old keys must match the number of new keys.")

    key_map = dict(zip(old_keys, new_keys))

    # Stream tensors and rename on the fly
    renamed_tensors = {}
    with MemoryEfficientSafeOpen(model_path_to_load) as handler:
        original_keys = handler.keys()
        pbar = comfy.utils.ProgressBar(len(original_keys))
        for key in tqdm(original_keys, desc="Renaming keys", unit="keys"):
            new_key = key_map.get(key, key)
            renamed_tensors[new_key] = handler.get_tensor(key)
            pbar.update(1)

    # Use [-1] for diffusion_models to get the actual diffusion_models folder, not legacy unet
    model_dir = folder_paths.get_folder_paths(model_type)[-1]
    output_path = os.path.join(model_dir, f"{output_filename.strip()}.safetensors")

    save_file(renamed_tensors, output_path, metadata)
    
    # Cleanup after operation
    del renamed_tensors
    cleanup_after_operation()

    return output_path


class ModelRenameKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelRenameKeys",
            display_name="Rename Diffusion Model Keys",
            category="ModelUtils/Keys",
            description="Loads a diffusion model, renames specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "diffusionmodel_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                ),
                io.String.Input("old_keys", multiline=True, default=""),
                io.String.Input("new_keys", multiline=True, default=""),
                io.String.Input("output_filename", default="renamed_model"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, diffusionmodel_name: str, old_keys: str, 
                new_keys: str, output_filename: str) -> io.NodeOutput:
        path = _rename_keys(diffusionmodel_name, "diffusion_models", 
                           old_keys, new_keys, output_filename)
        return io.NodeOutput(path)


class TextEncoderRenameKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderRenameKeys",
            display_name="Rename Text Encoder Keys",
            category="ModelUtils/Keys",
            description="Loads a text encoder, renames specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "textencoder_name",
                    options=folder_paths.get_filename_list("text_encoders"),
                ),
                io.String.Input("old_keys", multiline=True, default=""),
                io.String.Input("new_keys", multiline=True, default=""),
                io.String.Input("output_filename", default="renamed_textencoder"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, textencoder_name: str, old_keys: str, 
                new_keys: str, output_filename: str) -> io.NodeOutput:
        path = _rename_keys(textencoder_name, "text_encoders", 
                           old_keys, new_keys, output_filename)
        return io.NodeOutput(path)


class LoRARenameKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRARenameKeys",
            display_name="Rename LoRA Keys",
            category="ModelUtils/Keys",
            description="Loads a LoRA, renames specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                ),
                io.String.Input("old_keys", multiline=True, default=""),
                io.String.Input("new_keys", multiline=True, default=""),
                io.String.Input("output_filename", default="renamed_lora"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, lora_name: str, old_keys: str, 
                new_keys: str, output_filename: str) -> io.NodeOutput:
        path = _rename_keys(lora_name, "loras", 
                           old_keys, new_keys, output_filename)
        return io.NodeOutput(path)


class CheckpointRenameKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CheckpointRenameKeys",
            display_name="Rename Checkpoint Keys",
            category="ModelUtils/Keys",
            description="Loads a checkpoint, renames specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "ckpt_name",
                    options=folder_paths.get_filename_list("checkpoints"),
                ),
                io.String.Input("old_keys", multiline=True, default=""),
                io.String.Input("new_keys", multiline=True, default=""),
                io.String.Input("output_filename", default="renamed_checkpoint"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name: str, old_keys: str, 
                new_keys: str, output_filename: str) -> io.NodeOutput:
        path = _rename_keys(ckpt_name, "checkpoints", 
                           old_keys, new_keys, output_filename)
        return io.NodeOutput(path)


class EmbeddingRenameKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmbeddingRenameKeys",
            display_name="Rename Embedding Keys",
            category="ModelUtils/Keys",
            description="Loads an embedding, renames specified keys, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "embedding",
                    options=folder_paths.get_filename_list("embeddings"),
                ),
                io.String.Input("old_keys", multiline=True, default=""),
                io.String.Input("new_keys", multiline=True, default=""),
                io.String.Input("output_filename", default="renamed_embedding"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, embedding: str, old_keys: str, 
                new_keys: str, output_filename: str) -> io.NodeOutput:
        path = _rename_keys(embedding, "embeddings", 
                           old_keys, new_keys, output_filename)
        return io.NodeOutput(path)