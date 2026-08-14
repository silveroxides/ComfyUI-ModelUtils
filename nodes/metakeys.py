import os
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from unifiedefficientloader import MemoryEfficientSafeOpen
from .utils import convert_pt_to_safetensors


def _get_metakeys(model_name: str, model_type: str) -> tuple[str, str]:
    """Shared logic for all MetaKeys nodes."""
    model_path = folder_paths.get_full_path_or_raise(model_type, model_name)

    if model_path.endswith(('.pt', '.pth', '.bin', '.ckpt')):
        temp_safe_path = model_path + ".safetensors"
        if not os.path.exists(temp_safe_path):
            conversion_successful, error_message = convert_pt_to_safetensors(model_path, temp_safe_path)
            if not conversion_successful:
                return f"Conversion failed: {error_message}", ""
        model_path = temp_safe_path

    with MemoryEfficientSafeOpen(model_path, low_memory=True) as handler:
        metadata_str = str(handler.metadata() or {})
        keys = handler.keys()
        pbar = comfy.utils.ProgressBar(len(keys))
        lines = []
        for layer_name in tqdm(keys, desc="Reading layer shapes", unit="layers"):
            lines.append(f"{layer_name}, {list(handler.get_shape(layer_name))}\n")
            pbar.update(1)
        layer_shapes = "".join(lines)

    return metadata_str, layer_shapes


class ModelMetaKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelMetaKeys",
            display_name="Get Diffusion Model Metadata & Keys",
            category="ModelUtils/Keys",
            description="Loads a diffusion model and returns its metadata and layer names.",
            inputs=[
                io.Combo.Input(
                    "diffusionmodel_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                    tooltip="Diffusion model whose safetensors metadata and tensor keys will be listed.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="metadata"),
                io.String.Output(display_name="keys"),
            ],
        )

    @classmethod
    def execute(cls, diffusionmodel_name: str) -> io.NodeOutput:
        metadata, keys = _get_metakeys(diffusionmodel_name, "diffusion_models")
        return io.NodeOutput(metadata, keys)


class TextEncoderMetaKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncoderMetaKeys",
            display_name="Get Text Encoder Metadata & Keys",
            category="ModelUtils/Keys",
            description="Loads a text encoder and returns its metadata and layer names.",
            inputs=[
                io.Combo.Input(
                    "textencoder_name",
                    options=folder_paths.get_filename_list("text_encoders"),
                    tooltip="Text encoder whose safetensors metadata and tensor keys will be listed.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="metadata"),
                io.String.Output(display_name="keys"),
            ],
        )

    @classmethod
    def execute(cls, textencoder_name: str) -> io.NodeOutput:
        metadata, keys = _get_metakeys(textencoder_name, "text_encoders")
        return io.NodeOutput(metadata, keys)


class LoRAMetaKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAMetaKeys",
            display_name="Get LoRA Metadata & Keys",
            category="ModelUtils/Keys",
            description="Loads a LoRA and returns its metadata and layer names.",
            inputs=[
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="LoRA whose safetensors metadata and tensor keys will be listed.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="metadata"),
                io.String.Output(display_name="keys"),
            ],
        )

    @classmethod
    def execute(cls, lora_name: str) -> io.NodeOutput:
        metadata, keys = _get_metakeys(lora_name, "loras")
        return io.NodeOutput(metadata, keys)


class CheckpointMetaKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CheckpointMetaKeys",
            display_name="Get Checkpoint Metadata & Keys",
            category="ModelUtils/Keys",
            description="Loads a checkpoint and returns its metadata and layer names.",
            inputs=[
                io.Combo.Input(
                    "ckpt_name",
                    options=folder_paths.get_filename_list("checkpoints"),
                    tooltip="Checkpoint whose safetensors metadata and tensor keys will be listed.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="metadata"),
                io.String.Output(display_name="keys"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name: str) -> io.NodeOutput:
        metadata, keys = _get_metakeys(ckpt_name, "checkpoints")
        return io.NodeOutput(metadata, keys)


class EmbeddingMetaKeys(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmbeddingMetaKeys",
            display_name="Get Embedding Metadata & Keys",
            category="ModelUtils/Keys",
            description="Loads an embedding and returns its metadata and layer names.",
            inputs=[
                io.Combo.Input(
                    "embedding",
                    options=folder_paths.get_filename_list("embeddings"),
                    tooltip="Embedding whose safetensors metadata and tensor keys will be listed.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="metadata"),
                io.String.Output(display_name="keys"),
            ],
        )

    @classmethod
    def execute(cls, embedding: str) -> io.NodeOutput:
        metadata, keys = _get_metakeys(embedding, "embeddings")
        return io.NodeOutput(metadata, keys)
