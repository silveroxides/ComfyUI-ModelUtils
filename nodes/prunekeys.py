import os
import re
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from .utils import convert_pt_to_safetensors
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation

from unifiedefficientloader import MemoryEfficientSafeOpen
from .uel_io import atomic_uel_writer
from .artifact_paths import canonical_model_artifact_path


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

    patterns = [p.strip() for p in keys_to_prune_str.strip().split('\n') if p.strip()]

    if not patterns:
        raise ValueError("No keys/patterns provided to prune.")

    # Use [-1] for diffusion_models to get the actual diffusion_models folder, not legacy unet
    output_path, output_name = canonical_model_artifact_path(model_type, output_filename)

    # Stream tensors, filter on the fly, write immediately
    with MemoryEfficientSafeOpen(model_path_to_load, low_memory=True) as handler:
        metadata = handler.metadata() or {}
        with atomic_uel_writer(output_path, metadata) as writer:
            all_keys = handler.keys()
            pbar = comfy.utils.ProgressBar(len(all_keys))
            kept_keys = [
                key for key in all_keys
                if not (
                    any(re.search(pattern, key) for pattern in patterns)
                    if use_regex else any(pattern in key for pattern in patterns)
                )
            ]
            stream = handler.async_stream(
                kept_keys, batch_size=1, prefetch_batches=1, pin_memory=False
            )
            for batch in tqdm(stream, total=len(kept_keys), desc="Pruning keys", unit="keys"):
                key, tensor = batch[0]
                writer.write_batch([(key, tensor.contiguous())])
                handler.mark_processed(key)
                pbar.update(1)
            pbar.update(len(all_keys) - len(kept_keys))

    # Cleanup after operation
    cleanup_after_operation()

    return output_name


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
                    tooltip="Diffusion model from which matching tensor keys will be removed.",
                ),
                io.String.Input("keys_to_prune", multiline=True, default="", tooltip="One tensor-key pattern per line. Blank lines are ignored."),
                io.Boolean.Input("use_regex", default=False, tooltip="Interpret each line as a regular expression; otherwise match literal key text."),
                io.String.Input("output_filename", default="pruned_model", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
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
                    tooltip="Text encoder from which matching tensor keys will be removed.",
                ),
                io.String.Input("keys_to_prune", multiline=True, default="", tooltip="One tensor-key pattern per line. Blank lines are ignored."),
                io.Boolean.Input("use_regex", default=False, tooltip="Interpret each line as a regular expression; otherwise match literal key text."),
                io.String.Input("output_filename", default="pruned_textencoder", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
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
                    tooltip="LoRA from which matching tensor keys will be removed.",
                ),
                io.String.Input("keys_to_prune", multiline=True, default="", tooltip="One tensor-key pattern per line. Blank lines are ignored."),
                io.Boolean.Input("use_regex", default=False, tooltip="Interpret each line as a regular expression; otherwise match literal key text."),
                io.String.Input("output_filename", default="pruned_lora", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
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
                    tooltip="Checkpoint from which matching tensor keys will be removed.",
                ),
                io.String.Input("keys_to_prune", multiline=True, default="", tooltip="One tensor-key pattern per line. Blank lines are ignored."),
                io.Boolean.Input("use_regex", default=False, tooltip="Interpret each line as a regular expression; otherwise match literal key text."),
                io.String.Input("output_filename", default="pruned_checkpoint", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
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
                    tooltip="Embedding from which matching tensor keys will be removed.",
                ),
                io.String.Input("keys_to_prune", multiline=True, default="", tooltip="One tensor-key pattern per line. Blank lines are ignored."),
                io.Boolean.Input("use_regex", default=False, tooltip="Interpret each line as a regular expression; otherwise match literal key text."),
                io.String.Input("output_filename", default="pruned_embedding", tooltip="Output filename without extension, written under the matching ComfyUI model directory."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(cls, embedding: str, keys_to_prune: str,
                use_regex: bool, output_filename: str) -> io.NodeOutput:
        path = _prune_keys(embedding, "embeddings",
                          keys_to_prune, use_regex, output_filename)
        return io.NodeOutput(path)
