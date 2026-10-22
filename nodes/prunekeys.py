from contextlib import closing
import os
import re
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from .utils import convert_pt_to_safetensors
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation

from unifiedefficientloader import MemoryEfficientSafeOpen
from .uel_io import atomic_uel_writer, stream_work_units
from .artifact_paths import canonical_model_artifact_path
from .quantization_guard import (
    UPSTREAM_WRITER_STORAGE_CODES,
    _is_low_bit,
    storage_dtype_code,
    write_preserved_tensor,
)


def _prune_keys(
    model_name: str,
    model_type: str,
    keys_to_prune_str: str,
    use_regex: bool,
    output_filename: str,
    include_mode: bool = False,
) -> str:
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

    raw_patterns = [p.strip() for p in keys_to_prune_str.strip().split('\n') if p.strip()]
    if not raw_patterns:
        raise ValueError("No keys/patterns provided to prune.")

    compiled_patterns = []
    if use_regex:
        for pat in raw_patterns:
            try:
                compiled_patterns.append(re.compile(pat))
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern '{pat}': {exc}") from exc

    model_size_gb = estimate_model_size(model_path_to_load)
    if model_size_gb:
        prepare_for_large_operation(model_size_gb * 1.2)

    output_path, output_name = canonical_model_artifact_path(model_type, output_filename)

    handler = MemoryEfficientSafeOpen(model_path_to_load, low_memory=True)
    try:
        metadata = (handler.metadata() or {}).copy()
        all_keys = list(handler.keys())
        kept_keys = []
        for key in all_keys:
            matched = (
                any(pat.search(key) is not None for pat in compiled_patterns)
                if use_regex
                else any(pat in key for pat in raw_patterns)
            )
            keep = matched if include_mode else not matched
            if keep:
                kept_keys.append(key)

        work_units = []
        raw_preserve_keys = set()
        for key in kept_keys:
            code = storage_dtype_code(handler, key)
            if _is_low_bit(handler, key) or (
                code is not None and code not in UPSTREAM_WRITER_STORAGE_CODES
            ):
                raw_preserve_keys.add(key)
                work_units.append((key, {}))
            else:
                work_units.append((key, {0: [key]}))

        pbar = comfy.utils.ProgressBar(len(all_keys))
        with atomic_uel_writer(output_path, metadata) as writer, closing(
            stream_work_units({0: handler}, work_units, pin_memory=False)
        ) as streamed:
            for key, loaded in tqdm(streamed, total=len(work_units), desc="Pruning keys", unit="keys"):
                if key in raw_preserve_keys:
                    write_preserved_tensor(
                        writer, key, handler, output_key=key, force_raw=True
                    )
                else:
                    tensor = loaded[(0, key)]
                    try:
                        write_preserved_tensor(
                            writer, key, handler, output_key=key, tensor=tensor
                        )
                    finally:
                        del tensor
                pbar.update(1)
            pbar.update(len(all_keys) - len(work_units))
    finally:
        handler.__exit__(None, None, None)
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
                io.Boolean.Input("include_mode", default=False, tooltip="Keep only matching keys instead of removing them. Nonmatching keys are pruned from the output."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        diffusionmodel_name: str,
        keys_to_prune: str,
        use_regex: bool,
        output_filename: str,
        include_mode: bool = False,
    ) -> io.NodeOutput:
        path = _prune_keys(
            diffusionmodel_name,
            "diffusion_models",
            keys_to_prune,
            use_regex,
            output_filename,
            include_mode=include_mode,
        )
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
                io.Boolean.Input("include_mode", default=False, tooltip="Keep only matching keys instead of removing them. Nonmatching keys are pruned from the output."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        textencoder_name: str,
        keys_to_prune: str,
        use_regex: bool,
        output_filename: str,
        include_mode: bool = False,
    ) -> io.NodeOutput:
        path = _prune_keys(
            textencoder_name,
            "text_encoders",
            keys_to_prune,
            use_regex,
            output_filename,
            include_mode=include_mode,
        )
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
                io.Boolean.Input("include_mode", default=False, tooltip="Keep only matching keys instead of removing them. Nonmatching keys are pruned from the output."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        lora_name: str,
        keys_to_prune: str,
        use_regex: bool,
        output_filename: str,
        include_mode: bool = False,
    ) -> io.NodeOutput:
        path = _prune_keys(
            lora_name,
            "loras",
            keys_to_prune,
            use_regex,
            output_filename,
            include_mode=include_mode,
        )
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
                io.Boolean.Input("include_mode", default=False, tooltip="Keep only matching keys instead of removing them. Nonmatching keys are pruned from the output."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        ckpt_name: str,
        keys_to_prune: str,
        use_regex: bool,
        output_filename: str,
        include_mode: bool = False,
    ) -> io.NodeOutput:
        path = _prune_keys(
            ckpt_name,
            "checkpoints",
            keys_to_prune,
            use_regex,
            output_filename,
            include_mode=include_mode,
        )
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
                io.Boolean.Input("include_mode", default=False, tooltip="Keep only matching keys instead of removing them. Nonmatching keys are pruned from the output."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        embedding: str,
        keys_to_prune: str,
        use_regex: bool,
        output_filename: str,
        include_mode: bool = False,
    ) -> io.NodeOutput:
        path = _prune_keys(
            embedding,
            "embeddings",
            keys_to_prune,
            use_regex,
            output_filename,
            include_mode=include_mode,
        )
        return io.NodeOutput(path)
