import os
import folder_paths
import comfy.utils
from tqdm import tqdm
from comfy_api.latest import io
from safetensors import safe_open
from .utils import convert_pt_to_safetensors
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation

from unifiedefficientloader import MemoryEfficientSafeOpen, IncrementalSafetensorsWriter

REVERSE_RENAME_DICT = {
    "transformer_blocks.": "blocks.",
    "norm1.linear_1": "adaln_modulation_self_attn.1",
    "norm1.linear_2": "adaln_modulation_self_attn.2",
    "norm2.linear_1": "adaln_modulation_cross_attn.1",
    "norm2.linear_2": "adaln_modulation_cross_attn.2",
    "norm3.linear_1": "adaln_modulation_mlp.1",
    "norm3.linear_2": "adaln_modulation_mlp.2",
    "attn1.to_q": "self_attn.q_proj",
    "attn1.to_k": "self_attn.k_proj",
    "attn1.to_v": "self_attn.v_proj",
    "attn1.to_out.0": "self_attn.output_proj",
    "attn2.to_q": "cross_attn.q_proj",
    "attn2.to_k": "cross_attn.k_proj",
    "attn2.to_v": "cross_attn.v_proj",
    "attn2.to_out.0": "cross_attn.output_proj",
    "ff.net.0.proj": "mlp.layer1",
    "ff.net.2": "mlp.layer2",
    "norm_out.linear_1": "final_layer.adaln_modulation.1",
    "norm_out.linear_2": "final_layer.adaln_modulation.2",
    "proj_out": "final_layer.linear",
    "time_embed.t_embedder": "t_embedder.1",
    "time_embed.norm": "t_embedding_norm",
    "patch_embed.proj": "x_embedder.proj.1",
}

def _convert_diffusers_to_non_diffusers_anima_lora(lora_name: str, output_filename: str) -> str:
    """
    Converts a Diffusers Anima LoRA back to its non-diffusers format.
    """
    model_type = "loras"
    model_path = folder_paths.get_full_path_or_raise(model_type, lora_name)

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

    model_dir = folder_paths.get_folder_paths(model_type)[-1]
    output_path = os.path.join(model_dir, f"{output_filename.strip()}.safetensors")

    # Stream tensors, rename on the fly, write immediately
    writer = IncrementalSafetensorsWriter(output_path, metadata=metadata)
    writer.__enter__()
    try:
        with MemoryEfficientSafeOpen(model_path_to_load) as handler:
            original_keys = handler.keys()
            pbar = comfy.utils.ProgressBar(len(original_keys))
            for key in tqdm(original_keys, desc="Renaming Anima LoRA keys", unit="keys"):
                new_key = key

                if key.startswith("text_conditioner."):
                    new_key = f"diffusion_model.llm_adapter.{key.removeprefix('text_conditioner.')}"
                elif key.startswith("transformer."):
                    new_key = key.removeprefix("transformer.")
                    for old_part, new_part in REVERSE_RENAME_DICT.items():
                        new_key = new_key.replace(old_part, new_part)
                    new_key = f"diffusion_model.{new_key}"

                writer.write(new_key, handler.get_tensor(key).contiguous())
                pbar.update(1)
    finally:
        writer.__exit__(None, None, None)

    # Cleanup after operation
    cleanup_after_operation()

    return output_path


class AnimaLoraRename(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AnimaLoraRename",
            display_name="Anima LoRA Rename (Diffusers to Non-Diffusers)",
            category="ModelUtils/LoRA",
            description="Loads a Diffusers Anima LoRA, reverses its key names to the non-diffusers format, and saves it as a new safetensors file.",
            inputs=[
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                ),
                io.String.Input("output_filename", default="anima_lora_non_diffusers"),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name: str, output_filename: str) -> io.NodeOutput:
        path = _convert_diffusers_to_non_diffusers_anima_lora(lora_name, output_filename)
        return io.NodeOutput(path)
