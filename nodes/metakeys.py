import os
import folder_paths
from .utils import convert_pt_to_safetensors, load_metadata_from_safetensors



class BaseMetaKeys:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metadata", "keys")
    FUNCTION = "get_metakeys"
    CATEGORY = "ModelUtils/Keys"
    DESCRIPTION = "Loads a model and returns its metadata and layer names."

    def _get_metakeys(self, model_name, model_type):
        model_path = folder_paths.get_full_path_or_raise(model_type, model_name)
        model_weights = {}
        metadata = {}
        layer_shapes = ""

        if model_path.endswith(('.pt', '.pth', '.bin', '.ckpt')):
            temp_safe_path = model_path + ".safetensors"
            if not os.path.exists(temp_safe_path):
                conversion_successful, error_message = convert_pt_to_safetensors(model_path, temp_safe_path)
                if not conversion_successful:
                    return f"Conversion failed: {error_message}", ""
            model_path = temp_safe_path

        model_weights, metadata = load_metadata_from_safetensors(model_path)
        metadata_str = str(metadata)
        for layer_name, tensor in model_weights.items():
            shape = str(tensor.shape)
            shape = shape.replace("torch.Size(", "").replace(")", "")
            name_shape = f"{layer_name}, {shape}\n"
            layer_shapes += name_shape
        return metadata_str, layer_shapes



class UNetMetaKeys(BaseMetaKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
            }
        }

    def get_metakeys(self, unet_name):
        return self._get_metakeys(unet_name, "diffusion_models")

class CLIPMetaKeys(BaseMetaKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
            }
        }

    def get_metakeys(self, clip_name):
        return self._get_metakeys(clip_name, "text_encoders")

class LoRAMetaKeys(BaseMetaKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
            }
        }
    def get_metakeys(self, lora_name):
        return self._get_metakeys(lora_name, "loras")

class CheckpointMetaKeys(BaseMetaKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    def get_metakeys(self, ckpt_name):
        return self._get_metakeys(ckpt_name, "checkpoints")

class EmbeddingMetaKeys(BaseMetaKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding": (folder_paths.get_filename_list("embeddings"), ),
            }
        }

    def get_metakeys(self, embedding):
        return self._get_metakeys(embedding, "embeddings")

__all__ = ("UNetMetaKeys", "CLIPMetaKeys", "LoRAMetaKeys", "CheckpointMetaKeys", "EmbeddingMetaKeys")
