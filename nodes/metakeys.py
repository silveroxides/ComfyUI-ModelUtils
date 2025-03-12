import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os
import folder_paths
from comfy.model_management import get_torch_device


def convert_pt_to_safetensors(pt_path, safe_path):
    try:
        model = torch.load(pt_path, map_location="cpu")
        if "state_dict" in model:
            state_dict = model["state_dict"]
        elif "model" in model:
            state_dict = model["model"]
        else:
            state_dict = model
        state_dict = {k.replace("state_dict.", ""): v for k, v in state_dict.items()}
        metadata = {"format": "pt"}
        save_file(state_dict, safe_path, metadata)
        return True, ""
    except Exception as e:
        return False, str(e)

def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    try:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            model_weights = {}
            for key in f.keys():
                model_weights[key] = f.get_tensor(key)
            metadata = f.metadata()
        if metadata is None:
            metadata = {}
        return model_weights, metadata
    except Exception as e:
        print(f"Error loading safetensors file: {e}")
        return {}, {}

class BaseMetaKeys:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metadata", "keys")
    FUNCTION = "get_metakeys"
    CATEGORY = "utils"
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

__all__ = ("UNetMetaKeys", "CLIPMetaKeys", "LoRAMetaKeys", "CheckpointMetaKeys")
