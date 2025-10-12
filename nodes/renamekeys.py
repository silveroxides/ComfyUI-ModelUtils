import os
import folder_paths
from .utils import convert_pt_to_safetensors, load_metadata_from_safetensors
from safetensors.torch import save_file

class BaseRenameKeys:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "rename_keys"
    CATEGORY = "ModelUtils"
    DESCRIPTION = "Loads a model, renames specified keys, and saves it as a new safetensors file."

    def _rename_keys(self, model_name, model_type, old_keys_str, new_keys_str, output_filename):
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

        model_weights, metadata = load_metadata_from_safetensors(model_path_to_load)

        old_keys = [key.strip() for key in old_keys_str.strip().split('\n') if key.strip()]
        new_keys = [key.strip() for key in new_keys_str.strip().split('\n') if key.strip()]

        if len(old_keys) != len(new_keys):
            raise ValueError("The number of old keys must match the number of new keys.")

        key_map = dict(zip(old_keys, new_keys))

        original_keys = list(model_weights.keys())
        renamed_tensors = {}
        for key in original_keys:
            new_key = key_map.get(key, key)
            renamed_tensors[new_key] = model_weights[key]

        model_dir = folder_paths.get_folder_paths(model_type)[0]
        output_path = os.path.join(model_dir, f"{output_filename.strip()}.safetensors")

        save_file(renamed_tensors, output_path, metadata)

        return (output_path,)

class UNetRenameKeys(BaseRenameKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "old_keys": ("STRING", {"multiline": True, "default": ""}),
                "new_keys": ("STRING", {"multiline": True, "default": ""}),
                "output_filename": ("STRING", {"default": "renamed_unet"}),
            }
        }

    def rename_keys(self, unet_name, old_keys, new_keys, output_filename):
        return self._rename_keys(unet_name, "diffusion_models", old_keys, new_keys, output_filename)

class CLIPRenameKeys(BaseRenameKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                "old_keys": ("STRING", {"multiline": True, "default": ""}),
                "new_keys": ("STRING", {"multiline": True, "default": ""}),
                "output_filename": ("STRING", {"default": "renamed_clip"}),
            }
        }

    def rename_keys(self, clip_name, old_keys, new_keys, output_filename):
        return self._rename_keys(clip_name, "text_encoders", old_keys, new_keys, output_filename)

class LoRARenameKeys(BaseRenameKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "old_keys": ("STRING", {"multiline": True, "default": ""}),
                "new_keys": ("STRING", {"multiline": True, "default": ""}),
                "output_filename": ("STRING", {"default": "renamed_lora"}),
            }
        }

    def rename_keys(self, lora_name, old_keys, new_keys, output_filename):
        return self._rename_keys(lora_name, "loras", old_keys, new_keys, output_filename)

class CheckpointRenameKeys(BaseRenameKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "old_keys": ("STRING", {"multiline": True, "default": ""}),
                "new_keys": ("STRING", {"multiline": True, "default": ""}),
                "output_filename": ("STRING", {"default": "renamed_checkpoint"}),
            }
        }

    def rename_keys(self, ckpt_name, old_keys, new_keys, output_filename):
        return self._rename_keys(ckpt_name, "checkpoints", old_keys, new_keys, output_filename)