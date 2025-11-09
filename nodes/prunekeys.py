import os
import re
import folder_paths
from .utils import convert_pt_to_safetensors, load_metadata_from_safetensors
from safetensors.torch import save_file

class BasePruneKeys:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "prune_keys"
    CATEGORY = "ModelUtils/Keys"
    DESCRIPTION = "Loads a model, removes specified keys, and saves it as a new safetensors file."

    def _prune_keys(self, model_name, model_type, keys_to_prune_str, use_regex, output_filename):
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

        patterns = [p.strip() for p in keys_to_prune_str.strip().split('\n') if p.strip()]

        if not patterns:
            raise ValueError("No keys/patterns provided to prune.")

        pruned_tensors = {}
        for key, tensor in model_weights.items():
            is_match = False
            if use_regex:
                if any(re.search(pattern, key) for pattern in patterns):
                    is_match = True
            else:
                if any(pattern in key for pattern in patterns):
                    is_match = True

            if not is_match:
                pruned_tensors[key] = tensor

        model_dir = folder_paths.get_folder_paths(model_type)[0]
        output_path = os.path.join(model_dir, f"{output_filename.strip()}.safetensors")

        save_file(pruned_tensors, output_path, metadata)

        return (output_path,)

class UNetPruneKeys(BasePruneKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "keys_to_prune": ("STRING", {"multiline": True, "default": ""}),
                "use_regex": ("BOOLEAN", {"default": False}),
                "output_filename": ("STRING", {"default": "pruned_unet"}),
            }
        }

    def prune_keys(self, unet_name, keys_to_prune, use_regex, output_filename):
        return self._prune_keys(unet_name, "diffusion_models", keys_to_prune, use_regex, output_filename)

class CLIPPruneKeys(BasePruneKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                "keys_to_prune": ("STRING", {"multiline": True, "default": ""}),
                "use_regex": ("BOOLEAN", {"default": False}),
                "output_filename": ("STRING", {"default": "pruned_clip"}),
            }
        }

    def prune_keys(self, clip_name, keys_to_prune, use_regex, output_filename):
        return self._prune_keys(clip_name, "text_encoders", keys_to_prune, use_regex, output_filename)

class LoRAPruneKeys(BasePruneKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "keys_to_prune": ("STRING", {"multiline": True, "default": ""}),
                "use_regex": ("BOOLEAN", {"default": False}),
                "output_filename": ("STRING", {"default": "pruned_lora"}),
            }
        }

    def prune_keys(self, lora_name, keys_to_prune, use_regex, output_filename):
        return self._prune_keys(lora_name, "loras", keys_to_prune, use_regex, output_filename)

class CheckpointPruneKeys(BasePruneKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "keys_to_prune": ("STRING", {"multiline": True, "default": ""}),
                "use_regex": ("BOOLEAN", {"default": False}),
                "output_filename": ("STRING", {"default": "pruned_checkpoint"}),
            }
        }

    def prune_keys(self, ckpt_name, keys_to_prune, use_regex, output_filename):
        return self._prune_keys(ckpt_name, "checkpoints", keys_to_prune, use_regex, output_filename)

class EmbeddingPruneKeys(BasePruneKeys):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding": (folder_paths.get_filename_list("embeddings"), ),
                "keys_to_prune": ("STRING", {"multiline": True, "default": ""}),
                "use_regex": ("BOOLEAN", {"default": False}),
                "output_filename": ("STRING", {"default": "pruned_embedding"}),
            }
        }

    def prune_keys(self, embedding, keys_to_prune, use_regex, output_filename):
        return self._prune_keys(embedding, "embeddings", keys_to_prune, use_regex, output_filename)