import os
import torch
import folder_paths
import comfy.utils
from safetensors.torch import save_file
from .merger_ops import TWO_MODEL_MODES, THREE_MODEL_MODES
from .merger_utils import MemoryEfficientSafeOpen

def load_documentation_from_file(filename):
    """Loads documentation from a markdown file in the ../docs/ directory."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    docs_path = os.path.join(current_dir, '..', 'docs', filename)
    try:
        with open(docs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"# Documentation File Not Found\n\nPlease ensure `{filename}` exists in the `docs` directory of the custom node."

# --- UI Dictionary Helper Functions ---
def _create_two_model_inputs(model_folder_name):
    return {"required": {
        "execution_mode": (["MERGE", "DOCUMENTATION ONLY"],),
        "model_a": (["None"] + folder_paths.get_filename_list(model_folder_name),),
        "model_b": (["None"] + folder_paths.get_filename_list(model_folder_name),),
        "calc_mode": ([m.name for m in TWO_MODEL_MODES],),
        "alpha": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01}),
        "beta": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01}),
        "gamma": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.001}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "output_filename": ("STRING", {"default": "merged_2_model"}),
        "save_dtype": (["fp32", "fp16", "bf16"],),
        "process_device": (["cuda", "cpu"],),
    }}

def _create_three_model_inputs(model_folder_name):
    return {"required": {
        "execution_mode": (["MERGE", "DOCUMENTATION ONLY"],),
        "model_a": (["None"] + folder_paths.get_filename_list(model_folder_name),),
        "model_b": (["None"] + folder_paths.get_filename_list(model_folder_name),),
        "model_c": (["None"] + folder_paths.get_filename_list(model_folder_name),),
        "calc_mode": ([m.name for m in THREE_MODEL_MODES],),
        "alpha": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01}),
        "beta": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01}),
        "gamma": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01}),
        "delta": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "output_filename": ("STRING", {"default": "merged_3_model"}),
        "save_dtype": (["fp32", "fp16", "bf16"],),
        "process_device": (["cuda", "cpu"],),
    }}

# --- Base Classes for Backend Logic ONLY ---
class MergerLogic:
    def _execute_merge(self, model_names, calc_mode, all_modes, recipe_params, model_type):
        calc_mode_class = next((m for m in all_modes if m.name == calc_mode), None)
        if not calc_mode_class: raise ValueError(f"Calc mode '{calc_mode}' not found.")
        primary_model_name = model_names.get('model_a')
        if not primary_model_name or primary_model_name == "None": raise ValueError("Model A is required to run the merge.")
        handlers = {}
        for name in model_names.values():
            if name and name != "None":
                path = folder_paths.get_full_path(model_type, name)
                if not path: raise FileNotFoundError(f"Model '{name}' not found.")
                handlers[name] = MemoryEfficientSafeOpen(path)
        all_keys = handlers[primary_model_name].keys()
        metadata = handlers[primary_model_name].header.get("__metadata__", {})
        merged_state_dict = {}
        pbar = comfy.utils.ProgressBar(len(all_keys))
        save_dtype = recipe_params.pop('save_dtype')
        save_torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(save_dtype)
        recipe_params.update({"handlers": handlers}) # Add handlers, device and dtype are already in
        for key in all_keys:
            recipe = calc_mode_class.create_recipe(key=key, **recipe_params)
            result = recipe.merge()
            if isinstance(result, dict):
                for r_key, r_tensor in result.items(): merged_state_dict[r_key] = r_tensor.to(save_torch_dtype).cpu().clone()
            else: merged_state_dict[key] = result.to(save_torch_dtype).cpu().clone()
            pbar.update(1)
        for handler in handlers.values(): handler.__exit__(None, None, None)
        output_folder = "loras" if calc_mode == "SVD LoRA Extraction" else model_type
        output_dir = folder_paths.get_folder_paths(output_folder)[0]
        os.makedirs(output_dir, exist_ok=True)
        output_filename = recipe_params.get("output_filename")
        output_path = os.path.join(output_dir, f"{output_filename}.safetensors")
        save_file(merged_state_dict, output_path, metadata=metadata)
        return (f"{output_filename}.safetensors",)

class BaseTwoMerger(MergerLogic):
    FUNCTION = "merge"; CATEGORY = "ModelUtils/Merging"
    RETURN_TYPES = ("STRING", "STRING",); RETURN_NAMES = ("output_filename", "documentation",)

    def merge(self, execution_mode, model_a, model_b, calc_mode, alpha, beta, gamma, seed, output_filename, save_dtype, process_device):
        doc = load_documentation_from_file('merger_2_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return ("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "calc_mode": calc_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32
        }
        model_names = {"model_a": model_a, "model_b": model_b}
        filename, = self._execute_merge(model_names, calc_mode, TWO_MODEL_MODES, recipe_params, self.MODEL_TYPE)
        return (filename, doc)

class BaseThreeMerger(MergerLogic):
    FUNCTION = "merge"; CATEGORY = "ModelUtils/Merging"
    RETURN_TYPES = ("STRING", "STRING",); RETURN_NAMES = ("output_filename", "documentation",)

    def merge(self, execution_mode, model_a, model_b, model_c, calc_mode, alpha, beta, gamma, delta, seed, output_filename, save_dtype, process_device):
        doc = load_documentation_from_file('merger_3_model_modes.md')
        if execution_mode == "DOCUMENTATION ONLY":
            return ("Documentation mode active. No merge performed.", doc)

        recipe_params = {
            "model_a": model_a, "model_b": model_b, "model_c": model_c, "calc_mode": calc_mode,
            "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "seed": seed,
            "output_filename": output_filename, "save_dtype": save_dtype,
            "device": process_device, "dtype": torch.float32
        }
        model_names = {"model_a": model_a, "model_b": model_b, "model_c": model_c}
        filename, = self._execute_merge(model_names, calc_mode, THREE_MODEL_MODES, recipe_params, self.MODEL_TYPE)
        return (filename, doc)

# --- Concrete 2-Model Nodes ---
class CheckpointTwoMerger(BaseTwoMerger):
    MODEL_TYPE = "checkpoints"
    @classmethod
    def INPUT_TYPES(s): return _create_two_model_inputs(s.MODEL_TYPE)
class UNetTwoMerger(BaseTwoMerger):
    MODEL_TYPE = "diffusion_models"
    @classmethod
    def INPUT_TYPES(s): return _create_two_model_inputs(s.MODEL_TYPE)
class CLIPTwoMerger(BaseTwoMerger):
    MODEL_TYPE = "text_encoders"
    @classmethod
    def INPUT_TYPES(s): return _create_two_model_inputs(s.MODEL_TYPE)
class LoRATwoMerger(BaseTwoMerger):
    MODEL_TYPE = "loras"
    @classmethod
    def INPUT_TYPES(s): return _create_two_model_inputs(s.MODEL_TYPE)

# --- Concrete 3-Model Nodes ---
class CheckpointThreeMerger(BaseThreeMerger):
    MODEL_TYPE = "checkpoints"
    @classmethod
    def INPUT_TYPES(s): return _create_three_model_inputs(s.MODEL_TYPE)
class UNetThreeMerger(BaseThreeMerger):
    MODEL_TYPE = "diffusion_models"
    @classmethod
    def INPUT_TYPES(s): return _create_three_model_inputs(s.MODEL_TYPE)
class CLIPThreeMerger(BaseThreeMerger):
    MODEL_TYPE = "text_encoders"
    @classmethod
    def INPUT_TYPES(s): return _create_three_model_inputs(s.MODEL_TYPE)
class LoRAThreeMerger(BaseThreeMerger):
    MODEL_TYPE = "loras"
    @classmethod
    def INPUT_TYPES(s): return _create_three_model_inputs(s.MODEL_TYPE)

__all__ = [
    "CheckpointTwoMerger", "UNetTwoMerger", "CLIPTwoMerger", "LoRATwoMerger",
    "CheckpointThreeMerger", "UNetThreeMerger", "CLIPThreeMerger", "LoRAThreeMerger"
]