import torch
from safetensors import safe_open
from safetensors.torch import save_file
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
