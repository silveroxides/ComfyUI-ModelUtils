import torch

from .uel_io import atomic_uel_writer

def convert_pt_to_safetensors(pt_path, safe_path):
    try:
        model = torch.load(pt_path, map_location="cpu")
        if "state_dict" in model:
            state_dict = model["state_dict"]
        elif "model" in model:
            state_dict = model["model"]
        else:
            state_dict = model
        metadata = {"format": "pt"}
        with atomic_uel_writer(safe_path, metadata) as writer:
            for key in list(state_dict.keys()):
                tensor = state_dict.pop(key)
                writer.write_batch(
                    [(key.replace("state_dict.", ""), tensor.contiguous())]
                )
        return True, ""
    except Exception as e:
        return False, str(e)
