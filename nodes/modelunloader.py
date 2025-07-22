from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import comfy.model_management as mm

class OffloadModelsNode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "any": (IO.ANY, {}),
                "force": (IO.BOOLEAN, {"default": False, "tooltip": "If True, forcibly unloads models from GPU."}),
                "enable_offload": (IO.BOOLEAN, {"default": False, "tooltip": "If True, calls ComfyUI's `unload_all_models()` function."}),
            }
        }

    RETURN_TYPES = (IO.ANY, )
    RETURN_NAMES = ("any", )
    FUNCTION = 'function'
    CATEGORY = 'advanced'


    def function(self, any, force, enable_offload):
        if enable_offload == True:
            print("OffloadModels: `enable_offload` is True. Calling `mm.unload_all_models()`.")
            mm.unload_all_models()
            if Force == True:
                mm.soft_empty_cache(Force=True)
            else:
                mm.soft_empty_cache()
        return (any,)

__all__ = ("OffloadModelsNode")