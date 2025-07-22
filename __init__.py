from .nodes import *

NODE_CLASS_MAPPINGS = {
    "UNetMetaKeys": UNetMetaKeys,
    "CLIPMetaKeys": CLIPMetaKeys,
    "LoRAMetaKeys": LoRAMetaKeys,
    "CheckpointMetaKeys": CheckpointMetaKeys,
    "OffloadModels": OffloadModelsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNetMetaKeys": "Get UNet Metadata and Keys",
    "CLIPMetaKeys": "Get CLIP Metadata and Keys",
    "LoRAMetaKeys": "Get LoRA Metadata and Keys",
    "CheckpointMetaKeys": "Get Checkpoint Metadata and Keys",
    "OffloadModelsNode": "Offload Models with Passthrough",
}