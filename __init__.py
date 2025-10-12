from .nodes import *

NODE_CLASS_MAPPINGS = {
    "UNetMetaKeys": UNetMetaKeys,
    "CLIPMetaKeys": CLIPMetaKeys,
    "LoRAMetaKeys": LoRAMetaKeys,
    "CheckpointMetaKeys": CheckpointMetaKeys,
    "UNetRenameKeys": UNetRenameKeys,
    "CLIPRenameKeys": CLIPRenameKeys,
    "LoRARenameKeys": LoRARenameKeys,
    "CheckpointRenameKeys": CheckpointRenameKeys,
    "UNetPruneKeys": UNetPruneKeys,
    "CLIPPruneKeys": CLIPPruneKeys,
    "LoRAPruneKeys": LoRAPruneKeys,
    "CheckpointPruneKeys": CheckpointPruneKeys,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNetMetaKeys": "Get UNet Meta Keys",
    "CLIPMetaKeys": "Get CLIP Meta Keys",
    "LoRAMetaKeys": "Get LoRA Meta Keys",
    "CheckpointMetaKeys": "Get Checkpoint Meta Keys",
    "UNetRenameKeys": "Rename UNet Keys",
    "CLIPRenameKeys": "Rename CLIP Keys",
    "LoRARenameKeys": "Rename LoRA Keys",
    "CheckpointRenameKeys": "Rename Checkpoint Keys",
    "UNetPruneKeys": "Prune UNet Keys",
    "CLIPPruneKeys": "Prune CLIP Keys",
    "LoRAPruneKeys": "Prune LoRA Keys",
    "CheckpointPruneKeys": "Prune Checkpoint Keys",
}