from .nodes import *

NODE_CLASS_MAPPINGS = {
    "UNetMetaKeys": UNetMetaKeys,
    "CLIPMetaKeys": CLIPMetaKeys,
    "CheckpointPruneKeys": CheckpointPruneKeys,
    "CheckpointTwoMerger": CheckpointTwoMerger,
    "UNetTwoMerger": UNetTwoMerger,
    "CLIPTwoMerger": CLIPTwoMerger,
    "LoRATwoMerger": LoRATwoMerger,
    "CheckpointThreeMerger": CheckpointThreeMerger,
    "UNetThreeMerger": UNetThreeMerger,
    "CLIPThreeMerger": CLIPThreeMerger,
    "LoRAThreeMerger": LoRAThreeMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNetMetaKeys": "Get UNet Meta Keys",
    "CheckpointPruneKeys": "Prune Checkpoint Keys",
    "CheckpointTwoMerger": "Merge Checkpoints (2 Models)",
    "UNetTwoMerger": "Merge UNets (2 Models)",
    "CLIPTwoMerger": "Merge CLIPs (2 Models)",
    "LoRATwoMerger": "Merge LoRAs (2 Models)",
    "CheckpointThreeMerger": "Merge Checkpoints (3 Models)",
    "UNetThreeMerger": "Merge UNets (3 Models)",
    "CLIPThreeMerger": "Merge CLIPs (3 Models)",
    "LoRAThreeMerger": "Merge LoRAs (3 Models)",
}