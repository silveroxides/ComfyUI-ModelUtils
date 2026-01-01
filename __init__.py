from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes.metakeys import (
    ModelMetaKeys, TextEncoderMetaKeys, LoRAMetaKeys,
    CheckpointMetaKeys, EmbeddingMetaKeys
)
from .nodes.renamekeys import (
    ModelRenameKeys, TextEncoderRenameKeys, LoRARenameKeys,
    CheckpointRenameKeys, EmbeddingRenameKeys
)
from .nodes.prunekeys import (
    ModelPruneKeys, TextEncoderPruneKeys, LoRAPruneKeys,
    CheckpointPruneKeys, EmbeddingPruneKeys
)
from .nodes.merger import (
    ModelTwoMerger, TextEncoderTwoMerger, LoRATwoMerger,
    CheckpointTwoMerger, EmbeddingTwoMerger,
    ModelThreeMerger, TextEncoderThreeMerger, LoRAThreeMerger,
    CheckpointThreeMerger, EmbeddingThreeMerger
)
from .nodes.lora_extract import (
    LoRASaveFromFile, LoRACheckpointSaveFromFile
)
from .nodes.lycoris_extract import (
    LyCORISExtractFixed, LyCORISExtractThreshold,
    LyCORISExtractRatio, LyCORISExtractQuantile, LyCORISExtractFull
)


class ModelUtilsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            # MetaKeys
            ModelMetaKeys, TextEncoderMetaKeys, LoRAMetaKeys,
            CheckpointMetaKeys, EmbeddingMetaKeys,
            # RenameKeys
            ModelRenameKeys, TextEncoderRenameKeys, LoRARenameKeys,
            CheckpointRenameKeys, EmbeddingRenameKeys,
            # PruneKeys
            ModelPruneKeys, TextEncoderPruneKeys, LoRAPruneKeys,
            CheckpointPruneKeys, EmbeddingPruneKeys,
            # Two-Model Mergers
            ModelTwoMerger, TextEncoderTwoMerger, LoRATwoMerger,
            CheckpointTwoMerger, EmbeddingTwoMerger,
            # Three-Model Mergers
            ModelThreeMerger, TextEncoderThreeMerger, LoRAThreeMerger,
            CheckpointThreeMerger, EmbeddingThreeMerger,
            # LoRA Extraction
            LoRASaveFromFile, LoRACheckpointSaveFromFile,
            # LyCORIS Extraction
            LyCORISExtractFixed, LyCORISExtractThreshold,
            LyCORISExtractRatio, LyCORISExtractQuantile, LyCORISExtractFull,
        ]


async def comfy_entrypoint() -> ModelUtilsExtension:
    return ModelUtilsExtension()

