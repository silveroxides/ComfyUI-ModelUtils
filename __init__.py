from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes.lora_merger import LoRAMultiMerge, LoRAMultiMergeDARE, LoRAMultiMergeDAREEnhanced
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
from .nodes.lora_extract_svd import (
    LoRAExtractFixed, LoRAExtractRatio, LoRAExtractQuantile,
    LoRAExtractKnee, LoRAExtractFrobenius
)
from .nodes.dora_extract_wd import (
    DoRAExtractFixed, DoRAExtractRatio, DoRAExtractQuantile,
    DoRAExtractKnee, DoRAExtractFrobenius
)
from .nodes.dora_learned_wd import (
    DoRALearnedExtractFixed, DoRALearnedExtractRatio, DoRALearnedExtractQuantile,
    DoRALearnedExtractKnee, DoRALearnedExtractFrobenius
)
from .nodes.lora_resize import (
    LoRAResizeFixed, LoRAResizeRatio,
    LoRAResizeFrobenius, LoRAResizeCumulative,
    LoRAMergeToModel
)
from .nodes.downloader_nodes import (
    CheckpointInfoMetaDownloader, DiffusionModelInfoMetaDownloader, LoRAInfoMetaDownloader, EmbeddingInfoMetaDownloader,
    VAEInfoMetaDownloader, ControlNetInfoMetaDownloader, ManualPathInfoMetaDownloader
)
from .nodes.model_info_nodes import (
    CheckpointInfoLoader, LoRAInfoLoader, EmbeddingInfoLoader,
    VAEInfoLoader, ControlNetInfoLoader, DiffusionModelInfoLoader
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
            LoRAExtractFixed, LoRAExtractRatio, LoRAExtractQuantile,
            LoRAExtractKnee, LoRAExtractFrobenius,
            # DoRA Extraction
            DoRAExtractFixed, DoRAExtractRatio, DoRAExtractQuantile,
            DoRAExtractKnee, DoRAExtractFrobenius,
            # Learned DoRA Extraction
            DoRALearnedExtractFixed, DoRALearnedExtractRatio, DoRALearnedExtractQuantile,
            DoRALearnedExtractKnee, DoRALearnedExtractFrobenius,
            # LoRA Resize
            LoRAResizeFixed, LoRAResizeRatio,
            LoRAResizeFrobenius, LoRAResizeCumulative,
            # LoRA Multi-Merge
            LoRAMultiMerge, LoRAMultiMergeDARE, LoRAMultiMergeDAREEnhanced,
            # LoRA Merge To Model
            LoRAMergeToModel,
            # Downloaders
            CheckpointInfoMetaDownloader, DiffusionModelInfoMetaDownloader, LoRAInfoMetaDownloader, EmbeddingInfoMetaDownloader,
            VAEInfoMetaDownloader, ControlNetInfoMetaDownloader, ManualPathInfoMetaDownloader,
            # Info Loaders
            CheckpointInfoLoader, LoRAInfoLoader, EmbeddingInfoLoader,
            VAEInfoLoader, ControlNetInfoLoader, DiffusionModelInfoLoader,
        ]


async def comfy_entrypoint() -> ModelUtilsExtension:
    return ModelUtilsExtension()

