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
from .nodes.consensus_merger import CWB_MERGER_NODES
from .nodes.cwb_delta_lora_merger import DELTA_CWB_LORA_NODES
from .nodes.lodestone_merger import LODESTONE_MERGER_NODES
from .nodes.model_analysis import MODEL_ANALYSIS_NODES
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
from .nodes.text_encoder_extract import (
    TextEncoderLoRAExtractFixed, TextEncoderLoRAExtractRatio,
    TextEncoderLoRAExtractQuantile, TextEncoderLoRAExtractKnee,
    TextEncoderLoRAExtractFrobenius, TextEncoderDoRAExtractFixed,
    TextEncoderDoRAExtractRatio, TextEncoderDoRAExtractQuantile,
    TextEncoderDoRAExtractKnee, TextEncoderDoRAExtractFrobenius,
)
from .nodes.lora_resize import (
    LoRANormalizeAlpha,
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


from .nodes.lora_rename import AnimaLoraRename
from .nodes.minimax_h3_lora_convert import MiniMaxH3DiffusersLoRAConvert
from .nodes.dtype_conversion import DiffusionModelDtypeConversion
from .nodes.layer_parameters import LayerParameterConfiguration
from .nodes.lora_model_analysis import LoRAOnModelAnalysis

class ModelUtilsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LayerParameterConfiguration,
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
            # Dedicated CWB Mergers
            *CWB_MERGER_NODES,
            # Dedicated Delta CWB LoRA Mergers
            *DELTA_CWB_LORA_NODES,
            # Dedicated Lodestone LoRA Mergers
            *LODESTONE_MERGER_NODES,
            # Two-Model Analysis
            *MODEL_ANALYSIS_NODES,
            LoRAOnModelAnalysis,
            # LoRA Extraction
            LoRAExtractFixed, LoRAExtractRatio, LoRAExtractQuantile,
            LoRAExtractKnee, LoRAExtractFrobenius,
            # DoRA Extraction
            DoRAExtractFixed, DoRAExtractRatio, DoRAExtractQuantile,
            DoRAExtractKnee, DoRAExtractFrobenius,
            # Learned DoRA Extraction
            DoRALearnedExtractFixed, DoRALearnedExtractRatio, DoRALearnedExtractQuantile,
            DoRALearnedExtractKnee, DoRALearnedExtractFrobenius,
            # Text Encoder LoRA/DoRA Extraction
            TextEncoderLoRAExtractFixed, TextEncoderLoRAExtractRatio,
            TextEncoderLoRAExtractQuantile, TextEncoderLoRAExtractKnee,
            TextEncoderLoRAExtractFrobenius, TextEncoderDoRAExtractFixed,
            TextEncoderDoRAExtractRatio, TextEncoderDoRAExtractQuantile,
            TextEncoderDoRAExtractKnee, TextEncoderDoRAExtractFrobenius,
            # LoRA Resize
            LoRANormalizeAlpha,
            LoRAResizeFixed, LoRAResizeRatio,
            LoRAResizeFrobenius, LoRAResizeCumulative,
            # LoRA Multi-Merge
            LoRAMultiMerge, LoRAMultiMergeDARE, LoRAMultiMergeDAREEnhanced,
            # LoRA Merge To Model
            LoRAMergeToModel,
            # LoRA Utilities
            AnimaLoraRename,
            MiniMaxH3DiffusersLoRAConvert,
            # Dtype Conversion
            DiffusionModelDtypeConversion,
            # Downloaders
            CheckpointInfoMetaDownloader, DiffusionModelInfoMetaDownloader, LoRAInfoMetaDownloader, EmbeddingInfoMetaDownloader,
            VAEInfoMetaDownloader, ControlNetInfoMetaDownloader, ManualPathInfoMetaDownloader,
            # Info Loaders
            CheckpointInfoLoader, LoRAInfoLoader, EmbeddingInfoLoader,
            VAEInfoLoader, ControlNetInfoLoader, DiffusionModelInfoLoader,
        ]


async def comfy_entrypoint() -> ModelUtilsExtension:
    return ModelUtilsExtension()

