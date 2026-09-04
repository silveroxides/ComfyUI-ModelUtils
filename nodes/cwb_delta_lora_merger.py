import fnmatch
import gc
import re
from dataclasses import dataclass, field

import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io
from tqdm import tqdm
from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned

from .artifact_paths import canonical_model_artifact_path
from .consensus_merger import (
    CWB_CONFIG,
    CWBSettings,
    CWBDiagnostics,
    DENSE_CWB_PRESETS,
    merge_cwb_tensors,
    resolve_cwb_settings,
)
from .device_utils import (
    cleanup_after_operation,
    estimate_model_size,
    prepare_for_large_operation,
)
from .lora_alpha import normalize_lora_pair
from .lora_resize import (
    build_lora_layer_map,
    canonical_lora_key,
    parse_lora_layers,
    select_output_dtype,
    validate_canonical_blocks,
)
from .quantization_guard import inspect_low_bit_input
from .uel_io import AsyncTensorCursor, atomic_uel_writer


_OPERATION = "Delta CWB LoRA Merge"
_ALLOWED_ROLES = {
    "down", "up", "alpha", "diff", "format", "down_suffix", "up_suffix",
    "alpha_suffix",
}
_FACTOR_SUFFIXES = (
    ".lora_A.default.weight", ".lora_B.default.weight",
    ".lora_down.weight", ".lora_up.weight",
    "_lora.down.weight", "_lora.up.weight",
    ".lora_A.weight", ".lora_B.weight",
    ".lora.down.weight", ".lora.up.weight",
    ".lora_A", ".lora_B",
    ".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight",
)
_UNSUPPORTED_SUFFIXES = (
    ".lora_mid.weight", ".reshape_weight", ".dora_scale", ".w_norm",
    ".b_norm", ".set_weight",
)


@dataclass(frozen=True)
class DeltaCWBLoRASource:
    source_index: int
    block_name: str
    down_key: str | None
    up_key: str | None
    alpha_key: str | None
    diff_key: str | None
    source_dtype: torch.dtype

    @property
    def keys(self):
        return tuple(
            key for key in (self.down_key, self.up_key, self.alpha_key, self.diff_key)
            if key is not None
        )


@dataclass(frozen=True)
class DeltaCWBLayerUnit:
    core: str
    output_key: str
    sources: tuple[DeltaCWBLoRASource, ...]
    zero_contributors: int
    preserve_anchor: bool


@dataclass
class DeltaCWBReport:
    merged_layers: int = 0
    anchor_preserved_layers: int = 0
    zero_filled_contributors: int = 0
    rejected_unsupported_layers: int = 0
    cuda_cpu_fallbacks: int = 0
    output_dtypes: dict[str, int] = field(default_factory=dict)

    def record_dtype(self, dtype):
        name = str(dtype).removeprefix("torch.")
        self.output_dtypes[name] = self.output_dtypes.get(name, 0) + 1

    def render(self, diagnostics: CWBDiagnostics, preset_name: str, input_count: int, custom: bool):
        dtype_summary = ", ".join(
            f"{name}={count}" for name, count in sorted(self.output_dtypes.items())
        ) or "none"
        source = "connected custom configuration" if custom else preset_name
        return "\n".join([
            "DELTA CWB LORA MERGE SUMMARY",
            f"Settings: {source}",
            f"Input contributors: {input_count}",
            f"Merged layers: {self.merged_layers}",
            f"Anchor-preserved layers: {self.anchor_preserved_layers}",
            f"Explicit zero contributors: {self.zero_filled_contributors}",
            f"Rejected/unsupported layers: {self.rejected_unsupported_layers}",
            f"Accepted weighting contributors: {diagnostics.accepted_weighting_contributors}/{diagnostics.weighting_contributors}",
            f"All-rejected equal fallbacks: {diagnostics.all_rejected_fallbacks}",
            f"Zero-weight equal fallbacks: {diagnostics.zero_weight_fallbacks}",
            f"CUDA OOM CPU fallbacks: {self.cuda_cpu_fallbacks}",
            f"Output dtypes: {dtype_summary}",
        ])


def _compile_patterns(patterns, glob_patterns):
    entries = [entry.strip() for entry in patterns.splitlines() if entry.strip()]
    if glob_patterns:
        return tuple(entries)
    return tuple(re.compile(entry) for entry in entries)


def _matches_pattern(key, patterns, glob_patterns):
    if glob_patterns:
        return any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)
    return any(pattern.search(key) is not None for pattern in patterns)


def _factor_delta(down, up, layer):
    if down.ndim == 2 and up.ndim == 2:
        if down.shape[0] != up.shape[1]:
            raise ValueError(f"Delta CWB rank mismatch for '{layer}'.")
        return up.float() @ down.float()
    if down.ndim == 4 and up.ndim == 4 and tuple(up.shape[2:]) == (1, 1):
        rank, channels, height, width = down.shape
        if up.shape[1] != rank:
            raise ValueError(f"Delta CWB rank mismatch for '{layer}'.")
        delta = up.reshape(up.shape[0], rank).float() @ down.reshape(rank, -1).float()
        return delta.reshape(up.shape[0], channels, height, width)
    raise ValueError(
        f"Delta CWB layer '{layer}' requires linear factors or convolutional "
        "factors with a 1x1 up kernel."
    )


def _inspect_inputs(handlers):
    infos = []
    for index, handler in enumerate(handlers):
        low_bit = inspect_low_bit_input(handler, f"LoRA {index + 1}", _OPERATION)
        keys = list(handler.keys())
        pairs, passthrough = parse_lora_layers(keys)
        unsupported_passthrough = [
            key for key in passthrough if key.endswith(_UNSUPPORTED_SUFFIXES)
        ]
        if unsupported_passthrough:
            raise ValueError(
                f"[{_OPERATION}] LoRA {index + 1} contains unsupported companion "
                f"tensor '{unsupported_passthrough[0]}'."
            )
        incomplete = [key for key in passthrough if key.endswith(_FACTOR_SUFFIXES)]
        if incomplete:
            raise ValueError(
                f"[{_OPERATION}] LoRA {index + 1} contains an incomplete factor "
                f"pair at '{incomplete[0]}'."
            )
        infos.append({"pairs": pairs, "passthrough": passthrough, "low_bit": low_bit})
    return infos


def _build_layer_units(
    layer_map,
    infos,
    handlers,
    mismatch_mode,
    exclude_patterns,
    discard_patterns,
    glob_patterns,
):
    units = []
    stream_keys = [[] for _ in handlers]
    rejected = 0
    for core, mapped in layer_map.items():
        if not any(index == 0 for index, _ in mapped):
            continue
        output_key = canonical_lora_key(core, "diff")
        if _matches_pattern(output_key, discard_patterns, glob_patterns):
            rejected += 1
            continue
        preserve = _matches_pattern(output_key, exclude_patterns, glob_patterns)
        candidates = []
        shapes = {}
        for index, block in sorted(mapped):
            roles = infos[index]["pairs"][block]
            unsupported = set(roles) - _ALLOWED_ROLES
            if unsupported:
                raise ValueError(
                    f"[{_OPERATION}] Layer '{block}' has unsupported roles: "
                    f"{', '.join(sorted(unsupported))}."
                )
            if "down" in roles or "up" in roles:
                if not all(role in roles for role in ("down", "up")):
                    raise ValueError(f"[{_OPERATION}] Layer '{block}' has an incomplete factor pair.")
                down_key, up_key = roles["down"], roles["up"]
                if down_key in infos[index]["low_bit"] or up_key in infos[index]["low_bit"]:
                    raise ValueError(f"[{_OPERATION}] Cannot expand low-bit factors for '{block}'.")
                down_shape = tuple(handlers[index].get_shape(down_key))
                up_shape = tuple(handlers[index].get_shape(up_key))
                if len(down_shape) == 2 and len(up_shape) == 2:
                    if down_shape[0] != up_shape[1]:
                        raise ValueError(f"[{_OPERATION}] Rank mismatch for '{block}'.")
                    dense_shape = (up_shape[0], down_shape[1])
                elif len(down_shape) == 4 and len(up_shape) == 4 and up_shape[2:] == (1, 1):
                    if down_shape[0] != up_shape[1]:
                        raise ValueError(f"[{_OPERATION}] Rank mismatch for '{block}'.")
                    dense_shape = (up_shape[0], *down_shape[1:])
                else:
                    raise ValueError(
                        f"[{_OPERATION}] Layer '{block}' requires linear factors or "
                        "convolutional factors with a 1x1 up kernel."
                    )
                factor_dtypes = (
                    handlers[index].get_dtype(down_key),
                    handlers[index].get_dtype(up_key),
                )
                source_dtype = torch.float32 if torch.float32 in factor_dtypes else factor_dtypes[1]
                source = DeltaCWBLoRASource(
                    index, block, down_key, up_key, roles.get("alpha"), None, source_dtype
                )
            elif "diff" in roles:
                diff_key = roles["diff"]
                if diff_key in infos[index]["low_bit"]:
                    raise ValueError(f"[{_OPERATION}] Cannot merge low-bit delta '{block}'.")
                dense_shape = tuple(handlers[index].get_shape(diff_key))
                source = DeltaCWBLoRASource(
                    index, block, None, None, None, diff_key,
                    handlers[index].get_dtype(diff_key),
                )
            else:
                raise ValueError(f"[{_OPERATION}] Layer '{block}' has no supported representation.")
            candidates.append(source)
            shapes[index] = dense_shape

        anchor = next((source for source in candidates if source.source_index == 0), None)
        if anchor is None:
            continue
        anchor_shape = shapes[0]
        compatible = [
            source for source in candidates if shapes[source.source_index] == anchor_shape
        ]
        present = {source.source_index for source in compatible}
        mismatch_count = len(handlers) - len(present)
        if preserve:
            sources = (anchor,)
            zero_count = 0
        elif mismatch_count and mismatch_mode == "error":
            raise ValueError(
                f"[{_OPERATION}] Layer '{core}' is missing or incompatible in one or more inputs."
            )
        elif mismatch_count and mismatch_mode == "skip":
            sources = (anchor,)
            zero_count = 0
            preserve = True
        else:
            sources = tuple(compatible)
            zero_count = mismatch_count if mismatch_mode == "zeros" else 0

        for source in sources:
            stream_keys[source.source_index].extend(source.keys)
        units.append(DeltaCWBLayerUnit(core, output_key, sources, zero_count, preserve))
    return units, stream_keys, rejected


def _to_processing_device(tensor, device):
    if device.type == "cuda":
        return transfer_to_gpu_pinned(tensor, device, torch.float32)
    return tensor.to(device=device, dtype=torch.float32)


def _process_layer_on_device(
    unit, loaded, settings: CWBSettings, save_dtype, device, diagnostics
):
    deltas = []
    try:
        for source in unit.sources:
            if source.diff_key is not None:
                delta = _to_processing_device(
                    loaded[(source.source_index, source.diff_key)], device
                )
            else:
                down = loaded[(source.source_index, source.down_key)]
                up = loaded[(source.source_index, source.up_key)]
                alpha = (
                    loaded[(source.source_index, source.alpha_key)]
                    if source.alpha_key is not None else None
                )
                down, up = normalize_lora_pair(down, up, alpha, layer=source.block_name)
                delta = _factor_delta(
                    _to_processing_device(down, device),
                    _to_processing_device(up, device),
                    source.block_name,
                )
            deltas.append(delta)
        if unit.zero_contributors:
            deltas.extend(torch.zeros_like(deltas[0]) for _ in range(unit.zero_contributors))
        if unit.preserve_anchor:
            merged = deltas[0].clone()
        else:
            merged = merge_cwb_tensors(
                deltas,
                settings,
                reference_index=0,
                allow_similarity_alignment=False,
                diagnostics=diagnostics,
            )
        dtype = select_output_dtype(
            [source.source_dtype for source in unit.sources],
            save_dtype,
            is_1d_diff=merged.ndim == 1,
        )
        return merged.to(dtype=dtype, device="cpu").contiguous(), dtype
    finally:
        deltas.clear()


def _is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _release_cuda_oom():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _process_layer(unit, loaded, settings, save_dtype, requested_device, diagnostics):
    device = torch.device(requested_device)
    attempt_diagnostics = CWBDiagnostics()
    try:
        output, dtype = _process_layer_on_device(
            unit, loaded, settings, save_dtype, device, attempt_diagnostics
        )
        diagnostics.absorb_runtime(attempt_diagnostics)
        return output, dtype, False
    except Exception as error:
        if device.type != "cuda" or not _is_cuda_oom(error):
            raise
        _release_cuda_oom()
        retry_diagnostics = CWBDiagnostics()
        output, dtype = _process_layer_on_device(
            unit, loaded, settings, save_dtype, torch.device("cpu"), retry_diagnostics
        )
        diagnostics.absorb_runtime(retry_diagnostics)
        return output, dtype, True


class DeltaCWBLoRAMergerLogic:
    @staticmethod
    def execute(
        lora_paths,
        source_names,
        settings,
        preset_name,
        custom_settings,
        mismatch_mode,
        output_filename,
        save_dtype,
        process_device,
        exclude_patterns,
        discard_patterns,
        glob_patterns,
        force_clear_cache,
    ):
        prepare_for_large_operation(
            sum(estimate_model_size(path) for path in lora_paths) * 2.5,
            torch.device(process_device),
        )
        handlers = []
        cursors = []
        report = DeltaCWBReport()
        diagnostics = CWBDiagnostics()
        try:
            for path in lora_paths:
                handlers.append(MemoryEfficientSafeOpen(path, low_memory=True))
            infos = _inspect_inputs(handlers)
            layer_map = build_lora_layer_map(infos, None)
            validate_canonical_blocks(layer_map, _OPERATION)
            compiled_excludes = _compile_patterns(exclude_patterns, glob_patterns)
            compiled_discards = _compile_patterns(discard_patterns, glob_patterns)
            units, stream_keys, rejected = _build_layer_units(
                layer_map,
                infos,
                handlers,
                mismatch_mode,
                compiled_excludes,
                compiled_discards,
                glob_patterns,
            )
            report.rejected_unsupported_layers = rejected
            cursors = [
                AsyncTensorCursor(
                    handler,
                    keys,
                    pin_memory=torch.device(process_device).type == "cuda",
                )
                for handler, keys in zip(handlers, stream_keys)
            ]
            output_path, output_name = canonical_model_artifact_path("loras", output_filename)
            metadata = (handlers[0].metadata() or {}).copy()
            metadata.update({
                "merge_method": "delta_cwb",
                "source_models": ",".join(source_names),
                "alpha_normalized": "true",
                "output_representation": "full_direct_difference",
                "cwb_consensus_type": settings.consensus_type,
                "cwb_alignment_method": settings.alignment_method,
                "cwb_alignment_threshold": str(settings.alignment_threshold),
                "cwb_similarity_threshold": str(settings.similarity_threshold),
                "cwb_power_alpha": str(settings.power_alpha),
                "cwb_diversity_beta": str(settings.diversity_beta),
                "cwb_rescale_norm": str(settings.rescale_norm).lower(),
                "cwb_global_scale": str(settings.global_scale),
                "cwb_dynamic_similarity_contrast": str(settings.dynamic_similarity_contrast).lower(),
                "cwb_soft_comfort_bandpass": str(settings.soft_comfort_bandpass).lower(),
                "cwb_position_weight": str(settings.position_weight),
                "cwb_preserve_common_prefix": str(settings.preserve_common_prefix).lower(),
            })
            pbar = comfy.utils.ProgressBar(len(units))
            with atomic_uel_writer(output_path, metadata) as writer, torch.no_grad():
                for unit in tqdm(units, desc="Delta CWB LoRA layers", unit="layers"):
                    if force_clear_cache:
                        _release_cuda_oom()
                    loaded = {}
                    try:
                        for source in unit.sources:
                            for key in source.keys:
                                loaded[(source.source_index, key)] = cursors[
                                    source.source_index
                                ].take(key)
                        output, dtype, used_cpu = _process_layer(
                            unit,
                            loaded,
                            settings,
                            save_dtype,
                            process_device,
                            diagnostics,
                        )
                        writer.write_batch([(unit.output_key, output)])
                        report.merged_layers += int(not unit.preserve_anchor)
                        report.anchor_preserved_layers += int(unit.preserve_anchor)
                        report.zero_filled_contributors += unit.zero_contributors
                        report.cuda_cpu_fallbacks += int(used_cpu)
                        report.record_dtype(dtype)
                        del output
                    finally:
                        for source_index, key in list(loaded):
                            del loaded[(source_index, key)]
                            cursors[source_index].release(key)
                        loaded.clear()
                    pbar.update(1)
            for cursor in cursors:
                cursor.finish()
            return output_name, report.render(
                diagnostics, preset_name, len(lora_paths), custom_settings
            )
        finally:
            for cursor in cursors:
                cursor.close()
            for handler in handlers:
                handler.__exit__(None, None, None)
            cleanup_after_operation()

    @staticmethod
    def execute_node(kwargs, count):
        names = [kwargs[f"lora_{index}"] for index in range(1, count + 1)]
        paths = [folder_paths.get_full_path_or_raise("loras", name) for name in names]
        settings = resolve_cwb_settings(
            kwargs["cwb_preset"], DENSE_CWB_PRESETS, kwargs.get("cwb_config")
        )
        requested_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[kwargs["save_dtype"]]
        filename, report = DeltaCWBLoRAMergerLogic.execute(
            paths,
            names,
            settings,
            kwargs["cwb_preset"],
            kwargs.get("cwb_config") is not None,
            kwargs["mismatch_mode"],
            kwargs["output_filename"],
            requested_dtype,
            kwargs["process_device"],
            kwargs["exclude_patterns"],
            kwargs["discard_patterns"],
            kwargs["glob_patterns"],
            kwargs["force_clear_cache"],
        )
        return io.NodeOutput(filename, report)


def _fixed_inputs(count, default_filename):
    options = folder_paths.get_filename_list("loras")
    inputs = []
    if count:
        inputs.append(io.Combo.Input(
            "lora_1",
            options=options,
            tooltip="Metadata and layer anchor. Defines saved layers and preserved values.",
        ))
    inputs.extend(
        io.Combo.Input(
            f"lora_{index}",
            options=options,
            tooltip="Equal-prior LoRA contributor. CWB determines its effective row influence.",
        )
        for index in range(2, count + 1)
    )
    inputs.extend([
        io.Combo.Input(
            "cwb_preset",
            options=list(DENSE_CWB_PRESETS),
            default="balanced_mean",
            tooltip="Controls how strongly agreement, disagreement, and row magnitude affect the saved full weight changes. A connected CWB Config overrides it completely.",
        ),
        CWB_CONFIG.Input(
            "cwb_config",
            optional=True,
            tooltip="Optional complete settings override from CWB Custom Configuration.",
        ),
        io.Combo.Input(
            "mismatch_mode",
            options=["skip", "zeros", "error"],
            default="skip",
            tooltip="For a missing or incompatible anchored layer: preserve LoRA 1, include an explicit zero contributor, or abort.",
        ),
        io.String.Input(
            "output_filename",
            default=default_filename,
            tooltip="Relative filename for the atomically saved full-difference LoRA.",
        ),
        io.Combo.Input(
            "save_dtype",
            options=["fp32", "fp16", "bf16"],
            default="bf16",
            tooltip="Requested saved precision. Participating FP32 inputs preserve FP32.",
        ),
        io.Combo.Input(
            "process_device",
            options=["cuda", "cpu"],
            default="cuda",
            tooltip="Processing device. Only a failed CUDA layer retries on CPU.",
        ),
        io.String.Input(
            "exclude_patterns",
            default="",
            multiline=True,
            tooltip="Preserve matching canonical full-difference layers from LoRA 1.",
        ),
        io.String.Input(
            "discard_patterns",
            default="",
            multiline=True,
            tooltip="Omit matching canonical full-difference layers from the output.",
        ),
        io.Boolean.Input(
            "glob_patterns",
            default=False,
            tooltip="Use shell-style glob patterns instead of regular expressions.",
        ),
        io.Boolean.Input(
            "force_clear_cache",
            default=True,
            tooltip="Clear Python and CUDA caches before each layer to reduce retained memory.",
        ),
    ])
    return inputs


def _multi_inputs(default_filename):
    options = folder_paths.get_filename_list("loras")
    optional = ["None", *options]
    inputs = [
        io.Combo.Input(
            "lora_count",
            options=[str(index) for index in range(2, 9)],
            default="2",
            tooltip="Number of consecutive selected LoRAs to merge.",
        ),
        io.Combo.Input(
            "lora_1",
            options=options,
            tooltip="Metadata and layer anchor. Defines saved layers and preserved values.",
        ),
        io.Combo.Input(
            "lora_2",
            options=options,
            tooltip="Second equal-prior LoRA contributor.",
        ),
    ]
    inputs.extend(
        io.Combo.Input(
            f"lora_{index}",
            options=optional,
            default="None",
            tooltip="Additional equal-prior contributor; required when included by LoRA Count.",
        )
        for index in range(3, 9)
    )
    inputs.extend(_fixed_inputs(0, default_filename))
    return inputs


class DeltaCWBLoRATwoMerger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DeltaCWBLoRATwoMerger",
            display_name="Delta CWB Merge LoRAs (2)",
            category="ModelUtils/LoRA/Merge/Delta CWB",
            description="Expand two LoRAs to full weight changes and merge corresponding rows with CWB.",
            inputs=_fixed_inputs(2, "delta_cwb_merged_2_lora"),
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="cwb_report"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return DeltaCWBLoRAMergerLogic.execute_node(kwargs, 2)


class DeltaCWBLoRAThreeMerger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DeltaCWBLoRAThreeMerger",
            display_name="Delta CWB Merge LoRAs (3)",
            category="ModelUtils/LoRA/Merge/Delta CWB",
            description="Expand three LoRAs to full weight changes and merge corresponding rows with CWB.",
            inputs=_fixed_inputs(3, "delta_cwb_merged_3_lora"),
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="cwb_report"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return DeltaCWBLoRAMergerLogic.execute_node(kwargs, 3)


class DeltaCWBLoRAMultiMerger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DeltaCWBLoRAMultiMerger",
            display_name="Delta CWB LoRA Multi-Merge",
            category="ModelUtils/LoRA/Merge/Delta CWB",
            description="Expand 2-8 LoRAs to full weight changes and merge corresponding rows with CWB.",
            inputs=_multi_inputs("delta_cwb_merged_multi_lora"),
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="cwb_report"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        count = int(kwargs["lora_count"])
        missing = [
            index for index in range(1, count + 1)
            if not kwargs[f"lora_{index}"] or kwargs[f"lora_{index}"] == "None"
        ]
        if missing:
            raise ValueError(
                f"LoRA Count includes unselected input(s): {', '.join(map(str, missing))}."
            )
        return DeltaCWBLoRAMergerLogic.execute_node(kwargs, count)


DELTA_CWB_LORA_NODES = [
    DeltaCWBLoRATwoMerger,
    DeltaCWBLoRAThreeMerger,
    DeltaCWBLoRAMultiMerger,
]
