"""
LoRA Resize - Resize existing LoRAs in factor space.

Uses bounded asynchronous UEL streaming and rank-space SVD for fixed rank and
dynamic methods (sv_ratio, sv_fro, sv_cumulative).
"""
import os
import gc
import fnmatch
import re
import logging
from collections import Counter, defaultdict
import torch
import folder_paths
import comfy.utils
from comfy.weight_adapter import LoRAAdapter
from tqdm import tqdm
from comfy_api.latest import io
from .device_utils import estimate_model_size, prepare_for_large_operation, cleanup_after_operation

from unifiedefficientloader import MemoryEfficientSafeOpen, transfer_to_gpu_pinned
from .uel_io import atomic_uel_writer, stream_work_units
from .artifact_paths import canonical_model_artifact_path
from .lora_alpha import lora_alpha_scale, normalize_lora_pair
from .layer_parameters import parameter_input, resolve_layer_parameters
from .quantization_guard import DiffusionQuantization, inspect_low_bit_input, layer_has_low_bit, write_preserved_tensor
from typing import Optional, Dict, Tuple, List



# Reuse rank functions from extraction module
from .lora_extract_svd import (
    _index_sv_cumulative,
    _index_sv_fro,
    _compile_patterns,
)


MIN_SV = 1e-6


class AdapterErrorCapture(logging.Handler):
    """Observe errors ComfyUI logs and swallows for one adapter target."""

    def __init__(self, target_key):
        super().__init__(level=logging.ERROR)
        self.target_key = target_key
        self.messages = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("ERROR ") and self.target_key in message:
            self.messages.append(message)

LORA_PAIR_SUFFIXES = (
    (".lora_A.default.weight", ".lora_B.default.weight", ".alpha", "peft"),
    (".lora_down.weight", ".lora_up.weight", ".alpha", "comfy"),
    ("_lora.down.weight", "_lora.up.weight", ".alpha", "diffusers_legacy"),
    (".lora_A.weight", ".lora_B.weight", ".alpha", "diffusers"),
    (".lora.down.weight", ".lora.up.weight", ".alpha", "huggingface"),
    (".lora_A", ".lora_B", ".alpha", "mochi"),
    (".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight", ".alpha", "transformers"),
)

LORA_AUX_SUFFIXES = (
    (".lora_mid.weight", "mid"),
    (".reshape_weight", "reshape"),
    (".dora_scale", "dora_scale"),
    (".w_norm", "w_norm"),
    (".b_norm", "b_norm"),
    (".set_weight", "set_weight"),
)

CANONICAL_DOWN_SUFFIX = ".lora_A.weight"
CANONICAL_UP_SUFFIX = ".lora_B.weight"

BASE_MODEL_KEY_PREFIXES = (
    "model.diffusion_model.", "diffusion_model.", "transformer.", "model.", "net.",
)
LORA_BLOCK_PREFIXES = (
    "lora_unet_", "lora_transformer_", "lora_te1_", "lora_te2_", "lora_te_",
    "lycoris_", "diffusion_model.", "transformer.", "unet.",
)


def canonical_lora_block_name(block_name: str) -> str:
    """Return the preferred generic ComfyUI diffusion-model LoRA root."""
    root = block_name
    wrappers = (
        "unet.base_model.model.",
        "base_model.model.",
        "model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in wrappers:
            if root.startswith(prefix):
                root = root[len(prefix):]
                changed = True
                break

    dotted_prefixes = (
        "diffusion_model.",
        "transformer.",
        "unet.",
        "net.",
    )
    for prefix in dotted_prefixes:
        if root.startswith(prefix):
            return f"diffusion_model.{root[len(prefix):]}"

    # Flattened trainer aliases cannot be expanded to dotted model keys safely
    # without a reference model. Keep those roots loadable rather than guessing.
    if root.startswith(("lora_unet_", "lora_transformer_", "lycoris_", "lora_te")):
        return root
    return f"diffusion_model.{root}"


def canonical_lora_key(block_name: str, role: str) -> str:
    root = canonical_lora_block_name(block_name)
    suffixes = {
        "down": CANONICAL_DOWN_SUFFIX,
        "up": CANONICAL_UP_SUFFIX,
        "alpha": ".alpha",
        "mid": ".lora_mid.weight",
        "reshape": ".reshape_weight",
        "dora_scale": ".dora_scale",
        "diff": ".diff",
        "diff_b": ".diff_b",
        "w_norm": ".w_norm",
        "b_norm": ".b_norm",
        "set_weight": ".set_weight",
    }
    return f"{root}{suffixes[role]}"


def _strip_lora_prefix(value: str, prefixes) -> str:
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def build_lora_layer_map(
    lora_infos: List[Dict], base_model_path: Optional[str]
) -> Dict[str, List[Tuple[int, str]]]:
    """Map parsed LoRA blocks to a reference model using existing merge rules."""
    if not base_model_path:
        layer_map = {}
        for idx, info in enumerate(lora_infos):
            for block_name in info["pairs"]:
                layer_map.setdefault(block_name, []).append((idx, block_name))
        return layer_map

    with MemoryEfficientSafeOpen(base_model_path, low_memory=True) as base_handler:
        base_keys = list(base_handler.keys())

    normalized_base = {}
    for base_key in base_keys:
        normalized = _strip_lora_prefix(base_key, BASE_MODEL_KEY_PREFIXES)
        normalized_base[normalized] = base_key
        normalized_base[normalized.replace(".", "_")] = base_key

    layer_map = {}
    mapped = set()
    for idx, info in enumerate(lora_infos):
        for block_name, block_keys in info["pairs"].items():
            core = _strip_lora_prefix(block_name, LORA_BLOCK_PREFIXES)
            if core.endswith(".lora"):
                core = core[:-5]
            target = None
            if "down" in block_keys or "up" in block_keys:
                candidate = f"{core}.weight"
                target = normalized_base.get(candidate) or normalized_base.get(candidate.replace(".", "_"))
            elif "diff_b" in block_keys or "b_norm" in block_keys:
                candidate = f"{core}.bias"
                target = normalized_base.get(candidate) or normalized_base.get(candidate.replace(".", "_"))
            elif any(role in block_keys for role in ("diff", "w_norm", "set_weight")):
                target = normalized_base.get(core) or normalized_base.get(core.replace(".", "_"))
                if target is None:
                    candidate = f"{core}.weight"
                    target = normalized_base.get(candidate) or normalized_base.get(candidate.replace(".", "_"))

            if target is None:
                continue
            normalized_target = _strip_lora_prefix(target, BASE_MODEL_KEY_PREFIXES)
            if ("diff_b" in block_keys or "b_norm" in block_keys) and normalized_target.endswith(".bias"):
                output_core = normalized_target[:-5]
            elif any(role in block_keys for role in ("diff", "w_norm", "set_weight")) and normalized_target.endswith(".weight"):
                output_core = normalized_target[:-7]
            elif any(role in block_keys for role in ("diff", "w_norm", "set_weight")):
                output_core = normalized_target
            else:
                output_core = normalized_target[:-7] if normalized_target.endswith(".weight") else normalized_target
            output_core = f"diffusion_model.{output_core}"
            layer_map.setdefault(output_core, []).append((idx, block_name))
            mapped.add((idx, block_name))

    for idx, info in enumerate(lora_infos):
        for block_name, block_keys in info["pairs"].items():
            if (idx, block_name) not in mapped and any(
                role in block_keys for role in ("diff", "diff_b", "w_norm", "b_norm", "set_weight")
            ):
                layer_map.setdefault(block_name, []).append((idx, block_name))
    return layer_map


def layer_tensor_keys(layer: Dict[str, str]) -> Dict[str, str]:
    roles = {
        "down", "up", "alpha", "mid", "reshape", "dora_scale",
        "diff", "diff_b", "w_norm", "b_norm", "set_weight",
    }
    return {role: key for role, key in layer.items() if role in roles}


def layer_has_companions(layer: Dict[str, str]) -> bool:
    return any(role in layer for role in ("mid", "reshape", "dora_scale", "set_weight"))


def validate_canonical_blocks(block_names, operation: str) -> None:
    seen = {}
    for block_name in block_names:
        canonical = canonical_lora_block_name(block_name)
        previous = seen.get(canonical)
        if previous is not None and previous != block_name:
            raise ValueError(
                f"[{operation}] LoRA keys '{previous}' and '{block_name}' both normalize to '{canonical}'"
            )
        seen[canonical] = block_name


def is_direct_diff_key(key: str) -> bool:
    return key.endswith((".diff", ".diff_b", ".w_norm", ".b_norm"))


def select_output_dtype(
    source_dtypes: List[torch.dtype],
    requested_dtype: torch.dtype,
    force: bool = False,
    is_1d_diff: bool = False,
) -> torch.dtype:
    """Select an output dtype without losing precision-sensitive LoRA deltas."""
    if is_1d_diff:
        return torch.float32
    if not force and torch.float32 in source_dtypes:
        return torch.float32
    floating_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if source_dtypes and not any(dtype in floating_dtypes for dtype in source_dtypes):
        return source_dtypes[0]
    return requested_dtype


# =============================================================================
# LoRA Format Detection - Support various LoRA sources
# =============================================================================

def detect_lora_format(keys: List[str]) -> Dict:
    """
    Detect LoRA format and key patterns from various sources.

    Supports:
    - Kohya/A1111: lora_unet_*, lora_te_*
    - Diffusers: *.lora_A.weight, *.lora_B.weight
    - PEFT: base_model.model.*.lora_A/B
    - ComfyUI native: *.lora_down.weight, *.lora_up.weight
    - HuggingFace: *.lora.down.weight, *.lora.up.weight (dots instead of underscores)
    - Full diff: *.diff, *.diff_b (full weight differences, not low-rank)
    - Flux/SDXL: Various transformer key patterns

    Returns:
        Dict with format info and key patterns
    """
    format_info = {
        "format": "unknown",
        "down_suffix": ".lora_down.weight",
        "up_suffix": ".lora_up.weight",
        "alpha_suffix": ".alpha",
        "has_alpha": False,
        "key_count": 0,
        "is_full_diff": False,  # New flag for full diff format
    }

    # Check for alpha keys
    alpha_keys = [k for k in keys if '.alpha' in k or k.endswith('alpha')]
    format_info["has_alpha"] = len(alpha_keys) > 0

    detected = []
    for down_suffix, up_suffix, alpha_suffix, name in LORA_PAIR_SUFFIXES:
        if any(k.endswith(down_suffix) or k.endswith(up_suffix) for k in keys):
            detected.append((down_suffix, up_suffix, alpha_suffix, name))
    has_full_diff = any(is_direct_diff_key(k) for k in keys)
    if has_full_diff:
        detected.append((None, None, None, "full_diff"))

    if len(detected) > 1:
        format_info["format"] = "mixed"
    elif detected:
        down_suffix, up_suffix, alpha_suffix, name = detected[0]
        format_info.update({
            "format": name,
            "down_suffix": down_suffix,
            "up_suffix": up_suffix,
            "alpha_suffix": alpha_suffix,
        })
    format_info["is_full_diff"] = has_full_diff and not any(item[0] for item in detected)
    pairs, _ = parse_lora_layers(keys)
    format_info["key_count"] = len(pairs)

    return format_info



def parse_lora_layers(keys: List[str]) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Parse low-rank, direct-diff, and pass-through tensors in one full scan."""
    layers: Dict[str, Dict[str, str]] = {}
    consumed = set()

    for key in keys:
        if key.endswith(".diff_b"):
            block_name = key[:-7]
            layers.setdefault(block_name, {})["diff_b"] = key
            consumed.add(key)
            continue
        if key.endswith(".diff"):
            block_name = key[:-5]
            layers.setdefault(block_name, {})["diff"] = key
            consumed.add(key)
            continue
        matched_aux = False
        for suffix, role in LORA_AUX_SUFFIXES:
            if key.endswith(suffix):
                block_name = key[:-len(suffix)]
                layers.setdefault(block_name, {})[role] = key
                consumed.add(key)
                matched_aux = True
                break
        if matched_aux:
            continue
        for down_suffix, up_suffix, alpha_suffix, format_name in LORA_PAIR_SUFFIXES:
            if key.endswith(down_suffix):
                block_name = key[:-len(down_suffix)]
                layer = layers.setdefault(block_name, {})
                layer.update({
                    "down": key,
                    "down_suffix": down_suffix,
                    "up_suffix": up_suffix,
                    "alpha_suffix": alpha_suffix,
                    "format": format_name,
                })
                consumed.add(key)
                break
            if key.endswith(up_suffix):
                block_name = key[:-len(up_suffix)]
                layer = layers.setdefault(block_name, {})
                layer.update({
                    "up": key,
                    "down_suffix": down_suffix,
                    "up_suffix": up_suffix,
                    "alpha_suffix": alpha_suffix,
                    "format": format_name,
                })
                consumed.add(key)
                break

    for key in keys:
        if not key.endswith(".alpha"):
            continue
        block_name = key[:-6]
        if block_name in layers:
            layers[block_name]["alpha"] = key
            consumed.add(key)

    # Incomplete or auxiliary-only groups cannot be transformed, but all their
    # source tensors must survive as pass-through data.
    for block_name, layer in list(layers.items()):
        has_pair = "down" in layer and "up" in layer
        has_direct = any(
            role in layer
            for role in ("diff", "diff_b", "w_norm", "b_norm", "set_weight")
        )
        if has_pair or has_direct:
            continue
        for key in layer_tensor_keys(layer).values():
            consumed.discard(key)
        del layers[block_name]

    return layers, [key for key in keys if key not in consumed]


def extract_lora_pairs(keys: List[str], format_info: Optional[Dict] = None) -> Dict[str, Dict[str, str]]:
    """
    Group LoRA keys into down/up/alpha pairs.

    For full_diff format, groups .diff and .diff_b keys.

    Returns:
        Dict[block_name, {"down": key, "up": key, "alpha": key}]
        For full_diff: {"diff": key, "diff_b": key}
    """
    pairs, _ = parse_lora_layers(keys)
    return pairs


def detect_lora_rank(
    handler: MemoryEfficientSafeOpen,
    pairs: Dict,
    low_bit_keys: Optional[set[str]] = None,
) -> Tuple[int, float]:
    """
    Detect the rank and alpha of an existing LoRA.

    Returns:
        (network_dim, network_alpha)
    """
    network_dim = None
    network_alpha = None

    for block_name, block_keys in pairs.items():
        if "down" not in block_keys:
            continue

        # Get dim from down weight shape
        if network_dim is None:
            down_key = block_keys["down"]
            shape = handler.get_shape(down_key)
            # Linear: [rank, in_features] or Conv: [rank, in_ch, k, k]
            network_dim = shape[0]

        if network_dim is not None:
            break

    # Default alpha to dim if not found
    if network_alpha is None:
        network_alpha = float(network_dim) if network_dim else 1.0
    if network_dim is None:
        network_dim = 1

    return network_dim, network_alpha


# =============================================================================
# Factor-space resize engine
# =============================================================================

def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
        and "cuda" in str(error).lower()
    )


def _release_cuda_after_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _factor_matrices(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[int, int, int]]]:
    """Return down/up matrices and optional convolution output layout."""
    if lora_down.ndim == 2:
        if lora_up.ndim != 2:
            raise ValueError("Linear LoRA up/down tensors must both be two-dimensional")
        if lora_down.shape[0] != lora_up.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch: down={tuple(lora_down.shape)}, "
                f"up={tuple(lora_up.shape)}"
            )
        return lora_down, lora_up, None

    if lora_down.ndim != 4 or lora_up.ndim != 4:
        raise ValueError("LoRA resize supports linear and convolutional factor pairs")
    if tuple(lora_up.shape[2:]) != (1, 1):
        raise ValueError("Convolutional LoRA up tensor must use a 1x1 kernel")
    rank, in_channels, kernel_h, kernel_w = lora_down.shape
    if lora_up.shape[1] != rank:
        raise ValueError(
            f"LoRA rank mismatch: down={tuple(lora_down.shape)}, "
            f"up={tuple(lora_up.shape)}"
        )
    return (
        lora_down.reshape(rank, in_channels * kernel_h * kernel_w),
        lora_up.reshape(lora_up.shape[0], rank),
        (in_channels, kernel_h, kernel_w),
    )


def _resize_factors_on_device(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    max_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    scale: float,
    min_rank: int,
    process_device: str,
) -> Dict:
    """Resize a LoRA pair using only its rank-space core."""
    down_matrix, up_matrix, conv_layout = _factor_matrices(lora_down, lora_up)
    if str(process_device).startswith("cuda"):
        down_device = transfer_to_gpu_pinned(
            down_matrix, process_device, torch.float32
        )
        up_device = transfer_to_gpu_pinned(
            up_matrix, process_device, torch.float32
        )
    else:
        down_device = down_matrix.to(device=process_device, dtype=torch.float32)
        up_device = up_matrix.to(device=process_device, dtype=torch.float32)

    q_up, r_up = torch.linalg.qr(up_device, mode="reduced")
    q_down, r_down = torch.linalg.qr(down_device.T, mode="reduced")
    core = r_up @ r_down.T
    core_u, singular_values, core_vh = torch.linalg.svd(
        core, full_matrices=False
    )
    new_rank, new_alpha, stats = _compute_resize(
        singular_values,
        max_rank,
        dynamic_method,
        dynamic_param,
        scale,
        min_rank,
    )

    new_up = (q_up @ core_u[:, :new_rank]) * singular_values[:new_rank]
    new_down = core_vh[:new_rank, :] @ q_down.T

    if conv_layout is not None:
        in_channels, kernel_h, kernel_w = conv_layout
        new_up = new_up.reshape(new_up.shape[0], new_rank, 1, 1)
        new_down = new_down.reshape(
            new_rank, in_channels, kernel_h, kernel_w
        )

    result = {
        "lora_down": new_down.cpu().contiguous(),
        "lora_up": new_up.cpu().contiguous(),
        "new_rank": new_rank,
        "new_alpha": new_alpha,
        **stats,
    }
    del (
        down_device,
        up_device,
        q_up,
        r_up,
        q_down,
        r_down,
        core,
        core_u,
        singular_values,
        core_vh,
        new_up,
        new_down,
    )
    return result


def _resize_lora_factors(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    max_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    scale: float,
    min_rank: int = 1,
    process_device: str = "cpu",
) -> Dict:
    """Resize one pair with per-layer CUDA OOM fallback."""
    try:
        result = _resize_factors_on_device(
            lora_down,
            lora_up,
            max_rank,
            dynamic_method,
            dynamic_param,
            scale,
            min_rank,
            process_device,
        )
        result["cpu_fallback"] = False
        return result
    except BaseException as error:
        if not str(process_device).startswith("cuda") or not _is_cuda_oom(error):
            raise
        error.__traceback__ = None
        _release_cuda_after_oom()
        try:
            result = _resize_factors_on_device(
                lora_down,
                lora_up,
                max_rank,
                dynamic_method,
                dynamic_param,
                scale,
                min_rank,
                "cpu",
            )
        except BaseException as cpu_error:
            raise RuntimeError(
                f"CPU fallback failed while resizing LoRA layer: {cpu_error}"
            ) from error
        result["cpu_fallback"] = True
        return result


def _compute_resize(
    S: torch.Tensor,
    max_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    scale: float,
    min_rank: int = 1,
) -> Tuple[int, float, Dict]:
    """Compute new rank and alpha based on resize method."""

    if dynamic_method == "sv_ratio" and dynamic_param is not None:
        # Use kohya convention: S[0]/ratio
        min_sv = S[0] / dynamic_param
        new_rank = max(1, int(torch.sum(S > min_sv).item()))
    elif dynamic_method == "sv_cumulative" and dynamic_param is not None:
        new_rank = _index_sv_cumulative(S, dynamic_param, max_rank)
    elif dynamic_method == "sv_fro" and dynamic_param is not None:
        new_rank = _index_sv_fro(S, dynamic_param, max_rank)
    else:
        new_rank = max_rank

    available_rank = max(1, len(S))
    effective_min_rank = min(max(1, min_rank), max_rank, available_rank)
    if S[0] < MIN_SV:
        new_rank = effective_min_rank
    else:
        new_rank = max(
            effective_min_rank,
            min(new_rank, max_rank, available_rank),
        )

    new_alpha = float(scale * new_rank)

    # Compute retention stats
    s_sum = float(torch.sum(torch.abs(S)))
    s_rank = float(torch.sum(torch.abs(S[:new_rank]))) if new_rank <= len(S) else s_sum

    S_sq = S.pow(2)
    s_fro = float(torch.sqrt(torch.sum(S_sq)))
    s_red_fro = float(torch.sqrt(torch.sum(S_sq[:new_rank]))) if new_rank <= len(S) else s_fro

    stats = {
        "sum_retained": s_rank / s_sum if s_sum > 0 else 1.0,
        "fro_retained": s_red_fro / s_fro if s_fro > 0 else 1.0,
    }

    return new_rank, new_alpha, stats


def _resize_filter_inputs():
    return [
        io.String.Input("exclude_patterns", default="", multiline=True, tooltip="One pattern per line. Matching layer names or source/canonical tensor keys keep the whole layer at its original rank and precision, including alpha."),
        io.String.Input("discard_patterns", default="", multiline=True, tooltip="One pattern per line. Remove the whole matching layer, including factors and alpha. Discard takes precedence over exclude."),
        io.Boolean.Input("glob_patterns", default=False, tooltip="Use shell-style glob patterns instead of regular expressions for exclude and discard filters."),
        io.Boolean.Input("include_mode", default=False, tooltip="Treat exclude patterns as an include list. Matching layers are resized; nonmatching layers are preserved unchanged. Discard patterns still remove matching layers."),
    ]


def _build_resize_work_units(
    handler, pairs, passthrough_keys, low_bit_keys,
    exclude_patterns="", discard_patterns="", glob_patterns=False,
    include_mode=False, layer_parameters=None,
):
    """Plan raw-copy and streamed work without loading tensor payloads."""
    raw_units = []
    stream_units = []
    claimed = set()
    rank_counts = Counter()
    preserved_companion_groups = 0
    excludes, discards = [
        [re.compile(fnmatch.translate(line.strip()) if glob_patterns else line.strip())
         for line in text.splitlines() if line.strip()]
        for text in (exclude_patterns, discard_patterns)
    ]

    def matches(patterns, names):
        return any(pattern.search(name) for pattern in patterns for name in names)

    for block_name, block_keys in pairs.items():
        tensor_keys = layer_tensor_keys(block_keys)
        if "down" in block_keys:
            rank_counts[int(handler.get_shape(block_keys["down"])[0])] += 1

        entries = [
            (role, key, canonical_lora_key(block_name, role))
            for role, key in tensor_keys.items()
        ]
        related_passthrough = [
            (None, key, key)
            for key in passthrough_keys
            if key.startswith(f"{block_name}.") and key not in claimed
        ]

        group_entries = entries + related_passthrough
        names = [block_name, canonical_lora_block_name(block_name)]
        names.extend(name for _, key, output_key in group_entries for name in (key, output_key))
        if matches(discards, names):
            claimed.update(key for _, key, _ in group_entries)
            continue
        matched = matches(excludes, names)
        preserve = matched != include_mode

        if layer_has_low_bit(block_keys, low_bit_keys):
            raw_entries = entries + related_passthrough
            raw_units.append({"entries": raw_entries})
            claimed.update(key for _, key, _ in raw_entries)
            continue

        if preserve:
            stream_units.append({"kind": "preserve", "entries": group_entries})
            claimed.update(key for _, key, _ in group_entries)
        elif layer_has_companions(block_keys):
            preserved_companion_groups += 1
            stream_units.append({"kind": "preserve", "entries": entries})
        elif "down" in block_keys and "up" in block_keys:
            stream_units.append({
                "kind": "resize",
                "entries": entries,
                "resize_parameters": (layer_parameters or {}).get(
                    canonical_lora_block_name(block_name)
                ),
                "alpha_output": (
                    canonical_lora_key(block_name, "alpha")
                    if "alpha" in block_keys
                    else None
                ),
            })
        else:
            stream_units.append({"kind": "copy", "entries": entries})
        claimed.update(key for _, key, _ in entries)

    for key in passthrough_keys:
        if key in claimed:
            continue
        if matches(discards, [key]):
            continue
        entry = (None, key, key)
        matched = matches(excludes, [key])
        if key in low_bit_keys:
            raw_units.append({"entries": [entry]})
        else:
            stream_units.append({
                "kind": "preserve" if matched != include_mode else "copy",
                "entries": [entry],
            })
        claimed.add(key)

    return raw_units, stream_units, rank_counts, preserved_companion_groups


def _rank_summary(rank_counts: Counter) -> str:
    if not rank_counts:
        return "none"
    return ", ".join(
        f"{rank}x{count}" for rank, count in sorted(rank_counts.items())
    )


# =============================================================================
# Main Resize Function
# =============================================================================

def normalize_lora_alpha_file(
    lora_path: str,
    output_filename: str,
    verbose: bool = True,
    reference_model_path: Optional[str] = None,
) -> str:
    """Materialize LoRA alpha scaling and remove alpha tensors."""
    handler = MemoryEfficientSafeOpen(lora_path, low_memory=True)
    try:
        low_bit_keys = inspect_low_bit_input(
            handler, f"LoRA ({lora_path})", "LoRA Alpha Normalize"
        )
        metadata = (handler.metadata() or {}).copy()
        all_keys = handler.keys()
        pairs, _ = parse_lora_layers(all_keys)
        output_keys = {}
        mapped_blocks = {}
        if reference_model_path:
            layer_map = build_lora_layer_map(
                [{"pairs": pairs}], reference_model_path
            )
            validate_canonical_blocks(layer_map, "LoRA Alpha Normalize")
            mapped_blocks = {
                source_block: output_block
                for output_block, sources in layer_map.items()
                for _, source_block in sources
            }
            for block_name, output_block in mapped_blocks.items():
                for role, source_key in layer_tensor_keys(pairs[block_name]).items():
                    output_keys[source_key] = canonical_lora_key(output_block, role)

        work_units = []
        normalized_up_keys = set()
        consumed_alpha_keys = set()
        for block_name, block_keys in pairs.items():
            alpha_key = block_keys.get("alpha")
            if alpha_key is None:
                continue
            down_key = block_keys.get("down")
            up_key = block_keys.get("up")
            if down_key is None or up_key is None:
                raise ValueError(
                    f"Alpha tensor for '{block_name}' has no complete LoRA pair."
                )
            if down_key in low_bit_keys or up_key in low_bit_keys:
                raise ValueError(
                    f"Cannot alpha-normalize low-bit LoRA factors for '{block_name}'."
                )
            down_shape = handler.get_shape(down_key)
            up_shape = handler.get_shape(up_key)
            if len(down_shape) < 2 or len(up_shape) < 2:
                raise ValueError(
                    f"LoRA factors for '{block_name}' must have at least two dimensions."
                )
            rank = int(down_shape[0])
            if rank <= 0 or int(up_shape[1]) != rank:
                raise ValueError(f"LoRA factor rank mismatch for '{block_name}'.")
            work_units.append((block_name, {0: [up_key, alpha_key]}))
            normalized_up_keys.add(up_key)
            consumed_alpha_keys.add(alpha_key)

        all_alpha_keys = {key for key in all_keys if key.endswith(".alpha")}
        orphan_alpha_keys = sorted(all_alpha_keys - consumed_alpha_keys)
        if orphan_alpha_keys:
            raise ValueError(
                f"Alpha tensor has no complete LoRA pair: {orphan_alpha_keys[0]}"
            )
        if not work_units:
            raise ValueError("Input LoRA contains no alpha tensors.")

        metadata["alpha_normalized"] = "true"
        metadata["alpha_normalization"] = (
            "up factor multiplied by alpha divided by rank; alpha tensors removed"
        )
        output_path, _ = canonical_model_artifact_path("loras", output_filename)

        with atomic_uel_writer(output_path, metadata) as writer:
            for key in all_keys:
                if key in all_alpha_keys or key in normalized_up_keys:
                    continue
                write_preserved_tensor(
                    writer,
                    key,
                    handler,
                    output_key=output_keys.get(key),
                    force_raw=True,
                )

            with torch.no_grad():
                iterator = stream_work_units(
                    {0: handler}, work_units, pin_memory=False
                )
                try:
                    for block_name, loaded in iterator:
                        block_keys = pairs[block_name]
                        up_key = block_keys["up"]
                        alpha_key = block_keys["alpha"]
                        rank = int(handler.get_shape(block_keys["down"])[0])
                        up = loaded[(0, up_key)]
                        alpha = loaded[(0, alpha_key)]
                        if alpha.numel() != 1 or not torch.isfinite(alpha).all().item():
                            raise ValueError(
                                f"LoRA alpha for '{block_name}' must be a finite scalar."
                            )
                        scale = lora_alpha_scale(alpha, rank, layer=block_name)
                        output = up
                        try:
                            if scale != 1.0:
                                output = (
                                    up.to(dtype=torch.float32)
                                    .mul_(scale)
                                    .to(dtype=up.dtype)
                                    .contiguous()
                                )
                            writer.write_batch([(
                                output_keys.get(up_key, up_key),
                                output.cpu().contiguous(),
                            )])
                        finally:
                            del output
                            del up
                            del alpha
                finally:
                    iterator.close()

        if verbose:
            print(
                f"[LoRA Alpha Normalize] Removed {len(work_units)} alpha tensors"
            )
            if reference_model_path:
                print(
                    f"[LoRA Alpha Normalize] Reference-mapped {len(mapped_blocks)} LoRA layers"
                )
            print(f"[LoRA Alpha Normalize] Saved to {output_path}")
        return output_path
    finally:
        handler.__exit__(None, None, None)
        cleanup_after_operation()

def resize_lora_file(
    lora_path: str,
    new_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
    device: str,
    save_dtype: torch.dtype,
    output_filename: str,
    verbose: bool = True,
    force_clear_cache: bool = False,
    min_rank: int = 1,
    exclude_patterns: str = "",
    discard_patterns: str = "",
    glob_patterns: bool = False,
    include_mode: bool = False,
    layer_parameters=None,
) -> str:
    """Resize a LoRA through one bounded UEL stream and factor-space SVD."""
    if min_rank < 1:
        raise ValueError("min_rank must be at least 1")
    if min_rank > new_rank:
        raise ValueError("min_rank cannot exceed max rank")

    parameter_profiles = {
        None: ("resize:fixed", {"new_rank": new_rank}),
        "sv_ratio": (
            "resize:ratio", {"max_rank": new_rank, "ratio": dynamic_param}
        ),
        "sv_fro": (
            "resize:frobenius", {
                "max_rank": new_rank,
                "min_rank": min_rank,
                "target": dynamic_param,
            },
        ),
        "sv_cumulative": (
            "resize:cumulative", {"max_rank": new_rank, "target": dynamic_param}
        ),
    }
    try:
        parameter_profile, parameter_defaults = parameter_profiles[dynamic_method]
    except KeyError as error:
        raise ValueError(f"Unsupported resize method: {dynamic_method}") from error

    lora_size_gb = estimate_model_size(lora_path)
    prepare_for_large_operation(lora_size_gb, torch.device(device))

    handler = MemoryEfficientSafeOpen(lora_path, low_memory=True)

    try:
        low_bit_keys = inspect_low_bit_input(handler, f"LoRA ({lora_path})", "LoRA Resize")
        metadata = (handler.metadata() or {}).copy()
        all_keys = handler.keys()

        format_info = detect_lora_format(all_keys)
        pairs, passthrough_keys = parse_lora_layers(all_keys)
        validate_canonical_blocks(pairs, "LoRA Resize")
        resolved_layer_parameters = resolve_layer_parameters(
            layer_parameters,
            parameter_profile,
            (canonical_lora_block_name(block_name) for block_name in pairs),
            parameter_defaults,
            node_name="LoRA Resize",
        )
        raw_units, stream_units, rank_counts, preserved_companion_groups = (
            _build_resize_work_units(
                handler, pairs, passthrough_keys, low_bit_keys,
                exclude_patterns, discard_patterns, glob_patterns, include_mode,
                resolved_layer_parameters,
            )
        )
        rank_text = _rank_summary(rank_counts)

        if verbose:
            method_str = f"{dynamic_method}: {dynamic_param}" if dynamic_method else "fixed"
            print(f"[LoRA Resize] Format: {format_info['format']}, layers: {format_info['key_count']}")
            print(f"[LoRA Resize] Source ranks: {rank_text}")
            print(
                f"[LoRA Resize] Resizing with method={method_str}, "
                f"min_rank={min_rank}, max_rank={new_rank}"
            )

        if dynamic_method:
            resize_description = f"Dynamic resize with {dynamic_method}: {dynamic_param}"
        else:
            resize_description = f"Fixed resize with maximum rank {new_rank}"
        metadata["ss_training_comment"] = (
            f"{resize_description}; source ranks {rank_text}"
        )
        metadata["ss_network_dim"] = "Dynamic"
        metadata["ss_network_alpha"] = "Dynamic"
        metadata["modelutils_resize_filter_mode"] = (
            "include" if include_mode else "exclude"
        )

        output_path, _ = canonical_model_artifact_path("loras", output_filename)

        fro_list = []
        cpu_fallbacks = 0
        total_units = len(raw_units) + len(stream_units)
        pbar = comfy.utils.ProgressBar(total_units)

        writer_context = atomic_uel_writer(output_path, metadata)
        writer = writer_context.__enter__()
        stream = None
        try:
            with torch.no_grad():
                for unit in raw_units:
                    for _, key, output_key in unit["entries"]:
                        write_preserved_tensor(
                            writer,
                            key,
                            handler,
                            output_key,
                            force_raw=True,
                        )
                    pbar.update(1)

                stream_keys = [
                    key
                    for unit in stream_units
                    for _, key, _ in unit["entries"]
                ]
                if stream_keys:
                    stream = handler.async_stream(
                        stream_keys,
                        batch_size=1,
                        prefetch_batches=1,
                        pin_memory=str(device).startswith("cuda"),
                    )
                    stream_iterator = iter(stream)
                    iterator = tqdm(
                        stream_units, desc="Resizing layers", unit="layers"
                    )
                    for unit in iterator:
                        loaded = {}
                        unit_keys = []
                        outputs = []
                        result = None
                        try:
                            for role, expected_key, output_key in unit["entries"]:
                                batch = next(stream_iterator)
                                if len(batch) != 1 or batch[0][0] != expected_key:
                                    raise RuntimeError(
                                        "UEL resize stream returned tensors out of order"
                                    )
                                loaded[expected_key] = batch[0][1]
                                unit_keys.append(expected_key)

                            source_dtypes = [
                                handler.get_dtype(key) for key in unit_keys
                            ]
                            if unit["kind"] == "preserve":
                                outputs = [
                                    (output_key, loaded[key].cpu().contiguous())
                                    for _, key, output_key in unit["entries"]
                                ]
                            elif unit["kind"] == "copy":
                                for role, key, output_key in unit["entries"]:
                                    tensor = loaded[key]
                                    target_dtype = select_output_dtype(
                                        source_dtypes,
                                        save_dtype,
                                        is_1d_diff=(
                                            role in {"diff", "diff_b", "w_norm", "b_norm"}
                                            and tensor.ndim == 1
                                        ),
                                    )
                                    outputs.append(
                                        (
                                            output_key,
                                            tensor.to(target_dtype).cpu().contiguous(),
                                        )
                                    )
                            else:
                                by_role = {
                                    role: (key, output_key)
                                    for role, key, output_key in unit["entries"]
                                    if role is not None
                                }
                                down_key, down_output = by_role["down"]
                                up_key, up_output = by_role["up"]
                                source_rank = int(loaded[down_key].shape[0])
                                if "alpha" in by_role:
                                    alpha_key, _ = by_role["alpha"]
                                    scale = float(loaded[alpha_key].item()) / source_rank
                                else:
                                    scale = 1.0

                                resize_parameters = unit["resize_parameters"]
                                if resize_parameters is None:
                                    layer_max_rank = new_rank
                                    layer_dynamic_param = dynamic_param
                                    layer_min_rank = min_rank
                                elif dynamic_method is None:
                                    layer_max_rank = resize_parameters["new_rank"]
                                    layer_dynamic_param = None
                                    layer_min_rank = 1
                                elif dynamic_method == "sv_ratio":
                                    layer_max_rank = resize_parameters["max_rank"]
                                    layer_dynamic_param = resize_parameters["ratio"]
                                    layer_min_rank = 1
                                elif dynamic_method == "sv_fro":
                                    layer_max_rank = resize_parameters["max_rank"]
                                    layer_dynamic_param = resize_parameters["target"]
                                    layer_min_rank = resize_parameters["min_rank"]
                                else:
                                    layer_max_rank = resize_parameters["max_rank"]
                                    layer_dynamic_param = resize_parameters["target"]
                                    layer_min_rank = 1

                                result = _resize_lora_factors(
                                    loaded[down_key],
                                    loaded[up_key],
                                    layer_max_rank,
                                    dynamic_method,
                                    layer_dynamic_param,
                                    scale,
                                    layer_min_rank,
                                    device,
                                )
                                fro_list.append(result["fro_retained"])
                                cpu_fallbacks += int(result["cpu_fallback"])
                                layer_dtype = select_output_dtype(
                                    source_dtypes, save_dtype
                                )
                                outputs.extend(
                                    [
                                        (
                                            down_output,
                                            result["lora_down"].to(layer_dtype),
                                        ),
                                        (
                                            up_output,
                                            result["lora_up"].to(layer_dtype),
                                        ),
                                    ]
                                )
                                if unit["alpha_output"] is not None:
                                    outputs.append(
                                        (
                                            unit["alpha_output"],
                                            torch.tensor(
                                                result["new_alpha"],
                                                dtype=layer_dtype,
                                            ),
                                        )
                                    )
                                for role in ("diff", "diff_b", "w_norm", "b_norm"):
                                    if role not in by_role:
                                        continue
                                    key, output_key = by_role[role]
                                    tensor = loaded[key]
                                    target_dtype = select_output_dtype(
                                        source_dtypes,
                                        save_dtype,
                                        is_1d_diff=tensor.ndim == 1,
                                    )
                                    outputs.append(
                                        (
                                            output_key,
                                            tensor.to(target_dtype).cpu().contiguous(),
                                        )
                                    )

                            writer.write_batch(outputs)
                        finally:
                            outputs.clear()
                            if result is not None:
                                result.clear()
                            loaded.clear()
                            for key in unit_keys:
                                handler.mark_processed(key)
                            if force_clear_cache:
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            pbar.update(1)

                    try:
                        next(stream_iterator)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError("UEL resize stream returned extra tensors")
        except BaseException as exc:
            if stream is not None:
                close_stream = getattr(stream, "close", None)
                if close_stream is not None:
                    close_stream()
            writer_context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            if stream is not None:
                close_stream = getattr(stream, "close", None)
                if close_stream is not None:
                    close_stream()
            writer_context.__exit__(None, None, None)

        if verbose and fro_list:
            avg_fro = sum(fro_list) / len(fro_list)
            variance = sum((value - avg_fro) ** 2 for value in fro_list) / len(fro_list)
            std_fro = variance ** 0.5
            print(f"[LoRA Resize] Average Frobenius retention: {avg_fro:.1%} ± {std_fro:.3f}")
        if preserved_companion_groups:
            logging.warning(
                "[LoRA Resize] Preserved %d companion-bearing LoRA group(s) without rank transformation",
                preserved_companion_groups,
            )
        if verbose:
            print(f"[LoRA Resize] CUDA OOM CPU fallbacks: {cpu_fallbacks}")

        print(f"[LoRA Resize] Saved to {output_path}")
        return output_path

    finally:
        handler.__exit__(None, None, None)
        cleanup_after_operation()


# =============================================================================
# Node Definitions
# =============================================================================

class LoRANormalizeAlpha(io.ComfyNode):
    """Materialize LoRA alpha scaling into its up factors."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRANormalizeAlpha",
            display_name="LoRA Normalize Alpha",
            category="ModelUtils/LoRA/Utilities",
            description="Apply each alpha divided by rank scale to its LoRA up factor and save an alpha-free LoRA.",
            inputs=[
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="LoRA whose per-layer alpha scaling will be materialized.",
                ),
                io.String.Input(
                    "output_filename",
                    default="normalized_lora",
                    tooltip="Output filename without extension, written under ComfyUI's LoRA directory.",
                ),
                io.Combo.Input(
                    "reference_model",
                    options=["None", *folder_paths.get_filename_list("diffusion_models")],
                    default="None",
                    tooltip="Optional diffusion model used to normalize flattened LoRA layer names to its exact model paths.",
                ),
            ],
            outputs=[io.AnyType.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, output_filename, reference_model="None") -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        reference_path = None
        if reference_model != "None":
            reference_path = folder_paths.get_full_path_or_raise(
                "diffusion_models", reference_model
            )
        normalize_lora_alpha_file(
            lora_path,
            output_filename,
            reference_model_path=reference_path,
        )
        _, output_name = canonical_model_artifact_path("loras", output_filename)
        return io.NodeOutput(output_name)

class LoRAResizeFixed(io.ComfyNode):
    """Resize LoRA to a fixed rank."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAResizeFixed",
            display_name="LoRA Resize (Fixed Rank)",
            category="ModelUtils/LoRA/Resize",
            description="Resize existing LoRA to a specific supported rank using factor-space SVD.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                              tooltip="LoRA to resize"),
                io.Int.Input("new_rank", default=64, min=1, max=3072,
                            tooltip="Target rank"),
                io.String.Input("output_filename", default="resized_lora", tooltip="Output filename without extension, written under ComfyUI's LoRA directory."),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16", tooltip="Data type used to save resized LoRA tensors."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", tooltip="Device used for per-layer resize arithmetic; CUDA out-of-memory retries the affected layer on CPU."),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
                *_resize_filter_inputs(),
                parameter_input("resize:fixed"),
            ],
            outputs=[io.AnyType.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, new_rank, output_filename, save_dtype, device, force_clear_cache, exclude_patterns="", discard_patterns="", glob_patterns=False, include_mode=False, layer_parameters=None) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        resize_lora_file(
            lora_path, new_rank, None, None, device, dtype, output_filename,
            force_clear_cache=force_clear_cache,
            exclude_patterns=exclude_patterns,
            discard_patterns=discard_patterns,
            glob_patterns=glob_patterns,
            include_mode=include_mode,
            layer_parameters=layer_parameters,
        )
        _, output_name = canonical_model_artifact_path("loras", output_filename)
        return io.NodeOutput(output_name)


class LoRAResizeRatio(io.ComfyNode):
    """Resize LoRA keeping SVs above ratio threshold."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAResizeRatio",
            display_name="LoRA Resize (SV Ratio)",
            category="ModelUtils/LoRA/Resize",
            description="Dynamically resize LoRA, keeping singular values where S[i] > S[0]/ratio.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                              tooltip="LoRA to resize"),
                io.Int.Input("max_rank", default=128, min=1, max=3072,
                            tooltip="Maximum allowed rank"),
                io.Float.Input("ratio", default=2.0, min=1.0, max=100.0, step=0.1,
                              tooltip="Keep SVs where S[i] > S[0]/ratio"),
                io.String.Input("output_filename", default="resized_lora_ratio", tooltip="Output filename without extension, written under ComfyUI's LoRA directory."),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16", tooltip="Data type used to save resized LoRA tensors."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", tooltip="Device used for per-layer resize arithmetic; CUDA out-of-memory retries the affected layer on CPU."),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
                *_resize_filter_inputs(),
                parameter_input("resize:ratio"),
            ],
            outputs=[io.AnyType.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, ratio, output_filename, save_dtype, device, force_clear_cache, exclude_patterns="", discard_patterns="", glob_patterns=False, include_mode=False, layer_parameters=None) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        resize_lora_file(
            lora_path, max_rank, "sv_ratio", ratio, device, dtype, output_filename,
            force_clear_cache=force_clear_cache,
            exclude_patterns=exclude_patterns,
            discard_patterns=discard_patterns,
            glob_patterns=glob_patterns,
            include_mode=include_mode,
            layer_parameters=layer_parameters,
        )
        _, output_name = canonical_model_artifact_path("loras", output_filename)
        return io.NodeOutput(output_name)


class LoRAResizeFrobenius(io.ComfyNode):
    """Resize LoRA to preserve Frobenius norm target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAResizeFrobenius",
            display_name="LoRA Resize (Frobenius)",
            category="ModelUtils/LoRA/Resize",
            description="Dynamically resize LoRA to preserve target fraction of Frobenius norm.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                              tooltip="LoRA to resize"),
                io.Int.Input("max_rank", default=128, min=1, max=3072,
                            tooltip="Maximum allowed rank"),
                io.Int.Input("min_rank", default=1, min=1, max=3072,
                            tooltip="Minimum retained rank when layer dimensions permit"),
                io.Float.Input("target", default=0.9, min=0.1, max=1.0, step=0.01,
                              tooltip="Target Frobenius norm retention (0.9 = 90%)"),
                io.String.Input("output_filename", default="resized_lora_fro", tooltip="Output filename without extension, written under ComfyUI's LoRA directory."),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16", tooltip="Data type used to save resized LoRA tensors."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", tooltip="Device used for per-layer resize arithmetic; CUDA out-of-memory retries the affected layer on CPU."),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
                *_resize_filter_inputs(),
                parameter_input("resize:frobenius"),
            ],
            outputs=[io.AnyType.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, min_rank, target, output_filename, save_dtype, device, force_clear_cache, exclude_patterns="", discard_patterns="", glob_patterns=False, include_mode=False, layer_parameters=None) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        resize_lora_file(
            lora_path, max_rank, "sv_fro", target, device, dtype, output_filename,
            force_clear_cache=force_clear_cache,
            exclude_patterns=exclude_patterns,
            discard_patterns=discard_patterns,
            glob_patterns=glob_patterns,
            include_mode=include_mode,
            min_rank=min_rank,
            layer_parameters=layer_parameters,
        )
        _, output_name = canonical_model_artifact_path("loras", output_filename)
        return io.NodeOutput(output_name)


class LoRAResizeCumulative(io.ComfyNode):
    """Resize LoRA to preserve cumulative SV target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAResizeCumulative",
            display_name="LoRA Resize (Cumulative)",
            category="ModelUtils/LoRA/Resize",
            description="Dynamically resize LoRA to preserve target fraction of cumulative singular values.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                              tooltip="LoRA to resize"),
                io.Int.Input("max_rank", default=128, min=1, max=3072,
                            tooltip="Maximum allowed rank"),
                io.Float.Input("target", default=0.9, min=0.1, max=1.0, step=0.01,
                              tooltip="Target cumulative SV retention (0.9 = 90%)"),
                io.String.Input("output_filename", default="resized_lora_cumulative", tooltip="Output filename without extension, written under ComfyUI's LoRA directory."),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16", tooltip="Data type used to save resized LoRA tensors."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", tooltip="Device used for per-layer resize arithmetic; CUDA out-of-memory retries the affected layer on CPU."),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
                *_resize_filter_inputs(),
                parameter_input("resize:cumulative"),
            ],
            outputs=[io.AnyType.Output(display_name="output_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lora_name, max_rank, target, output_filename, save_dtype, device, force_clear_cache, exclude_patterns="", discard_patterns="", glob_patterns=False, include_mode=False, layer_parameters=None) -> io.NodeOutput:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        resize_lora_file(
            lora_path, max_rank, "sv_cumulative", target, device, dtype, output_filename,
            force_clear_cache=force_clear_cache,
            exclude_patterns=exclude_patterns,
            discard_patterns=discard_patterns,
            glob_patterns=glob_patterns,
            include_mode=include_mode,
            layer_parameters=layer_parameters,
        )
        _, output_name = canonical_model_artifact_path("loras", output_filename)
        return io.NodeOutput(output_name)




# =============================================================================
# LoRA Merge To Model (Save merged model, skip extraction)
# =============================================================================

def merge_loras_to_model(
    lora_paths: List[str],
    lora_weights: List[float],
    base_model_path: str,
    device: str,
    save_dtype: torch.dtype,
    output_filename: str,
    skip_patterns_str: str = "",
    verbose: bool = True,
    lazy_load: bool = True,
    force_clear_cache: bool = False,
    include_1d_diffs: bool = False,
    return_report: bool = False,
    include_mode: bool = False,
) -> str | tuple[str, str]:
    """
    Merge multiple LoRAs into a base model and save the result directly.

    Unlike merge_multi_loras_via_base, this function:
    - Does NOT extract the result back to a LoRA
    - Saves the merged full model to the base model's directory

    Args:
        lora_paths: List of paths to LoRA files
        lora_weights: List of weight strengths (0.0-2.0) for each LoRA
        base_model_path: Path to base model
        device: Processing device
        save_dtype: Output dtype
        output_filename: Output filename (without extension)
        skip_patterns_str: Regex patterns for layers to skip
        verbose: Print progress info
        lazy_load: Low memory mode: load tensors from disk on demand
        force_clear_cache: Clear CUDA cache after each layer
        return_report: Return the path and merge report instead of only the path
        include_mode: Use skip_patterns_str to select matching base keys instead

    Returns:
        Path to saved merged model, optionally paired with the merge report
    """
    # Estimate memory and prepare
    total_size_gb = estimate_model_size(base_model_path)
    for lp in lora_paths:
        total_size_gb += estimate_model_size(lp)

    if verbose:
        print(f"[LoRA Merge To Model] Preparing memory for {total_size_gb:.2f}GB operation...")
        print(f"[LoRA Merge To Model] Merging {len(lora_paths)} LoRAs with weights: {lora_weights}")
    prepare_for_large_operation(total_size_gb * 1.5, torch.device(device))

    # Open all files - use low_memory for base model to avoid OS page caching
    base_handler = MemoryEfficientSafeOpen(base_model_path, low_memory=lazy_load)
    lora_handlers = [MemoryEfficientSafeOpen(lp, low_memory=lazy_load) for lp in lora_paths]


    try:
        base_quant = DiffusionQuantization(
            base_handler, base_model_path, "LoRA Merge To Model base model"
        )
        base_low_bit_keys = base_quant.isolated_low_bit_keys
        lora_low_bit_keys = [
            inspect_low_bit_input(
                handler, f"LoRA {i + 1} ({lora_paths[i]})", "LoRA Merge To Model"
            )
            for i, handler in enumerate(lora_handlers)
        ]
        # Detect format and extract pairs for each LoRA
        lora_infos = []

        for i, handler in enumerate(lora_handlers):
            keys = handler.keys()
            format_info = detect_lora_format(keys)
            pairs, _ = parse_lora_layers(keys)
            network_dim, network_alpha = detect_lora_rank(
                handler, pairs, lora_low_bit_keys[i]
            )
            lora_infos.append({
                "index": i,
                "path": lora_paths[i],
                "handler": handler,
                "format_info": format_info,
                "pairs": pairs,
                "network_dim": network_dim,
                "network_alpha": network_alpha,
                "weight": lora_weights[i],
                "low_bit_keys": lora_low_bit_keys[i],
            })
            if verbose:
                print(f"[LoRA Merge To Model] LoRA {i+1}: {format_info['format']}, {len(pairs)} layers, dim={network_dim}")

        # Common prefixes
        BASE_PREFIXES = ["model.diffusion_model.", "diffusion_model.", "transformer.", "model.", "net."]
        LORA_PREFIXES = [
            "lora_unet_", "lora_transformer_", "lora_te1_", "lora_te2_", "lora_te_",
            "lycoris_", "model.diffusion_model.", "diffusion_model.",
            "unet.base_model.model.", "base_model.model.", "transformer.", "unet."
        ]

        def extract_core_layer_base(key: str) -> str:
            result = key
            if result.endswith(".weight"):
                result = result[:-7]
            elif result.endswith(".bias"):
                result = result[:-5]
            for prefix in BASE_PREFIXES:
                if result.startswith(prefix):
                    result = result[len(prefix):]
                    break
            return result

        def extract_core_layer_lora(block_name: str) -> str:
            result = block_name
            changed = True
            while changed:
                changed = False
                for prefix in LORA_PREFIXES:
                    if result.startswith(prefix):
                        result = result[len(prefix):]
                        changed = True
                        break
            # Strip trailing .lora suffix (HuggingFace format leaves this after suffix removal)
            if result.endswith(".lora"):
                result = result[:-5]
            return result.replace(".", "_")


        base_keys = list(base_quant.data_keys)
        base_aliases = set()
        for key in base_keys:
            normalized = key
            for prefix in BASE_PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
            base_aliases.add(normalized)
            base_aliases.add(normalized.replace(".", "_"))

        # Build lookups for low-rank layers and exact direct patches.
        lora_lookup = {}
        direct_lookup = {}
        mapped_groups = set()
        applied_groups = set()
        group_catalog = {}
        group_events = defaultdict(list)
        for info in lora_infos:
            for block_name, block_keys in info["pairs"].items():
                group_catalog[(info["index"], block_name)] = (info, block_keys)
                selected_weight_direct = next(
                    (role for role in ("set_weight", "diff", "w_norm") if role in block_keys),
                    None,
                )
                if ("down" in block_keys or "up" in block_keys) and selected_weight_direct is None:
                    core = extract_core_layer_lora(block_name)
                    lora_lookup.setdefault(core, []).append((info, block_name, block_keys))

                direct_roles = []
                if selected_weight_direct is not None:
                    direct_roles.append(selected_weight_direct)
                selected_bias_direct = next(
                    (role for role in ("diff_b", "b_norm") if role in block_keys),
                    None,
                )
                if selected_bias_direct is not None:
                    direct_roles.append(selected_bias_direct)

                for direct_name in direct_roles:
                    if direct_name not in block_keys:
                        continue
                    direct_key = block_keys[direct_name]
                    core = block_name
                    changed = True
                    while changed:
                        changed = False
                        for prefix in LORA_PREFIXES:
                            if core.startswith(prefix):
                                core = core[len(prefix):]
                                changed = True
                                break
                    if direct_name in ("diff_b", "b_norm"):
                        targets = [f"{core}.bias"]
                    elif direct_name in ("w_norm", "set_weight"):
                        targets = [f"{core}.weight"]
                    else:
                        # Exact non-weight state keys win; module diffs fall back to .weight.
                        exact_aliases = {core, core.replace(".", "_")}
                        targets = [core] if exact_aliases & base_aliases else [f"{core}.weight"]
                    contribution = (info, block_name, direct_name, direct_key, block_keys)
                    for target in targets:
                        direct_lookup.setdefault(target, []).append(contribution)
                        underscored_target = target.replace(".", "_")
                        if underscored_target != target:
                            direct_lookup.setdefault(underscored_target, []).append(contribution)

        # Compile skip patterns
        skip_patterns = _compile_patterns(skip_patterns_str)

        # Preserve metadata from base model
        base_metadata = base_quant.output_metadata()
        base_metadata["merge_comment"] = f"Merged {len(lora_paths)} LoRAs with weights: {lora_weights}"

        # Build output path before loop
        output_path, _ = canonical_model_artifact_path(
            "diffusion_models", output_filename
        )

        base_outcomes = {
            "PATCHED": [],
            "BASE-ONLY BIASES RETAINED": [],
            "OTHER BASE-ONLY TENSORS RETAINED": [],
            "GUARDED": [],
            "MAPPED BUT UNCHANGED": [],
            "OMITTED BY SKIP PATTERN": [],
        }

        def record_group_event(info, block_name, status, target, reason):
            group_id = (info["index"], block_name)
            mapped_groups.add(group_id)
            group_events[group_id].append({
                "status": status,
                "target": target,
                "reason": reason,
            })

        def low_bit_description(info, block_keys):
            affected = sorted(set(block_keys.values()) & info["low_bit_keys"])
            return ", ".join(
                f"{key} ({info['handler'].get_dtype(key)})" for key in affected
            )

        def group_description(group_id, events):
            info, block_keys = group_catalog[group_id]
            block_name = group_id[1]
            sources = "\n".join(
                f"  source {role}: {key}"
                for role, key in sorted(layer_tensor_keys(block_keys).items())
            )
            details = "\n".join(
                f"  target {event['target']}: {event['reason']}" for event in events
            )
            return f"L{info['index'] + 1}::{block_name}\n{sources}\n{details}"

        stats = {
            "merged": 0,
            "base_bias_retained": 0,
            "base_only_retained": 0,
            "guarded_retained": 0,
            "mapped_unmodified": 0,
            "skipped": 0,
            "disabled_1d": 0,
            "shape_mismatch": 0,
        }
        pbar = comfy.utils.ProgressBar(len(base_keys))

        if verbose:
            print(f"[LoRA Merge To Model] Processing {len(base_keys)} base model keys...")

        handler_map = {-1: base_handler}
        handler_map.update({info["index"]: info["handler"] for info in lora_infos})
        work_units = []
        for planned_base_key in base_keys:
            planned_core = planned_base_key
            for prefix in BASE_PREFIXES:
                if planned_core.startswith(prefix):
                    planned_core = planned_core[len(prefix):]
                    break
            planned_direct = direct_lookup.get(planned_core, [])
            if not planned_direct:
                planned_direct = direct_lookup.get(planned_core.replace(".", "_"), [])
            planned_low_rank = []
            if planned_base_key.endswith(".weight"):
                core = extract_core_layer_base(planned_base_key)
                planned_low_rank = lora_lookup.get(core.replace(".", "_"), [])
            skipped = any(pattern.search(planned_base_key) for pattern in skip_patterns) != include_mode
            entries = {}
            if planned_base_key not in base_low_bit_keys and not skipped:
                entries[-1] = base_quant.required_keys(planned_base_key)
                for info, _, _, direct_key, block_keys in planned_direct:
                    if not layer_has_low_bit(block_keys, info["low_bit_keys"]):
                        entries.setdefault(info["index"], []).append(direct_key)
                for info, _, block_keys in planned_low_rank:
                    if not layer_has_low_bit(block_keys, info["low_bit_keys"]):
                        entries.setdefault(info["index"], []).extend(
                            layer_tensor_keys(block_keys).values()
                        )
                entries = {
                    source: list(dict.fromkeys(keys)) for source, keys in entries.items()
                }
            work_units.append((planned_base_key, entries))

        streamed_units = stream_work_units(
            handler_map,
            work_units,
            pin_memory=str(device).startswith("cuda"),
        )

        writer_context = atomic_uel_writer(output_path, base_metadata)
        writer = writer_context.__enter__()
        try:
            with torch.no_grad():
                for base_key, loaded in tqdm(
                    streamed_units,
                    total=len(work_units),
                    desc="Merging to model",
                    unit="keys",
                ):
                    core_with_suffix = base_key
                    for prefix in BASE_PREFIXES:
                        if core_with_suffix.startswith(prefix):
                            core_with_suffix = core_with_suffix[len(prefix):]
                            break
                    direct_contributions = direct_lookup.get(core_with_suffix, [])
                    if not direct_contributions:
                        direct_contributions = direct_lookup.get(core_with_suffix.replace(".", "_"), [])

                    low_rank_contributions = []
                    if base_key.endswith(".weight"):
                        core = extract_core_layer_base(base_key)
                        low_rank_contributions = lora_lookup.get(core.replace(".", "_"), [])

                    mapped_groups.update(
                        (info["index"], block_name)
                        for info, block_name, _, _, _ in direct_contributions
                    )
                    mapped_groups.update(
                        (info["index"], block_name)
                        for info, block_name, _ in low_rank_contributions
                    )

                    if base_key in base_low_bit_keys:
                        write_preserved_tensor(writer, base_key, base_handler)
                        stats["guarded_retained"] += 1
                        base_outcomes["GUARDED"].append(
                            f"{base_key}: isolated low-bit base tensor "
                            f"({base_handler.get_dtype(base_key)}) preserved unchanged"
                        )
                        for info, block_name, _, direct_key, _ in direct_contributions:
                            record_group_event(
                                info, block_name, "guarded", base_key,
                                f"base target is low-bit; source={direct_key}",
                            )
                        for info, block_name, block_keys in low_rank_contributions:
                            record_group_event(
                                info, block_name, "guarded", base_key,
                                "base target is low-bit; sources="
                                + ", ".join(sorted(layer_tensor_keys(block_keys).values())),
                            )
                        pbar.update(1)
                        continue

                    skip_pattern = next(
                        (pattern.pattern for pattern in skip_patterns if pattern.search(base_key)),
                        None,
                    )
                    if (skip_pattern is not None) != include_mode:
                        skip_reason = "no include pattern matched" if include_mode else f"pattern={skip_pattern}"
                        stats["skipped"] += 1
                        base_outcomes["OMITTED BY SKIP PATTERN"].append(
                            f"{base_key}: {skip_reason}"
                        )
                        for info, block_name, _, direct_key, _ in direct_contributions:
                            record_group_event(
                                info, block_name, "skipped", base_key,
                                f"target omitted: {skip_reason}; source={direct_key}",
                            )
                        for info, block_name, block_keys in low_rank_contributions:
                            record_group_event(
                                info, block_name, "skipped", base_key,
                                f"target omitted: {skip_reason}; sources="
                                + ", ".join(sorted(layer_tensor_keys(block_keys).values())),
                            )
                        pbar.update(1)
                        continue

                    # Load base weight only after guarded and skipped keys are classified.
                    cpu_base = (
                        base_quant.decode(
                            base_key,
                            {source_key: loaded[(-1, source_key)] for source_key in base_quant.required_keys(base_key)},
                            torch.float32,
                        )
                        if base_key in base_quant.quantized_keys else loaded[(-1, base_key)]
                    )

                    if direct_contributions or low_rank_contributions:
                        if device == 'cuda':
                            base_weight = transfer_to_gpu_pinned(cpu_base, device, torch.float32)
                        else:
                            base_weight = cpu_base.to(device=device, dtype=torch.float32)
                        del cpu_base

                        base_dtype = base_quant.logical_dtype(base_key, save_dtype)
                        applied_1d_diff = False
                        applied = False

                        for info, block_name, direct_role, direct_key, block_keys in direct_contributions:
                            if layer_has_low_bit(block_keys, info["low_bit_keys"]):
                                record_group_event(
                                    info, block_name, "guarded", base_key,
                                    "adapter group contains isolated low-bit tensor(s): "
                                    + low_bit_description(info, block_keys),
                                )
                                continue
                            cpu_patch = loaded[(info["index"], direct_key)]
                            is_additive = direct_role != "set_weight"
                            if is_additive and cpu_patch.ndim == 1 and not include_1d_diffs:
                                stats["disabled_1d"] += 1
                                record_group_event(
                                    info, block_name, "disabled_1d", base_key,
                                    f"source={direct_key}, shape={tuple(cpu_patch.shape)}; "
                                    "include_1d_diffs is disabled",
                                )
                                del cpu_patch
                                continue
                            if tuple(cpu_patch.shape) != tuple(base_weight.shape):
                                logging.warning(
                                    "[LoRA Merge To Model] Shape mismatch for %s: %s != %s; skipped",
                                    direct_key,
                                    tuple(cpu_patch.shape),
                                    tuple(base_weight.shape),
                                )
                                stats["shape_mismatch"] += 1
                                record_group_event(
                                    info, block_name, "shape_mismatch", base_key,
                                    f"source={direct_key}, source_shape={tuple(cpu_patch.shape)}, "
                                    f"target_shape={tuple(base_weight.shape)}",
                                )
                                del cpu_patch
                                continue
                            applied_1d_diff = applied_1d_diff or (
                                is_additive and cpu_patch.ndim == 1
                            )
                            if device == 'cuda':
                                patch_tensor = transfer_to_gpu_pinned(cpu_patch, device, torch.float32)
                            else:
                                patch_tensor = cpu_patch.to(device=device, dtype=torch.float32)
                            del cpu_patch
                            if direct_role == "set_weight":
                                base_weight.copy_(patch_tensor)
                            else:
                                base_weight.add_(patch_tensor, alpha=info["weight"])
                            del patch_tensor
                            applied = True
                            applied_groups.add((info["index"], block_name))
                            record_group_event(
                                info, block_name, "applied", base_key,
                                f"role={direct_role}, source={direct_key}, "
                                f"strength={info['weight']}",
                            )

                        # Let ComfyUI's LoRAAdapter own alpha, LoCon, reshape,
                        # and DoRA reconstruction while retaining this node's
                        # streaming file pipeline.
                        for info, block_name, block_keys in low_rank_contributions:
                            if "down" not in block_keys or "up" not in block_keys:
                                record_group_event(
                                    info, block_name, "load_failure", base_key,
                                    "recognized low-rank group is missing an A/down or B/up tensor",
                                )
                                continue
                            if layer_has_low_bit(block_keys, info["low_bit_keys"]):
                                record_group_event(
                                    info, block_name, "guarded", base_key,
                                    "adapter group contains isolated low-bit tensor(s): "
                                    + low_bit_description(info, block_keys),
                                )
                                continue
                            tensor_keys = layer_tensor_keys(block_keys)
                            tensors = {
                                key: loaded[(info["index"], key)]
                                for key in tensor_keys.values()
                            }
                            down_key = block_keys["down"]
                            up_key = block_keys["up"]
                            alpha_tensor = tensors.get(block_keys.get("alpha"))
                            tensors[down_key], tensors[up_key] = normalize_lora_pair(
                                tensors[down_key],
                                tensors[up_key],
                                alpha_tensor,
                                layer=block_name,
                            )
                            loaded[(info["index"], down_key)] = tensors[down_key]
                            loaded[(info["index"], up_key)] = tensors[up_key]
                            if "alpha" in block_keys:
                                tensors.pop(block_keys["alpha"], None)
                            alpha = None
                            dora_scale = tensors.get(block_keys.get("dora_scale"))
                            try:
                                adapter = LoRAAdapter.load(
                                    block_name,
                                    tensors,
                                    alpha,
                                    dora_scale,
                                    set(),
                                )
                            except Exception as exc:
                                logging.exception(
                                    "[LoRA Merge To Model] Failed to load adapter group %s",
                                    block_name,
                                )
                                record_group_event(
                                    info, block_name, "load_failure", base_key,
                                    f"{type(exc).__name__}: {exc}",
                                )
                                del tensors
                                continue
                            if adapter is None:
                                logging.warning(
                                    "[LoRA Merge To Model] Recognized LoRA pair '%s' could not be loaded; skipped",
                                    block_name,
                                )
                                record_group_event(
                                    info, block_name, "load_failure", base_key,
                                    "ComfyUI LoRAAdapter.load returned no adapter",
                                )
                                del tensors
                                continue
                            candidate_weight = base_weight.clone()
                            error_capture = AdapterErrorCapture(base_key)
                            root_logger = logging.getLogger()
                            root_logger.addHandler(error_capture)
                            raised_error = None
                            try:
                                candidate_weight = adapter.calculate_weight(
                                    candidate_weight,
                                    base_key,
                                    info["weight"],
                                    1.0,
                                    None,
                                    lambda value: value,
                                    torch.float32,
                                    None,
                                )
                            except Exception as exc:
                                raised_error = exc
                                logging.exception(
                                    "[LoRA Merge To Model] Adapter calculation failed for %s",
                                    base_key,
                                )
                            finally:
                                root_logger.removeHandler(error_capture)

                            calculation_errors = list(error_capture.messages)
                            if raised_error is not None:
                                calculation_errors.append(
                                    f"{type(raised_error).__name__}: {raised_error}"
                                )
                            if calculation_errors:
                                record_group_event(
                                    info, block_name, "calculation_failure", base_key,
                                    " | ".join(calculation_errors),
                                )
                                del candidate_weight, adapter, tensors
                                continue

                            del base_weight
                            base_weight = candidate_weight
                            del adapter, tensors
                            applied = True
                            applied_groups.add((info["index"], block_name))
                            record_group_event(
                                info, block_name, "applied", base_key,
                                "ComfyUI LoRAAdapter calculation completed; sources="
                                + ", ".join(sorted(tensor_keys.values()))
                                + f", strength={info['weight']}",
                            )

                        target_dtype = select_output_dtype(
                            [base_dtype],
                            save_dtype,
                            is_1d_diff=applied_1d_diff,
                        )
                        writer.write(base_key, base_weight.to(target_dtype).cpu().contiguous())
                        del base_weight
                        stats["merged" if applied else "mapped_unmodified"] += 1
                        if applied:
                            sources = []
                            for group_id, events in group_events.items():
                                if any(
                                    event["target"] == base_key
                                    and event["status"] == "applied"
                                    for event in events
                                ):
                                    sources.append(
                                        f"LoRA {group_id[0] + 1}::{group_id[1]}"
                                    )
                            base_outcomes["PATCHED"].append(
                                f"{base_key} <- {', '.join(sources)}"
                            )
                        else:
                            reasons = []
                            for events in group_events.values():
                                reasons.extend(
                                    event["reason"] for event in events
                                    if event["target"] == base_key
                                )
                            base_outcomes["MAPPED BUT UNCHANGED"].append(
                                f"{base_key}: {' | '.join(reasons)}"
                            )
                    else:
                        target_dtype = select_output_dtype(
                            [base_quant.logical_dtype(base_key, save_dtype)],
                            save_dtype,
                        )
                        writer.write(base_key, cpu_base.to(target_dtype).contiguous())
                        del cpu_base
                        if base_key.endswith(".bias"):
                            stats["base_bias_retained"] += 1
                            base_outcomes["BASE-ONLY BIASES RETAINED"].append(base_key)
                        else:
                            stats["base_only_retained"] += 1
                            base_outcomes["OTHER BASE-ONLY TENSORS RETAINED"].append(base_key)

                    if force_clear_cache:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    pbar.update(1)
        except BaseException as exc:
            streamed_units.close()
            writer_context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            streamed_units.close()
            writer_context.__exit__(None, None, None)

        adapter_outcomes = {
            "APPLIED": [],
            "PARTIALLY APPLIED": [],
            "1D DISABLED": [],
            "GUARDED": [],
            "SKIPPED BY PATTERN": [],
            "SHAPE MISMATCH": [],
            "LOAD FAILURE": [],
            "CALCULATION FAILURE": [],
            "MAPPED BUT UNCHANGED": [],
            "UNRESOLVED": [],
        }
        status_section = {
            "disabled_1d": "1D DISABLED",
            "guarded": "GUARDED",
            "skipped": "SKIPPED BY PATTERN",
            "shape_mismatch": "SHAPE MISMATCH",
            "load_failure": "LOAD FAILURE",
            "calculation_failure": "CALCULATION FAILURE",
        }
        for group_id in group_catalog:
            events = group_events.get(group_id, [])
            statuses = [event["status"] for event in events]
            if not events:
                info, block_keys = group_catalog[group_id]
                sources = "\n".join(
                    f"  source {role}: {key}"
                    for role, key in sorted(layer_tensor_keys(block_keys).items())
                )
                adapter_outcomes["UNRESOLVED"].append(
                    f"L{info['index'] + 1}::{group_id[1]}\n"
                    f"{sources}\n  reason: no matching base tensor"
                )
            elif all(status == "applied" for status in statuses):
                adapter_outcomes["APPLIED"].append(group_description(group_id, events))
            elif "applied" in statuses:
                adapter_outcomes["PARTIALLY APPLIED"].append(
                    group_description(group_id, events)
                )
            else:
                section = next(
                    (
                        status_section[status]
                        for status in (
                            "calculation_failure", "load_failure", "shape_mismatch",
                            "guarded", "disabled_1d", "skipped",
                        )
                        if status in statuses
                    ),
                    "MAPPED BUT UNCHANGED",
                )
                adapter_outcomes[section].append(group_description(group_id, events))

        error_sections = (
            "PARTIALLY APPLIED", "SHAPE MISMATCH", "LOAD FAILURE",
            "CALCULATION FAILURE", "MAPPED BUT UNCHANGED", "UNRESOLVED",
        )
        exclusion_sections = ("1D DISABLED", "GUARDED", "SKIPPED BY PATTERN")
        has_errors = any(adapter_outcomes[name] for name in error_sections)
        has_exclusions = any(adapter_outcomes[name] for name in exclusion_sections)
        if has_errors:
            outcome = "COMPLETED WITH ERRORS"
        elif has_exclusions:
            outcome = "COMPLETED WITH EXCLUSIONS"
        else:
            outcome = "SUCCESS"

        total_groups = len(group_catalog)
        fully_applied = len(adapter_outcomes["APPLIED"])
        report_lines = [
            f"{outcome}: {fully_applied}/{total_groups} adapter groups fully applied",
            "",
            "SUMMARY",
            f"- Base tensors: {len(base_keys)}",
            f"- Base tensors patched: {len(base_outcomes['PATCHED'])}",
            f"- Adapter groups fully applied: {fully_applied}",
            f"- Adapter groups partially applied: {len(adapter_outcomes['PARTIALLY APPLIED'])}",
            f"- Adapter groups excluded: {sum(len(adapter_outcomes[name]) for name in exclusion_sections)}",
            f"- Adapter groups failed or unresolved: {sum(len(adapter_outcomes[name]) for name in error_sections)}",
            "",
            "INPUTS",
        ]
        report_lines.extend(
            f"- L{info['index'] + 1}: {os.path.basename(info['path'])}, "
            f"strength={info['weight']}"
            for info in lora_infos
        )

        base_detail_sections = (
            "BASE-ONLY BIASES RETAINED",
            "OTHER BASE-ONLY TENSORS RETAINED",
            "GUARDED",
            "MAPPED BUT UNCHANGED",
            "OMITTED BY SKIP PATTERN",
        )
        adapter_detail_sections = tuple(
            section for section in adapter_outcomes if section != "APPLIED"
        )
        detailed_base = [
            (section, base_outcomes[section])
            for section in base_detail_sections
            if base_outcomes[section]
        ]
        detailed_adapter = [
            (section, adapter_outcomes[section])
            for section in adapter_detail_sections
            if adapter_outcomes[section]
        ]

        report_lines.extend(["", "BASE TENSORS NOT PATCHED"])
        if detailed_base:
            for section, entries in detailed_base:
                report_lines.append(f"[{section}] ({len(entries)})")
                report_lines.extend(f"- {entry}" for entry in entries)
        else:
            report_lines.append("- None")

        report_lines.extend(["", "ADAPTER GROUP ISSUES"])
        if detailed_adapter:
            for section, entries in detailed_adapter:
                report_lines.append(f"[{section}] ({len(entries)})")
                report_lines.extend(f"- {entry}" for entry in entries)
        else:
            report_lines.append("- None")

        report = "\n".join(report_lines).rstrip()
        if verbose:
            retained = sum(len(base_outcomes[name]) for name in base_detail_sections)
            issues = sum(len(entries) for _, entries in detailed_adapter)
            print(
                f"[LoRA Merge To Model] {outcome}: "
                f"{fully_applied}/{total_groups} groups applied, "
                f"{len(base_outcomes['PATCHED'])} tensors patched, "
                f"{retained} base tensors not patched, {issues} adapter issues"
            )
            print("[LoRA Merge To Model] Detailed diagnostics returned in node output.")

        print(f"[LoRA Merge To Model] Saved to {output_path}")
        return (output_path, report) if return_report else output_path

    finally:
        base_handler.__exit__(None, None, None)
        for handler in lora_handlers:
            handler.__exit__(None, None, None)
        cleanup_after_operation()


class LoRAMergeToModel(io.ComfyNode):
    """Merge multiple LoRAs into base model and save as full model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoRAMergeToModel",
            display_name="LoRA Merge To Model",
            category="ModelUtils/LoRA/Merge",
            description="Merge 1-8 LoRAs into a base model and save the result. Saves to base model directory.",
            inputs=[
                io.Combo.Input("base_model", options=folder_paths.get_filename_list("diffusion_models"),
                              tooltip="Base model the LoRAs were trained on"),
                io.Combo.Input("lora_count", options=["1", "2", "3", "4", "5", "6", "7", "8"], default="2",
                              tooltip="Number of LoRAs to merge"),
                # LoRA 1
                io.Combo.Input("lora_1", options=folder_paths.get_filename_list("loras"),
                              tooltip="First LoRA"),
                io.Float.Input("weight_1", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 1"),
                # LoRA 2
                io.Combo.Input("lora_2", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Second LoRA"),
                io.Float.Input("weight_2", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 2"),
                # LoRA 3
                io.Combo.Input("lora_3", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Third LoRA"),
                io.Float.Input("weight_3", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 3"),
                # LoRA 4
                io.Combo.Input("lora_4", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Fourth LoRA"),
                io.Float.Input("weight_4", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 4"),
                # LoRA 5
                io.Combo.Input("lora_5", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Fifth LoRA"),
                io.Float.Input("weight_5", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 5"),
                # LoRA 6
                io.Combo.Input("lora_6", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Sixth LoRA"),
                io.Float.Input("weight_6", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 6"),
                # LoRA 7
                io.Combo.Input("lora_7", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Seventh LoRA"),
                io.Float.Input("weight_7", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 7"),
                # LoRA 8
                io.Combo.Input("lora_8", options=["None"] + folder_paths.get_filename_list("loras"),
                              default="None", tooltip="Eighth LoRA"),
                io.Float.Input("weight_8", default=1.0, min=-10.0, max=10.0, step=0.01,
                              tooltip="Weight strength for LoRA 8"),
                # Settings
                io.String.Input("skip_patterns", default="", multiline=True,
                               tooltip="Regex patterns for base tensor keys to omit, or to include when Include Mode is enabled. Guarded low-bit base tensors are always preserved."),
                io.String.Input("output_filename", default="merged_model", tooltip="Output filename without extension, written under ComfyUI's diffusion-model directory."),
                io.Combo.Input("save_dtype", options=["fp16", "bf16", "fp32"], default="fp16", tooltip="Data type used to save the model after applying the LoRA."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda", tooltip="Device used for per-layer LoRA application; CUDA out-of-memory retries the affected layer on CPU."),
                io.Boolean.Input("lazy_load", default=True, tooltip="Low memory mode: load tensors from disk on demand"),
                io.Boolean.Input("force_clear_cache", default=False, tooltip="Clear CUDA cache after each layer (slower but saves VRAM)"),
                io.Boolean.Input("include_1d_diffs", default=False,
                                 tooltip="Apply 1D direct-diff tensors as FP32. Disabled preserves prior behavior."),
                io.Boolean.Input("include_mode", default=False,
                                 tooltip="Use Skip Patterns as an include-only filter. Only matching base tensors are processed and saved; an empty filter selects nothing. Guarded low-bit base tensors remain preserved."),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
                io.String.Output(display_name="merge_report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, base_model, lora_count,
                lora_1, weight_1, lora_2, weight_2, lora_3, weight_3, lora_4, weight_4,
                lora_5, weight_5, lora_6, weight_6, lora_7, weight_7, lora_8, weight_8,
                skip_patterns, output_filename, save_dtype, device, lazy_load, force_clear_cache,
                include_1d_diffs, include_mode=False) -> io.NodeOutput:

        # Build LoRA list based on count
        count = int(lora_count)
        lora_names = [lora_1, lora_2, lora_3, lora_4, lora_5, lora_6, lora_7, lora_8][:count]
        lora_weights = [weight_1, weight_2, weight_3, weight_4, weight_5, weight_6, weight_7, weight_8][:count]

        # Filter out "None" entries
        valid_loras = [(name, weight) for name, weight in zip(lora_names, lora_weights) if name != "None"]
        if not valid_loras:
            raise ValueError("At least one LoRA must be selected")

        lora_names, lora_weights = zip(*valid_loras)
        lora_paths = [folder_paths.get_full_path_or_raise("loras", name) for name in lora_names]
        base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[save_dtype]

        path, report = merge_loras_to_model(
            lora_paths=list(lora_paths),
            lora_weights=list(lora_weights),
            base_model_path=base_path,
            device=device,
            save_dtype=dtype,
            output_filename=output_filename,
            skip_patterns_str=skip_patterns,
            lazy_load=lazy_load,
            force_clear_cache=force_clear_cache,
            include_1d_diffs=include_1d_diffs,
            return_report=True,
            include_mode=include_mode,
        )
        _, output_name = canonical_model_artifact_path("diffusion_models", output_filename)
        return io.NodeOutput(output_name, report)
