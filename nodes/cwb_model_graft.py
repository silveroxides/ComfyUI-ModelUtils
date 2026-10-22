"""Standalone training-free Consensus-Weighted Blending model grafting node."""

from __future__ import annotations

from contextlib import closing
import json
import re

import comfy.utils
import folder_paths
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from tqdm import tqdm
from unifiedefficientloader import MemoryEfficientSafeOpen

from .artifact_paths import canonical_model_artifact_path
from .consensus_merger import (
    CWB_CONFIG,
    CWBDiagnostics,
    CWBSettings,
    DENSE_CWB_PRESETS,
    _clear_previous_layer,
    _is_float_dtype,
    _merge_tensors_to_cpu,
    _requested_dtype,
    resolve_cwb_settings,
)
from .device_utils import (
    cleanup_after_operation,
    estimate_model_size,
    prepare_for_large_operation,
)
from .merger import (
    _compile_patterns,
    _matches_any_pattern,
    load_documentation_from_file,
)
from .quantization_guard import (
    DiffusionQuantization,
    diffusion_key_map,
    write_preserved_tensor,
)
from .uel_io import atomic_uel_writer, stream_work_units


_MAIN_BLOCK_INDEX_RE = re.compile(
    r"(?:^|\.)(?:layers|blocks|double_blocks|single_blocks|transformer_blocks)\.(\d+)\."
)
_ATTN_KEY_TOKENS = (
    ".qkv.",
    ".qkv_proj.",
    ".proj.q.",
    ".proj.k.",
    ".proj.v.",
    ".proj.o.",
    ".attn.",
    ".self_attn.",
    ".q_norm.",
    ".k_norm.",
    ".to_q.",
    ".to_k.",
    ".to_v.",
    ".to_out.",
)
_FFN_KEY_TOKENS = (
    ".ffn_up_gate.",
    ".ffn_up.",
    ".ffn_gate.",
    ".ffn_down.",
    ".mlp.",
    ".ff.",
    ".fc1.",
    ".fc2.",
    ".w1.",
    ".w2.",
    ".w3.",
)


def _extract_main_block_index(key: str) -> int | None:
    if "text_embedder." in key or "token_refiner." in key:
        return None
    match = _MAIN_BLOCK_INDEX_RE.search(key)
    return int(match.group(1)) if match else None


def _classify_graft_operator(key: str) -> str | None:
    padded = f".{key}."
    if any(token in padded for token in _ATTN_KEY_TOKENS):
        return "attention"
    if any(token in padded for token in _FFN_KEY_TOKENS):
        return "ffn"
    return None


def _block_rule_allows(
    block_idx: int | None,
    rule: str,
    protect_early_blocks: int,
    max_block_index: int,
) -> bool:
    if rule == "none":
        return False
    if block_idx is None:
        return False
    if block_idx < protect_early_blocks:
        return False
    if rule == "all_after_protected":
        return True
    if rule == "interleaved_even":
        return (block_idx % 2) == 0
    if rule == "interleaved_odd":
        return (block_idx % 2) == 1
    if rule == "back_half_only":
        midpoint = (max_block_index + 1) // 2
        return block_idx >= max(protect_early_blocks, midpoint)
    return False


def _parse_custom_key_map(raw_map: str) -> dict[str, str]:
    """Parse lines of 'donor_key -> base_key' or 'donor_key = base_key' into {base_key: donor_key}."""
    mapping: dict[str, str] = {}
    if not raw_map or not raw_map.strip():
        return mapping
    for line_num, raw_line in enumerate(raw_map.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            donor_part, base_part = line.split("->", 1)
        elif "=" in line:
            donor_part, base_part = line.split("=", 1)
        else:
            raise ValueError(
                f"Custom key map line {line_num}: expected 'donor_key -> base_key'."
            )
        donor_key = donor_part.strip()
        base_key = base_part.strip()
        if not donor_key or not base_key:
            raise ValueError(
                f"Custom key map line {line_num}: both donor_key and base_key must be non-empty."
            )
        mapping[base_key] = donor_key
    return mapping


def _align_donor_trailing_shape(
    base_tensor: torch.Tensor, donor_tensor: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    """Align donor trailing dimensions to base_tensor so CWB similarity alignment can match rows."""
    if donor_tensor.ndim != base_tensor.ndim:
        if donor_tensor.numel() == base_tensor.numel():
            return donor_tensor.reshape(base_tensor.shape), True
        flat_donor = donor_tensor.reshape(1, 1, -1).to(dtype=torch.float32)
        resized = F.interpolate(
            flat_donor, size=base_tensor.numel(), mode="linear", align_corners=False
        )
        return resized.reshape(base_tensor.shape).to(dtype=donor_tensor.dtype), True

    if donor_tensor.shape == base_tensor.shape:
        return donor_tensor, False

    if base_tensor.ndim <= 1:
        return donor_tensor, True

    if donor_tensor.shape[1:] == base_tensor.shape[1:]:
        return donor_tensor, True

    target_trailing = int(torch.tensor(base_tensor.shape[1:]).prod().item())
    donor_2d = donor_tensor.reshape(donor_tensor.shape[0], 1, -1).to(dtype=torch.float32)
    resized_2d = F.interpolate(
        donor_2d, size=target_trailing, mode="linear", align_corners=False
    )
    reshaped = resized_2d.reshape(donor_tensor.shape[0], *base_tensor.shape[1:]).to(
        dtype=donor_tensor.dtype
    )
    return reshaped, True


def _graft_pair_to_cpu(
    base_tensor: torch.Tensor,
    donor_tensor: torch.Tensor,
    settings: CWBSettings,
    device: str,
    target_dtype: torch.dtype,
    *,
    operation_label: str,
    diagnostics: CWBDiagnostics,
) -> torch.Tensor:
    aligned_donor, shape_changed = _align_donor_trailing_shape(base_tensor, donor_tensor)
    merged = _merge_tensors_to_cpu(
        [base_tensor, aligned_donor],
        settings,
        device,
        torch.float32,
        reference_index=0,
        allow_similarity_alignment=True,
        operation_label=operation_label,
        diagnostics=diagnostics,
    )
    if shape_changed or settings.rescale_norm:
        base_norm = torch.linalg.vector_norm(base_tensor.to(dtype=torch.float32))
        merged_norm = torch.linalg.vector_norm(merged)
        if merged_norm.item() > 1e-8 and base_norm.item() > 0.0:
            merged = merged * (base_norm / merged_norm)
    return merged.to(dtype=target_dtype).cpu().contiguous()


def _graft_with_subpart(
    key: str,
    op_type: str,
    subpart: str,
    base_tensor: torch.Tensor,
    donor_tensor: torch.Tensor,
    settings: CWBSettings,
    device: str,
    target_dtype: torch.dtype,
    diagnostics: CWBDiagnostics,
) -> torch.Tensor | None:
    """Return grafted tensor or None if the subpart rule keeps the base tensor untouched."""
    padded = f".{key}."
    if op_type == "attention":
        is_qkv = any(tok in padded for tok in (".qkv.", ".qkv_proj."))
        is_q_or_k = any(
            tok in padded for tok in (".proj.q.", ".proj.k.", ".to_q.", ".to_k.", ".q_norm.", ".k_norm.")
        )
        is_v = any(tok in padded for tok in (".proj.v.", ".to_v."))
        is_out = any(tok in padded for tok in (".proj.o.", ".out_proj.", ".to_out."))
        has_known_attn_part = is_qkv or is_q_or_k or is_v or is_out

        if has_known_attn_part and subpart == "output_only":
            if not is_out:
                return None
        elif has_known_attn_part and subpart == "value_and_output":
            if is_q_or_k:
                return None
            if is_qkv and base_tensor.ndim >= 1 and base_tensor.shape[0] % 3 == 0:
                aligned_donor, _ = _align_donor_trailing_shape(base_tensor, donor_tensor)
                if aligned_donor.shape[0] % 3 == 0:
                    q_a, k_a, v_a = base_tensor.chunk(3, dim=0)
                    _, _, v_b = aligned_donor.chunk(3, dim=0)
                    v_merged = _graft_pair_to_cpu(
                        v_a,
                        v_b,
                        settings,
                        device,
                        target_dtype,
                        operation_label=f"{key}[v]",
                        diagnostics=diagnostics,
                    )
                    return torch.cat(
                        [
                            q_a.to(dtype=target_dtype).cpu(),
                            k_a.to(dtype=target_dtype).cpu(),
                            v_merged,
                        ],
                        dim=0,
                    ).contiguous()
            elif not (is_v or is_out):
                return None

    elif op_type == "ffn":
        is_up_gate = ".ffn_up_gate." in padded
        is_up = any(tok in padded for tok in (".ffn_up.", ".fc1.", ".w1."))
        is_gate = any(tok in padded for tok in (".ffn_gate.", ".w3."))
        is_down = any(tok in padded for tok in (".ffn_down.", ".fc2.", ".w2.", ".ff.net.2."))
        has_known_ffn_part = is_up_gate or is_up or is_gate or is_down

        if has_known_ffn_part and subpart == "down_only":
            if not is_down:
                return None
        elif has_known_ffn_part and subpart == "up_only":
            if is_down or is_gate:
                return None
            if is_up_gate and base_tensor.ndim >= 1 and base_tensor.shape[0] % 2 == 0:
                aligned_donor, _ = _align_donor_trailing_shape(base_tensor, donor_tensor)
                if aligned_donor.shape[0] % 2 == 0:
                    up_a, gate_a = base_tensor.chunk(2, dim=0)
                    up_b, _ = aligned_donor.chunk(2, dim=0)
                    up_merged = _graft_pair_to_cpu(
                        up_a,
                        up_b,
                        settings,
                        device,
                        target_dtype,
                        operation_label=f"{key}[up]",
                        diagnostics=diagnostics,
                    )
                    return torch.cat(
                        [
                            up_merged,
                            gate_a.to(dtype=target_dtype).cpu(),
                        ],
                        dim=0,
                    ).contiguous()
            elif not is_up:
                return None

    return _graft_pair_to_cpu(
        base_tensor,
        donor_tensor,
        settings,
        device,
        target_dtype,
        operation_label=key,
        diagnostics=diagnostics,
    )


def _resolve_donor_qkv_plan(
    base_key: str,
    donor_keys_set: set[str],
    custom_map: dict[str, str],
) -> tuple[str, list[str], int | None] | None:
    """Resolve donor key(s) for base_key.

    Returns:
        (mode, donor_logical_keys, slice_index)
        mode is 'direct', 'fuse_qkv' (cat q,k,v), or 'split_qkv' (chunk 3 and pick slice_index).
    """
    if base_key in custom_map:
        mapped = custom_map[base_key]
        if mapped in donor_keys_set:
            return ("direct", [mapped], None)
        return None
    if base_key in donor_keys_set:
        return ("direct", [base_key], None)

    if base_key.endswith(".proj.qkv.weight"):
        prefix = base_key[: -len("qkv.weight")]
        parts = [f"{prefix}{p}.weight" for p in ("q", "k", "v")]
        if all(k in donor_keys_set for k in parts):
            return ("fuse_qkv", parts, None)

    for idx, part in enumerate(("q", "k", "v")):
        suffix = f".proj.{part}.weight"
        if base_key.endswith(suffix):
            fused = f"{base_key[:-len(suffix)]}.proj.qkv.weight"
            if fused in donor_keys_set:
                return ("split_qkv", [fused], idx)

    return None


class CWBModelGraft(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        models = folder_paths.get_filename_list("diffusion_models")
        presets = list(DENSE_CWB_PRESETS)
        return io.Schema(
            node_id="CWBModelGraft",
            display_name="CWB Model Graft",
            category="ModelUtils/Merging",
            description="Graft selected blocks, sub-projections, or mapped layers from a donor diffusion model into a base diffusion model using Consensus-Weighted Blending.",
            inputs=[
                io.Combo.Input(
                    "execution_mode",
                    options=["MERGE", "DOCUMENTATION ONLY"],
                    default="MERGE",
                    tooltip="MERGE writes a grafted model; DOCUMENTATION ONLY opens no model files.",
                ),
                io.Combo.Input(
                    "model_a",
                    options=models,
                    tooltip="Base diffusion model that anchors structure, metadata, and protected blocks.",
                ),
                io.Combo.Input(
                    "model_b",
                    options=models,
                    tooltip="Donor diffusion model supplying parts to graft into Model A.",
                ),
                io.Combo.Input(
                    "attention_preset",
                    options=presets,
                    default="robust_medn",
                    tooltip="CWB preset applied to grafted attention layers.",
                ),
                CWB_CONFIG.Input(
                    "attention_config",
                    optional=True,
                    tooltip="Optional custom CWB configuration override for attention layers.",
                ),
                io.Combo.Input(
                    "ffn_preset",
                    options=presets,
                    default="balanced_mean",
                    tooltip="CWB preset applied to grafted feed-forward (MLP) layers.",
                ),
                CWB_CONFIG.Input(
                    "ffn_config",
                    optional=True,
                    tooltip="Optional custom CWB configuration override for feed-forward (MLP) layers.",
                ),
                io.Int.Input(
                    "protect_early_blocks",
                    default=2,
                    min=0,
                    max=256,
                    step=1,
                    tooltip="Number of initial transformer blocks (starting from 0) kept 100% from Model A to preserve global layout.",
                ),
                io.Combo.Input(
                    "attention_block_rule",
                    options=[
                        "interleaved_even",
                        "interleaved_odd",
                        "back_half_only",
                        "all_after_protected",
                        "none",
                    ],
                    default="interleaved_even",
                    tooltip="Which transformer blocks after the protected start have their attention layers grafted.",
                ),
                io.Combo.Input(
                    "ffn_block_rule",
                    options=[
                        "all_after_protected",
                        "interleaved_even",
                        "interleaved_odd",
                        "back_half_only",
                        "none",
                    ],
                    default="all_after_protected",
                    tooltip="Which transformer blocks after the protected start have their feed-forward layers grafted.",
                ),
                io.Combo.Input(
                    "attention_subpart",
                    options=["all", "value_and_output", "output_only"],
                    default="all",
                    tooltip="Select whether to graft all attention weights, only value and output projections, or only the output projection.",
                ),
                io.Combo.Input(
                    "ffn_subpart",
                    options=["all", "up_only", "down_only"],
                    default="all",
                    tooltip="Select whether to graft all feed-forward weights, only the up projection, or only the down projection.",
                ),
                io.String.Input(
                    "custom_key_map",
                    default="",
                    multiline=True,
                    tooltip="Optional explicit key mappings (one per line as 'donor_key -> base_key') to graft across differently named layers.",
                ),
                io.String.Input(
                    "output_filename",
                    default="cwb_grafted_model",
                    tooltip="Filename without extension under ComfyUI's diffusion_models directory.",
                ),
                io.Combo.Input(
                    "save_dtype",
                    options=["fp32", "fp16", "bf16"],
                    default="bf16",
                    tooltip="Requested dtype for grafted floating tensors.",
                ),
                io.Combo.Input(
                    "process_device",
                    options=["cuda", "cpu"],
                    default="cuda",
                    tooltip="Per-tensor FP32 CWB computation device; CUDA OOM retries on CPU.",
                ),
                io.String.Input(
                    "exclude_patterns",
                    default="",
                    multiline=True,
                    tooltip="Newline or whitespace separated patterns to preserve from Model A (or whitelist when include_mode is enabled).",
                ),
                io.String.Input(
                    "discard_patterns",
                    default="",
                    multiline=True,
                    tooltip="Newline or whitespace separated patterns to omit from the output file.",
                ),
                io.Boolean.Input(
                    "glob_patterns",
                    default=False,
                    tooltip="Interpret exclude and discard entries as shell-style globs instead of regular expressions.",
                ),
                io.Boolean.Input(
                    "lazy_load",
                    default=True,
                    tooltip="Stream only one logical work unit at a time through UEL and release completed tensors immediately.",
                ),
                io.Boolean.Input(
                    "force_clear_cache",
                    default=False,
                    tooltip="Run Python garbage collection and empty CUDA cache before each tensor.",
                ),
                io.Boolean.Input(
                    "override_dtype",
                    default=False,
                    tooltip="Force grafted floating tensors to Save Dtype instead of preserving Model A's float precision.",
                ),
                io.Boolean.Input(
                    "include_mode",
                    default=False,
                    tooltip="Use Exclude Patterns as a whitelist instead. Only matching layers are eligible for grafting; nonmatching layers are preserved from Model A.",
                ),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_filename"),
                io.String.Output(display_name="documentation"),
                io.String.Output(display_name="cwb_report"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        documentation = load_documentation_from_file("cwb_model_graft.md")
        if kwargs["execution_mode"] == "DOCUMENTATION ONLY":
            return io.NodeOutput(
                "Documentation mode active. No merge performed.",
                documentation,
                "CWB report unavailable because no merge was performed.",
            )
        filename, report = cls._run_graft(kwargs)
        return io.NodeOutput(filename, documentation, report)

    @classmethod
    def _run_graft(cls, params: dict) -> tuple[str, str]:
        model_names = [params["model_a"], params["model_b"]]
        paths = []
        for name in model_names:
            path = folder_paths.get_full_path("diffusion_models", name)
            if not path:
                raise FileNotFoundError(f"diffusion_models input '{name}' was not found.")
            paths.append(path)

        total_size = sum(estimate_model_size(path) for path in paths)
        device = params["process_device"]
        if total_size:
            prepare_for_large_operation(total_size * 1.2, torch.device(device))

        custom_map = _parse_custom_key_map(params.get("custom_key_map", ""))
        attn_settings = resolve_cwb_settings(
            params["attention_preset"],
            DENSE_CWB_PRESETS,
            params.get("attention_config"),
        )
        ffn_settings = resolve_cwb_settings(
            params["ffn_preset"],
            DENSE_CWB_PRESETS,
            params.get("ffn_config"),
        )
        diagnostics = CWBDiagnostics()
        protect_early = int(params.get("protect_early_blocks", 2))
        attn_rule = params.get("attention_block_rule", "interleaved_even")
        ffn_rule = params.get("ffn_block_rule", "all_after_protected")
        attn_subpart = params.get("attention_subpart", "all")
        ffn_subpart = params.get("ffn_subpart", "all")
        glob_mode = bool(params.get("glob_patterns", False))
        include_mode = bool(params.get("include_mode", False))
        exclude = _compile_patterns(params.get("exclude_patterns", ""), glob_mode=glob_mode)
        discard = _compile_patterns(params.get("discard_patterns", ""), glob_mode=glob_mode)
        requested_dtype = _requested_dtype(params["save_dtype"])

        handlers = [
            MemoryEfficientSafeOpen(path, low_memory=params["lazy_load"])
            for path in paths
        ]
        try:
            quantizers = [
                DiffusionQuantization(handler, path, f"CWB Graft input {idx + 1}")
                for idx, (handler, path) in enumerate(zip(handlers, paths))
            ]
            key_maps = [
                diffusion_key_map(q.data_keys, f"CWB Graft input {idx + 1}")
                for idx, q in enumerate(quantizers)
            ]
            low_bit_sets = [
                {
                    logical
                    for logical, physical in mapping.items()
                    if physical in q.isolated_low_bit_keys
                }
                for mapping, q in zip(key_maps, quantizers)
            ]

            base_keys = sorted(key_maps[0])
            donor_keys_set = set(key_maps[1])
            block_indices = [
                idx
                for k in base_keys
                if (idx := _extract_main_block_index(k)) is not None
            ]
            max_block_index = max(block_indices) if block_indices else 0

            def matches_filter(key: str, patterns: list) -> bool:
                aliases = [key, key_maps[0][key]]
                return any(
                    _matches_any_pattern(alias, patterns, glob_mode=glob_mode)
                    for alias in aliases
                )

            def should_graft_key(key: str) -> tuple[bool, str | None, tuple[str, list[str], int | None] | None]:
                if matches_filter(key, discard):
                    return (False, None, None)
                has_filter_rules = bool(exclude)
                matched_filter = matches_filter(key, exclude)
                if include_mode and not matched_filter:
                    return (False, None, None)
                if not include_mode and matched_filter:
                    return (False, None, None)

                donor_plan = _resolve_donor_qkv_plan(key, donor_keys_set, custom_map)
                if donor_plan is None:
                    return (False, None, None)
                _, donor_logical_keys, _ = donor_plan
                if key in low_bit_sets[0] or any(dk in low_bit_sets[1] for dk in donor_logical_keys):
                    return (False, None, None)

                op_type = _classify_graft_operator(key)
                if key in custom_map or (include_mode and has_filter_rules and matched_filter):
                    return (True, op_type or "ffn", donor_plan)

                block_idx = _extract_main_block_index(key)
                if op_type == "attention":
                    if _block_rule_allows(block_idx, attn_rule, protect_early, max_block_index):
                        return (True, "attention", donor_plan)
                elif op_type == "ffn":
                    if _block_rule_allows(block_idx, ffn_rule, protect_early, max_block_index):
                        return (True, "ffn", donor_plan)
                return (False, None, None)

            work_units = []
            graft_plans: dict[str, tuple[str, tuple[str, list[str], int | None]]] = {}
            for key in base_keys:
                if matches_filter(key, discard):
                    work_units.append((key, {}))
                    continue
                graft_ok, op_type, donor_plan = should_graft_key(key)
                base_phys = key_maps[0][key]
                if not graft_ok:
                    entries = (
                        {0: quantizers[0].required_keys(base_phys)}
                        if base_phys in quantizers[0].quantized_keys
                        else {}
                    )
                    work_units.append((key, entries))
                    continue

                assert op_type is not None and donor_plan is not None
                graft_plans[key] = (op_type, donor_plan)
                donor_req: list[str] = []
                for dk in donor_plan[1]:
                    d_phys = key_maps[1][dk]
                    for rk in quantizers[1].required_keys(d_phys):
                        if rk not in donor_req:
                            donor_req.append(rk)
                work_units.append(
                    (
                        key,
                        {
                            0: quantizers[0].required_keys(base_phys),
                            1: donor_req,
                        },
                    )
                )

            output_path, output_rel = canonical_model_artifact_path(
                "diffusion_models", params["output_filename"]
            )
            output_metadata = quantizers[0].output_metadata()
            output_metadata["cwb.graft"] = json.dumps(
                {
                    "base_model": params["model_a"],
                    "donor_model": params["model_b"],
                    "attention_preset": params["attention_preset"],
                    "ffn_preset": params["ffn_preset"],
                    "protect_early_blocks": protect_early,
                    "attention_block_rule": attn_rule,
                    "ffn_block_rule": ffn_rule,
                    "attention_subpart": attn_subpart,
                    "ffn_subpart": ffn_subpart,
                },
                sort_keys=True,
            )

            def decode_logical(index: int, logical_key: str, loaded: dict) -> torch.Tensor:
                phys = key_maps[index][logical_key]
                if phys in quantizers[index].quantized_keys:
                    parts = {
                        part: loaded[(index, part)]
                        for part in quantizers[index].required_keys(phys)
                    }
                    return quantizers[index].decode(phys, parts, torch.float32, device="cpu")
                return loaded[(index, phys)]

            def preserve_base(writer, key: str, loaded: dict, *, force_raw: bool = False):
                phys = key_maps[0][key]
                if phys in quantizers[0].quantized_keys and (0, phys) in loaded:
                    decoded = decode_logical(0, key, loaded)
                    writer.write(key, decoded.to(requested_dtype).cpu().contiguous())
                else:
                    write_preserved_tensor(
                        writer,
                        phys,
                        handlers[0],
                        output_key=key,
                        force_raw=force_raw,
                        tensor=loaded.get((0, phys)),
                    )

            handler_map = {0: handlers[0], 1: handlers[1]}
            grafted_count = 0
            preserved_count = 0
            pbar = comfy.utils.ProgressBar(len(work_units))
            with atomic_uel_writer(output_path, output_metadata) as writer, torch.no_grad(), closing(
                stream_work_units(
                    handler_map,
                    work_units,
                    pin_memory=str(device).startswith("cuda"),
                )
            ) as streamed:
                for key, loaded in tqdm(streamed, total=len(work_units), desc="CWB grafting tensors", unit="tensors"):
                    _clear_previous_layer(params)
                    if matches_filter(key, discard):
                        pbar.update(1)
                        continue
                    if key not in graft_plans:
                        preserved_count += 1
                        preserve_base(writer, key, loaded, force_raw=key in low_bit_sets[0])
                        pbar.update(1)
                        continue

                    op_type, (mode, donor_logical_keys, slice_idx) = graft_plans[key]
                    base_tensor = decode_logical(0, key, loaded)
                    if mode == "direct":
                        donor_tensor = decode_logical(1, donor_logical_keys[0], loaded)
                    elif mode == "fuse_qkv":
                        donor_tensor = torch.cat(
                            [decode_logical(1, dk, loaded) for dk in donor_logical_keys],
                            dim=0,
                        )
                    else:
                        fused_donor = decode_logical(1, donor_logical_keys[0], loaded)
                        donor_tensor = fused_donor.chunk(3, dim=0)[slice_idx or 0]

                    base_phys = key_maps[0][key]
                    base_dtype = quantizers[0].logical_dtype(base_phys, requested_dtype)
                    if not _is_float_dtype(base_dtype) or not _is_float_dtype(donor_tensor.dtype):
                        preserved_count += 1
                        preserve_base(writer, key, loaded)
                        pbar.update(1)
                        continue

                    target_dtype = (
                        requested_dtype
                        if params.get("override_dtype", False)
                        else base_dtype
                    )
                    active_settings = attn_settings if op_type == "attention" else ffn_settings
                    active_subpart = attn_subpart if op_type == "attention" else ffn_subpart
                    grafted = _graft_with_subpart(
                        key,
                        op_type,
                        active_subpart,
                        base_tensor,
                        donor_tensor,
                        active_settings,
                        device,
                        target_dtype,
                        diagnostics,
                    )
                    if grafted is None:
                        preserved_count += 1
                        preserve_base(writer, key, loaded)
                    else:
                        grafted_count += 1
                        writer.write(key, grafted)
                    del base_tensor, donor_tensor, grafted
                    pbar.update(1)

            report = diagnostics.render(
                f"attn={params['attention_preset']}, ffn={params['ffn_preset']}",
                2,
                params.get("attention_config") is not None or params.get("ffn_config") is not None,
            )
            report += f"\nGrafted tensors: {grafted_count}\nPreserved base tensors: {preserved_count}\n"
            return output_rel, report
        finally:
            for handler in handlers:
                handler.__exit__(None, None, None)
            cleanup_after_operation()


CWB_GRAFT_NODES = [
    CWBModelGraft,
]
