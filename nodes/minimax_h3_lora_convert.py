from __future__ import annotations

from dataclasses import dataclass
import os
import re

import comfy.utils
from comfy_api.latest import io
import folder_paths
import torch
from unifiedefficientloader import MemoryEfficientSafeOpen

from .artifact_paths import canonical_model_artifact_path
from .quantization_guard import inspect_low_bit_input
from .uel_io import atomic_uel_writer

_FACTOR_RE = re.compile(
    r"^(?P<module>.+)\.(?:lora_(?P<factor_ab>[AB])|lora\.(?P<factor_word>down|up))(?:\.default)?\.weight$"
)
_BLOCK_RE = re.compile(r"^transformer_blocks\.(?P<index>\d+)\.(?P<tail>.+)$")
_REFINER_RE = re.compile(r"^token_refiner\.refiner_blocks\.(?P<index>\d+)\.(?P<tail>.+)$")
_QKV_RE = re.compile(r"^attn\.to_(?P<part>[qkv])$")

_DIRECT_TAILS = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
}

_EXTRA_PREFIX_MAP = {
    "context_embedder": "condition_proj",
    "proj_in": "video_patch_proj",
    "proj_out": "final_layer.video_out",
    "audio_proj_in": "audio_patch_proj",
    "audio_proj_out": "final_layer.audio_out",
    "norm_out.norm": "final_layer.norm",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
}


@dataclass(frozen=True)
class _ConversionUnit:
    target: str
    sources: tuple[str, ...]
    qkv: bool
    is_swiglu: bool = False


def _target_module(module: str) -> tuple[str, str | None, bool]:
    """Map Diffusers module name to ComfyUI native target module.

    Returns:
        (target_module, qkv_part, is_swiglu_fc1)
    """
    clean_mod = module.removeprefix("transformer.")
    match = _BLOCK_RE.fullmatch(clean_mod)
    if match:
        prefix = f"diffusion_model.blocks.{match.group('index')}"
        tail = match.group("tail")
    else:
        match = _REFINER_RE.fullmatch(clean_mod)
        if match:
            prefix = f"diffusion_model.token_refiner.blocks.{match.group('index')}"
            tail = match.group("tail")
        else:
            for old_pfx, new_pfx in _EXTRA_PREFIX_MAP.items():
                if clean_mod.startswith(old_pfx):
                    target = f"diffusion_model.{new_pfx}{clean_mod[len(old_pfx):]}"
                    return target, None, False
            raise ValueError(f"Unsupported MiniMax H3 Diffusers module: {module}")

    qkv = _QKV_RE.fullmatch(tail)
    if qkv:
        return f"{prefix}.attn.qkv_proj", qkv.group("part"), False

    if tail in _DIRECT_TAILS:
        target = f"{prefix}.{_DIRECT_TAILS[tail]}"
        is_swiglu = (tail == "ff.net.0.proj")
        return target, None, is_swiglu

    raise ValueError(f"Unsupported MiniMax H3 Diffusers module tail: {module}")


def _build_plan(keys: list[str]) -> list[_ConversionUnit]:
    """Build conversion work units from input keys."""
    factors: dict[str, dict[str, str]] = {}
    for key in keys:
        match = _FACTOR_RE.fullmatch(key)
        if not match:
            raise ValueError(
                f"Input is not a compatible MiniMax H3 Diffusers PEFT LoRA; unsupported key: {key}"
            )
        mod = match.group("module")
        factor = match.group("factor_ab") or (
            "A" if match.group("factor_word") == "down" else "B"
        )
        pair = factors.setdefault(mod, {})
        if factor in pair:
            raise ValueError(f"Duplicate LoRA factor for: {mod}")
        pair[factor] = key

    missing = [mod for mod, pair in factors.items() if set(pair.keys()) != {"A", "B"}]
    if missing:
        raise ValueError(f"Incomplete LoRA A/B pair for: {missing[0]}")

    direct_units: list[_ConversionUnit] = []
    qkv_groups: dict[str, dict[str, tuple[str, str]]] = {}

    for mod, pair in factors.items():
        target, qkv_part, is_swiglu = _target_module(mod)
        source_pair = (pair["A"], pair["B"])
        if qkv_part is None:
            direct_units.append(
                _ConversionUnit(
                    target=target,
                    sources=source_pair,
                    qkv=False,
                    is_swiglu=is_swiglu,
                )
            )
        else:
            qkv_groups.setdefault(target, {})[qkv_part] = source_pair

    qkv_units: list[_ConversionUnit] = []
    for target, parts in qkv_groups.items():
        if set(parts.keys()) != {"q", "k", "v"}:
            raise ValueError(f"Incomplete Q/K/V LoRA group for: {target}")
        sources = tuple(k for part in ("q", "k", "v") for k in parts[part])
        qkv_units.append(_ConversionUnit(target=target, sources=sources, qkv=True))

    units = sorted(direct_units + qkv_units, key=lambda u: u.target)
    targets = [u.target for u in units]
    if len(targets) != len(set(targets)):
        raise ValueError("Conversion would produce duplicate ComfyUI module keys")

    return units


def _validate_pair(down: torch.Tensor, up: torch.Tensor, label: str) -> None:
    if down.ndim != 2 or up.ndim != 2:
        raise ValueError(f"LoRA factors for '{label}' must be 2D matrices")
    if down.shape[0] != up.shape[1]:
        raise ValueError(
            f"LoRA factor rank mismatch for '{label}': down={list(down.shape)}, up={list(up.shape)}"
        )


def _convert_qkv(
    tensors: dict[str, torch.Tensor],
    sources: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse separate Q, K, V LoRA factors into unified block-diagonal qkv_proj factors."""
    pairs = [
        (tensors[sources[i]], tensors[sources[i + 1]])
        for i in range(0, 6, 2)
    ]
    for part, (down, up) in zip(("Q", "K", "V"), pairs):
        _validate_pair(down, up, part)

    in_dims = {down.shape[1] for down, _ in pairs}
    if len(in_dims) != 1:
        raise ValueError("Q/K/V LoRA input dimensions do not match")

    dtypes = {t.dtype for pair in pairs for t in pair}
    if len(dtypes) != 1:
        raise ValueError("Q/K/V LoRA dtypes do not match")

    source_dtype = pairs[0][0].dtype
    ranks = [down.shape[0] for down, _ in pairs]
    output_dims = [up.shape[0] for _, up in pairs]
    total_rank = sum(ranks)
    total_out = sum(output_dims)

    # Concatenate down matrices vertically: [total_rank, in_dim]
    exact_down = torch.cat([down for down, _ in pairs], dim=0).contiguous()

    # Place up matrices block-diagonally: [total_out, total_rank]
    exact_up = torch.zeros(
        (total_out, total_rank),
        dtype=source_dtype,
        device=pairs[0][1].device,
    )
    row = 0
    col = 0
    for (_, part_up), out_dim, part_rank in zip(pairs, output_dims, ranks):
        exact_up[row : row + out_dim, col : col + part_rank].copy_(part_up)
        row += out_dim
        col += part_rank

    return exact_down, exact_up.contiguous()


def _convert_swiglu(
    down: torch.Tensor, up: torch.Tensor, label: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap SwiGLU output halves from Diffusers [value; gate] to ComfyUI [gate; value]."""
    _validate_pair(down, up, label)
    if up.shape[0] % 2 != 0:
        raise ValueError(
            f"SwiGLU LoRA B factor for '{label}' has odd output dimension {up.shape[0]}"
        )
    val_b, gate_b = up.chunk(2, dim=0)
    swapped_up = torch.cat([gate_b, val_b], dim=0).contiguous()
    return down.contiguous(), swapped_up


def convert_minimax_h3_diffusers_lora(
    input_path: str,
    output_path: str,
) -> str:
    """Stream and convert Diffusers MiniMax H3 LoRA into ComfyUI native format."""
    if os.path.normcase(os.path.abspath(input_path)) == os.path.normcase(
        os.path.abspath(output_path)
    ):
        raise ValueError("Output path must differ from the source LoRA path")

    handler = MemoryEfficientSafeOpen(input_path, low_memory=True)
    try:
        inspect_low_bit_input(handler, f"LoRA ({input_path})", "MiniMax H3 LoRA Convert")
        keys = list(handler.keys())
        source_count = len(keys)
        units = _build_plan(keys)

        metadata = {str(k): str(v) for k, v in (handler.metadata() or {}).items()}
        metadata.update(
            {
                "modelutils_conversion": "diffusers_minimax_h3_to_comfyui",
                "modelutils_qkv_fusion": "lossless_block_diagonal",
                "modelutils_mlp_swiglu": "swapped_gate_value",
            }
        )

        progress = comfy.utils.ProgressBar(len(units))
        output_count = 0
        fused_qkv_count = 0

        with atomic_uel_writer(output_path, metadata) as writer:
            for unit in units:
                stream = handler.async_stream(
                    list(unit.sources),
                    batch_size=1,
                    prefetch_batches=1,
                    pin_memory=False,
                )
                tensors = {}
                try:
                    for batch in stream:
                        for k, t in batch:
                            tensors[k] = t

                    if unit.qkv:
                        down, up = _convert_qkv(tensors, unit.sources)
                        fused_qkv_count += 1
                    elif unit.is_swiglu:
                        down_key, up_key = unit.sources
                        down, up = _convert_swiglu(
                            tensors[down_key], tensors[up_key], unit.target
                        )
                    else:
                        down_key, up_key = unit.sources
                        down = tensors[down_key].contiguous()
                        up = tensors[up_key].contiguous()
                        _validate_pair(down, up, unit.target)

                    writer.write_batch(
                        [
                            (f"{unit.target}.lora_A.weight", down),
                            (f"{unit.target}.lora_B.weight", up),
                        ]
                    )
                    output_count += 2
                finally:
                    tensors.clear()
                    for key in unit.sources:
                        handler.mark_processed(key)
                    stream.close()

                progress.update(1)
    finally:
        handler.__exit__(None, None, None)

    lines = [
        "MiniMax H3 Diffusers LoRA conversion complete",
        f"Source tensors: {source_count}",
        f"Output tensors: {output_count}",
        f"Fused QKV modules: {fused_qkv_count}",
        f"Output size: {os.path.getsize(output_path)} bytes",
        f"Output: {output_path}",
    ]
    return "\n".join(lines)


class MiniMaxH3DiffusersLoRAConvert(io.ComfyNode):
    """Convert Diffusers MiniMax H3 LoRAs to ComfyUI's native layout."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DiffusersLoRAConvert",
            display_name="MiniMax H3 Diffusers LoRA Convert",
            category="ModelUtils/LoRA",
            description="Converts Diffusers/PEFT MiniMax H3 LoRAs to ComfyUI's native fused layout (lossless block-diagonal QKV and SwiGLU-aligned MLP).",
            inputs=[
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="Diffusers-format MiniMax H3 LoRA to convert.",
                ),
                io.String.Input(
                    "output_filename",
                    default="minimax_h3_comfyui",
                    tooltip="Output filename without extension, written under ComfyUI's LoRA directory.",
                ),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
                io.String.Output(display_name="conversion_report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        lora_name: str,
        output_filename: str,
    ) -> io.NodeOutput:
        input_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        output_path, output_name = canonical_model_artifact_path(
            "loras", output_filename
        )
        report = convert_minimax_h3_diffusers_lora(input_path, output_path)
        return io.NodeOutput(output_name, report)
