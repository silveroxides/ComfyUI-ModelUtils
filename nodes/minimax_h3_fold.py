from __future__ import annotations

import fnmatch
import json
import math
import os
import re

import comfy.utils
from comfy_api.latest import io
import folder_paths
import torch
from unifiedefficientloader import MemoryEfficientSafeOpen

from .artifact_paths import canonical_model_artifact_path
from .device_utils import cleanup_after_operation
from .quantization_guard import inspect_low_bit_input
from .uel_io import atomic_uel_writer


def _compile_patterns(pattern_string: str, glob_mode: bool = False) -> list:
    """Compile newline or whitespace separated patterns."""
    if not pattern_string or not pattern_string.strip():
        return []
    patterns = []
    for pattern in pattern_string.split():
        if not pattern:
            continue
        if glob_mode:
            patterns.append(pattern)
        else:
            try:
                patterns.append(re.compile(pattern))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
    return patterns


def _matches_any_pattern(key: str, patterns: list, glob_mode: bool = False) -> bool:
    if not patterns:
        return False
    if glob_mode:
        return any(fnmatch.fnmatch(key, f"*{p}*") for p in patterns)
    return any(p.search(key) is not None for p in patterns)


def time_embedding_curve(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample silu(time_embedder(t)) on the curve grid."""
    t = torch.linspace(0.0, 1.0, grid_size, dtype=torch.float32, device=device)
    half = w1.shape[1] // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    w1_d, b1_d, w2_d, b2_d = (x.to(device, torch.float32) for x in (w1, b1, w2, b2))
    hidden = torch.nn.functional.silu(emb @ w1_d.T + b1_d)
    return torch.nn.functional.silu(hidden @ w2_d.T + b2_d)


def _fold_curve_basis(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    grid_size: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Calculate SVD curve basis and adaln_t_table."""
    compute_device = device
    try:
        curve = time_embedding_curve(w1, b1, w2, b2, grid_size, compute_device)
        u, s, vh = torch.linalg.svd(curve, full_matrices=False)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        cleanup_after_operation()
        compute_device = torch.device("cpu")
        curve = time_embedding_curve(w1, b1, w2, b2, grid_size, compute_device)
        u, s, vh = torch.linalg.svd(curve, full_matrices=False)

    target_rank = min(rank, vh.shape[0])
    table = u[:, :target_rank] * s[:target_rank]
    basis = vh[:target_rank].T
    error = ((table @ basis.T - curve).norm() / curve.norm()).item()
    return table.float().cpu().contiguous(), basis.float().cpu().contiguous(), error


def _detect_time_embedder_keys(all_keys: list[str]) -> tuple[str, dict[str, str]]:
    for prefix in ("", "diffusion_model.", "model.diffusion_model."):
        candidates = {
            "w1": f"{prefix}time_embedder.proj_in.weight",
            "b1": f"{prefix}time_embedder.proj_in.bias",
            "w2": f"{prefix}time_embedder.proj_out.weight",
            "b2": f"{prefix}time_embedder.proj_out.bias",
        }
        if all(k in all_keys for k in candidates.values()):
            return prefix, candidates
    raise KeyError(
        "Model is missing required time_embedder projection tensors (proj_in/proj_out)."
    )


def _is_adaln(key: str) -> bool:
    return ".adaln_proj.linear." in key or key.startswith("adaln_proj.linear.")


def fold_minimax_h3_diffusion_model(
    input_path: str,
    output_path: str,
    *,
    exclude_patterns: str = "",
    include_mode: bool = False,
    discard_patterns: str = "",
    glob_patterns: bool = False,
    curve_rank: int = 8,
    curve_grid: int = 1025,
    process_device: str = "cuda",
) -> str:
    """Stream and fold MiniMax H3 AdaLN linears in a diffusion model."""
    if os.path.normcase(os.path.abspath(input_path)) == os.path.normcase(
        os.path.abspath(output_path)
    ):
        raise ValueError("Output path must differ from the source model path.")

    device = torch.device(
        process_device if torch.cuda.is_available() and process_device == "cuda" else "cpu"
    )

    compiled_excludes = _compile_patterns(exclude_patterns, glob_patterns)
    compiled_discards = _compile_patterns(discard_patterns, glob_patterns)

    handler = MemoryEfficientSafeOpen(input_path, low_memory=True)
    try:
        inspect_low_bit_input(handler, f"Model ({input_path})", "MiniMax H3 Fold AdaLN")
        all_keys = list(handler.keys())
        prefix, time_keys = _detect_time_embedder_keys(all_keys)

        # Step 1: Load time embedder and calculate basis
        stream = handler.async_stream(
            list(time_keys.values()),
            batch_size=1,
            prefetch_batches=1,
            pin_memory=False,
        )
        time_tensors = {}
        try:
            for batch in stream:
                for k, t in batch:
                    time_tensors[k] = t
            table, basis, error = _fold_curve_basis(
                time_tensors[time_keys["w1"]],
                time_tensors[time_keys["b1"]],
                time_tensors[time_keys["w2"]],
                time_tensors[time_keys["b2"]],
                curve_grid,
                curve_rank,
                device,
            )
        finally:
            time_tensors.clear()
            for k in time_keys.values():
                handler.mark_processed(k)
            stream.close()

        # Determine which AdaLN layers are folded vs kept full width
        any_adaln_kept_unfolded = False
        for k in all_keys:
            if _is_adaln(k) and not _matches_any_pattern(k, compiled_discards, glob_patterns):
                matched = _matches_any_pattern(k, compiled_excludes, glob_patterns)
                should_fold = matched if include_mode else not matched
                if not should_fold:
                    any_adaln_kept_unfolded = True
                    break

        metadata = dict(handler.metadata() or {})
        if "config" in metadata:
            try:
                cfg = json.loads(metadata["config"])
                if isinstance(cfg, dict):
                    tf = cfg.setdefault("transformer", {})
                    tf["adaln_curve_grid"] = table.shape[0]
                    tf["time_embed_dim"] = table.shape[1]
                    metadata["config"] = json.dumps(cfg, separators=(",", ":"))
            except Exception:
                pass
        metadata["modelutils_adaln_fold"] = f"rank{table.shape[1]}_curve_grid_{table.shape[0]}"

        table_key = f"{prefix}adaln_t_table"
        time_embedder_prefix = f"{prefix}time_embedder."

        counts = {
            "folded": 0,
            "preserved": 0,
            "discarded": 0,
            "time_embedder_dropped": 0,
        }

        basis_device = basis.to(device)

        with atomic_uel_writer(output_path, metadata) as writer:
            # Emit adaln_t_table
            writer.write_batch([(table_key, table)])

            # Filter remaining keys
            remaining_keys = [k for k in all_keys if k != table_key]
            progress = comfy.utils.ProgressBar(len(remaining_keys))

            for key in remaining_keys:
                if _matches_any_pattern(key, compiled_discards, glob_patterns):
                    counts["discarded"] += 1
                    handler.mark_processed(key)
                    progress.update(1)
                    continue

                if key.startswith(time_embedder_prefix) and not any_adaln_kept_unfolded:
                    counts["time_embedder_dropped"] += 1
                    handler.mark_processed(key)
                    progress.update(1)
                    continue

                # Stream this tensor
                tstream = handler.async_stream(
                    [key], batch_size=1, prefetch_batches=1, pin_memory=False
                )
                try:
                    batch = next(tstream)
                    tensor = batch[0][1]

                    if _is_adaln(key):
                        matched = _matches_any_pattern(key, compiled_excludes, glob_patterns)
                        should_fold = matched if include_mode else not matched
                        if should_fold:
                            if key.endswith(".weight"):
                                if tensor.ndim != 2 or tensor.shape[1] != basis.shape[0]:
                                    raise ValueError(
                                        f"{key} has shape {tuple(tensor.shape)}, expected [*, {basis.shape[0]}]"
                                    )
                                folded_w = (
                                    tensor.to(device, torch.float32) @ basis_device
                                ).to(torch.float16).cpu().contiguous()
                                writer.write_batch([(key, folded_w)])
                            else:
                                writer.write_batch([(key, tensor.to(torch.float16).contiguous())])
                            counts["folded"] += 1
                        else:
                            writer.write_batch([(key, tensor.contiguous())])
                            counts["preserved"] += 1
                    else:
                        writer.write_batch([(key, tensor.contiguous())])
                        counts["preserved"] += 1
                finally:
                    handler.mark_processed(key)
                    tstream.close()

                progress.update(1)
    finally:
        handler.__exit__(None, None, None)
        cleanup_after_operation()

    report_lines = [
        "MiniMax H3 AdaLN Fold complete",
        f"Curve rank: {curve_rank}, grid points: {curve_grid}",
        f"Reconstruction relative error: {error:.4e}",
        f"AdaLN layers folded: {counts['folded']}",
        f"Tensors preserved: {counts['preserved']}",
        f"Tensors discarded: {counts['discarded']}",
        f"Time embedder dropped: {counts['time_embedder_dropped']}",
        f"Output size: {os.path.getsize(output_path)} bytes",
        f"Output: {output_path}",
    ]
    return "\n".join(report_lines)


class MiniMaxH3FoldAdaLN(io.ComfyNode):
    """Fold MiniMax H3 full-width AdaLN linears [*, 2688] onto rank-8 time-embedding curve basis."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FoldAdaLN",
            display_name="MiniMax H3 Fold AdaLN",
            category="ModelUtils/DiffusionModels",
            description="Folds MiniMax H3 full-width AdaLN linear layers onto a low-rank basis of the silu(time_embedder) curve, replacing the time embedder with adaln_t_table.",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                    tooltip="MiniMax H3 diffusion model with full-width AdaLN linears [*, 2688] to fold.",
                ),
                io.String.Input(
                    "output_filename",
                    default="minimax_h3_folded",
                    tooltip="Output filename without extension, written under ComfyUI's diffusion_models directory.",
                ),
                io.String.Input(
                    "discard_patterns",
                    default="",
                    multiline=True,
                    tooltip="Newline-separated regex or glob patterns for tensors to omit entirely from the saved model (e.g. drop specific transformer blocks or refiner).",
                ),
                io.Boolean.Input(
                    "glob_patterns",
                    default=False,
                    tooltip="When True, filter patterns use shell globs (* matches any sequence, dots are literal). When False (default), patterns are Python regex matched as substrings.",
                ),
                io.Int.Input(
                    "curve_rank",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Basis rank for the silu(time_embedder) curve (default 8, matching ComfyUI H3 checkpoints).",
                ),
                io.Int.Input(
                    "curve_grid",
                    default=1025,
                    min=65,
                    max=4097,
                    step=64,
                    tooltip="Number of interpolation grid points sampled for timestep t in [0, 1] (default 1025).",
                ),
                io.Combo.Input(
                    "process_device",
                    options=["cuda", "cpu"],
                    default="cuda",
                    tooltip="Device used for SVD curve decomposition and matrix projection; CUDA OOM automatically retries on CPU.",
                ),
                io.String.Input(
                    "exclude_patterns",
                    default="",
                    multiline=True,
                    tooltip="Newline-separated regex or glob patterns. Matching AdaLN layers remain unfolded at full width [*, 2688].",
                ),
                io.Boolean.Input(
                    "include_mode",
                    default=False,
                    tooltip="When True, use exclude_patterns as a whitelist: only matching AdaLN layers are folded; all others remain full width.",
                ),
            ],
            outputs=[
                io.AnyType.Output(display_name="output_path"),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        output_filename: str,
        discard_patterns: str = "",
        glob_patterns: bool = False,
        curve_rank: int = 8,
        curve_grid: int = 1025,
        process_device: str = "cuda",
        exclude_patterns: str = "",
        include_mode: bool = False,
    ) -> io.NodeOutput:
        input_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", model_name
        )
        output_path, output_name = canonical_model_artifact_path(
            "diffusion_models", output_filename
        )
        report = fold_minimax_h3_diffusion_model(
            input_path,
            output_path,
            exclude_patterns=exclude_patterns,
            include_mode=include_mode,
            discard_patterns=discard_patterns,
            glob_patterns=glob_patterns,
            curve_rank=curve_rank,
            curve_grid=curve_grid,
            process_device=process_device,
        )
        return io.NodeOutput(output_name, report)
