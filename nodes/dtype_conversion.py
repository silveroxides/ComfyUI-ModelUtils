"""Asynchronous dtype conversion for diffusion-model safetensors files."""

from __future__ import annotations

import os
import re
import uuid
from collections import Counter
from contextlib import contextmanager

import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io
from tqdm import tqdm
from unifiedefficientloader import (
    IncrementalSafetensorsWriter,
    MemoryEfficientSafeOpen,
)

from .device_utils import cleanup_after_operation


TARGET_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _compile_exclusions(pattern_text: str) -> list[re.Pattern]:
    patterns = []
    for line_number, value in enumerate(pattern_text.splitlines(), start=1):
        pattern = value.strip()
        if not pattern:
            continue
        try:
            patterns.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(
                f"Invalid exclusion regex on line {line_number}: {exc}"
            ) from exc
    return patterns


def _is_excluded(key: str, patterns: list[re.Pattern]) -> bool:
    return any(pattern.search(key) for pattern in patterns)


def _output_path(output_filename: str) -> str:
    name = output_filename.strip()
    if not name:
        raise ValueError("Output filename must not be empty.")
    if name.lower().endswith(".safetensors"):
        name = name[:-12]
    output_dir = os.path.join(folder_paths.models_dir, "diffusion_models")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{name}.safetensors")


@contextmanager
def _atomic_writer(output_path: str, metadata: dict):
    directory = os.path.dirname(output_path) or "."
    basename = os.path.basename(output_path)
    temporary_path = os.path.join(
        directory, f".{basename}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with IncrementalSafetensorsWriter(
            temporary_path, metadata=metadata, max_workers=1
        ) as writer:
            yield writer
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
        raise


def convert_diffusion_model_dtype(
    model_name: str,
    target_dtype: str,
    exclude_patterns: str,
    output_filename: str,
) -> tuple[str, str]:
    try:
        destination_dtype = TARGET_DTYPES[target_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported target dtype: {target_dtype}") from exc

    source_path = folder_paths.get_full_path_or_raise(
        "diffusion_models", model_name
    )
    output_path = _output_path(output_filename)
    if os.path.normcase(os.path.abspath(source_path)) == os.path.normcase(
        os.path.abspath(output_path)
    ):
        raise ValueError("Output path must differ from the input model path.")

    exclusions = _compile_exclusions(exclude_patterns)
    counts = Counter()
    original_dtypes = Counter()

    try:
        with MemoryEfficientSafeOpen(source_path, low_memory=True) as loader:
            metadata = (loader.metadata() or {}).copy()
            with _atomic_writer(output_path, metadata) as writer:
                keys = list(loader.keys())
                progress = comfy.utils.ProgressBar(len(keys))
                stream = loader.async_stream(
                    keys,
                    batch_size=1,
                    prefetch_batches=1,
                    pin_memory=False,
                )
                for batch in tqdm(
                    stream,
                    total=len(keys),
                    desc="Converting diffusion-model dtype",
                    unit="tensors",
                ):
                    key, source = batch[0]
                    original_dtypes[str(source.dtype).removeprefix("torch.")] += 1
                    output = source
                    if _is_excluded(key, exclusions):
                        counts["excluded"] += 1
                    elif not source.is_floating_point():
                        counts["non_floating"] += 1
                    elif source.dtype == destination_dtype:
                        counts["already_target"] += 1
                    else:
                        output = source.to(dtype=destination_dtype).contiguous()
                        counts["converted"] += 1

                    writer.write_batch([(key, output)])
                    loader.mark_processed(key)
                    progress.update(1)
                    del output
                    del source
                    batch.clear()
                    del batch
    finally:
        cleanup_after_operation()

    total = sum(counts.values())
    dtype_summary = ", ".join(
        f"{name}={count}" for name, count in sorted(original_dtypes.items())
    )
    report = "\n".join(
        (
            "DIFFUSION MODEL DTYPE CONVERSION",
            f"Input: {model_name}",
            f"Output: {output_path}",
            f"Target dtype: {target_dtype}",
            f"Total tensors: {total}",
            f"Converted floating tensors: {counts['converted']}",
            f"Excluded tensors preserved: {counts['excluded']}",
            f"Already target dtype: {counts['already_target']}",
            f"Non-floating tensors preserved: {counts['non_floating']}",
            f"Original dtype counts: {dtype_summary or 'None'}",
        )
    )
    return output_path, report


class DiffusionModelDtypeConversion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffusionModelDtypeConversion",
            display_name="Diffusion Model Dtype Conversion",
            category="ModelUtils/Conversion",
            description=(
                "Streams a diffusion-model safetensors file into fp32, fp16, "
                "or bf16. Floating tensors matching an exclusion regex retain "
                "their original dtype; non-floating tensors are always preserved."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                    tooltip="Diffusion-model file whose floating tensors will be converted.",
                ),
                io.Combo.Input(
                    "target_dtype", options=["fp32", "fp16", "bf16"],
                    tooltip="Target dtype for floating tensors not matched by an exclusion pattern.",
                ),
                io.String.Input(
                    "exclude_patterns",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Optional Python regex patterns, one per line. Matching "
                        "tensor keys retain their original dtype."
                    ),
                ),
                io.String.Input(
                    "output_filename", default="converted_model",
                    tooltip="Output filename without extension, written under ComfyUI's diffusion-model directory.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
                io.String.Output(display_name="conversion_report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        target_dtype: str,
        exclude_patterns: str,
        output_filename: str,
    ) -> io.NodeOutput:
        output_path, report = convert_diffusion_model_dtype(
            model_name,
            target_dtype,
            exclude_patterns,
            output_filename,
        )
        return io.NodeOutput(output_path, report)
