"""Fail-safe detection for quantized or isolated low-bit safetensors inputs."""

import logging
import json
import struct

import torch


MAX_ISOLATED_LOW_BIT_TENSORS = 2


def _low_bit_dtypes():
    names = (
        "int8",
        "uint8",
        "float4_e2m1fn_x2",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    return {getattr(torch, name) for name in names if hasattr(torch, name)}


LOW_BIT_DTYPES = _low_bit_dtypes()
LOW_BIT_STORAGE_CODES = {"I8", "U8", "F4"}
UPSTREAM_WRITER_STORAGE_CODES = {
    "F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8",
    "U64", "U32", "U16", "BOOL", "C64", "F8_E5M2", "F8_E4M3",
}


def _header(handler):
    if getattr(handler, "_header", None) is not None:
        return handler._header
    with open(handler.filename, "rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        return json.loads(source.read(header_size).decode("utf-8"))


def storage_dtype_code(handler, key: str) -> str | None:
    entry = _header(handler).get(key)
    return entry.get("dtype") if entry else None


def _is_low_bit(handler, key: str) -> bool:
    storage_code = storage_dtype_code(handler, key)
    if storage_code in LOW_BIT_STORAGE_CODES or (
        storage_code is not None and storage_code.startswith("F8_")
    ):
        return True
    return handler.get_dtype(key) in LOW_BIT_DTYPES


def inspect_low_bit_input(handler, label: str, operation: str) -> set[str]:
    """Reject quantized models; return isolated low-bit keys that must not be transformed."""
    keys = list(handler.keys())
    metadata = handler.metadata() or {}
    quant_keys = [key for key in keys if key.endswith(".comfy_quant")]
    has_quant_metadata = "_quantization_metadata" in metadata
    has_legacy_scaled_fp8 = any(key.endswith("scaled_fp8") for key in keys)
    if quant_keys or has_quant_metadata or has_legacy_scaled_fp8:
        raise ValueError(
            f"[{operation}] {label} uses ComfyUI quantization metadata; quantized inputs are unsupported"
        )

    low_bit_keys = {key for key in keys if _is_low_bit(handler, key)}
    if len(low_bit_keys) > MAX_ISOLATED_LOW_BIT_TENSORS:
        raise ValueError(
            f"[{operation}] {label} contains {len(low_bit_keys)} low-bit tensors; "
            "casted or quantized inputs are unsupported"
        )
    if low_bit_keys:
        logging.warning(
            "[%s] %s contains %d isolated low-bit tensor(s); "
            "preserving/excluding unchanged: %s",
            operation,
            label,
            len(low_bit_keys),
            ", ".join(sorted(low_bit_keys)),
        )
    return low_bit_keys


def layer_has_low_bit(block_keys: dict, low_bit_keys: set[str]) -> bool:
    return any(
        block_keys.get(name) in low_bit_keys
        for name in ("down", "up", "alpha", "diff", "diff_b")
    )


def write_preserved_tensor(writer, key: str, handler) -> None:
    """Copy one tensor's original safetensors bytes and dtype into an active writer."""
    header = _header(handler)
    entry = header[key]
    if entry["dtype"] in UPSTREAM_WRITER_STORAGE_CODES:
        writer.write(key, handler.get_tensor(key).cpu().contiguous())
        return

    source_start, source_end = entry["data_offsets"]
    byte_size = source_end - source_start

    with writer._lock:
        if key in writer._manifest:
            raise ValueError(f"Tensor '{key}' has already been written.")
        target_start = writer._current_data_offset
        target_end = target_start + byte_size
        writer._manifest[key] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [target_start, target_end],
        }
        writer._current_data_offset = target_end

    source_path = handler.filename
    source_absolute = 8 + getattr(handler, "_header_size", 0) + source_start
    if not getattr(handler, "_header_size", None):
        with open(source_path, "rb") as source:
            source_absolute = 8 + struct.unpack("<Q", source.read(8))[0] + source_start
    target_absolute = 8 + writer.max_header_bytes + target_start

    def copy_raw():
        try:
            with open(source_path, "rb") as source:
                source.seek(source_absolute)
                remaining = byte_size
                with writer._lock:
                    writer._file.seek(target_absolute)
                    while remaining:
                        chunk = source.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise EOFError(f"Unexpected EOF while preserving tensor '{key}'")
                        writer._file.write(chunk)
                        remaining -= len(chunk)
        finally:
            writer._semaphore.release()

    writer._semaphore.acquire()
    writer._futures.append(writer._executor.submit(copy_raw))
