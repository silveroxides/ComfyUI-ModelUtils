"""
Memory-efficient safetensors utilities with pinned memory support.
Forces direct file reads to minimize OS page cache usage and avoids VRAM accumulation.
"""
import os
import gc
import json
import struct
import torch
from safetensors import safe_open
from typing import Dict, List, Optional, Any

# Module-level configuration for pinned transfer
_verbose = False
_pinned_transfer_stats = {"pinned": 0, "fallback": 0}

def set_verbose(enabled: bool):
    """Enable/disable verbose output for pinned transfers."""
    global _verbose
    _verbose = enabled

def get_pinned_transfer_stats():
    """Return pinned transfer statistics."""
    return _pinned_transfer_stats.copy()

def reset_pinned_transfer_stats():
    """Reset transfer statistics."""
    global _pinned_transfer_stats
    _pinned_transfer_stats = {"pinned": 0, "fallback": 0}

def transfer_to_gpu_pinned(
    tensor: torch.Tensor,
    device: str = 'cuda',
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Transfer tensor to GPU using pinned memory for faster transfer."""
    global _pinned_transfer_stats

    if tensor.device.type != 'cpu' or not torch.cuda.is_available():
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    if not str(device).startswith('cuda'):
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    try:
        pinned = tensor.pin_memory()
        if dtype is not None:
            result = pinned.to(device=device, dtype=dtype, non_blocking=True)
        else:
            result = pinned.to(device=device, non_blocking=True)

        torch.cuda.current_stream().synchronize()
        _pinned_transfer_stats["pinned"] += 1
        return result
    except Exception as e:
        _pinned_transfer_stats["fallback"] += 1
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)


class MemoryEfficientSafeOpen:
    """Memory-efficient safetensors file reader with streaming and pinned memory support.

    Operates strictly in streaming mode (low memory). Tensors are read byte-by-byte from
    their exact offsets without using safe_open or memory mapping, ensuring tensors do not
    remain persistently in OS page caches or RAM.
    """

    def __init__(self, filename: str, device: str = 'cpu', **kwargs):
        # Ignore mmap_mode, low_memory, or any other legacy kwargs. We always stream.
        self.filename = filename
        self.device = device
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self) -> List[str]:
        """Return all tensor keys (excluding metadata)."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        """Return file metadata."""
        return self.header.get("__metadata__", {})

    def get_shape(self, key: str) -> tuple:
        """Get tensor shape without loading tensor data."""
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in file")
        return tuple(self.header[key]["shape"])

    def get_ndim(self, key: str) -> int:
        """Get tensor ndim without loading tensor data."""
        return len(self.get_shape(key))

    def keys_sorted_by_offset(self) -> List[str]:
        keys_with_offsets = []
        for key in self.keys():
            offset = self.header[key]["data_offsets"][0]
            keys_with_offsets.append((key, offset))
        keys_with_offsets.sort(key=lambda x: x[1])
        return [k for k, _ in keys_with_offsets]

    def get_tensor(self, key: str) -> torch.Tensor:
        """Load a single tensor by key directly from file byte offset."""
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start != offset_end:
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = bytearray(offset_end - offset_start)
            self.file.readinto(tensor_bytes)
        else:
            tensor_bytes = None

        return self._deserialize_tensor(tensor_bytes, metadata)

    def get_tensor_to_gpu(
        self,
        key: str,
        device: str = 'cuda',
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        cpu_tensor = self.get_tensor(key)
        return transfer_to_gpu_pinned(cpu_tensor, device, dtype)

    def get_tensors_parallel(
        self,
        keys: Optional[List[str]] = None,
        workers: Optional[int] = None,
        sort_by_offset: bool = True
    ) -> Dict[str, torch.Tensor]:
        if keys is None:
            keys = self.keys()

        if workers is None:
            try:
                from .device_utils import get_optimal_workers, estimate_model_size
                model_size = estimate_model_size(self.filename)
                workers = get_optimal_workers(model_size)
            except ImportError:
                workers = min(4, os.cpu_count() or 4)

        if sort_by_offset:
            keys_with_offsets = [(k, self.header[k]["data_offsets"][0]) for k in keys]
            keys_with_offsets.sort(key=lambda x: x[1])
            keys = [k for k, _ in keys_with_offsets]

        result = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _read_tensor_from_file,
                    self.filename,
                    key,
                    self.header[key],
                    self.header_size
                ): key
                for key in keys
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    tensor = future.result()
                    if tensor is not None:
                        result[key] = tensor
                except Exception as e:
                    print(f"[MemoryEfficientSafeOpen] Failed to load {key}: {e}")

        return result

    def get_tensor_as_dict(self, key: str) -> Dict[str, Any]:
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            return {}

        self.file.seek(self.header_size + 8 + offset_start)
        tensor_bytes = self.file.read(offset_end - offset_start)
        return json.loads(tensor_bytes.decode("utf-8"))

    def _read_header(self):
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
            "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
            "U8": torch.uint8, "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}")


def _read_tensor_from_file(
    filepath: str,
    key: str,
    metadata: Dict,
    header_size: int
) -> Optional[torch.Tensor]:
    """Helper function to read a single tensor from file (for parallel execution).

    Each worker opens its own file handle for thread safety.
    """
    offset_start, offset_end = metadata["data_offsets"]
    dtype_str = metadata["dtype"]
    shape = metadata["shape"]

    if offset_start == offset_end:
        return None

    dtype_map = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
        "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
        "U8": torch.uint8, "BOOL": torch.bool,
    }
    if hasattr(torch, "float8_e5m2"):
        dtype_map["F8_E5M2"] = torch.float8_e5m2
    if hasattr(torch, "float8_e4m3fn"):
        dtype_map["F8_E4M3"] = torch.float8_e4m3fn

    dtype = dtype_map.get(dtype_str, torch.float32)

    with open(filepath, "rb") as f:
        f.seek(header_size + 8 + offset_start)
        tensor_bytes = f.read(offset_end - offset_start)

    byte_tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.uint8)

    if dtype_str in ["F8_E5M2", "F8_E4M3"]:
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)

    return byte_tensor.view(dtype).reshape(shape)


def get_union_keys(*handlers) -> List[str]:
    """Returns the union of all tensor keys across multiple model handlers."""
    all_keys = set()
    for handler in handlers:
        if handler is not None:
            all_keys.update(handler.keys())
    return sorted(all_keys)


def get_intersection_keys(*handlers) -> List[str]:
    """Returns only keys present in ALL model handlers."""
    handlers = [h for h in handlers if h is not None]
    if not handlers:
        return []
    all_keys = set(handlers[0].keys())
    for handler in handlers[1:]:
        all_keys &= set(handler.keys())
    return sorted(all_keys)