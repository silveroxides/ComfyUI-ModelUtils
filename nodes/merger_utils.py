import json
import struct
import torch

class MemoryEfficientSafeOpen:
    def __init__(self, filename, device='cpu'):
        self.filename = filename
        self.device = device
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        tensor_bytes = None
        if offset_start != offset_end:
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

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
            byte_tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.uint8)

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
            raise ValueError(f"Unsupported float8 type: {dtype_str}. Your PyTorch version may be too old.")


def get_union_keys(*handlers):
    """Returns the union of all tensor keys across multiple model handlers.
    
    Useful when you want to iterate over all unique keys present in any model.
    """
    all_keys = set()
    for handler in handlers:
        if handler is not None:
            all_keys.update(handler.keys())
    return sorted(all_keys)


def get_intersection_keys(*handlers):
    """Returns only keys present in ALL model handlers.
    
    Useful when you want to iterate only over keys that exist in every model.
    """
    handlers = [h for h in handlers if h is not None]
    if not handlers:
        return []
    all_keys = set(handlers[0].keys())
    for handler in handlers[1:]:
        all_keys &= set(handler.keys())
    return sorted(all_keys)