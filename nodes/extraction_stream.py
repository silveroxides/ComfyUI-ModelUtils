"""Bounded paired UEL streaming helpers for model extraction nodes."""

import torch

from .uel_io import AsyncTensorCursor

def is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "cuda" in str(error).lower()
        and "out of memory" in str(error).lower()
    )


def release_cuda_oom():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_source_tensors(cpu_a, cpu_b, device, dtype=torch.float32):
    """Move one source pair, retrying a confirmed CUDA OOM on CPU."""
    use_cuda = str(device).startswith("cuda")
    try:
        if use_cuda:
            tensor_a = cpu_a.to(device=device, dtype=dtype, non_blocking=True)
            tensor_b = cpu_b.to(device=device, dtype=dtype, non_blocking=True) if cpu_b is not None else None
        else:
            tensor_a = cpu_a.to(device=device, dtype=dtype)
            tensor_b = cpu_b.to(device=device, dtype=dtype) if cpu_b is not None else None
        return tensor_a, tensor_b, device
    except BaseException as error:
        if not use_cuda or not is_cuda_oom(error):
            raise
        release_cuda_oom()
        return (
            cpu_a.to(device="cpu", dtype=dtype),
            cpu_b.to(device="cpu", dtype=dtype) if cpu_b is not None else None,
            "cpu",
        )


def retry_cuda_oom_on_cpu(cuda_call, cpu_call, device):
    """Retry one compute operation on CPU only for a confirmed CUDA OOM."""
    try:
        return cuda_call()
    except BaseException as error:
        if not str(device).startswith("cuda") or not is_cuda_oom(error):
            raise
        release_cuda_oom()
        return cpu_call()


def lora_difference(cpu_a, cpu_b, device):
    """Construct A minus B with transfer and arithmetic CUDA OOM fallback."""
    tensor_a, tensor_b, actual_device = move_source_tensors(
        cpu_a, cpu_b, device
    )
    if tensor_b is None:
        return tensor_a, actual_device
    try:
        return tensor_a - tensor_b, actual_device
    except BaseException as error:
        if not str(actual_device).startswith("cuda") or not is_cuda_oom(error):
            raise
        del tensor_a, tensor_b
        release_cuda_oom()
        return cpu_a.float() - cpu_b.float(), "cpu"


def dora_difference(cpu_a, cpu_b, device, *, keep_sources=False):
    """Construct a directional DoRA delta with complete CUDA OOM fallback."""
    tensor_a, tensor_b, actual_device = move_source_tensors(
        cpu_a, cpu_b, device
    )

    def compute(a, b):
        if b is None:
            base = torch.zeros_like(a)
            result = (a, None, base, a)
            return result if keep_sources else result[:2]
        if a.ndim < 2:
            result = (None, None, b, a)
            return result if keep_sources else result[:2]
        is_conv = a.ndim == 4
        if is_conv:
            out_channels = a.shape[0]
            norm_a = torch.linalg.norm(a.reshape(out_channels, -1), dim=1, keepdim=True)
            norm_b = torch.linalg.norm(b.reshape(out_channels, -1), dim=1, keepdim=True)
        else:
            norm_a = torch.linalg.norm(a, dim=1, keepdim=True)
            norm_b = torch.linalg.norm(b, dim=1, keepdim=True)
        ratio = norm_b / (norm_a + 1e-8)
        if is_conv:
            ratio = ratio.view(-1, 1, 1, 1)
            scale = norm_a.view(-1, 1, 1, 1)
        else:
            scale = norm_a
        result = (a * ratio - b, scale, b, a)
        return result if keep_sources else result[:2]

    try:
        return (*compute(tensor_a, tensor_b), actual_device)
    except BaseException as error:
        if not str(actual_device).startswith("cuda") or not is_cuda_oom(error):
            raise
        del tensor_a, tensor_b
        release_cuda_oom()
        cpu_b_float = cpu_b.float() if cpu_b is not None else None
        return (*compute(cpu_a.float(), cpu_b_float), "cpu")


def paired_async_tensors(
    handler_a,
    handler_b,
    work_units,
    *,
    pin_memory: bool,
    quantization_a=None,
    quantization_b=None,
    compute_dtype=torch.float32,
):
    """Yield logical A/B tensors and release their source keys per work unit."""
    def source_keys(quantization, logical_key):
        if logical_key is None:
            return ()
        return tuple(
            quantization.required_keys(logical_key)
            if quantization is not None
            else (logical_key,)
        )

    required_a = [source_keys(quantization_a, unit[1]) for unit in work_units]
    required_b = [source_keys(quantization_b, unit[2]) for unit in work_units]
    keys_a = [key for keys in required_a for key in keys]
    keys_b = [key for keys in required_b for key in keys]
    cursor_a = AsyncTensorCursor(handler_a, keys_a, pin_memory=pin_memory)
    cursor_b = AsyncTensorCursor(handler_b, keys_b, pin_memory=pin_memory)
    try:
        for (logical_key, key_a, key_b), keys_for_a, keys_for_b in zip(
            work_units, required_a, required_b
        ):
            tensors_a = {}
            tensors_b = {}
            consumed_a = []
            consumed_b = []
            tensor_a = tensor_b = None
            try:
                for source_key in keys_for_a:
                    tensors_a[source_key] = cursor_a.take(source_key)
                    consumed_a.append(source_key)
                for source_key in keys_for_b:
                    tensors_b[source_key] = cursor_b.take(source_key)
                    consumed_b.append(source_key)
                tensor_a = (
                    quantization_a.decode(
                        key_a, tensors_a, compute_dtype, device="cpu"
                    )
                    if quantization_a is not None and key_a in quantization_a.quantized_keys
                    else tensors_a.get(key_a)
                )
                tensor_b = (
                    quantization_b.decode(
                        key_b, tensors_b, compute_dtype, device="cpu"
                    )
                    if quantization_b is not None and key_b in quantization_b.quantized_keys
                    else tensors_b.get(key_b)
                )
                yield logical_key, tensor_a, tensor_b
            finally:
                del tensor_a, tensor_b, tensors_a, tensors_b
                for source_key in consumed_a:
                    cursor_a.release(source_key)
                for source_key in consumed_b:
                    cursor_b.release(source_key)
        cursor_a.finish()
        cursor_b.finish()
    finally:
        cursor_a.close()
        cursor_b.close()
