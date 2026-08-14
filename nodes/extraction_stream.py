"""Bounded paired UEL streaming helpers for model extraction nodes."""

from collections import deque

import torch

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


class AsyncTensorCursor:
    """Consume one ordered, bounded UEL stream and release each consumed key."""

    def __init__(self, handler, keys, *, pin_memory: bool):
        self.handler = handler
        self.expected = tuple(keys)
        self.position = 0
        self.pending = deque()
        self.stream = None
        if self.expected:
            self.stream = handler.async_stream(
                list(self.expected),
                batch_size=1,
                prefetch_batches=1,
                pin_memory=pin_memory,
            )

    def take(self, expected_key):
        if self.position >= len(self.expected):
            raise RuntimeError(f"Unexpected tensor request: {expected_key}")
        planned_key = self.expected[self.position]
        if planned_key != expected_key:
            raise RuntimeError(
                f"UEL stream order mismatch: expected {planned_key}, requested {expected_key}"
            )
        if not self.pending:
            self.pending.extend(next(self.stream))
        key, tensor = self.pending.popleft()
        if key != expected_key:
            raise RuntimeError(f"UEL yielded {key} while {expected_key} was expected")
        self.position += 1
        return tensor

    def release(self, key):
        self.handler.mark_processed(key)

    def finish(self):
        if self.position != len(self.expected) or self.pending:
            raise RuntimeError("UEL stream ended before all planned tensors were consumed")
        if self.stream is not None:
            try:
                next(self.stream)
            except StopIteration:
                return
            raise RuntimeError("UEL stream yielded unplanned tensors")

    def close(self):
        if self.stream is not None:
            self.stream.close()


def paired_async_tensors(handler_a, handler_b, work_units, *, pin_memory: bool):
    """Yield ordered A/B source tensors and release them after each work unit."""
    keys_a = [unit[1] for unit in work_units if unit[1] is not None]
    keys_b = [unit[2] for unit in work_units if unit[2] is not None]
    cursor_a = AsyncTensorCursor(handler_a, keys_a, pin_memory=pin_memory)
    cursor_b = AsyncTensorCursor(handler_b, keys_b, pin_memory=pin_memory)
    try:
        for logical_key, key_a, key_b in work_units:
            tensor_a = cursor_a.take(key_a) if key_a is not None else None
            tensor_b = cursor_b.take(key_b) if key_b is not None else None
            try:
                yield logical_key, tensor_a, tensor_b
            finally:
                del tensor_a, tensor_b
                if key_a is not None:
                    cursor_a.release(key_a)
                if key_b is not None:
                    cursor_b.release(key_b)
        cursor_a.finish()
        cursor_b.finish()
    finally:
        cursor_a.close()
        cursor_b.close()
