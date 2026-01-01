"""
Device capabilities and memory management utilities.

Integrates with ComfyUI's model_management for safe memory operations.
"""
import os
import gc
import psutil
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

# Import ComfyUI's model management functions
try:
    from comfy.model_management import (
        get_torch_device,
        get_free_memory,
        get_total_memory,
        free_memory,
        soft_empty_cache,
        unload_all_models,
        is_device_cuda,
    )
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False


@dataclass
class DeviceCapabilities:
    """Device capabilities for I/O and compute optimization."""
    device_type: str  # 'cuda', 'cpu', 'mps', 'xpu'
    total_vram_gb: float
    free_vram_gb: float
    total_ram_gb: float
    free_ram_gb: float
    cpu_cores: int
    recommended_workers: int
    max_tensor_size_gb: float  # Largest tensor we can safely load
    supports_pinned_memory: bool
    supports_async_streams: bool


def get_device_capabilities(device: Optional[torch.device] = None) -> DeviceCapabilities:
    """
    Analyze device capabilities for optimal I/O configuration.
    
    Args:
        device: Target device (default: ComfyUI's torch device)
        
    Returns:
        DeviceCapabilities dataclass with recommended settings
    """
    if device is None:
        if COMFY_AVAILABLE:
            device = get_torch_device()
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    device_type = device.type if hasattr(device, 'type') else 'cpu'
    
    # Get RAM info
    ram_info = psutil.virtual_memory()
    total_ram_gb = ram_info.total / (1024**3)
    free_ram_gb = ram_info.available / (1024**3)
    
    # Get VRAM info
    if device_type == 'cuda' and torch.cuda.is_available():
        if COMFY_AVAILABLE:
            total_vram = get_total_memory(device)
            free_vram = get_free_memory(device)
        else:
            free_vram, total_vram = torch.cuda.mem_get_info(device)
        total_vram_gb = total_vram / (1024**3)
        free_vram_gb = free_vram / (1024**3)
        supports_pinned = True
        supports_async = True
    elif device_type == 'mps':
        total_vram_gb = total_ram_gb * 0.5  # MPS shares RAM
        free_vram_gb = free_ram_gb * 0.5
        supports_pinned = False
        supports_async = False
    else:
        total_vram_gb = 0
        free_vram_gb = 0
        supports_pinned = False
        supports_async = False
    
    # CPU cores
    cpu_cores = os.cpu_count() or 4
    
    # Recommended workers based on hardware
    if device_type == 'cuda':
        # GPU: fewer workers since I/O is not the bottleneck
        recommended_workers = min(4, cpu_cores // 2)
    else:
        # CPU: more workers for parallel I/O
        recommended_workers = min(8, cpu_cores)
    
    # Max tensor size: leave headroom for operations
    if device_type == 'cuda':
        max_tensor_gb = min(free_vram_gb * 0.8, free_ram_gb * 0.5)
    else:
        max_tensor_gb = free_ram_gb * 0.6
    
    return DeviceCapabilities(
        device_type=device_type,
        total_vram_gb=total_vram_gb,
        free_vram_gb=free_vram_gb,
        total_ram_gb=total_ram_gb,
        free_ram_gb=free_ram_gb,
        cpu_cores=cpu_cores,
        recommended_workers=recommended_workers,
        max_tensor_size_gb=max_tensor_gb,
        supports_pinned_memory=supports_pinned,
        supports_async_streams=supports_async,
    )


def estimate_model_size(file_path: str) -> float:
    """Estimate model size in GB from file size."""
    try:
        return os.path.getsize(file_path) / (1024**3)
    except OSError:
        return 0.0


def can_fit_in_memory(
    model_paths: list[str],
    device: Optional[torch.device] = None,
    safety_factor: float = 0.8
) -> Tuple[bool, str]:
    """
    Check if models can fit in available memory.
    
    Args:
        model_paths: List of model file paths
        device: Target device
        safety_factor: Fraction of free memory to use (default 0.8)
        
    Returns:
        (can_fit, reason_message)
    """
    caps = get_device_capabilities(device)
    
    total_size = sum(estimate_model_size(p) for p in model_paths)
    
    if caps.device_type == 'cuda':
        available = caps.free_vram_gb * safety_factor
        mem_type = "VRAM"
    else:
        available = caps.free_ram_gb * safety_factor
        mem_type = "RAM"
    
    if total_size <= available:
        return True, f"OK: {total_size:.2f}GB needed, {available:.2f}GB {mem_type} available"
    else:
        return False, f"Insufficient memory: {total_size:.2f}GB needed but only {available:.2f}GB {mem_type} available"


def prepare_for_large_operation(
    estimated_memory_gb: float,
    device: Optional[torch.device] = None
) -> bool:
    """
    Prepare device for a large memory operation by freeing resources.
    
    Args:
        estimated_memory_gb: Estimated memory needed in GB
        device: Target device
        
    Returns:
        True if preparation succeeded
    """
    if not COMFY_AVAILABLE:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    
    if device is None:
        device = get_torch_device()
    
    memory_bytes = int(estimated_memory_gb * 1024**3)
    
    # Use ComfyUI's free_memory which unloads models intelligently
    try:
        free_memory(memory_bytes, device)
        return True
    except Exception as e:
        print(f"[DeviceUtils] Warning: Could not free memory: {e}")
        return False


def cleanup_after_operation():
    """Clean up memory after a large operation."""
    gc.collect()
    
    if COMFY_AVAILABLE:
        soft_empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_optimal_workers(
    model_size_gb: float,
    caps: Optional[DeviceCapabilities] = None
) -> int:
    """
    Calculate optimal number of I/O workers based on model size and device.
    
    Args:
        model_size_gb: Model size in GB
        caps: Pre-calculated device capabilities (optional)
        
    Returns:
        Recommended number of workers
    """
    if caps is None:
        caps = get_device_capabilities()
    
    # For very large models, fewer workers to avoid memory pressure
    if model_size_gb > caps.free_ram_gb * 0.5:
        return max(1, caps.recommended_workers // 2)
    
    # For small models, use more workers
    if model_size_gb < 2.0:
        return min(8, caps.cpu_cores)
    
    return caps.recommended_workers


def get_processing_device_string(prefer_cuda: bool = True) -> str:
    """
    Get the best available processing device as a string.
    
    Args:
        prefer_cuda: Prefer CUDA over CPU if available
        
    Returns:
        Device string ('cuda', 'mps', 'cpu')
    """
    if prefer_cuda:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def print_device_info():
    """Print device capabilities for debugging."""
    caps = get_device_capabilities()
    print(f"\n{'='*50}")
    print("  Device Capabilities")
    print(f"{'='*50}")
    print(f"  Device Type: {caps.device_type}")
    print(f"  Total VRAM: {caps.total_vram_gb:.2f} GB")
    print(f"  Free VRAM: {caps.free_vram_gb:.2f} GB")
    print(f"  Total RAM: {caps.total_ram_gb:.2f} GB")
    print(f"  Free RAM: {caps.free_ram_gb:.2f} GB")
    print(f"  CPU Cores: {caps.cpu_cores}")
    print(f"  Recommended Workers: {caps.recommended_workers}")
    print(f"  Max Tensor Size: {caps.max_tensor_size_gb:.2f} GB")
    print(f"  Pinned Memory: {caps.supports_pinned_memory}")
    print(f"  Async Streams: {caps.supports_async_streams}")
    print(f"{'='*50}\n")
