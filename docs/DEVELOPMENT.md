# ComfyUI-ModelUtils - Complete Project History

## Version Timeline

| Version | Commits | Key Features |
|---------|---------|--------------|
| 0.1.1 | `70d6c13`-`d67cbca` | Initial nodes (MetaKeys) |
| 0.1.2 | `6b662a4` | Icon assets |
| 0.1.4 | `cbe3502` | Key renaming/pruning nodes |
| 0.1.5 | `78b3198` | Model merging nodes |
| 0.1.6 | `31989be` | Embedding nodes, namespace fixes |
| 0.1.7 | `b6fae8d` | Mismatch handling, layer filtering, V3 migration |
| **0.2.0** | `3df073f` | LoRA extraction, memory optimization, pinned memory |

---

## Phase 1: Initial Release (v0.1.1)

**Commit:** `70d6c13` - "Add first nodes, toml and publish"

### Files Created
- `nodes/metakeys.py` - Get model metadata and layer keys
- `nodes/utils.py` - Shared utilities
- `__init__.py` - Node registration
- `pyproject.toml` - Package configuration

### Nodes
- `ModelMetaKeys` - Get diffusion model metadata
- `CheckpointMetaKeys` - Get checkpoint metadata
- `TextEncoderMetaKeys` - Get text encoder metadata
- `LoRAMetaKeys` - Get LoRA metadata
- `EmbeddingMetaKeys` - Get embedding metadata

---

## Phase 2: Key Manipulation (v0.1.4)

**Commit:** `4f08fe6` - "add key renaming and key pruning nodes"

### Files Created
- `nodes/renamekeys.py` - Rename tensor keys
- `nodes/prunekeys.py` - Remove tensor keys by pattern

### Nodes (5 variants each for model types)
- `ModelRenameKeys`, `CheckpointRenameKeys`, `TextEncoderRenameKeys`, `LoRARenameKeys`, `EmbeddingRenameKeys`
- `ModelPruneKeys`, `CheckpointPruneKeys`, `TextEncoderPruneKeys`, `LoRAPruneKeys`, `EmbeddingPruneKeys`

---

## Phase 3: Model Merging (v0.1.5)

**Commit:** `7959370` - "Add model merging nodes"

### Files Created
- `nodes/merger.py` - Main merging logic
- `nodes/merger_ops.py` - Merge operation modes
- `nodes/merger_utils.py` - `MemoryEfficientSafeOpen` class
- `docs/merger_2_model_modes.md` - Documentation
- `docs/merger_3_model_modes.md` - Documentation

### Nodes
- `CheckpointTwoMerger`, `DiffusionModelTwoMerger`, `TextEncoderTwoMerger`, `LoRATwoMerger`
- `CheckpointThreeMerger`, `DiffusionModelThreeMerger`, `TextEncoderThreeMerger`, `LoRAThreeMerger`

### Merge Modes (TWO_MODEL_MODES)
- Weighted Sum, Add Difference, Smooth Add Difference
- Train Difference, Sum Twice, Triple Sum
- Euclidean Add Difference, Distribution Crossover, Ties
- Slerp, Lerp, SVD Supermerger, SVD LoRA Extraction

### Merge Modes (THREE_MODEL_MODES)
- Add Difference (A + (B - C) × α)

### MemoryEfficientSafeOpen (initial)
```python
class MemoryEfficientSafeOpen:
    def __init__(self, filename, device='cpu', mmap_mode=True)
    def keys() -> List[str]
    def get_tensor(key) -> Tensor
    # mmap for zero-copy reads
```

---

## Phase 4: Embedding Nodes (v0.1.6)

**Commit:** `0d035ef` - "Add embedding nodes"

### Additions
- Added embedding variants to all node types
- Namespace conflict fixes (`68f22fa`, `5ea63ab`)
- Refactored names for universality (`232dc76`)

---

## Phase 5: Mismatch Handling (v0.1.7)

**Commits:** `19bc047` → `de27186`

### Features Added
- `mismatch_mode` parameter: skip/zeros/error
- Layer filtering with regex patterns
- `exclude_patterns` - Keep model A only (no merge)
- `discard_patterns` - Remove from output entirely

### Files Modified
- `merger.py` - Added mismatch handling logic
- `merger_ops.py` - Added `MissingTensorBehavior` enum, `MissingTensorError`

---

## Phase 6: V3 API Migration (v0.1.7)

**Commit:** `a3116cb` - "migrate to v3 schema and add LoRA extraction node"

### Changes
- Migrated all nodes from legacy API to `comfy_api.latest.io`
- `io.Schema`, `io.ComfyNode`, `io.NodeOutput`
- Added initial LoRA extraction node (diff-based)

---

## Phase 7: LyCORIS Extraction (pre-0.2.0)

**Commits:** `92d3c55` → `18679cc`

### Files Created
- `nodes/lycoris_extract.py` - LyCORIS-based extraction
- `nodes/lora_extract.py` - Simple diff extraction

### Initial Implementation
- Used `lycoris-lora` external dependency
- `lycoris.utils.extract_linear`, `extract_conv`
- Added `requirements.txt` with `lycoris-lora`

### Nodes Split (`18679cc`)
- `LyCORISExtractFixed`, `LyCORISExtractThreshold`
- `LyCORISExtractRatio`, `LyCORISExtractQuantile`

---

## Phase 8: Native SVD Port (0.2.0)

**Commits:** `6d7b2d1` → `eef7377`

### Removed
- `lycoris-lora` dependency
- `lycoris_extract.py`
- `lora_extract.py`

### NEW: [lora_extract_svd.py](file:///f:/ComfyUI/custom_nodes/ComfyUI-ModelUtils/nodes/lora_extract_svd.py)

**8 Rank Selection Functions:**
```python
_index_sv_fixed(S, dim)           # Fixed rank
_index_sv_threshold(S, threshold) # Absolute threshold
_index_sv_ratio(S, ratio)         # Relative to max
_index_sv_cumulative(S, target)   # Cumulative sum %
_index_sv_fro(S, target)          # Frobenius norm
_index_sv_knee(S)                 # Knee detection
_index_sv_cumulative_knee(S)      # Cumsum knee
_index_sv_rel_decrease(S, tau)    # Ratio drop
```

**SVD Functions:**
```python
_svd_extract_linear_lowrank(weight, rank, device, clamp_quantile)
_svd_extract_linear(weight, mode, mode_param, device, max_rank, clamp_quantile)
_svd_extract_conv(weight, mode, mode_param, device, max_rank, clamp_quantile)
_detect_fused_layer(key, shape)  # QKV/MLP detection
_extract_chunked_layer(...)      # Chunked extraction
```

**Nodes:**
- `LoRAExtractFixed` - Uses `svd_lowrank` (10x faster)
- `LoRAExtractRatio` - Relative threshold
- `LoRAExtractQuantile` - Cumulative target
- `LoRAExtractKnee` - Auto knee detection
- `LoRAExtractFrobenius` - Norm preservation

---

## Phase 9: Memory Management (0.2.0)

**Commit:** `6d7b2d1` - "refactor nodes and implement memory management"

### NEW: [device_utils.py](file:///f:/ComfyUI/custom_nodes/ComfyUI-ModelUtils/nodes/device_utils.py)

Integrates with `comfy.model_management`:

```python
@dataclass
class DeviceCapabilities:
    device_type, total_vram_gb, free_vram_gb, total_ram_gb, free_ram_gb
    cpu_cores, recommended_workers, max_tensor_size_gb
    supports_pinned_memory, supports_async_streams

get_device_capabilities(device) -> DeviceCapabilities
estimate_model_size(path) -> float  # GB
can_fit_in_memory(paths, device, safety_factor) -> (bool, str)
prepare_for_large_operation(gb, device) -> bool  # Calls free_memory()
cleanup_after_operation()  # gc.collect() + soft_empty_cache()
get_optimal_workers(size_gb) -> int
```

### Updates to Existing Nodes
- `metakeys.py` - Header-only reading (no tensor loading)
- `renamekeys.py` - `MemoryEfficientSafeOpen` + memory mgmt
- `prunekeys.py` - `MemoryEfficientSafeOpen` + memory mgmt
- `merger.py` - Memory prep/cleanup

---

## Phase 10: Pinned Memory (0.2.0)

**Commit:** `b770786` - "Add pinned memory for faster merging and extraction"

### [merger_utils.py](file:///f:/ComfyUI/custom_nodes/ComfyUI-ModelUtils/nodes/merger_utils.py) Enhancements

```python
# Context manager
class PinnedTensor:
    def __enter__(self) -> Tensor  # Pins memory
    def __exit__(...)              # Unpins

# Transfer helper (2-3x faster)
def transfer_to_gpu_pinned(tensor, device='cuda', dtype=None) -> Tensor

# MemoryEfficientSafeOpen additions
def get_tensor_to_gpu(key, device, dtype)  # Pinned transfer
def keys_sorted_by_offset() -> List[str]   # Sequential I/O
def get_tensors_parallel(keys, workers, sort_by_offset) -> Dict  # Parallel
def get_tensor_as_dict(key) -> Dict        # JSON config parsing
```

---

## Phase 11: SVD Optimization (0.2.0)

**Commit:** `65c1ae3` - "add svd_lowrank for fixed rank extraction"

### Changes
- Added `_svd_extract_linear_lowrank()` using `torch.svd_lowrank`
- Routes fixed mode to lowrank for **10x speedup**
- Changed `chunk_large_layers` default to `False`

---

## Phase 12: Bug Fixes (0.2.0)

**Commit:** `eef7377` - "Added .contiguous() to saves"

### Fix
Non-contiguous tensor error from SVD slices:
```python
tensor.to(dtype).cpu().contiguous()  # Added .contiguous()
```

---

## Current File Structure

```
ComfyUI-ModelUtils/
├── __init__.py
├── pyproject.toml
├── requirements.txt (empty - no external deps)
├── README.md
├── docs/
│   ├── merger_2_model_modes.md
│   └── merger_3_model_modes.md
├── nodes/
│   ├── utils.py
│   ├── metakeys.py
│   ├── renamekeys.py
│   ├── prunekeys.py
│   ├── merger.py
│   ├── merger_ops.py
│   ├── merger_utils.py
│   ├── device_utils.py
│   └── lora_extract_svd.py
└── assets/
    └── icon.png
```

---

## Performance Summary

| Feature | Improvement |
|---------|-------------|
| Fixed-rank SVD | **10x** (svd_lowrank) |
| CPU→GPU transfer | **2-3x** (pinned memory) |
| Metadata reading | **100x+** (header only) |
| Parallel I/O | **2-4x** (ThreadPoolExecutor) |
| Memory management | Auto-unload models before heavy ops |
