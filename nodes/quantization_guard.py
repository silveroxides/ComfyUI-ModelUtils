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


_QUANT_SIDECARS = (
    "comfy_quant", "weight_scale", "weight_scale_2", "input_scale",
    "pre_quant_scale", "weight_s_rel", "weight_s_channel",
    "weight_correction", "weight_codebook", "scale_weight", "scale_input",
)


class DiffusionQuantization:
    """Index and decode Core quantized diffusion weights one layer at a time.

    The handler is a UEL low-memory handler. Header inspection uses only its
    keys, shapes, dtypes, and metadata; tensor values are supplied to ``decode``
    by the caller's bounded UEL work unit.
    """

    def __init__(self, handler, model_path: str, label: str):
        self.handler = handler
        self.model_path = model_path
        self.label = label
        self._keys = tuple(handler.keys())
        self._key_set = set(self._keys)
        self._metadata = dict(handler.metadata() or {})
        self._layer_configs = {}
        self._architecture_shapes = None
        self._legacy_scaled_fp8 = any(key.endswith("scaled_fp8") for key in self._keys)

        encoded = self._metadata.get("_quantization_metadata")
        if encoded:
            try:
                layers = json.loads(encoded).get("layers", {})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"[{label}] Invalid _quantization_metadata in {model_path}: {exc}"
                ) from exc
            if not isinstance(layers, dict):
                raise ValueError(f"[{label}] Invalid quantization layers metadata in {model_path}")
            self._layer_configs.update(layers)

        quantized = set()
        for sidecar in self._keys:
            if not sidecar.endswith(".comfy_quant"):
                continue
            stem = sidecar[:-len(".comfy_quant")]
            weight_key = f"{stem}.weight"
            if weight_key not in self._key_set:
                raise ValueError(f"[{label}] Quantization sidecar has no weight: {sidecar}")
            quantized.add(weight_key)

        for stem in self._layer_configs:
            weight_key = f"{stem}.weight"
            if weight_key in self._key_set:
                quantized.add(weight_key)

        if self._legacy_scaled_fp8:
            for sidecar in self._keys:
                if sidecar.endswith(".scale_weight"):
                    stem = sidecar[:-len(".scale_weight")]
                    weight_key = f"{stem}.weight"
                    if weight_key in self._key_set:
                        quantized.add(weight_key)
                        self._layer_configs.setdefault(stem, {"format": "float8_e4m3fn"})

        self.quantized_keys = frozenset(quantized)
        sidecars = set()
        for key in self.quantized_keys:
            stem = key[:-len(".weight")]
            for name in _QUANT_SIDECARS:
                candidate = f"{stem}.{name}"
                if candidate in self._key_set:
                    sidecars.add(candidate)
        for key in self._keys:
            if key.endswith(".comfy_quant") and any(
                key == f"{weight[:-len('.weight')]}.comfy_quant" for weight in self.quantized_keys
            ):
                sidecars.add(key)
        if self._legacy_scaled_fp8:
            sidecars.update(key for key in self._keys if key.endswith("scaled_fp8"))
        self._sidecars = frozenset(sidecars)
        self.data_keys = tuple(key for key in self._keys if key not in self._sidecars)

        self.isolated_low_bit_keys = {
            key for key in self.data_keys
            if key not in self.quantized_keys and _is_low_bit(handler, key)
        }
        if len(self.isolated_low_bit_keys) > MAX_ISOLATED_LOW_BIT_TENSORS:
            raise ValueError(
                f"[{label}] {model_path} contains {len(self.isolated_low_bit_keys)} low-bit tensors; "
                "casted or quantized inputs are unsupported"
            )
        if self.isolated_low_bit_keys:
            logging.warning(
                "[%s] %s contains %d isolated low-bit tensor(s); preserving/excluding unchanged: %s",
                label, model_path, len(self.isolated_low_bit_keys),
                ", ".join(sorted(self.isolated_low_bit_keys)),
            )

    def required_keys(self, key: str) -> list[str]:
        if key not in self.quantized_keys:
            return [key]
        stem = key[:-len(".weight")]
        return [key] + [
            f"{stem}.{name}" for name in _QUANT_SIDECARS
            if f"{stem}.{name}" in self._key_set
        ]

    def _architecture_shape(self, key: str) -> tuple[int, ...]:
        if self._architecture_shapes is None:
            from comfy import model_detection

            # Core detection examines keys, shapes and dtypes. Meta tensors supply
            # that inventory without materializing any checkpoint tensor data.
            state = {
                name: torch.empty(
                    tuple(self.handler.get_shape(name)),
                    dtype=self.handler.get_dtype(name), device="meta",
                )
                for name in self._keys
            }
            prefix = model_detection.unet_prefix_from_state_dict(state)
            detected = {
                name[len(prefix):]: value
                for name, value in state.items() if name.startswith(prefix)
            }
            try:
                config = model_detection.model_config_from_unet(
                    detected, "", metadata=self._metadata,
                )
                if config is None:
                    raise ValueError("Core could not detect this diffusion architecture")
                with torch.device("meta"):
                    model = config.get_model(detected, "", device=torch.device("meta"))
                modules = dict(getattr(model, "diffusion_model", model).named_modules())
                shapes = {}
                for weight_key in self.quantized_keys:
                    module_name = weight_key[len(prefix):].removesuffix(".weight")
                    module = modules.get(module_name)
                    if module is None:
                        continue
                    shape = getattr(module, "_orig_shape", None)
                    if shape is None:
                        weight = getattr(module, "weight", None)
                        shape = getattr(weight, "shape", None)
                    if shape is not None:
                        shapes[weight_key] = tuple(shape)
                self._architecture_shapes = shapes
            except Exception as exc:
                raise ValueError(
                    f"[{self.label}] Cannot determine exact quantized layer shapes from "
                    f"Core architecture for {self.model_path}: {exc}"
                ) from exc
            finally:
                state.clear()
                detected.clear()
                if "model" in locals():
                    del model

        shape = self._architecture_shapes.get(key)
        if shape is None:
            raise ValueError(
                f"[{self.label}] Core architecture has no logical shape for quantized layer {key}"
            )
        stored = tuple(self.handler.get_shape(key))
        if len(shape) != 2 or len(stored) != 2:
            raise ValueError(f"[{self.label}] Invalid padded quantized shape for {key}: {shape}, {stored}")
        stem = key[:-len(".weight")]
        config = self._layer_configs.get(stem)
        nvfp4 = (isinstance(config, dict) and config.get("format") == "nvfp4") or (
            f"{stem}.weight_scale_2" in self._key_set
        )
        if nvfp4:
            expected = ((shape[0] + 15) // 16 * 16, (shape[1] + 15) // 16 * 8)
        else:
            expected = ((shape[0] + 31) // 32 * 32, (shape[1] + 31) // 32 * 32)
        if stored != expected:
            raise ValueError(
                f"[{self.label}] Core architecture shape {shape} disagrees with packed storage {stored} for {key}"
            )
        return shape

    def _format(self, key: str, tensors=None):
        stem = key[:-len(".weight")]
        config = self._layer_configs.get(stem)
        sidecar = f"{stem}.comfy_quant"
        if tensors is None and config is None and sidecar in self._key_set:
            stream = self.handler.async_stream(
                [sidecar], batch_size=1, prefetch_batches=1, pin_memory=False,
            )
            yielded_keys = []
            try:
                batch = next(stream)
                yielded_keys = [name for name, _ in batch]
                if len(batch) != 1 or batch[0][0] != sidecar:
                    raise RuntimeError(f"[{self.label}] Unexpected quantization sidecar for {key}")
                tensors = {sidecar: batch[0][1]}
            finally:
                try:
                    for yielded_key in yielded_keys:
                        self.handler.mark_processed(yielded_key)
                finally:
                    stream.close()
        if tensors is not None and sidecar in tensors:
            raw = tensors[sidecar].detach().to(device="cpu").contiguous().reshape(-1).tolist()
            try:
                config = json.loads(bytes(raw).decode("utf-8"))
            except (UnicodeDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"[{self.label}] Invalid quantization sidecar {sidecar}: {exc}") from exc
            self._layer_configs[stem] = config
        if config is None:
            raise ValueError(f"[{self.label}] Missing quantization config for {key}")
        if not isinstance(config, dict):
            raise ValueError(f"[{self.label}] Invalid quantization config for {key}: expected object")
        fmt = config.get("format")
        try:
            from comfy.quant_ops import QUANT_ALGOS
        except ImportError as exc:
            raise RuntimeError("ComfyUI quantization support is unavailable") from exc
        if fmt not in QUANT_ALGOS:
            raise ValueError(f"[{self.label}] Unsupported ComfyUI quantization format {fmt!r} for {key}")
        return fmt, config

    def logical_shape(self, key: str) -> tuple[int, ...]:
        if key not in self.quantized_keys:
            return tuple(self.handler.get_shape(key))
        fmt, _ = self._format(key)
        if fmt == "nvfp4" or fmt == "mxfp8":
            return self._architecture_shape(key)
        shape = tuple(self.handler.get_shape(key))
        if fmt in {"convrot_w4a4", "asym_w4a8_int8"}:
            if len(shape) != 2:
                raise ValueError(f"[{self.label}] Expected packed matrix weight for {key}, got {shape}")
            return shape[0], shape[1] * 2
        return shape

    def logical_dtype(self, key: str, compute_dtype: torch.dtype) -> torch.dtype:
        if key not in self.quantized_keys:
            return self.handler.get_dtype(key)
        return compute_dtype

    def decode(self, key: str, tensors, compute_dtype: torch.dtype, device="cpu") -> torch.Tensor:
        if key not in self.quantized_keys:
            raise ValueError(f"[{self.label}] {key} is not a recognized quantized diffusion weight")
        fmt, config = self._format(key, tensors)
        try:
            from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
        except ImportError as exc:
            raise RuntimeError("ComfyUI quantization support is unavailable") from exc

        stem = key[:-len(".weight")]
        device = torch.device(device)
        shape = self.logical_shape(key)

        def tensor(name, *, dtype=None, required=False):
            tensor_key = f"{stem}.{name}"
            if tensor_key not in tensors and name == "weight_scale":
                legacy_key = f"{stem}.scale_weight"
                if legacy_key in tensors:
                    tensor_key = legacy_key
            value = tensors.get(tensor_key)
            if value is None:
                if required:
                    raise ValueError(f"[{self.label}] Missing quantization tensor {tensor_key} for {key}")
                return None
            value = value.to(device=device)
            if dtype is not None:
                value = value.view(dtype=dtype)
            return value

        params_conf = config.get("params", {})
        if not isinstance(params_conf, dict):
            params_conf = {}
        if fmt in ("float8_e4m3fn", "float8_e5m2"):
            values = {"scale": tensor("weight_scale", required=True)}
        elif fmt == "mxfp8":
            values = {"scale": tensor("weight_scale", dtype=torch.float8_e8m0fnu, required=True)}
        elif fmt == "nvfp4":
            values = {
                "scale": tensor("weight_scale_2", required=True),
                "block_scale": tensor("weight_scale", dtype=torch.float8_e4m3fn, required=True),
            }
        elif fmt == "int8_tensorwise":
            values = {"scale": tensor("weight_scale", required=True)}
            if config.get("convrot", params_conf.get("convrot", False)):
                values.update({
                    "convrot": True,
                    "convrot_groupsize": int(config.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
                })
        elif fmt == "convrot_w4a4":
            values = {
                "scale": tensor("weight_scale", required=True),
                "convrot_groupsize": int(config.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
                "quant_group_size": 64,
                "linear_dtype": config.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
            }
        elif fmt == "asym_w4a8_int8":
            if f"{stem}.weight_correction" in tensors:
                raise ValueError(
                    f"[{self.label}] Core's W4A8 loader does not apply weight_correction for {key}"
                )
            scale_value = tensors.get(f"{stem}.weight_s_rel")
            values = {
                "scale": tensor("weight_s_rel", dtype=torch.float8_e4m3fn if scale_value is not None and scale_value.dtype == torch.uint8 else None, required=True),
                "s_channel": tensor("weight_s_channel", required=True),
                "codebook": tensor("weight_codebook"),
                "group_size": int(config.get("group_size", params_conf.get("group_size", 16))),
                "convrot_groupsize": int(config.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
            }
        else:
            raise ValueError(f"[{self.label}] Unsupported ComfyUI quantization format {fmt!r} for {key}")

        layout_type = QUANT_ALGOS[fmt]["comfy_tensor_layout"]
        layout = get_layout_class(layout_type)
        params = layout.Params(**values, orig_dtype=compute_dtype, orig_shape=shape)
        qdata = tensors[key].to(device=device, dtype=QUANT_ALGOS[fmt]["storage_t"])
        return QuantizedTensor(qdata, layout_type, params).dequantize()

    def output_metadata(self) -> dict:
        metadata = dict(self._metadata)
        metadata.pop("_quantization_metadata", None)
        return metadata


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
        for name in (
            "down", "up", "mid", "reshape", "dora_scale",
            "diff", "diff_b", "w_norm", "b_norm", "set_weight",
        )
    )


def write_preserved_tensor(
    writer,
    key: str,
    handler,
    output_key: str | None = None,
    *,
    force_raw: bool = False,
    tensor=None,
) -> None:
    """Copy one tensor's original safetensors bytes and dtype into an active writer."""
    destination = output_key or key
    header = _header(handler)
    entry = header[key]
    if not force_raw and entry["dtype"] in UPSTREAM_WRITER_STORAGE_CODES:
        if tensor is not None:
            writer.write(destination, tensor.cpu().contiguous())
            return
        stream = handler.async_stream(
            [key], batch_size=1, prefetch_batches=1, pin_memory=False
        )
        yielded = False
        try:
            batch = next(stream)
            if len(batch) != 1 or batch[0][0] != key:
                raise RuntimeError(f"UEL yielded an unexpected preserved tensor for {key}")
            yielded = True
            writer.write(destination, batch[0][1].cpu().contiguous())
        finally:
            if yielded:
                handler.mark_processed(key)
            stream.close()
        return

    source_start, source_end = entry["data_offsets"]
    byte_size = source_end - source_start

    with writer._lock:
        if destination in writer._manifest:
            raise ValueError(f"Tensor '{destination}' has already been written.")
        target_start = writer._current_data_offset
        target_end = target_start + byte_size
        writer._manifest[destination] = {
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
