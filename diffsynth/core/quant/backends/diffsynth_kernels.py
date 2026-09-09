import json
from dataclasses import dataclass, field

import torch

from ..base import BackendConfig, QuantBackend, register_quant_backend
from ..config import register_quant_method

try:
    import diffsynth_kernels as dk

    DIFFSYNTH_KERNELS_AVAILABLE = True
except ImportError:
    DIFFSYNTH_KERNELS_AVAILABLE = False


@dataclass
class DiffSynthKernelsConfig(BackendConfig):
    execution_backend: str = "auto"

    def __post_init__(self):
        if self.execution_backend not in ("auto", "eager", "cuda"):
            raise ValueError(
                "execution_backend must be one of: 'auto', 'eager', 'cuda'"
            )

    def compression_options(self):
        return {}


@dataclass
class DiffSynthKernelsTileANSConfig(DiffSynthKernelsConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    raw_lane_threshold: float = 7.9
    dtype: torch.dtype = field(init=False, default=None)
    compress_method: str = field(init=False, default="tile_ans")

    def compression_options(self):
        return {
            "tile_elements": self.tile_elements,
            "probability_bits": self.probability_bits,
            "raw_lane_threshold": self.raw_lane_threshold,
        }


@dataclass
class DiffSynthKernelsTileANSFP32Config(DiffSynthKernelsTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 10
    dtype: torch.dtype = field(init=False, default=torch.float32)


@dataclass
class DiffSynthKernelsTileANSFP16Config(DiffSynthKernelsTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 11
    dtype: torch.dtype = field(init=False, default=torch.float16)


@dataclass
class DiffSynthKernelsTileANSBF16Config(DiffSynthKernelsTileANSConfig):
    tile_elements: int = 0
    probability_bits: int = 11
    dtype: torch.dtype = field(init=False, default=torch.bfloat16)


@dataclass
class DiffSynthKernelsTileANSFP8E4M3FNConfig(DiffSynthKernelsTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    dtype: torch.dtype = field(init=False, default=torch.float8_e4m3fn)


@dataclass
class DiffSynthKernelsTileANSFP8E4M3FNUZConfig(DiffSynthKernelsTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    dtype: torch.dtype = field(init=False, default=torch.float8_e4m3fnuz)


@dataclass
class DiffSynthKernelsTileANSFP8E5M2Config(DiffSynthKernelsTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    dtype: torch.dtype = field(init=False, default=torch.float8_e5m2)


@dataclass
class DiffSynthKernelsTileANSFP8E5M2FNUZConfig(DiffSynthKernelsTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    dtype: torch.dtype = field(init=False, default=torch.float8_e5m2fnuz)


@dataclass
class DiffSynthKernelsDFloat11BF16Config(DiffSynthKernelsConfig):
    bytes_per_thread: int = 16
    threads_per_block: int = 128
    dtype: torch.dtype = field(init=False, default=torch.bfloat16)
    compress_method: str = field(init=False, default="dfloat11")

    def compression_options(self):
        return {
            "bytes_per_thread": self.bytes_per_thread,
            "threads_per_block": self.threads_per_block,
        }


class DiffSynthKernelsLinear(torch.nn.Linear):
    _STATE_PREFIX = "_diffsynth_kernels."

    def __init__(
        self,
        in_features,
        out_features,
        bias,
        *,
        compute_dtype,
        compression_dtype,
        compress_method,
        execution_backend="auto",
    ):
        with torch.device("meta"):
            super().__init__(
                in_features,
                out_features,
                bias=bias,
                dtype=compute_dtype,
            )
        self.compute_dtype = compute_dtype
        self.compression_dtype = compression_dtype
        self.compress_method = compress_method
        self.execution_backend = execution_backend
        self.weight = None
        self._compressed_weight = None
        resolved = dk.resolve_compression(
            compression_dtype,
            compress_method=compress_method,
        )
        self._compression_buffer_names = resolved.buffer_names
        for name in self._compression_buffer_names:
            self.register_buffer(name, None, persistent=False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    @property
    def qweight(self):
        for name in self._compression_buffer_names:
            buffer = getattr(self, name)
            if buffer is not None:
                return buffer
        return self.bias

    def _compressed(self):
        if self._compressed_weight is None:
            raise RuntimeError("DiffSynthKernelsLinear has no compressed weight loaded")
        return self._compressed_weight

    def _set_compressed(self, compressed):
        if compressed.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Compressed weight shape {compressed.shape} does not match Linear shape "
                f"{(self.out_features, self.in_features)}"
            )
        if compressed.dtype != self.compression_dtype:
            raise ValueError(
                f"Compressed weight dtype {compressed.dtype} does not match configured "
                f"dtype {self.compression_dtype}"
            )
        if compressed.compress_method != self.compress_method:
            raise ValueError(
                f"Compressed weight method '{compressed.compress_method}' does not match "
                f"configured method '{self.compress_method}'"
            )
        if set(compressed.buffers) != set(self._compression_buffer_names):
            raise ValueError(
                f"Compressed buffers {sorted(compressed.buffers)} do not match expected "
                f"{list(self._compression_buffer_names)}"
            )
        for name in self._compression_buffer_names:
            self._buffers[name] = compressed.buffers[name]
        self._compressed_weight = dk.CompressedTensor(
            header=compressed.header,
            buffers={name: self._buffers[name] for name in self._compression_buffer_names},
            shape=compressed.shape,
            dtype=compressed.dtype,
        )

    def forward(self, x):
        weight = dk.decompress(
            self._compressed(),
            execution_backend=self.execution_backend,
        ).to(x.dtype)
        return torch.nn.functional.linear(x, weight, self.bias)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._compressed_weight is None:
            return
        state = self._compressed_weight.state_dict(prefix + self._STATE_PREFIX)
        for key, value in state.items():
            destination[key] = value if keep_vars else value.detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        compressed_prefix = prefix + self._STATE_PREFIX
        header_key = compressed_prefix + "header"
        if header_key in state_dict:
            compressed = dk.CompressedTensor.from_state_dict(
                state_dict,
                prefix=compressed_prefix,
            )
            self._set_compressed(compressed)
            state_dict.pop(header_key)
            for name in compressed.buffers:
                state_dict.pop(compressed_prefix + "buffers." + name)
        else:
            legacy = {
                name: state_dict.pop(prefix + name)
                for name in self._compression_buffer_names
                if prefix + name in state_dict
            }
            if legacy:
                if set(legacy) != set(self._compression_buffer_names):
                    error_msgs.append(
                        f"Incomplete legacy compressed buffers for '{prefix[:-1]}'"
                    )
                else:
                    resolved = dk.resolve_compression(
                        self.compression_dtype,
                        compress_method=self.compress_method,
                    )
                    self._set_compressed(
                        dk.CompressedTensor(
                            header={
                                "version": dk.ENVELOPE_VERSION,
                                "compress_method": self.compress_method,
                                "codec_version": resolved.codec_version,
                                "resolved_options": {},
                            },
                            buffers=legacy,
                            shape=(self.out_features, self.in_features),
                            dtype=self.compression_dtype,
                        )
                    )
            elif strict:
                missing_keys.append(header_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _apply(self, fn, recurse=True):
        original = {
            name: self._buffers.pop(name)
            for name in self._compression_buffer_names
        }
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            for name, buffer in original.items():
                if buffer is None:
                    self._buffers[name] = None
                    continue
                transformed = fn(buffer)
                if transformed.dtype != buffer.dtype:
                    transformed = buffer.to(device=transformed.device)
                self._buffers[name] = transformed
        if self._compressed_weight is not None:
            self._compressed_weight = dk.CompressedTensor(
                header=self._compressed_weight.header,
                buffers={name: self._buffers[name] for name in self._compression_buffer_names},
                shape=self._compressed_weight.shape,
                dtype=self._compressed_weight.dtype,
            )
        return self

    def __deepcopy__(self, memo):
        clone = type(self)(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            compute_dtype=self.compute_dtype,
            compression_dtype=self.compression_dtype,
            compress_method=self.compress_method,
            execution_backend=self.execution_backend,
        )
        if self._compressed_weight is not None:
            clone._set_compressed(
                dk.CompressedTensor(
                    header=self._compressed_weight.header,
                    buffers={
                        name: self._buffers[name].detach().clone()
                        for name in self._compression_buffer_names
                    },
                    shape=self._compressed_weight.shape,
                    dtype=self._compressed_weight.dtype,
                )
            )
        if self.bias is not None:
            clone.bias = torch.nn.Parameter(
                self.bias.detach().clone(),
                requires_grad=False,
            )
        memo[id(self)] = clone
        return clone


@register_quant_backend("diffsynth_kernels")
class DiffSynthKernelsQuantBackend(QuantBackend):
    def validate_environment(self):
        if not DIFFSYNTH_KERNELS_AVAILABLE:
            raise ImportError(
                "diffsynth-kernels is required for this quantization method. Install "
                "the package and the CuPy extra matching the CUDA major version."
            )

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": True,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (DiffSynthKernelsLinear,)

    def flatten_state_dict(self, state_dict):
        return state_dict, {
            "diffsynth.quantization.schema": "1",
            "diffsynth.quantization.backends": json.dumps(["diffsynth_kernels"]),
        }

    def unflatten_state_dict(self, state_dict, metadata):
        return state_dict

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        linear.requires_grad_(False)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        compressed = dk.compress(
            linear.weight.data,
            format=self.config.dtype,
            compress_method=self.config.compress_method,
            execution_backend=self.config.execution_backend,
            **self.config.compression_options(),
        )
        quant_linear = DiffSynthKernelsLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=linear.weight.dtype,
            compression_dtype=self.config.dtype,
            compress_method=self.config.compress_method,
            execution_backend=self.config.execution_backend,
        )
        quant_linear._set_compressed(compressed)
        if linear.bias is not None:
            quant_linear.bias = torch.nn.Parameter(
                linear.bias.data,
                requires_grad=False,
            )
        return quant_linear if model_device is None else quant_linear.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return DiffSynthKernelsLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=compute_dtype,
            compression_dtype=self.config.dtype,
            compress_method=self.config.compress_method,
            execution_backend=self.config.execution_backend,
        )

    def dequantize_to_linear(
        self,
        module,
        compute_dtype,
        compute_device=None,
        model_device=None,
    ):
        if compute_device is not None:
            module = module.to(device=compute_device)
        weight = dk.decompress(
            module._compressed(),
            execution_backend=self.config.execution_backend,
        ).to(compute_dtype)
        linear = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(
                module.bias.data.to(dtype=compute_dtype, device=weight.device),
                requires_grad=False,
            )
        return linear if model_device is None else linear.to(device=model_device)


_METHODS = (
    (
        "diffsynth_kernels_tile_ans_fp32",
        DiffSynthKernelsTileANSFP32Config,
        "Lossless Tile-ANS compression after conversion to FP32",
    ),
    (
        "diffsynth_kernels_tile_ans_fp16",
        DiffSynthKernelsTileANSFP16Config,
        "Lossless Tile-ANS compression after conversion to FP16",
    ),
    (
        "diffsynth_kernels_dfloat11_bf16",
        DiffSynthKernelsDFloat11BF16Config,
        "Lossless DFloat11 compression after conversion to BF16",
    ),
    (
        "diffsynth_kernels_tile_ans_bf16",
        DiffSynthKernelsTileANSBF16Config,
        "Lossless Tile-ANS compression after conversion to BF16",
    ),
    (
        "diffsynth_kernels_tile_ans_fp8_e4m3fn",
        DiffSynthKernelsTileANSFP8E4M3FNConfig,
        "Lossless Tile-ANS compression after conversion to FP8 E4M3FN",
    ),
    (
        "diffsynth_kernels_tile_ans_fp8_e4m3fnuz",
        DiffSynthKernelsTileANSFP8E4M3FNUZConfig,
        "Lossless Tile-ANS compression after conversion to FP8 E4M3FNUZ",
    ),
    (
        "diffsynth_kernels_tile_ans_fp8_e5m2",
        DiffSynthKernelsTileANSFP8E5M2Config,
        "Lossless Tile-ANS compression after conversion to FP8 E5M2",
    ),
    (
        "diffsynth_kernels_tile_ans_fp8_e5m2fnuz",
        DiffSynthKernelsTileANSFP8E5M2FNUZConfig,
        "Lossless Tile-ANS compression after conversion to FP8 E5M2FNUZ",
    ),
)

for method_name, config_class, label in _METHODS:
    register_quant_method(
        method_name,
        "diffsynth_kernels",
        config_class.from_kwargs,
        label=label,
    )


__all__ = [
    "DiffSynthKernelsConfig",
    "DiffSynthKernelsDFloat11BF16Config",
    "DiffSynthKernelsLinear",
    "DiffSynthKernelsQuantBackend",
    "DiffSynthKernelsTileANSBF16Config",
    "DiffSynthKernelsTileANSConfig",
    "DiffSynthKernelsTileANSFP16Config",
    "DiffSynthKernelsTileANSFP32Config",
    "DiffSynthKernelsTileANSFP8E4M3FNConfig",
    "DiffSynthKernelsTileANSFP8E4M3FNUZConfig",
    "DiffSynthKernelsTileANSFP8E5M2Config",
    "DiffSynthKernelsTileANSFP8E5M2FNUZConfig",
]
