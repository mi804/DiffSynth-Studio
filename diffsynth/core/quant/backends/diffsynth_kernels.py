import json
import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real

import torch

from ..base import BackendConfig, QuantBackend, register_quant_backend
from ..config import register_quant_method

try:
    import diffsynth_kernels as dk

    CompressionMethod = dk.CompressionMethod
    _REQUIRED_KERNELS_API = (
        CompressionMethod.BF16_MANTISSA,
        CompressionMethod.BF16_ADAPTIVE,
        dk.CompressionKind,
        dk.resolve_compression,
    )
    DIFFSYNTH_KERNELS_AVAILABLE = True
    _DIFFSYNTH_KERNELS_IMPORT_ERROR = None
except (ImportError, AttributeError) as error:
    dk = None
    DIFFSYNTH_KERNELS_AVAILABLE = False
    _DIFFSYNTH_KERNELS_IMPORT_ERROR = error

    class CompressionMethod(str, Enum):
        """Keep this optional backend importable until its dependency is installed."""

        DEFAULT = "default"
        DFLOAT11 = "dfloat11"
        TILE_ANS = "tile_ans"
        BF16_MANTISSA = "bf16_mantissa"
        BF16_ADAPTIVE = "bf16_adaptive"

        def __str__(self):
            return self.value


@dataclass
class DiffSynthKernelsConfig(BackendConfig):
    execution_backend: str = "auto"

    def __post_init__(self):
        if self.execution_backend not in ("auto", "eager", "cuda"):
            raise ValueError(
                "execution_backend must be one of: 'auto', 'eager', 'cuda'"
            )
        dtype = getattr(self, "dtype", None)
        compress_method = getattr(self, "compress_method", None)
        if DIFFSYNTH_KERNELS_AVAILABLE and dtype is not None and compress_method is not None:
            dk.resolve_compression(
                dtype,
                compress_method=compress_method,
                **self.compression_options(),
            )

    def compression_options(self):
        return {}


@dataclass
class DiffSynthKernelsTileANSConfig(DiffSynthKernelsConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    raw_lane_threshold: float = 7.9
    dtype: torch.dtype = field(init=False, default=None)
    compress_method: CompressionMethod = field(
        init=False,
        default=CompressionMethod.TILE_ANS,
    )

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
    compress_method: CompressionMethod = field(
        init=False,
        default=CompressionMethod.DFLOAT11,
    )

    def compression_options(self):
        return {
            "bytes_per_thread": self.bytes_per_thread,
            "threads_per_block": self.threads_per_block,
        }


@dataclass
class DiffSynthKernelsBF16MantissaConfig(DiffSynthKernelsConfig):
    mantissa_bits: int = 4
    tile_elements: int = 0
    probability_bits: int = 0
    raw_lane_threshold: float = 7.9
    dtype: torch.dtype = field(init=False, default=torch.bfloat16)
    compress_method: CompressionMethod = field(
        init=False,
        default=CompressionMethod.BF16_MANTISSA,
    )

    def compression_options(self):
        return {
            "mantissa_bits": self.mantissa_bits,
            "tile_elements": self.tile_elements,
            "probability_bits": self.probability_bits,
            "raw_lane_threshold": self.raw_lane_threshold,
        }


@dataclass
class DiffSynthKernelsBF16AdaptiveConfig(DiffSynthKernelsConfig):
    target_bpp: float = 8.0
    bpp_tolerance: float = 0.5
    codebook_size: int | None = None
    side_dtype: torch.dtype | None = None
    sample_rows: int = 512
    iterations: int = 10
    full_refine_steps: int = 1
    fixed_refine_steps: int = 1
    tile_elements: int = 8192
    probability_bits: int = 0
    raw_lane_threshold: float = 7.9
    use_cached_lambda: bool = True
    dtype: torch.dtype = field(init=False, default=torch.bfloat16)
    compress_method: CompressionMethod = field(
        init=False,
        default=CompressionMethod.BF16_ADAPTIVE,
    )

    def compression_options(self):
        return {
            "target_bpp": self.target_bpp,
            "bpp_tolerance": self.bpp_tolerance,
            "codebook_size": self.codebook_size,
            "side_dtype": self.side_dtype,
            "sample_rows": self.sample_rows,
            "iterations": self.iterations,
            "full_refine_steps": self.full_refine_steps,
            "fixed_refine_steps": self.fixed_refine_steps,
            "tile_elements": self.tile_elements,
            "probability_bits": self.probability_bits,
            "raw_lane_threshold": self.raw_lane_threshold,
            "use_cached_lambda": self.use_cached_lambda,
        }


@dataclass
class DiffSynthKernelsBF16AdaptiveOnlineConfig(
    DiffSynthKernelsBF16AdaptiveConfig
):
    bpp_tolerance: float = field(init=False, default=0.5)
    sample_rows: int = field(init=False, default=512)
    iterations: int = field(init=False, default=10)
    full_refine_steps: int = field(init=False, default=1)
    fixed_refine_steps: int = field(init=False, default=1)
    use_cached_lambda: bool = field(init=False, default=True)

    def __post_init__(self):
        if (
            isinstance(self.target_bpp, bool)
            or not isinstance(self.target_bpp, Real)
            or not math.isfinite(float(self.target_bpp))
            or not 1.0 <= float(self.target_bpp) <= 11.0
        ):
            raise ValueError("target_bpp must be finite and in [1.0, 11.0]")
        super().__post_init__()

    def compression_options(self):
        options = super().compression_options()
        options["target_bpp"] = max(1.0, float(self.target_bpp) - 0.3)
        return options


@dataclass
class DiffSynthKernelsBF16AdaptiveOfflineConfig(
    DiffSynthKernelsBF16AdaptiveConfig
):
    bpp_tolerance: float = field(init=False, default=0.0)
    sample_rows: int = field(init=False, default=512)
    iterations: int = field(init=False, default=50)
    full_refine_steps: int = field(init=False, default=2)
    fixed_refine_steps: int = field(init=False, default=1)
    use_cached_lambda: bool = field(init=False, default=False)


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
        self.execution_backend = execution_backend
        self.weight = None
        self._compressed_weight = None
        resolved = dk.resolve_compression(
            compression_dtype,
            compress_method=compress_method,
        )
        self.compress_method = resolved.compress_method
        self.compression_kind = resolved.compression_kind
        self._compression_codec_version = resolved.codec_version
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
        if not isinstance(compressed, dk.CompressedTensor):
            raise TypeError(
                "compressed weight must be a diffsynth_kernels.CompressedTensor"
            )
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
        if compressed.compression_kind != self.compression_kind:
            raise ValueError(
                f"Compressed weight kind '{compressed.compression_kind}' does not match "
                f"configured kind '{self.compression_kind}'"
            )
        if compressed.codec_version != self._compression_codec_version:
            raise ValueError(
                f"Compressed weight codec version {compressed.codec_version} does not match "
                f"configured version {self._compression_codec_version}"
            )
        if (
            compressed.compression_kind == dk.CompressionKind.LOSSY
            and compressed.header.get("version") != dk.ENVELOPE_VERSION
        ):
            raise ValueError("Lossy compressed weights require a standard v2 header")
        if set(compressed.buffers) != set(self._compression_buffer_names):
            raise ValueError(
                f"Compressed buffers {sorted(compressed.buffers)} do not match expected "
                f"{list(self._compression_buffer_names)}"
            )
        validated = dk.CompressedTensor(
            header=compressed.header,
            buffers={name: compressed.buffers[name] for name in self._compression_buffer_names},
            shape=compressed.shape,
            dtype=compressed.dtype,
        )
        for name in self._compression_buffer_names:
            self._buffers[name] = validated.buffers[name]
        self._compressed_weight = validated

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
                if self.compression_kind != dk.CompressionKind.LOSSLESS:
                    error_msgs.append(
                        f"Legacy headerless compressed buffers for '{prefix[:-1]}' are only "
                        "supported by lossless diffsynth-kernels methods; lossy weights "
                        "require a standard v2 header"
                    )
                elif set(legacy) != set(self._compression_buffer_names):
                    error_msgs.append(
                        f"Incomplete legacy compressed buffers for '{prefix[:-1]}'"
                    )
                else:
                    self._set_compressed(
                        dk.CompressedTensor(
                            header={
                                "version": dk.ENVELOPE_VERSION,
                                "compress_method": self.compress_method,
                                "codec_version": self._compression_codec_version,
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
                "diffsynth-kernels>=0.2.0 with the unified compression API is required "
                "for this quantization method. Install the package and the CuPy extra "
                "matching the CUDA major version."
            ) from _DIFFSYNTH_KERNELS_IMPORT_ERROR

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
        resolved = dk.resolve_compression(
            self.config.dtype,
            compress_method=self.config.compress_method,
            **self.config.compression_options(),
        )
        compressed = dk.compress(
            linear.weight.data,
            format=resolved.dtype,
            compress_method=resolved.compress_method,
            execution_backend=self.config.execution_backend,
            **resolved.options,
        )
        quant_linear = DiffSynthKernelsLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=linear.weight.dtype,
            compression_dtype=resolved.dtype,
            compress_method=resolved.compress_method,
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
        "diffsynth_kernels_bf16_mantissa",
        DiffSynthKernelsBF16MantissaConfig,
        "Lossy BF16 mantissa rounding with Tile-ANS compression",
    ),
    (
        "diffsynth_kernels_bf16_adaptive",
        DiffSynthKernelsBF16AdaptiveConfig,
        "Lossy adaptive BF16 codebook compression at a target bit rate",
    ),
    (
        "diffsynth_kernels_bf16_adaptive_online",
        DiffSynthKernelsBF16AdaptiveOnlineConfig,
        "Fast online adaptive BF16 quantization near a requested bit rate",
    ),
    (
        "diffsynth_kernels_bf16_adaptive_offline",
        DiffSynthKernelsBF16AdaptiveOfflineConfig,
        "Strict high-quality offline adaptive BF16 quantization",
    ),
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
    "DiffSynthKernelsBF16AdaptiveConfig",
    "DiffSynthKernelsBF16AdaptiveOnlineConfig",
    "DiffSynthKernelsBF16AdaptiveOfflineConfig",
    "DiffSynthKernelsBF16MantissaConfig",
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
