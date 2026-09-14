import json
import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real

import torch

from ..base import BackendConfig, QuantBackend, register_quant_backend
from ..config import register_quant_method

try:
    import entropack as ep

    CompressionMethod = ep.CompressionMethod
    _REQUIRED_ENTROPACK_API = (
        CompressionMethod.LATTICE_RANS,
        ep.CompressionKind,
        ep.resolve_compression,
    )
    ENTROPACK_AVAILABLE = True
    _ENTROPACK_IMPORT_ERROR = None
except (ImportError, AttributeError) as error:
    ep = None
    ENTROPACK_AVAILABLE = False
    _ENTROPACK_IMPORT_ERROR = error

    class CompressionMethod(str, Enum):
        """Keep this optional backend importable until its dependency is installed."""

        DEFAULT = "default"
        DFLOAT11 = "dfloat11"
        TILE_ANS = "tile_ans"
        LATTICE_RANS = "lattice_rans"

        def __str__(self):
            return self.value


@dataclass
class EntroPackConfig(BackendConfig):
    execution_backend: str = "auto"

    def __post_init__(self):
        if self.execution_backend not in ("auto", "eager", "cuda"):
            raise ValueError(
                "execution_backend must be one of: 'auto', 'eager', 'cuda'"
            )
        dtype = getattr(self, "dtype", None)
        compress_method = getattr(self, "compress_method", None)
        if ENTROPACK_AVAILABLE and dtype is not None and compress_method is not None:
            ep.resolve_compression(
                dtype,
                compress_method=compress_method,
                **self.compression_options(),
            )

    def compression_options(self):
        return {}


@dataclass
class EntroPackTileANSConfig(EntroPackConfig):
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
class EntroPackTileANSFP32Config(EntroPackTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 10
    dtype: torch.dtype = field(init=False, default=torch.float32)


@dataclass
class EntroPackTileANSFP16Config(EntroPackTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 11
    dtype: torch.dtype = field(init=False, default=torch.float16)


@dataclass
class EntroPackTileANSBF16Config(EntroPackTileANSConfig):
    tile_elements: int = 0
    probability_bits: int = 11
    dtype: torch.dtype = field(init=False, default=torch.bfloat16)


@dataclass
class EntroPackTileANSFP8E4M3FNConfig(EntroPackTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    dtype: torch.dtype = field(init=False, default=torch.float8_e4m3fn)


@dataclass
class EntroPackTileANSFP8E4M3FNUZConfig(EntroPackTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    dtype: torch.dtype = field(init=False, default=torch.float8_e4m3fnuz)


@dataclass
class EntroPackTileANSFP8E5M2Config(EntroPackTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    dtype: torch.dtype = field(init=False, default=torch.float8_e5m2)


@dataclass
class EntroPackTileANSFP8E5M2FNUZConfig(EntroPackTileANSConfig):
    tile_elements: int = 8192
    probability_bits: int = 0
    dtype: torch.dtype = field(init=False, default=torch.float8_e5m2fnuz)


@dataclass
class EntroPackDFloat11BF16Config(EntroPackConfig):
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
class EntroPackBF16E8Config(EntroPackConfig):
    """Lossy E8-lattice VQ at a target bit rate.

    Continuous ``target_bpp`` in [1, 11] (the lattice scale is found by a clean bisection on the
    real coded byte count -- no lambda search, no cache, no tolerance band). ``prob_bits=None``
    (default) lets the encoder pick the smallest rANS table that resolves the coordinate alphabet at
    the chosen scale -- a tiny table at low bpp (fast decode) growing to 13/14 bits at high bpp --
    so the full 1-11 bpp range is supported; pass an int in {9..14} to pin it. ``tile_elements=None``
    lets the encoder pick the largest rANS tile that still saturates the GPU; pass an int to pin it.
    """

    target_bpp: float = 3.0
    side_dtype: torch.dtype | None = None
    prob_bits: int | None = None
    tile_elements: int | None = None
    row_rdo_iterations: int = 0
    row_rdo_candidates: int = 5
    scale_search_iterations: int = 12
    scale_search_max_vectors: int = 262144
    dtype: torch.dtype = field(init=False, default=torch.bfloat16)
    compress_method: CompressionMethod = field(
        init=False,
        default=CompressionMethod.LATTICE_RANS,
    )

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
        options = {
            "target_bpp": self.target_bpp,
            "side_dtype": self.side_dtype,
            "prob_bits": self.prob_bits,
            "row_rdo_iterations": self.row_rdo_iterations,
            "row_rdo_candidates": self.row_rdo_candidates,
            "scale_search_iterations": self.scale_search_iterations,
            "scale_search_max_vectors": self.scale_search_max_vectors,
        }
        if self.tile_elements is not None:
            options["tile_elements"] = self.tile_elements
        return options


class EntroPackLinear(torch.nn.Linear):
    _STATE_PREFIX = "_entropack."

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
        resolved = ep.resolve_compression(
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
            raise RuntimeError("EntroPackLinear has no compressed weight loaded")
        return self._compressed_weight

    def _set_compressed(self, compressed):
        if not isinstance(compressed, ep.CompressedTensor):
            raise TypeError(
                "compressed weight must be an entropack.CompressedTensor"
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
            compressed.compression_kind == ep.CompressionKind.LOSSY
            and compressed.header.get("version") != ep.ENVELOPE_VERSION
        ):
            raise ValueError("Lossy compressed weights require a standard v2 header")
        if set(compressed.buffers) != set(self._compression_buffer_names):
            raise ValueError(
                f"Compressed buffers {sorted(compressed.buffers)} do not match expected "
                f"{list(self._compression_buffer_names)}"
            )
        validated = ep.CompressedTensor(
            header=compressed.header,
            buffers={name: compressed.buffers[name] for name in self._compression_buffer_names},
            shape=compressed.shape,
            dtype=compressed.dtype,
        )
        for name in self._compression_buffer_names:
            self._buffers[name] = validated.buffers[name]
        self._compressed_weight = validated

    def forward(self, x):
        weight = ep.decompress(
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
            compressed = ep.CompressedTensor.from_state_dict(
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
                if self.compression_kind != ep.CompressionKind.LOSSLESS:
                    error_msgs.append(
                        f"Legacy headerless compressed buffers for '{prefix[:-1]}' are only "
                        "supported by lossless entropack methods; lossy weights "
                        "require a standard v2 header"
                    )
                elif set(legacy) != set(self._compression_buffer_names):
                    error_msgs.append(
                        f"Incomplete legacy compressed buffers for '{prefix[:-1]}'"
                    )
                else:
                    self._set_compressed(
                        ep.CompressedTensor(
                            header={
                                "version": ep.ENVELOPE_VERSION,
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
            self._compressed_weight = ep.CompressedTensor(
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
                ep.CompressedTensor(
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


@register_quant_backend("entropack")
class EntroPackQuantBackend(QuantBackend):
    def validate_environment(self):
        if not ENTROPACK_AVAILABLE:
            raise ImportError(
                "entropack>=0.3.0 with the unified compression API is required "
                "for this quantization method. Install the package and the CuPy extra "
                "matching the CUDA major version."
            ) from _ENTROPACK_IMPORT_ERROR

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": True,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (EntroPackLinear,)

    def flatten_state_dict(self, state_dict):
        return state_dict, {
            "diffsynth.quantization.schema": "1",
            "diffsynth.quantization.backends": json.dumps(["entropack"]),
        }

    def unflatten_state_dict(self, state_dict, metadata):
        return state_dict

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        linear.requires_grad_(False)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        resolved = ep.resolve_compression(
            self.config.dtype,
            compress_method=self.config.compress_method,
            **self.config.compression_options(),
        )
        compressed = ep.compress(
            linear.weight.data,
            format=resolved.dtype,
            compress_method=resolved.compress_method,
            execution_backend=self.config.execution_backend,
            **resolved.options,
        )
        quant_linear = EntroPackLinear(
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
        return EntroPackLinear(
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
        weight = ep.decompress(
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
        "entropack_bf16_e8",
        EntroPackBF16E8Config,
        "Lossy E8-lattice vector quantization of BF16 weights at a target bit rate",
    ),
    (
        "entropack_tile_ans_fp32",
        EntroPackTileANSFP32Config,
        "Lossless Tile-ANS compression after conversion to FP32",
    ),
    (
        "entropack_tile_ans_fp16",
        EntroPackTileANSFP16Config,
        "Lossless Tile-ANS compression after conversion to FP16",
    ),
    (
        "entropack_dfloat11_bf16",
        EntroPackDFloat11BF16Config,
        "Lossless DFloat11 compression after conversion to BF16",
    ),
    (
        "entropack_tile_ans_bf16",
        EntroPackTileANSBF16Config,
        "Lossless Tile-ANS compression after conversion to BF16",
    ),
    (
        "entropack_tile_ans_fp8_e4m3fn",
        EntroPackTileANSFP8E4M3FNConfig,
        "Lossless Tile-ANS compression after conversion to FP8 E4M3FN",
    ),
    (
        "entropack_tile_ans_fp8_e4m3fnuz",
        EntroPackTileANSFP8E4M3FNUZConfig,
        "Lossless Tile-ANS compression after conversion to FP8 E4M3FNUZ",
    ),
    (
        "entropack_tile_ans_fp8_e5m2",
        EntroPackTileANSFP8E5M2Config,
        "Lossless Tile-ANS compression after conversion to FP8 E5M2",
    ),
    (
        "entropack_tile_ans_fp8_e5m2fnuz",
        EntroPackTileANSFP8E5M2FNUZConfig,
        "Lossless Tile-ANS compression after conversion to FP8 E5M2FNUZ",
    ),
)

for method_name, config_class, label in _METHODS:
    register_quant_method(
        method_name,
        "entropack",
        config_class.from_kwargs,
        label=label,
    )


__all__ = [
    "EntroPackBF16E8Config",
    "EntroPackConfig",
    "EntroPackDFloat11BF16Config",
    "EntroPackLinear",
    "EntroPackQuantBackend",
    "EntroPackTileANSBF16Config",
    "EntroPackTileANSConfig",
    "EntroPackTileANSFP16Config",
    "EntroPackTileANSFP32Config",
    "EntroPackTileANSFP8E4M3FNConfig",
    "EntroPackTileANSFP8E4M3FNUZConfig",
    "EntroPackTileANSFP8E5M2Config",
    "EntroPackTileANSFP8E5M2FNUZConfig",
]
