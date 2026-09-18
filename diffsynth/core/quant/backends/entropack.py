import json
import math
from dataclasses import dataclass, field
from numbers import Real

import torch

from ..base import BackendConfig, QuantBackend, register_quant_backend
from ..config import register_quant_method

try:
    import entropack as ep

    _REQUIRED_ENTROPACK_API = (
        ep.CompressedLinear,
        ep.CompressedFP8Linear,
        ep.CompressedINT8Linear,
        ep.resolve_compression,
    )
    ENTROPACK_AVAILABLE = True
    _ENTROPACK_IMPORT_ERROR = None
except (ImportError, AttributeError) as error:
    ep = None
    ENTROPACK_AVAILABLE = False
    _ENTROPACK_IMPORT_ERROR = error

def _is_rate(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))

@dataclass
class EntroPackConfig(BackendConfig):

    linear_kind: str = field(init=False, default="bf16")
    dtype: torch.dtype = field(init=False, default=torch.bfloat16)
    scheme: str = field(init=False, default="auto")

    def __post_init__(self):
        if not ENTROPACK_AVAILABLE:
            return
        resolved = ep.resolve_compression(
            self.container_dtype, compress_method=self.scheme, **self.compression_options(),
        )
        resolved.scheme.validate_options(resolved.options)

    @property
    def container_dtype(self) -> torch.dtype:
        return self.dtype

    @property
    def linear_cls(self):
        return {
            "bf16": ep.CompressedLinear,
            "fp8": ep.CompressedFP8Linear,
            "int8": ep.CompressedINT8Linear,
        }[self.linear_kind]

    def compression_options(self):
        return {}

    def linear_kwargs(self):
        return {"scheme": self.scheme, **self.compression_options()}

@dataclass
class EntroPackDFloat11BF16Config(EntroPackConfig):
    bytes_per_thread: int = 16
    threads_per_block: int = 128
    scheme: str = field(init=False, default="dfloat11")

    def compression_options(self):
        return {"bytes_per_thread": self.bytes_per_thread, "threads_per_block": self.threads_per_block}

@dataclass
class EntroPackTileANSBF16Config(EntroPackConfig):
    tile_elements: int = 0
    probability_bits: int = 11
    raw_lane_threshold: float = 7.9
    scheme: str = field(init=False, default="tile_ans")

    def compression_options(self):
        return {
            "tile_elements": self.tile_elements,
            "probability_bits": self.probability_bits,
            "raw_lane_threshold": self.raw_lane_threshold,
        }

@dataclass
class EntroPackLatticeConfig(EntroPackConfig):

    target_bpp: float | None = 4.0
    prob_bits: int | None = None
    tile_elements: int | None = None
    row_rdo_iterations: int = 0
    row_rdo_candidates: int = 5
    scale_search_iterations: int = 12
    scale_search_max_vectors: int = 262144

    def compression_options(self):
        options = {
            "prob_bits": self.prob_bits,
            "row_rdo_iterations": self.row_rdo_iterations,
            "row_rdo_candidates": self.row_rdo_candidates,
            "scale_search_iterations": self.scale_search_iterations,
            "scale_search_max_vectors": self.scale_search_max_vectors,
        }
        if self.tile_elements is not None:
            options["tile_elements"] = self.tile_elements
        return options

@dataclass
class EntroPackLatticeRANSBF16Config(EntroPackLatticeConfig):

    target_bpp: float = 3.0
    scheme: str = field(init=False, default="lattice_rans")

    def __post_init__(self):
        if not _is_rate(self.target_bpp) or not 1.0 <= float(self.target_bpp) <= 11.0:
            raise ValueError(f"target_bpp must be finite and in [1.0, 11.0] to code a BF16 weight, got {self.target_bpp!r}")
        super().__post_init__()

    def compression_options(self):
        return {"target_bpp": self.target_bpp, **super().compression_options()}

@dataclass
class EntroPackW8A8Config(EntroPackLatticeConfig):

    target_bpp: float | None = 4.0

    def __post_init__(self):
        if self.target_bpp is not None:
            if not _is_rate(self.target_bpp) or not 0.0 < float(self.target_bpp) < 8.0:
                raise ValueError(
                    f"target_bpp must be finite and in (0.0, 8.0) to code an 8-bit weight, or None to "
                    f"store the codes uncoded; got {self.target_bpp!r}"
                )
        self.scheme = "raw" if self.target_bpp is None else "lattice_rans"
        super().__post_init__()

    def linear_kwargs(self):
        return {"target_bpp": self.target_bpp, **EntroPackLatticeConfig.compression_options(self)}

@dataclass
class EntroPackFP8Config(EntroPackW8A8Config):
    linear_kind: str = field(init=False, default="fp8")

    @property
    def container_dtype(self):
        return torch.float8_e4m3fn

@dataclass
class EntroPackINT8Config(EntroPackW8A8Config):
    linear_kind: str = field(init=False, default="int8")

    @property
    def container_dtype(self):
        return torch.int8

@register_quant_backend("entropack")
class EntroPackQuantBackend(QuantBackend):
    def validate_environment(self):
        if not ENTROPACK_AVAILABLE:
            raise ImportError(
                "entropack with the CompressedLinear / CompressedFP8Linear / CompressedINT8Linear "
                "classes is required for this quantization method. Install the package and the CuPy extra "
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
        return (ep.CompressedLinear, ep.CompressedFP8Linear, ep.CompressedINT8Linear)

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
        quantized = self.config.linear_cls.from_linear(linear, dtype=self.config.dtype, **self.config.linear_kwargs())
        return quantized if model_device is None else quantized.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return self.config.linear_cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            dtype=self.config.dtype, **self.config.linear_kwargs(),
        )

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        weight = module.dequantize(compute_device).to(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(
                module.bias.data.to(dtype=compute_dtype, device=weight.device), requires_grad=False,
            )
        return linear if model_device is None else linear.to(device=model_device)

_METHODS = (
    ("entropack_dfloat11_bf16", EntroPackDFloat11BF16Config, "Lossless DFloat11 compression of BF16 weights"),
    ("entropack_tile_ans_bf16", EntroPackTileANSBF16Config, "Lossless Tile-ANS compression of BF16 weights"),
    (
        "entropack_lattice_rans_bf16",
        EntroPackLatticeRANSBF16Config,
        "Lossy E8-lattice vector quantization of BF16 weights at a target bit rate",
    ),
    ("entropack_fp8", EntroPackFP8Config, "W8A8 float8-e4m3 with the weight stored as coded float8"),
    ("entropack_int8", EntroPackINT8Config, "W8A8 int8 with the weight stored as coded int8"),
)

for method_name, config_class, label in _METHODS:
    register_quant_method(method_name, "entropack", config_class.from_kwargs, label=label)

__all__ = [
    "EntroPackConfig",
    "EntroPackDFloat11BF16Config",
    "EntroPackFP8Config",
    "EntroPackINT8Config",
    "EntroPackLatticeConfig",
    "EntroPackLatticeRANSBF16Config",
    "EntroPackQuantBackend",
    "EntroPackTileANSBF16Config",
    "EntroPackW8A8Config",
]
