import json
from dataclasses import dataclass, field

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

@dataclass
class EntroPackConfig(BackendConfig):

    linear_kind: str = field(init=False, default="bf16")
    scheme: str = field(init=False, default="auto")
    target_bit_per_param: float | None = field(init=False, default=None)
    options: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if not ENTROPACK_AVAILABLE or self.scheme == "auto":
            return
        ep.get_scheme(self.scheme).validate_options(self.linear_options())

    @property
    def linear_cls(self):
        return {
            "bf16": ep.CompressedLinear,
            "fp8": ep.CompressedFP8Linear,
            "int8": ep.CompressedINT8Linear,
        }[self.linear_kind]

    def linear_options(self) -> dict:
        options = dict(self.options)
        if self.target_bit_per_param is not None:
            options["target_bpp"] = self.target_bit_per_param
        return options

    def linear_kwargs(self) -> dict:
        kwargs = self.linear_options()
        if self.scheme != "auto":
            kwargs["scheme"] = self.scheme
        return kwargs

@dataclass
class EntroPackLosslessConfig(EntroPackConfig):
    scheme: str = "auto"
    options: dict = field(default_factory=dict)

@dataclass
class EntroPackLossyQuantConfig(EntroPackConfig):
    scheme: str = field(init=False, default="lattice_rans")
    target_bit_per_param: float | None = 4.0
    options: dict = field(default_factory=dict)

@dataclass
class EntroPackLossyQuantFP8Config(EntroPackConfig):
    linear_kind: str = field(init=False, default="fp8")
    target_bit_per_param: float | None = 4.0
    options: dict = field(default_factory=dict)

@dataclass
class EntroPackLossyQuantINT8Config(EntroPackConfig):
    linear_kind: str = field(init=False, default="int8")
    target_bit_per_param: float | None = 4.0
    options: dict = field(default_factory=dict)

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
        quantized = self.config.linear_cls.from_linear(linear, **self.config.linear_kwargs())
        return quantized if model_device is None else quantized.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return self.config.linear_cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            dtype=compute_dtype, **self.config.linear_kwargs(),
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

register_quant_method("entropack_lossless_compression", "entropack", EntroPackLosslessConfig.from_kwargs, label="lossless")
register_quant_method("entropack_lossy_quant", "entropack", EntroPackLossyQuantConfig.from_kwargs, label="lossy, lattice rate")
register_quant_method("entropack_lossy_quant_fp8", "entropack", EntroPackLossyQuantFP8Config.from_kwargs, label="W8A8, fp8")
register_quant_method("entropack_lossy_quant_int8", "entropack", EntroPackLossyQuantINT8Config.from_kwargs, label="W8A8, int8")
