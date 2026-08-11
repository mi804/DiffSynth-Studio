import importlib.util

import torch
from ..base import QuantBackend, register_quant_backend
from ..config import register_quant_method


class TorchaoLinear(torch.nn.Linear):
    """Marker class for torchao-quantized Linears."""


@register_quant_backend("torchao")
class TorchaoQuantBackend(QuantBackend):
    """Adapter over torchao `quantize_` (weight-only configs); the quantization lives in the weight tensor subclass, not the module class."""

    def _hardware_requirement(self):
        """
        `((major, minor), reason)` the current config needs, or `None` when unconstrained.
        Resolved from the config's fields, since one config class can map to several kernels
        with different requirements.
        """
        config = self.config
        name = type(config).__name__
        if name in ("Int4WeightOnlyConfig", "Float8DynamicActivationInt4WeightConfig"):
            # The mslk packings need SM>=9.0: "preshuffled" issues WGMMA, and "plain" runs an
            # mslk cutlass kernel needing TMA (on SM 8.0 it dies with "cutlass cannot
            # initialize"). Only the torch-native packing runs below that.
            if getattr(config, "int4_packing_format", "plain") in ("plain", "preshuffled"):
                return (9, 0), (
                    "the mslk int4 kernels (WGMMA / TMA instructions); select the torch-native "
                    'packing with backend_config_kwargs={"int4_packing_format": '
                    '"tile_packed_to_4d"} to run on older devices'
                )
            return None
        if name == "Float8DynamicActivationFloat8WeightConfig":
            return (8, 9), "fp8 tensor cores"
        if name in ("MXDynamicActivationMXWeightConfig", "NVFP4DynamicActivationNVFP4WeightConfig"):
            return (10, 0), (
                "block-scaled fp4/fp8 matmul (blockwise scales with swizzled layouts); "
                "pass `kernel_preference=KernelPreference.EMULATED` where the config supports it "
                "to run the numerics on any device instead"
            )
        return None

    def _require_mslk(self):
        if importlib.util.find_spec("mslk") is None:
            raise ImportError(
                f"`{type(self.config).__name__}` runs on the mslk int4 kernels "
                "(`torch.ops.mslk.f8i4bf16_*`), but `mslk` is not installed. Install the build "
                "matching your torch version from https://github.com/meta-pytorch/MSLK."
            )

    def validate_environment(self):
        try:
            import torchao  # noqa: F401
        except ImportError:
            raise ImportError(
                "torchao is required for torchao quantization methods. "
                'Please install it via `pip install torchao` or `pip install "diffsynth[quant]"`.'
            )
        if type(self.config).__name__ == "Float8DynamicActivationInt4WeightConfig":
            self._require_mslk()
        requirement = self._hardware_requirement()
        if requirement is None or not torch.cuda.is_available():
            return
        # The emulated path dequantizes and runs an ordinary matmul, so it needs no special
        # hardware; only the hardware-backed kernels do.
        if getattr(getattr(self.config, "kernel_preference", None), "name", None) == "EMULATED":
            return
        (major, minor), reason = requirement
        capability = torch.cuda.get_device_capability()
        if capability < (major, minor):
            raise RuntimeError(
                f"`{type(self.config).__name__}` needs CUDA compute capability >= {major}.{minor} "
                f"for {reason}, but {torch.cuda.get_device_name()} reports "
                f"SM {capability[0]}.{capability[1]}."
            )

    # `Int8DynamicActivationInt8WeightConfig` is the only config *measured* to be
    # unserializable: its weight is a `LinearActivationQuantizedTensor`, which torchao's
    # safetensors `flatten_tensor_state_dict` rejects outright. The other
    # dynamic-activation configs could not be measured here (they need newer GPUs, see
    # `_CONFIG_MIN_CAPABILITY`), so they keep the optimistic default and their real
    # behaviour should be settled by running the quant test matrix on suitable hardware.
    _NON_SERIALIZABLE_CONFIGS = frozenset({
        "Int8DynamicActivationInt8WeightConfig",
    })

    # Configs whose quantized Linear passes a finite gradient to its input (verified with
    # `check_differentiable`), so LoRA can train through the frozen layer. The weight-only
    # int8/fp8 tensor subclasses are differentiable; int4 (tile-packed kernel) and the
    # dynamic-activation configs are not.
    _DIFFERENTIABLE_CONFIGS = frozenset({
        "Int8WeightOnlyConfig",
        "Float8WeightOnlyConfig",
    })

    def capabilities(self):
        config_name = type(self.config).__name__
        return {
            "is_serializable": config_name not in self._NON_SERIALIZABLE_CONFIGS,
            "is_differentiable": config_name in self._DIFFERENTIABLE_CONFIGS,
            "is_compileable": True,
            "requires_calibration": "StaticActivation" in config_name,
        }

    def quantized_linear_classes(self):
        return (TorchaoLinear,)

    def checkpoint_key_patterns(self):
        return ("weight", "_weight_qdata", "_weight_scale", "_weight_zero_point", "bias")

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        from torchao.quantization import quantize_
        linear.requires_grad_(False)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        quant_linear = TorchaoLinear(linear.in_features, linear.out_features, bias=linear.bias is not None, device="meta")
        quant_linear.weight = linear.weight
        quant_linear.bias = linear.bias
        quantize_(quant_linear, self.config)
        if model_device is not None:
            quant_linear = quant_linear.to(device=model_device)
        return quant_linear

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return TorchaoLinear(linear.in_features, linear.out_features, bias=linear.bias is not None, device="meta")

    def flatten_state_dict(self, state_dict):
        self._require_safetensors_support()
        from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
        flattened = flatten_tensor_state_dict(state_dict)
        if isinstance(flattened, tuple):
            return flattened
        return flattened, {}

    def unflatten_state_dict(self, state_dict, metadata):
        self._require_safetensors_support()
        import json
        from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict
        from torchao.prototype.safetensors.safetensors_utils import is_metadata_torchao
        if not is_metadata_torchao(metadata):
            raise ValueError(
                "This checkpoint carries no torchao metadata (no `tensor_names` entry in the "
                "safetensors header), so its tensor subclasses cannot be rebuilt. It was most "
                "likely not saved by torchao."
            )
        tensor_names = json.loads(metadata["tensor_names"])
        root_names = [name for name in tensor_names if "." not in name]
        if root_names:
            metadata = {**metadata, "tensor_names": json.dumps([name for name in tensor_names if "." in name])}
        rebuilt = unflatten_tensor_state_dict(state_dict, metadata)
        if isinstance(rebuilt, tuple):
            rebuilt = rebuilt[0]
        for name in root_names:
            if name in state_dict:
                rebuilt[name] = state_dict[name]
        return rebuilt

    def _require_safetensors_support(self):
        from torchao import __version__ as torchao_version
        try:
            from torchao.prototype.safetensors.safetensors_support import (  # noqa: F401
                flatten_tensor_state_dict,
                unflatten_tensor_state_dict,
            )
        except ImportError:
            raise ImportError(
                f"Serializing torchao quantized weights to safetensors needs torchao >= 0.16 "
                f"(found {torchao_version}), which provides "
                "`torchao.prototype.safetensors`."
            )

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        weight = module.weight
        # Activation-quant configs wrap the quantized weight in a
        # `LinearActivationQuantizedTensor` that does not implement `aten.dequantize`
        # itself; its real quantized weight sits in `original_weight_tensor`.
        weight = getattr(weight, "original_weight_tensor", weight)
        fp_weight = (weight.dequantize() if hasattr(weight, "dequantize") else weight.data).to(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        return linear if model_device is None else linear.to(device=model_device)


def _int8_weight_only(backend_config_kwargs):
    from torchao.quantization import Int8WeightOnlyConfig
    backend_config_kwargs.setdefault("version", 2)
    return Int8WeightOnlyConfig(**backend_config_kwargs)


def _int4_weight_only(backend_config_kwargs):
    from torchao.quantization import Int4WeightOnlyConfig
    packing_format = backend_config_kwargs.get("int4_packing_format", "plain")
    if packing_format in ("plain", "preshuffled") and importlib.util.find_spec("mslk") is None:
        raise ImportError(
            f'The int4 packing format "{packing_format}" requires `mslk` '
            "(https://github.com/meta-pytorch/MSLK, install the build matching your torch version), "
            'or select a packing format shipped with torch, e.g. backend_config_kwargs={"int4_packing_format": "tile_packed_to_4d"} '
            "(runs on any CUDA device, but much slower on large-token workloads)."
        )
    return Int4WeightOnlyConfig(**backend_config_kwargs)


def _fp8_weight_only(backend_config_kwargs):
    from torchao.quantization import Float8WeightOnlyConfig
    return Float8WeightOnlyConfig(**backend_config_kwargs)


def _int8_dynamic_activation_int8_weight(backend_config_kwargs):
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig
    return Int8DynamicActivationInt8WeightConfig(**backend_config_kwargs)


def _float8_dynamic_activation_float8_weight(backend_config_kwargs):
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    return Float8DynamicActivationFloat8WeightConfig(**backend_config_kwargs)


def _float8_dynamic_activation_int4_weight(backend_config_kwargs):
    from torchao.quantization import Float8DynamicActivationInt4WeightConfig
    return Float8DynamicActivationInt4WeightConfig(**backend_config_kwargs)


def _mx_dynamic_activation_mx_weight(elem_dtype):
    def factory(backend_config_kwargs):
        from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
        dtype = getattr(torch, elem_dtype, None)
        if dtype is None:
            raise ImportError(f"This torch build has no `torch.{elem_dtype}`, required by the MX config.")
        kwargs = {"block_size": 32, "activation_dtype": dtype, "weight_dtype": dtype}
        kwargs.update(backend_config_kwargs)
        return MXDynamicActivationMXWeightConfig(**kwargs)
    return factory


def _nvfp4_dynamic_activation_nvfp4_weight(backend_config_kwargs):
    from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
    return NVFP4DynamicActivationNVFP4WeightConfig(**backend_config_kwargs)


register_quant_method("torchao_int8_w8a16", "torchao", _int8_weight_only, label="8bit, int8, weight-only")
register_quant_method("torchao_int4_w4a16", "torchao", _int4_weight_only,
                      label="W4A16, int4 weight-only (mslk packings need SM>=9.0; use "
                            'int4_packing_format="tile_packed_to_4d" on older devices)')
register_quant_method("torchao_fp8_w8a16", "torchao", _fp8_weight_only, label="8bit, fp8, weight-only")
register_quant_method("torchao_int8_w8a8", "torchao", _int8_dynamic_activation_int8_weight,
                      label="W8A8, int8 weight + int8 dynamic activation")
register_quant_method("torchao_fp8_w8a8", "torchao", _float8_dynamic_activation_float8_weight,
                      label="W8A8, fp8 weight + fp8 dynamic activation (needs SM>=8.9)")
register_quant_method("torchao_int4_w4a8", "torchao", _float8_dynamic_activation_int4_weight,
                      label="W4A8, int4 weight + fp8 dynamic activation (needs SM>=9.0 + `pip install mslk`)")
register_quant_method("torchao_mxfp8_w8a8", "torchao", _mx_dynamic_activation_mx_weight("float8_e4m3fn"),
                      label="W8A8, MXFP8 microscaling weight + activation (hardware path TBD, needs SM>=10.0; "
                            "use kernel_preference=EMULATED to run the numerics on any device)")
register_quant_method("torchao_mxfp4_w4a4", "torchao", _mx_dynamic_activation_mx_weight("float4_e2m1fn_x2"),
                      label="W4A4, MXFP4 microscaling weight + activation (hardware path TBD, needs SM>=10.0; "
                            "use kernel_preference=EMULATED to run the numerics on any device)")
register_quant_method("torchao_nvfp4_w4a4", "torchao", _nvfp4_dynamic_activation_nvfp4_weight,
                      label="W4A4, NVFP4 weight + activation (hardware path TBD, needs SM>=10.0)")
