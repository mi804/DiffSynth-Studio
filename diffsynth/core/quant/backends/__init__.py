import importlib

_LAZY_BACKENDS = {
    "bitsandbytes": ".bitsandbytes",
    "torchao": ".torchao",
    "comfy_kitchen": ".comfy_kitchen",
    "diffsynth_kernels_bf16_mantissa": ".diffsynth_kernels",
    "diffsynth_kernels_bf16_adaptive": ".diffsynth_kernels",
    "diffsynth_kernels_bf16_adaptive_online": ".diffsynth_kernels",
    "diffsynth_kernels_bf16_adaptive_offline": ".diffsynth_kernels",
    "diffsynth_kernels_tile_ans_fp32": ".diffsynth_kernels",
    "diffsynth_kernels_tile_ans_fp16": ".diffsynth_kernels",
    "diffsynth_kernels_dfloat11_bf16": ".diffsynth_kernels",
    "diffsynth_kernels_tile_ans_bf16": ".diffsynth_kernels",
    "diffsynth_kernels_tile_ans_fp8_e4m3fn": ".diffsynth_kernels",
    "diffsynth_kernels_tile_ans_fp8_e4m3fnuz": ".diffsynth_kernels",
    "diffsynth_kernels_tile_ans_fp8_e5m2": ".diffsynth_kernels",
    "diffsynth_kernels_tile_ans_fp8_e5m2fnuz": ".diffsynth_kernels",
}
_loaded = set()


def load_backend(name):
    module = _LAZY_BACKENDS.get(name)
    if module is not None and name not in _loaded:
        importlib.import_module(module, __name__)
        _loaded.add(name)


def load_all_backends():
    for name in _LAZY_BACKENDS:
        load_backend(name)


def load_backend_for_method(method):
    from ..config import QUANT_METHODS
    for name in _LAZY_BACKENDS:
        load_backend(name)
        if method in QUANT_METHODS:
            return
