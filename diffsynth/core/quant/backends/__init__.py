import importlib

_LAZY_BACKENDS = {
    "bitsandbytes": ".bitsandbytes",
    "torchao": ".torchao",
    "comfy_kitchen": ".comfy_kitchen",
    "entropack_bf16_e8": ".entropack",
    "entropack_tile_ans_fp32": ".entropack",
    "entropack_tile_ans_fp16": ".entropack",
    "entropack_dfloat11_bf16": ".entropack",
    "entropack_tile_ans_bf16": ".entropack",
    "entropack_tile_ans_fp8_e4m3fn": ".entropack",
    "entropack_tile_ans_fp8_e4m3fnuz": ".entropack",
    "entropack_tile_ans_fp8_e5m2": ".entropack",
    "entropack_tile_ans_fp8_e5m2fnuz": ".entropack",
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
