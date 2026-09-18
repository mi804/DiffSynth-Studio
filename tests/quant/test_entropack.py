import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

import entropack as ep
from diffsynth.core.quant import (
    QUANT_METHODS,
    MixedQuantizeConfig,
    QuantizeConfig,
    check_differentiable,
)
from diffsynth.core.quant.backends.entropack import (
    EntroPackLossyQuantConfig,
    EntroPackLossyQuantFP8Config,
    EntroPackLossyQuantINT8Config,
)

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="entropack's Linear classes need a CUDA device")

DEVICE = torch.device("cuda")
METHODS = (
    ("entropack_lossless_compression", ep.CompressedLinear, torch.bfloat16),
    ("entropack_lossy_quant", ep.CompressedLinear, torch.bfloat16),
    ("entropack_lossy_quant_fp8", ep.CompressedFP8Linear, torch.float8_e4m3fn),
    ("entropack_lossy_quant_int8", ep.CompressedINT8Linear, torch.int8),
)
W8A8 = METHODS[2:]
LOSSLESS = METHODS[:1]
LOSSY_METHOD = "entropack_lossy_quant"
SHAPES = (("lossless", 64), ("lossy", 512), ("w8a8", 512))

def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)

def _model(in_features=256, out_features=128):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(in_features, out_features, dtype=torch.bfloat16, device=DEVICE))
    with torch.no_grad():
        model[0].weight.mul_(0.02)
        if model[0].bias is not None:
            model[0].bias.mul_(0.02)
    return model

def _config(method, **kwargs):
    return QuantizeConfig(method=method, backend_config_kwargs=kwargs)

def _quantize(method, **kwargs):
    model = _model()
    _config(method, **kwargs).quantize_model(model)
    return model

def _mixed_source():
    return torch.nn.ModuleDict({
        name: torch.nn.Linear(width, 32, dtype=torch.bfloat16, device=DEVICE) for name, width in SHAPES
    })

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_a_registered_method_builds_the_linear_it_names(method, cls, container):
    layer = _quantize(method)[0]

    assert type(layer) is cls
    assert layer.container_dtype is container
    assert layer.compressed_weight.dtype is container
    assert layer.compressed_bits < 16.0
    assert layer(torch.randn(8, 256, dtype=torch.bfloat16, device=DEVICE)).shape == (8, 128)

@pytest.mark.parametrize(("method", "cls", "container"), LOSSLESS)
def test_a_lossless_floating_method_gives_the_weight_back_bit_for_bit(method, cls, container):
    source = _model()
    expected = source[0].weight.detach()
    config = _config(method)
    config.quantize_model(source)

    restored = config.backend.dequantize_to_linear(source[0], compute_dtype=torch.bfloat16)
    assert torch.equal(_bits(restored.weight), _bits(expected))

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_state_dict_roundtrip(method, cls, container):
    source = _quantize(method)
    config = _config(method)
    state = source.state_dict()
    assert state["0._entropack.header"].dtype is torch.uint8

    restored = _model()
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.bfloat16)
    restored.load_state_dict(state, assign=True)

    expected, actual = source[0].compressed_weight, restored[0].compressed_weight
    assert actual.header == expected.header
    for name in expected.buffers:
        assert torch.equal(actual.buffers[name], expected.buffers[name])
    x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEVICE)
    assert torch.equal(restored(x), source(x))

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_safetensors_roundtrip(tmp_path, method, cls, container):
    source = _quantize(method)
    config = _config(method)
    tensors, metadata = config.flatten_state_dict({k: v.cpu() for k, v in source.state_dict().items()})
    path = tmp_path / f"{method}.safetensors"
    save_file(tensors, path, metadata=metadata)

    restored = _model()
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.bfloat16)
    restored.load_state_dict(config.unflatten_state_dict(load_file(path), metadata), assign=True)

    assert metadata["diffsynth.quantization.schema"] == "1"
    x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEVICE)
    assert torch.equal(restored(x), source(x))

def test_mixed_methods_roundtrip():
    config = MixedQuantizeConfig(
        configs=[
            QuantizeConfig(method="entropack_lossless_compression", target_modules=["lossless"]),
            QuantizeConfig(method=LOSSY_METHOD, target_modules=["lossy"], backend_config_kwargs={"target_bit_per_param": 4.0}),
            QuantizeConfig(method="entropack_lossy_quant_int8", target_modules=["w8a8"],
                           backend_config_kwargs={"target_bit_per_param": 4.0}),
        ]
    )
    source = _mixed_source()
    config.quantize_model(source)
    state, metadata = config.flatten_state_dict(source.state_dict())

    restored = _mixed_source()
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.bfloat16)
    restored.load_state_dict(config.unflatten_state_dict(state, metadata), assign=True)

    assert restored["lossless"].compressed_weight.lossless is True
    assert restored["lossy"].compressed_weight.lossless is False
    assert type(restored["w8a8"]) is ep.CompressedINT8Linear
    for name, width in SHAPES:
        x = torch.randn(2, width, dtype=torch.bfloat16, device=DEVICE)
        assert torch.equal(restored[name](x), source[name](x))

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_a_dtype_cast_does_not_retype_the_stored_bits(method, cls, container):
    model = _quantize(method)
    original = {name: (buffer.dtype, buffer.clone())
                for name, buffer in model[0].compressed_weight.buffers.items()}

    model.to(dtype=torch.float16)

    for name, buffer in model[0].compressed_weight.buffers.items():
        dtype, value = original[name]
        assert buffer.dtype == dtype
        assert torch.equal(buffer, value)

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_a_deep_copy_owns_the_same_bits(method, cls, container):
    model = _quantize(method)
    cloned = copy.deepcopy(model[0])

    expected, actual = model[0].compressed_weight, cloned.compressed_weight
    assert actual.header == expected.header
    for name in expected.buffers:
        assert torch.equal(actual.buffers[name], expected.buffers[name])
        assert actual.buffers[name].data_ptr() != expected.buffers[name].data_ptr()

def test_a_pinned_or_removed_field_cannot_be_overridden():
    pinned = {
        "entropack_lossless_compression": ("dtype", "linear_kind", "target_bit_per_param", "execution_backend", "group_size"),
        "entropack_lossy_quant": ("dtype", "linear_kind", "scheme", "execution_backend", "group_size"),
        "entropack_lossy_quant_fp8": ("dtype", "linear_kind", "scheme", "execution_backend", "group_size"),
        "entropack_lossy_quant_int8": ("dtype", "linear_kind", "scheme", "execution_backend", "group_size"),
    }
    for method, field_names in pinned.items():
        for field_name in field_names:
            with pytest.raises(ValueError, match="not accepted"):
                QuantizeConfig(method=method, backend_config_kwargs={field_name: "invalid"})

def test_the_lossy_config_defaults_to_four_bits_and_passes_its_options_through():
    assert QUANT_METHODS[LOSSY_METHOD].backend == "entropack"

    config = EntroPackLossyQuantConfig()
    assert config.scheme == "lattice_rans"
    assert config.target_bit_per_param == 4.0
    assert config.options == {}
    assert config.linear_kwargs() == {"scheme": "lattice_rans", "target_bpp": 4.0}

    config = EntroPackLossyQuantConfig.from_kwargs({"target_bit_per_param": 3.0, "options": {"prob_bits": 12}})
    assert config.linear_kwargs() == {"scheme": "lattice_rans", "target_bpp": 3.0, "prob_bits": 12}

@pytest.mark.parametrize(("kwargs", "message"), (
    ({"target_bit_per_param": 0.9}, "target_bpp"),
    ({"target_bit_per_param": 11.1}, "target_bpp"),
    ({"target_bit_per_param": float("nan")}, "target_bpp"),
    ({"target_bit_per_param": True}, "target_bpp"),
    ({"options": {"prob_bits": 8}}, "prob_bits"),
))
def test_the_lossy_config_rejects_a_rate_it_cannot_honour(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        EntroPackLossyQuantConfig.from_kwargs(kwargs)

@pytest.mark.parametrize(("method", "cls", "container"), W8A8)
@pytest.mark.parametrize("target_bit_per_param", [8.0, 9, 16.0, 0.0, -1.0, float("nan"), True, "4"])
def test_a_w8a8_rate_at_or_above_the_codes_own_width_is_refused(method, cls, container, target_bit_per_param):
    with pytest.raises(ValueError, match="only pays below 8"):
        _quantize(method, target_bit_per_param=target_bit_per_param)

@pytest.mark.parametrize(("method", "cls", "container"), W8A8)
@pytest.mark.parametrize("target_bit_per_param", [None, 4.0])
def test_a_w8a8_method_can_store_its_codes_uncoded(method, cls, container, target_bit_per_param):
    layer = _quantize(method, target_bit_per_param=target_bit_per_param)[0]

    assert layer.scheme_name == ("raw" if target_bit_per_param is None else "lattice_rans")
    assert layer.codes(DEVICE).dtype is container
    assert torch.isfinite(layer(torch.randn(8, 256, dtype=torch.bfloat16, device=DEVICE))).all()
    assert layer.compressed_bits < (9.0 if target_bit_per_param is None else 6.0)

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_every_method_declares_itself_differentiable(method, cls, container):
    assert _config(method).backend.capabilities()["is_differentiable"] is True

def test_set_compressed_refuses_a_container_built_for_another_layer():
    weight = torch.randn(8, 16, dtype=torch.bfloat16, device=DEVICE) * 0.02
    coded = ep.compress(weight, compress_method="lattice_rans", target_bpp=4.0, execution_backend="cuda")
    dfloat = ep.compress(weight, compress_method="dfloat11", execution_backend="cuda")
    shell = ep.CompressedLinear(16, 8, bias=False, dtype=torch.bfloat16, scheme="lattice_rans", target_bpp=4.0)
    taller = ep.CompressedLinear(16, 9, bias=False, dtype=torch.bfloat16, scheme="lattice_rans", target_bpp=4.0)
    wrong_dtype = copy.deepcopy(coded)
    wrong_dtype.dtype = torch.float16

    with pytest.raises(ValueError, match="uses 'dfloat11'"):
        shell.set_compressed(dfloat)
    with pytest.raises(ValueError, match="does not match this Linear"):
        taller.set_compressed(coded)
    with pytest.raises(ValueError, match="configured for"):
        shell.set_compressed(wrong_dtype)
    with pytest.raises(TypeError, match="CompressedTensor"):
        shell.set_compressed({"header": {}, "buffers": {}})
    with pytest.raises(ValueError, match="configured for"):
        ep.CompressedINT8Linear(16, 8, bias=False, target_bpp=4.0).set_compressed(coded)

def test_dequant_once_restores_a_plain_linear():
    model = _quantize(LOSSY_METHOD, target_bit_per_param=4.0)
    config = QuantizeConfig(method=LOSSY_METHOD, backend_config_kwargs={"target_bit_per_param": 4.0}, mode="dequant_once")
    expected = model[0].dequantize().to(torch.bfloat16)

    config.dequantize_model(model, compute_dtype=torch.bfloat16)

    assert type(model[0]) is torch.nn.Linear
    assert torch.equal(_bits(model[0].weight), _bits(expected))

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_input_and_lora_branch_gradients_reach_every_method(method, cls, container):
    model = _quantize(method)
    assert check_differentiable(model[0], verbose=False)

    rank = 4
    lora_a = torch.nn.Parameter(torch.randn(rank, 256, dtype=torch.bfloat16, device=DEVICE))
    lora_b = torch.nn.Parameter(torch.randn(128, rank, dtype=torch.bfloat16, device=DEVICE))
    x = torch.randn(2, 8, 256, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
    output = model[0](x) + (x @ lora_a.t()) @ lora_b.t()
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all() and (x.grad != 0).any()
    assert x.grad.shape == x.shape
    assert lora_a.grad is not None and torch.isfinite(lora_a.grad).all()
    assert lora_b.grad is not None and torch.isfinite(lora_b.grad).all()
    assert all(not buffer.requires_grad for buffer in model[0].compressed_weight.buffers.values())
    assert all(parameter.grad is None for parameter in model[0].parameters())

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_tracking_a_gradient_does_not_change_the_forward(method, cls, container):
    model = _quantize(method)
    x = torch.randn(2, 8, 256, dtype=torch.bfloat16, device=DEVICE)

    with torch.no_grad():
        inference = model[0](x)
    tracked = model[0](x.clone().requires_grad_(True))

    assert tracked.requires_grad
    assert torch.equal(tracked.detach(), inference)

@pytest.mark.parametrize(("method", "cls", "container"), W8A8)
def test_a_w8a8_gradient_is_the_dequantized_weight_transpose(method, cls, container):
    layer = _quantize(method, target_bit_per_param=4.0)[0]
    x = torch.randn(3, 8, 256, dtype=torch.bfloat16, device=DEVICE)
    incoming = torch.randn(3, 8, 128, dtype=torch.bfloat16, device=DEVICE)

    tracked = x.clone().requires_grad_(True)
    layer(tracked).backward(incoming)

    expected = torch.mm(incoming.reshape(-1, 128), layer.dequantize(DEVICE).to(incoming.dtype)).reshape(x.shape)
    assert torch.equal(tracked.grad, expected)

@pytest.mark.parametrize(("method", "cls", "container"), METHODS)
def test_a_shape_the_codec_has_to_pad_still_round_trips(method, cls, container):
    for in_features, out_features in ((3420, 128), (1001, 64), (1, 1)):
        model = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16, device=DEVICE))
        _config(method).quantize_model(model)

        assert type(model[0]) is cls
        assert model[0].in_features == in_features
        assert model[0].compressed_weight.shape == (out_features, in_features)
        x = torch.randn(2, in_features, dtype=torch.bfloat16, device=DEVICE)
        assert model[0](x).shape == (2, out_features)
