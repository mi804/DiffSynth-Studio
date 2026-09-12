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
    EntroPackBF16E8Config,
    EntroPackLinear,
)


METHODS = (
    ("entropack_tile_ans_fp32", torch.float32),
    ("entropack_tile_ans_fp16", torch.float16),
    ("entropack_dfloat11_bf16", torch.bfloat16),
    ("entropack_tile_ans_bf16", torch.bfloat16),
    ("entropack_tile_ans_fp8_e4m3fn", torch.float8_e4m3fn),
    ("entropack_tile_ans_fp8_e4m3fnuz", torch.float8_e4m3fnuz),
    ("entropack_tile_ans_fp8_e5m2", torch.float8_e5m2),
    ("entropack_tile_ans_fp8_e5m2fnuz", torch.float8_e5m2fnuz),
)

LOSSY_METHOD = "entropack_bf16_e8"
LOSSY_KWARGS = {"execution_backend": "eager", "target_bpp": 8.0}


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@pytest.mark.parametrize(("method", "dtype"), METHODS)
def test_registered_method_converts_then_compresses_bitwise(method, dtype):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    expected = model[0].weight.detach().to(dtype)
    config = QuantizeConfig(
        method=method,
        backend_config_kwargs={"execution_backend": "eager"},
    )

    config.quantize_model(model)

    assert isinstance(model[0], EntroPackLinear)
    assert model[0].compression_dtype == dtype
    restored = config.backend.dequantize_to_linear(model[0], compute_dtype=dtype)
    assert torch.equal(_bits(restored.weight), _bits(expected))


@pytest.mark.parametrize(
    "method",
    (
        "entropack_dfloat11_bf16",
        "entropack_tile_ans_bf16",
    ),
)
def test_state_dict_roundtrip(method):
    source = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    config = QuantizeConfig(
        method=method,
        backend_config_kwargs={"execution_backend": "eager"},
    )
    config.quantize_model(source)
    state = source.state_dict()

    restored = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.float32)
    restored.load_state_dict(state, assign=True)

    expected_weight = source[0]._compressed()
    actual_weight = restored[0]._compressed()
    assert actual_weight.header == expected_weight.header
    for name in expected_weight.buffers:
        assert torch.equal(actual_weight.buffers[name], expected_weight.buffers[name])
    x = torch.randn(2, 33)
    assert torch.equal(restored(x), source(x))


@pytest.mark.parametrize(
    "method",
    (
        "entropack_dfloat11_bf16",
        "entropack_tile_ans_bf16",
    ),
)
def test_safetensors_roundtrip(tmp_path, method):
    source = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    config = QuantizeConfig(
        method=method,
        backend_config_kwargs={"execution_backend": "eager"},
    )
    config.quantize_model(source)
    tensors, metadata = config.flatten_state_dict(
        {name: tensor.cpu() for name, tensor in source.state_dict().items()}
    )
    path = tmp_path / "compressed.safetensors"
    save_file(tensors, path, metadata=metadata)

    restored = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.float32)
    restored.load_state_dict(
        config.unflatten_state_dict(load_file(path), metadata),
        assign=True,
    )

    assert metadata["diffsynth.quantization.schema"] == "1"
    x = torch.randn(2, 33)
    assert torch.equal(restored(x), source(x))


@pytest.mark.parametrize(
    "method",
    (
        "entropack_dfloat11_bf16",
        "entropack_tile_ans_bf16",
    ),
)
def test_legacy_buffer_keys_load(method):
    source = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    config = QuantizeConfig(
        method=method,
        backend_config_kwargs={"execution_backend": "eager"},
    )
    config.quantize_model(source)
    compressed = source[0]._compressed()
    legacy_state = {
        **{f"0.{name}": value for name, value in compressed.buffers.items()},
        "0.bias": source[0].bias,
    }

    restored = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.float32)
    restored.load_state_dict(legacy_state, assign=True)

    x = torch.randn(2, 33)
    assert torch.equal(restored(x), source(x))


def test_mixed_bf16_methods_roundtrip():
    config = MixedQuantizeConfig(
        configs=[
            QuantizeConfig(
                method="entropack_dfloat11_bf16",
                target_modules=["0"],
                backend_config_kwargs={"execution_backend": "eager"},
            ),
            QuantizeConfig(
                method="entropack_tile_ans_bf16",
                target_modules=["1"],
                backend_config_kwargs={"execution_backend": "eager"},
            ),
        ]
    )
    source = torch.nn.Sequential(
        torch.nn.Linear(16, 16, dtype=torch.float32),
        torch.nn.Linear(16, 16, dtype=torch.float32),
    )
    config.quantize_model(source)
    state, metadata = config.flatten_state_dict(source.state_dict())

    restored = torch.nn.Sequential(
        torch.nn.Linear(16, 16, dtype=torch.float32),
        torch.nn.Linear(16, 16, dtype=torch.float32),
    )
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.float32)
    restored.load_state_dict(config.unflatten_state_dict(state, metadata), assign=True)

    assert restored[0].compress_method == "dfloat11"
    assert restored[1].compress_method == "tile_ans"
    x = torch.randn(2, 16)
    assert torch.equal(restored(x), source(x))


def test_dtype_conversion_does_not_retype_compressed_buffers():
    model = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    config = QuantizeConfig(
        method="entropack_tile_ans_bf16",
        backend_config_kwargs={"execution_backend": "eager"},
    )
    config.quantize_model(model)
    original = {
        name: (buffer.dtype, buffer.clone())
        for name, buffer in model[0]._compressed().buffers.items()
    }

    model.to(dtype=torch.float16)

    for name, buffer in model[0]._compressed().buffers.items():
        dtype, value = original[name]
        assert buffer.dtype == dtype
        assert torch.equal(buffer, value)


def test_deepcopy_preserves_compressed_weight():
    model = torch.nn.Sequential(torch.nn.Linear(33, 17, dtype=torch.float32))
    config = QuantizeConfig(
        method="entropack_dfloat11_bf16",
        backend_config_kwargs={"execution_backend": "eager"},
    )
    config.quantize_model(model)

    cloned = copy.deepcopy(model[0])

    expected = model[0]._compressed()
    actual = cloned._compressed()
    assert actual.header == expected.header
    for name in expected.buffers:
        assert torch.equal(actual.buffers[name], expected.buffers[name])


@pytest.mark.parametrize(
    "method",
    (
        "entropack_tile_ans_fp16",
        LOSSY_METHOD,
    ),
)
def test_pinned_dtype_and_compress_method_reject_overrides(method):
    with pytest.raises(ValueError, match="not accepted"):
        QuantizeConfig(
            method=method,
            backend_config_kwargs={"dtype": torch.float32},
        )
    with pytest.raises(ValueError, match="not accepted"):
        QuantizeConfig(
            method=method,
            backend_config_kwargs={"compress_method": "dfloat11"},
        )


def _lossy_config(**config_kwargs):
    return QuantizeConfig(
        method=LOSSY_METHOD,
        backend_config_kwargs=LOSSY_KWARGS,
        **config_kwargs,
    )


def _lossy_model():
    return torch.nn.Sequential(torch.nn.Linear(512, 64, dtype=torch.float32))


def test_lossy_method_is_registered_with_final_defaults():
    assert QUANT_METHODS[LOSSY_METHOD].backend == "entropack"

    config = EntroPackBF16E8Config()
    assert config.compress_method is ep.CompressionMethod.BF16_E8
    assert config.dtype is torch.bfloat16
    assert config.target_bpp == 3.0
    assert config.side_dtype is None
    assert config.prob_bits is None
    assert config.tile_elements is None
    assert config.row_rdo_iterations == 0
    assert config.row_rdo_candidates == 5
    assert config.scale_search_iterations == 12
    assert config.scale_search_max_vectors == 262144
    assert config.compression_options() == {
        "target_bpp": 3.0,
        "side_dtype": None,
        "prob_bits": None,
        "row_rdo_iterations": 0,
        "row_rdo_candidates": 5,
        "scale_search_iterations": 12,
        "scale_search_max_vectors": 262144,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"target_bpp": 0.9}, "target_bpp"),
        ({"target_bpp": 11.1}, "target_bpp"),
        ({"side_dtype": torch.float16}, "side_dtype"),
        ({"prob_bits": 8}, "prob_bits"),
    ),
)
def test_lossy_config_rejects_invalid_values(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        EntroPackBF16E8Config.from_kwargs(kwargs)


@pytest.mark.parametrize("field_name", ("quality", "group_size", "dtype", "compress_method"))
def test_lossy_config_rejects_removed_unknown_and_pinned_fields(field_name):
    with pytest.raises(ValueError, match="not accepted"):
        EntroPackBF16E8Config.from_kwargs({field_name: "invalid"})


def test_lossy_reconstruction_uses_shared_linear():
    torch.manual_seed(10)
    model = _lossy_model()
    original = model[0].weight.detach().to(torch.bfloat16)
    config = _lossy_config()

    config.quantize_model(model)

    assert type(model[0]) is EntroPackLinear
    compressed = model[0]._compressed()
    assert compressed.header["version"] == ep.ENVELOPE_VERSION == 2
    assert compressed.compression_kind is ep.CompressionKind.LOSSY
    assert model[0].compression_kind is ep.CompressionKind.LOSSY
    restored = ep.decompress(compressed, execution_backend="eager")
    assert restored.dtype is torch.bfloat16
    assert restored.shape == original.shape
    assert not torch.equal(_bits(restored), _bits(original))
    relative_error = torch.linalg.vector_norm(restored.float() - original.float()) / torch.linalg.vector_norm(original.float())
    assert relative_error < 0.1


def test_lossy_v2_state_dict_and_safetensors_roundtrip(tmp_path):
    torch.manual_seed(11)
    source = _lossy_model()
    config = _lossy_config()
    config.quantize_model(source)
    state = source.state_dict()
    assert state["0._entropack.header"].dtype is torch.uint8

    restored = _lossy_model()
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.float32)
    restored.load_state_dict(state, assign=True)

    expected = source[0]._compressed()
    actual = restored[0]._compressed()
    assert actual.header == expected.header
    assert actual.compression_kind is ep.CompressionKind.LOSSY
    for name in expected.buffers:
        assert torch.equal(actual.buffers[name], expected.buffers[name])

    tensors, metadata = config.flatten_state_dict(state)
    path = tmp_path / f"{LOSSY_METHOD}.safetensors"
    save_file(tensors, path, metadata=metadata)
    from_file = _lossy_model()
    config.prepare_for_prequantized_load(from_file, compute_dtype=torch.float32)
    from_file.load_state_dict(
        config.unflatten_state_dict(load_file(path), metadata),
        assign=True,
    )
    for name in expected.buffers:
        assert torch.equal(from_file[0]._compressed().buffers[name], expected.buffers[name])
    x = torch.randn(2, 512)
    assert torch.equal(restored(x), source(x))
    assert torch.equal(from_file(x), source(x))


def test_set_compressed_rejects_wrong_method_dtype_codec_and_buffer_schema():
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    lossy = ep.compress(
        weight,
        compress_method=ep.CompressionMethod.BF16_E8,
        execution_backend="eager",
    )
    dfloat = ep.compress(
        weight,
        compress_method=ep.CompressionMethod.DFLOAT11,
        execution_backend="eager",
    )
    shell = EntroPackLinear(
        16,
        8,
        bias=False,
        compute_dtype=torch.float32,
        compression_dtype=torch.bfloat16,
        compress_method=ep.CompressionMethod.BF16_E8,
        execution_backend="eager",
    )

    with pytest.raises(ValueError, match="method"):
        shell._set_compressed(dfloat)

    wrong_dtype = copy.deepcopy(lossy)
    wrong_dtype.dtype = torch.float16
    with pytest.raises(ValueError, match="dtype"):
        shell._set_compressed(wrong_dtype)

    wrong_codec = copy.deepcopy(lossy)
    wrong_codec.header["codec_version"] += 1
    with pytest.raises(ValueError, match="codec version"):
        shell._set_compressed(wrong_codec)

    wrong_buffers = copy.deepcopy(lossy)
    wrong_buffers.buffers.pop(next(iter(wrong_buffers.buffers)))
    with pytest.raises(ValueError, match="buffers"):
        shell._set_compressed(wrong_buffers)


def test_lossy_requires_v2_header_and_rejects_headerless_buffers():
    model = torch.nn.Sequential(torch.nn.Linear(512, 64))
    config = _lossy_config()
    config.quantize_model(model)
    compressed = model[0]._compressed()

    v1 = copy.deepcopy(compressed)
    v1.header.update(
        version=1,
        format_id="bf16",
        method_id=ep.CompressionMethod.BF16_E8,
    )
    with pytest.raises(ValueError, match="standard v2 header"):
        model[0]._set_compressed(v1)

    legacy_state = {
        **{f"0.{name}": value for name, value in compressed.buffers.items()},
        "0.bias": model[0].bias,
    }
    shell = torch.nn.Sequential(torch.nn.Linear(512, 64))
    config.prepare_for_prequantized_load(shell, compute_dtype=torch.float32)
    with pytest.raises(RuntimeError, match="headerless.*only supported by lossless"):
        shell.load_state_dict(legacy_state, assign=True)


def test_mixed_lossless_and_lossy_methods_roundtrip():
    config = MixedQuantizeConfig(
        configs=[
            QuantizeConfig(
                method="entropack_dfloat11_bf16",
                target_modules=["lossless"],
                backend_config_kwargs={"execution_backend": "eager"},
            ),
            QuantizeConfig(
                method=LOSSY_METHOD,
                target_modules=["lossy"],
                backend_config_kwargs=LOSSY_KWARGS,
            ),
        ]
    )
    source = torch.nn.ModuleDict(
        {
            "lossless": torch.nn.Linear(64, 32),
            "lossy": torch.nn.Linear(512, 64),
        }
    )
    config.quantize_model(source)
    state, metadata = config.flatten_state_dict(source.state_dict())

    restored = torch.nn.ModuleDict(
        {
            "lossless": torch.nn.Linear(64, 32),
            "lossy": torch.nn.Linear(512, 64),
        }
    )
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.float32)
    restored.load_state_dict(config.unflatten_state_dict(state, metadata), assign=True)

    assert restored["lossless"].compression_kind is ep.CompressionKind.LOSSLESS
    assert restored["lossy"].compression_kind is ep.CompressionKind.LOSSY
    lossless_input = torch.randn(2, 64)
    assert torch.equal(
        restored["lossless"](lossless_input), source["lossless"](lossless_input)
    )
    lossy_input = torch.randn(2, 512)
    assert torch.equal(restored["lossy"](lossy_input), source["lossy"](lossy_input))


def test_lossy_dtype_preserving_apply_and_deepcopy():
    model = _lossy_model()
    config = _lossy_config()
    config.quantize_model(model)
    original = {
        name: (buffer.dtype, buffer.clone())
        for name, buffer in model[0]._compressed().buffers.items()
    }

    model.to(dtype=torch.float16)
    cloned = copy.deepcopy(model[0])

    for module in (model[0], cloned):
        for name, buffer in module._compressed().buffers.items():
            dtype, value = original[name]
            assert buffer.dtype == dtype
            assert torch.equal(buffer, value)
        assert module._compressed().header == model[0]._compressed().header


def test_lossy_dequant_once_restores_plain_linear():
    torch.manual_seed(12)
    model = _lossy_model()
    config = _lossy_config(mode="dequant_once")
    config.quantize_model(model)
    expected = ep.decompress(model[0]._compressed(), execution_backend="eager")

    config.dequantize_model(model, compute_dtype=torch.bfloat16)

    assert type(model[0]) is torch.nn.Linear
    assert torch.equal(_bits(model[0].weight), _bits(expected))


def test_lossy_input_and_lora_branch_gradients():
    torch.manual_seed(13)
    model = _lossy_model()
    config = _lossy_config()
    config.quantize_model(model)
    assert check_differentiable(model[0], verbose=False)

    rank = 4
    lora_a = torch.nn.Parameter(torch.randn(rank, 512, dtype=torch.bfloat16))
    lora_b = torch.nn.Parameter(torch.randn(64, rank, dtype=torch.bfloat16))
    x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
    output = model[0](x) + (x @ lora_a.t()) @ lora_b.t()
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert lora_a.grad is not None and torch.isfinite(lora_a.grad).all()
    assert lora_b.grad is not None and torch.isfinite(lora_b.grad).all()
    assert all(not buffer.requires_grad for buffer in model[0]._compressed().buffers.values())


def test_lossy_rejects_shapes_the_codec_cannot_tile():
    # lattice_rans codes 8-dim vectors per row, so a row shorter than 8 has no vector plane.
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with pytest.raises(ValueError, match="multiple of 8"):
        _lossy_config().quantize_model(model)
