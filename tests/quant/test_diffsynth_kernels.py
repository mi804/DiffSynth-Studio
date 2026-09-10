import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

import diffsynth_kernels as dk
from diffsynth.core.quant import (
    QUANT_METHODS,
    MixedQuantizeConfig,
    QuantizeConfig,
    check_differentiable,
)
from diffsynth.core.quant.backends.diffsynth_kernels import (
    DiffSynthKernelsBF16AdaptiveConfig,
    DiffSynthKernelsBF16AdaptiveOfflineConfig,
    DiffSynthKernelsBF16AdaptiveOnlineConfig,
    DiffSynthKernelsBF16MantissaConfig,
    DiffSynthKernelsLinear,
)


METHODS = (
    ("diffsynth_kernels_tile_ans_fp32", torch.float32),
    ("diffsynth_kernels_tile_ans_fp16", torch.float16),
    ("diffsynth_kernels_dfloat11_bf16", torch.bfloat16),
    ("diffsynth_kernels_tile_ans_bf16", torch.bfloat16),
    ("diffsynth_kernels_tile_ans_fp8_e4m3fn", torch.float8_e4m3fn),
    ("diffsynth_kernels_tile_ans_fp8_e4m3fnuz", torch.float8_e4m3fnuz),
    ("diffsynth_kernels_tile_ans_fp8_e5m2", torch.float8_e5m2),
    ("diffsynth_kernels_tile_ans_fp8_e5m2fnuz", torch.float8_e5m2fnuz),
)


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

    assert isinstance(model[0], DiffSynthKernelsLinear)
    assert model[0].compression_dtype == dtype
    restored = config.backend.dequantize_to_linear(model[0], compute_dtype=dtype)
    assert torch.equal(_bits(restored.weight), _bits(expected))


@pytest.mark.parametrize(
    "method",
    (
        "diffsynth_kernels_dfloat11_bf16",
        "diffsynth_kernels_tile_ans_bf16",
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
        "diffsynth_kernels_dfloat11_bf16",
        "diffsynth_kernels_tile_ans_bf16",
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
        "diffsynth_kernels_dfloat11_bf16",
        "diffsynth_kernels_tile_ans_bf16",
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
                method="diffsynth_kernels_dfloat11_bf16",
                target_modules=["0"],
                backend_config_kwargs={"execution_backend": "eager"},
            ),
            QuantizeConfig(
                method="diffsynth_kernels_tile_ans_bf16",
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
        method="diffsynth_kernels_tile_ans_bf16",
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
        method="diffsynth_kernels_dfloat11_bf16",
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
        "diffsynth_kernels_tile_ans_fp16",
        "diffsynth_kernels_bf16_mantissa",
        "diffsynth_kernels_bf16_adaptive",
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


LOSSY_METHODS = (
    (
        "diffsynth_kernels_bf16_mantissa",
        {"execution_backend": "eager", "mantissa_bits": 4},
    ),
    (
        "diffsynth_kernels_bf16_adaptive",
        {
            "execution_backend": "eager",
            "target_bpp": 8.0,
            "codebook_size": 32,
            "sample_rows": 16,
            "iterations": 1,
            "full_refine_steps": 1,
            "fixed_refine_steps": 1,
        },
    ),
)


def _lossy_config(method, kwargs, **config_kwargs):
    return QuantizeConfig(
        method=method,
        backend_config_kwargs=kwargs,
        **config_kwargs,
    )


def _lossy_model():
    return torch.nn.Sequential(torch.nn.Linear(512, 64, dtype=torch.float32))


def test_lossy_methods_are_registered_with_final_defaults():
    assert QUANT_METHODS["diffsynth_kernels_bf16_mantissa"].backend == "diffsynth_kernels"
    assert QUANT_METHODS["diffsynth_kernels_bf16_adaptive"].backend == "diffsynth_kernels"
    assert QUANT_METHODS["diffsynth_kernels_bf16_adaptive_online"].backend == "diffsynth_kernels"
    assert QUANT_METHODS["diffsynth_kernels_bf16_adaptive_offline"].backend == "diffsynth_kernels"

    mantissa = DiffSynthKernelsBF16MantissaConfig()
    assert mantissa.compress_method is dk.CompressionMethod.BF16_MANTISSA
    assert mantissa.dtype is torch.bfloat16
    assert mantissa.mantissa_bits == 4
    assert (mantissa.tile_elements, mantissa.probability_bits, mantissa.raw_lane_threshold) == (
        0,
        0,
        7.9,
    )

    adaptive = DiffSynthKernelsBF16AdaptiveConfig()
    assert adaptive.compress_method is dk.CompressionMethod.BF16_ADAPTIVE
    assert adaptive.dtype is torch.bfloat16
    assert adaptive.target_bpp == 8.0
    assert adaptive.bpp_tolerance == 0.5
    assert adaptive.codebook_size is None
    assert adaptive.side_dtype is None
    assert adaptive.sample_rows == 512
    assert adaptive.iterations == 10
    assert adaptive.full_refine_steps == 1
    assert adaptive.fixed_refine_steps == 1
    assert adaptive.use_cached_lambda is True
    assert (adaptive.tile_elements, adaptive.probability_bits, adaptive.raw_lane_threshold) == (
        8192,
        0,
        7.9,
    )

    online = DiffSynthKernelsBF16AdaptiveOnlineConfig(target_bpp=3.0)
    assert online.target_bpp == 3.0
    assert online.compression_options()["target_bpp"] == pytest.approx(2.7)
    assert online.compression_options()["bpp_tolerance"] == 0.5
    assert online.compression_options()["iterations"] == 10
    assert online.compression_options()["use_cached_lambda"] is True

    offline = DiffSynthKernelsBF16AdaptiveOfflineConfig(target_bpp=3.0)
    assert offline.compression_options()["target_bpp"] == 3.0
    assert offline.compression_options()["bpp_tolerance"] == 0.0
    assert offline.compression_options()["iterations"] == 50
    assert offline.compression_options()["full_refine_steps"] == 2
    assert offline.compression_options()["use_cached_lambda"] is False


@pytest.mark.parametrize(
    ("config_class", "kwargs", "message"),
    (
        (DiffSynthKernelsBF16MantissaConfig, {"mantissa_bits": 7}, "mantissa_bits"),
        (DiffSynthKernelsBF16AdaptiveConfig, {"target_bpp": 0.9}, "target_bpp"),
        (DiffSynthKernelsBF16AdaptiveConfig, {"target_bpp": 11.1}, "target_bpp"),
        (DiffSynthKernelsBF16AdaptiveConfig, {"side_dtype": torch.float16}, "side_dtype"),
    ),
)
def test_lossy_config_rejects_invalid_values(config_class, kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        config_class.from_kwargs(kwargs)


@pytest.mark.parametrize("field_name", ("quality", "group_size", "dtype", "compress_method"))
def test_adaptive_config_rejects_removed_unknown_and_pinned_fields(field_name):
    with pytest.raises(ValueError, match="not accepted"):
        DiffSynthKernelsBF16AdaptiveConfig.from_kwargs({field_name: "invalid"})


@pytest.mark.parametrize(("method", "kwargs"), LOSSY_METHODS)
def test_lossy_reconstruction_uses_shared_linear(method, kwargs):
    torch.manual_seed(10)
    model = _lossy_model()
    original = model[0].weight.detach().to(torch.bfloat16)
    config = _lossy_config(method, kwargs)

    config.quantize_model(model)

    assert type(model[0]) is DiffSynthKernelsLinear
    compressed = model[0]._compressed()
    assert compressed.header["version"] == dk.ENVELOPE_VERSION == 2
    assert compressed.compression_kind is dk.CompressionKind.LOSSY
    assert model[0].compression_kind is dk.CompressionKind.LOSSY
    restored = dk.decompress(compressed, execution_backend="eager")
    assert restored.dtype is torch.bfloat16
    assert restored.shape == original.shape
    assert not torch.equal(_bits(restored), _bits(original))
    relative_error = torch.linalg.vector_norm(restored.float() - original.float()) / torch.linalg.vector_norm(original.float())
    assert relative_error < 0.1


@pytest.mark.parametrize(("method", "kwargs"), LOSSY_METHODS)
def test_lossy_v2_state_dict_and_safetensors_roundtrip(tmp_path, method, kwargs):
    torch.manual_seed(11)
    source = _lossy_model()
    config = _lossy_config(method, kwargs)
    config.quantize_model(source)
    state = source.state_dict()
    assert state["0._diffsynth_kernels.header"].dtype is torch.uint8

    restored = _lossy_model()
    config.prepare_for_prequantized_load(restored, compute_dtype=torch.float32)
    restored.load_state_dict(state, assign=True)

    expected = source[0]._compressed()
    actual = restored[0]._compressed()
    assert actual.header == expected.header
    assert actual.compression_kind is dk.CompressionKind.LOSSY
    for name in expected.buffers:
        assert torch.equal(actual.buffers[name], expected.buffers[name])

    tensors, metadata = config.flatten_state_dict(state)
    path = tmp_path / f"{method}.safetensors"
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
    mantissa = dk.compress(
        weight,
        compress_method=dk.CompressionMethod.BF16_MANTISSA,
        execution_backend="eager",
    )
    dfloat = dk.compress(
        weight,
        compress_method=dk.CompressionMethod.DFLOAT11,
        execution_backend="eager",
    )
    shell = DiffSynthKernelsLinear(
        16,
        8,
        bias=False,
        compute_dtype=torch.float32,
        compression_dtype=torch.bfloat16,
        compress_method=dk.CompressionMethod.BF16_MANTISSA,
        execution_backend="eager",
    )

    with pytest.raises(ValueError, match="method"):
        shell._set_compressed(dfloat)

    wrong_dtype = copy.deepcopy(mantissa)
    wrong_dtype.dtype = torch.float16
    with pytest.raises(ValueError, match="dtype"):
        shell._set_compressed(wrong_dtype)

    wrong_codec = copy.deepcopy(mantissa)
    wrong_codec.header["codec_version"] += 1
    with pytest.raises(ValueError, match="codec version"):
        shell._set_compressed(wrong_codec)

    wrong_buffers = copy.deepcopy(mantissa)
    wrong_buffers.buffers.pop(next(iter(wrong_buffers.buffers)))
    with pytest.raises(ValueError, match="buffers"):
        shell._set_compressed(wrong_buffers)


def test_lossy_requires_v2_header_and_rejects_headerless_buffers():
    model = torch.nn.Sequential(torch.nn.Linear(16, 8))
    config = _lossy_config(
        "diffsynth_kernels_bf16_mantissa",
        {"execution_backend": "eager"},
    )
    config.quantize_model(model)
    compressed = model[0]._compressed()

    v1 = copy.deepcopy(compressed)
    v1.header.update(
        version=1,
        format_id="bf16",
        method_id=dk.CompressionMethod.BF16_MANTISSA,
    )
    with pytest.raises(ValueError, match="standard v2 header"):
        model[0]._set_compressed(v1)

    legacy_state = {
        **{f"0.{name}": value for name, value in compressed.buffers.items()},
        "0.bias": model[0].bias,
    }
    shell = torch.nn.Sequential(torch.nn.Linear(16, 8))
    config.prepare_for_prequantized_load(shell, compute_dtype=torch.float32)
    with pytest.raises(RuntimeError, match="headerless.*only supported by lossless"):
        shell.load_state_dict(legacy_state, assign=True)


def test_mixed_lossless_and_lossy_methods_roundtrip():
    config = MixedQuantizeConfig(
        configs=[
            QuantizeConfig(
                method="diffsynth_kernels_dfloat11_bf16",
                target_modules=["lossless"],
                backend_config_kwargs={"execution_backend": "eager"},
            ),
            QuantizeConfig(
                method="diffsynth_kernels_bf16_adaptive",
                target_modules=["lossy"],
                backend_config_kwargs=LOSSY_METHODS[1][1],
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

    assert restored["lossless"].compression_kind is dk.CompressionKind.LOSSLESS
    assert restored["lossy"].compression_kind is dk.CompressionKind.LOSSY
    lossless_input = torch.randn(2, 64)
    assert torch.equal(
        restored["lossless"](lossless_input), source["lossless"](lossless_input)
    )
    lossy_input = torch.randn(2, 512)
    assert torch.equal(restored["lossy"](lossy_input), source["lossy"](lossy_input))


def test_lossy_dtype_preserving_apply_and_deepcopy():
    model = _lossy_model()
    config = _lossy_config(
        "diffsynth_kernels_bf16_mantissa",
        {"execution_backend": "eager"},
    )
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
    config = _lossy_config(
        "diffsynth_kernels_bf16_mantissa",
        {"execution_backend": "eager"},
        mode="dequant_once",
    )
    config.quantize_model(model)
    expected = dk.decompress(model[0]._compressed(), execution_backend="eager")

    config.dequantize_model(model, compute_dtype=torch.bfloat16)

    assert type(model[0]) is torch.nn.Linear
    assert torch.equal(_bits(model[0].weight), _bits(expected))


@pytest.mark.parametrize(("method", "kwargs"), LOSSY_METHODS)
def test_lossy_input_and_lora_branch_gradients(method, kwargs):
    torch.manual_seed(13)
    model = _lossy_model()
    config = _lossy_config(method, kwargs)
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


def test_mantissa_has_no_hidden_minimum_layer_size():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    config = _lossy_config(
        "diffsynth_kernels_bf16_mantissa",
        {"execution_backend": "eager"},
    )
    config.quantize_model(model)
    assert isinstance(model[0], DiffSynthKernelsLinear)
