import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

from diffsynth.core.quant import MixedQuantizeConfig, QuantizeConfig
from diffsynth.core.quant.backends.diffsynth_kernels import DiffSynthKernelsLinear


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


def test_pinned_dtype_and_compress_method_reject_overrides():
    with pytest.raises(ValueError, match="not accepted"):
        QuantizeConfig(
            method="diffsynth_kernels_tile_ans_fp16",
            backend_config_kwargs={"dtype": torch.float32},
        )
    with pytest.raises(ValueError, match="not accepted"):
        QuantizeConfig(
            method="diffsynth_kernels_tile_ans_fp16",
            backend_config_kwargs={"compress_method": "dfloat11"},
        )
