tier_a = [
    "img_mod.1", "txt_mod.1",
    "timestep_embedder.linear_1", "timestep_embedder.linear_2",
    "img_in", "txt_in", "proj_out", "norm_out.linear",
]
INT8_PATTERNS = ["img_mod.1", "txt_mod.1"]
# qwen-image-2512-dit
mixed_final = MixedQuantizeConfig(configs=[
    QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=tier_a + INT8_PATTERNS),
    QuantizeConfig(method="torchao_int8_w8a16", target_modules=INT8_PATTERNS),
])

MODEL_CONFIGS.append({
    "model_hash": dit_hash,
    "model_name": "qwen_image_dit",
    "model_class": "diffsynth.models.qwen_image_dit.QwenImageDiT",
    "quant_config": {
        "configs": [
            {"method": "bitsandbytes_nf4", "exclude_modules": tier_a + INT8_PATTERNS},
            {"method": "torchao_int8_w8a16", "target_modules": INT8_PATTERNS},
        ],
        "load_prequantized": True,
    },
})
# qwen-image-2512-text-encoder
MODEL_CONFIGS.append({
    "model_hash": text_encoder_hash,
    "model_name": "qwen_image_text_encoder",
    "model_class": "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder",
    "quant_config": {"method": "bitsandbytes_nf4", "load_prequantized": True},
})