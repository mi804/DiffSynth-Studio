# 训练 FLUX LoRA

以下文件将会被用于构建 FLUX 模型。 你可以从[huggingface](https://huggingface.co/black-forest-labs/FLUX.1-dev)或[modelscope](https://www.modelscope.cn/models/ai-modelscope/flux.1-dev)下载，也可以使用以下代码下载这些文件:

```python
from diffsynth import download_models

download_models(["FLUX.1-dev"])
```

```
models/FLUX/
└── FLUX.1-dev
    ├── ae.safetensors
    ├── flux1-dev.safetensors
    ├── text_encoder
    │   └── model.safetensors
    └── text_encoder_2
        ├── config.json
        ├── model-00001-of-00002.safetensors
        ├── model-00002-of-00002.safetensors
        └── model.safetensors.index.json
```

使用以下命令启动训练任务：

```
CUDA_VISIBLE_DEVICES="0" python examples/train/flux/train_flux_lora.py \
  --pretrained_text_encoder_path models/FLUX/FLUX.1-dev/text_encoder/model.safetensors \
  --pretrained_text_encoder_2_path models/FLUX/FLUX.1-dev/text_encoder_2 \
  --pretrained_dit_path models/FLUX/FLUX.1-dev/flux1-dev.safetensors \
  --pretrained_vae_path models/FLUX/FLUX.1-dev/ae.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "bf16" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

有关参数的更多信息，请使用 `python examples/train/flux/train_flux_lora.py -h` 查看详细信息。

训练完成后，使用 `model_manager.load_lora` 加载 LoRA 以进行推理。

```python
from diffsynth import ModelManager, FluxImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
                                 "models/FLUX/FLUX.1-dev/text_encoder_2",
                                 "models/FLUX/FLUX.1-dev/ae.safetensors",
                                 "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
                             ])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = SDXLImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_with_lora.jpg")
```