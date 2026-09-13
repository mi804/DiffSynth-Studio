# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "z_image/Z-Image-Turbo/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/z_image/Z-Image-Turbo \
  --dataset_metadata_path data/diffsynth_example_dataset/z_image/Z-Image-Turbo/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_paths '[["models/Tongyi-MAI/Z-Image-Turbo/transformer/diffusion_pytorch_model-00001-of-00003.safetensors", "models/Tongyi-MAI/Z-Image-Turbo/transformer/diffusion_pytorch_model-00002-of-00003.safetensors", "models/Tongyi-MAI/Z-Image-Turbo/transformer/diffusion_pytorch_model-00003-of-00003.safetensors"], ["models/Tongyi-MAI/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors", "models/Tongyi-MAI/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors", "models/Tongyi-MAI/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors"], "models/Tongyi-MAI/Z-Image-Turbo/vae/diffusion_pytorch_model.safetensors"]' \
  --tokenizer_path "models/Tongyi-MAI/Z-Image-Turbo/tokenizer/" \
  --quant_options '["models/Tongyi-MAI/Z-Image-Turbo/transformer/diffusion_pytorch_model-00001-of-00003.safetensors", "models/Tongyi-MAI/Z-Image-Turbo/transformer/diffusion_pytorch_model-00002-of-00003.safetensors", "models/Tongyi-MAI/Z-Image-Turbo/transformer/diffusion_pytorch_model-00003-of-00003.safetensors"]:bitsandbytes_nf4;["models/Tongyi-MAI/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors", "models/Tongyi-MAI/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors", "models/Tongyi-MAI/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors"]:bitsandbytes_nf4' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image-Turbo_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8
