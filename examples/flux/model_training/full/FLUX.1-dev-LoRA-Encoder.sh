accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_lora_encoder.csv \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors,DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev:model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.lora_encoder." \
  --output_path "./models/train/FLUX.1-dev-LoRA-Encoder_full" \
  --trainable_models "lora_encoder" \
  --extra_inputs "lora_encoder_inputs" \
  --use_gradient_checkpointing
