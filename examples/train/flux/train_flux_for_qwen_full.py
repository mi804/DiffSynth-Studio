from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.trainers.flux_for_qwen_full import FluxForQwen, add_general_parsers, launch_training_task
from diffsynth.models.lora import FluxLoRAConverter
import torch, argparse
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/train/flux/controlnet.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_path",
        type=str,
        default='models/FLUX/FLUX.1-dev/text_encoder/model.safetensors',
        help="Path to pretrained text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder/model.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_2_path",
        type=str,
        default='models/FLUX/FLUX.1-dev/text_encoder_2',
        help="Path to pretrained t5 text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder_2`.",
    )
    parser.add_argument(
        "--pretrained_dit_path",
        type=str,
        default='models/FLUX/FLUX.1-dev/flux1-dev.safetensors',
        help="Path to pretrained dit model. For example, `models/FLUX/FLUX.1-dev/flux1-dev.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default='models/FLUX/FLUX.1-dev/ae.safetensors',
        help="Path to pretrained vae model. For example, `models/FLUX/FLUX.1-dev/ae.safetensors`.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--align_to_opensource_format",
        default=False,
        action="store_true",
        help="Whether to export lora files aligned with other opensource format.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["float8_e4m3fn"],
        help="Whether to use quantization when training the model, and in which format.",
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help="Preset LoRA path.",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="wandb_api_key.",
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update args with YAML configuration
    for key, value in config.items():
        setattr(args, key, value)
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    model = FluxForQwen(
        torch_dtype={"32": torch.float32, "bf16": torch.bfloat16}.get(args.precision, torch.float16),
        pretrained_weights=[args.pretrained_dit_path, args.pretrained_text_encoder_path, args.pretrained_text_encoder_2_path, args.pretrained_vae_path],
        preset_lora_path=args.preset_lora_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else None,
        quantize={"float8_e4m3fn": torch.float8_e4m3fn}.get(args.quantize, None),
        lr_warmup_steps=args.lr_warmup_steps,
    )
    launch_training_task(model, args)
