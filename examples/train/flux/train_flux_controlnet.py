from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.trainers.controlled_text_to_image import LightningModelForFluxControlNet, add_general_parsers, launch_training_task
from diffsynth.models.lora import FluxLoRAConverter
import torch, os, argparse
import yaml
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
os.environ["TOKENIZERS_PARALLELISM"] = "True"


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
    model = LightningModelForFluxControlNet(
        torch_dtype={"32": torch.float32, "bf16": torch.bfloat16}.get(args.precision, torch.bfloat16),
        pretrained_weights=[args.pretrained_dit_path, args.pretrained_text_encoder_path, args.pretrained_text_encoder_2_path, args.pretrained_vae_path],
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing)
    logger = TensorBoardLogger(save_dir='workdirs/flux-controlnet/tensorboard', name='flux_controlnet')
    launch_training_task(model, args, logger)
