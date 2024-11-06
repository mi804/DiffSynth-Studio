import lightning as pl
from peft import LoraConfig, inject_adapter_in_model
import torch, os
from ..data.simple_text_image import TextImageDataset
from ..data.controlled_text_image import ControlledTextImageDataset
from modelscope.hub.api import HubApi
from ..models.model_manager import ModelManager
from ..models.flux_controlnet import FluxControlNet
from ..pipelines.flux_image import FluxImagePipeline
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class LightningModelForFluxControlNet(pl.LightningModule):
    def __init__(
        self,
        pretrained_weights=[],
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
        state_dict_converter=None,
        torch_dtype=torch.float16,
    ):
        super().__init__()
        # Set parameters
        # Load models

        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        model_manager.load_models(pretrained_weights)

        self.pipe = FluxImagePipeline.from_model_manager(model_manager)

        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.freeze_parameters()
        self.controlnet = FluxControlNet(num_joint_blocks=5, num_single_blocks=1)
        self.init_controlnet()

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.state_dict_converter = state_dict_converter

    def load_models(self):
        # This function is implemented in other modules
        self.pipe = None

    def init_controlnet(self):
        # controlnet_module_name: dit_module_name
        named_module_mapping = {
            'time_embedder': 'time_embedder',
            'guidance_embedder': 'guidance_embedder',
            'pooled_text_embedder': 'pooled_text_embedder',
            'context_embedder': 'context_embedder',
            'x_embedder': 'x_embedder',
            'controlprompt_embedder': 'context_embedder',
        }
        # direct loading
        for controlnet_module_name, dit_module_name in named_module_mapping.items():
            source_module = getattr(self.pipe.dit, dit_module_name)
            target_module = getattr(self.controlnet, controlnet_module_name)
            target_module.load_state_dict(source_module.state_dict())

        # blocks loading
        for i in range(len(self.controlnet.blocks)):
            source_module = self.pipe.dit.blocks[i]
            target_module = self.controlnet.blocks[i]
            target_module.load_state_dict(source_module.state_dict())

        # single blocks loading
        for i in range(len(self.controlnet.single_blocks)):
            source_module = self.pipe.dit.single_blocks[i]
            target_module = self.controlnet.single_blocks[i]
            target_module.load_state_dict(source_module.state_dict())

        # zero initialization
        self.linear_zero_init(self.controlnet.controlnet_x_embedder)
        self.linear_zero_init(self.controlnet.crossattn.to_out)
        for module in self.controlnet.controlnet_blocks:
            self.linear_zero_init(module)
        for module in self.controlnet.controlnet_single_blocks:
            self.linear_zero_init(module)

    def linear_zero_init(self, linear_layer):
        nn.init.zeros_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()

    def training_step(self, batch, batch_idx):
        # Data
        text, image, mask, control_prompt = batch["text"], batch["image"], batch["mask"], batch["control_prompt"]

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True)
        latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents, guidance=3.5)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Prepare controlnet input
        controlnet_prompt_emb = self.pipe.encode_prompt(control_prompt, positive=True)
        control_image = self.pipe.vae_encoder(mask.to(dtype=self.pipe.torch_dtype, device=self.device))

        # forward
        # controlnet
        controlnet_res_stack, controlnet_single_res_stack = self.controlnet(
            noisy_latents,
            control_image,
            controlnet_prompt_emb,
            timestep,
            **prompt_emb,
            **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )

        # dit
        prompt_emb, pooled_prompt_emb, text_ids = prompt_emb['prompt_emb'], prompt_emb['pooled_prompt_emb'], prompt_emb['text_ids']
        image_ids, guidance = extra_input['image_ids'], extra_input['guidance']
        hidden_states = noisy_latents
        dit = self.pipe.denoising_model()

        conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
        if dit.guidance_embedder is not None:
            guidance = guidance * 1000
            conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)
        prompt_emb = dit.context_embedder(prompt_emb)
        image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))

        height, width = hidden_states.shape[-2:]
        hidden_states = dit.patchify(hidden_states)
        hidden_states = dit.x_embedder(hidden_states)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # Joint Blocks
        for block_id, block in enumerate(dit.blocks):
            if self.training and self.use_gradient_checkpointing:
                hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, prompt_emb, conditioning, image_rotary_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
            # ControlNet
            hidden_states = hidden_states + controlnet_res_stack[block_id]

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        for block_id, block in enumerate(dit.single_blocks):
            if self.training and self.use_gradient_checkpointing:
                hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, prompt_emb, conditioning, image_rotary_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
            # ControlNet
            hidden_states[:, prompt_emb.shape[1]:] = hidden_states[:, prompt_emb.shape[1]:] + controlnet_single_res_stack[block_id]
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        hidden_states = dit.final_norm_out(hidden_states, conditioning)
        hidden_states = dit.final_proj_out(hidden_states)
        hidden_states = dit.unpatchify(hidden_states, height, width)

        # Compute loss
        loss = torch.nn.functional.mse_loss(hidden_states.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        state_dict = self.controlnet.state_dict()
        checkpoint.update(state_dict)

    def on_load_checkpoint(self, checkpoint):
        self.controlnet.load_state_dict(checkpoint)


def add_general_parsers(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        required=False,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16", "16-mixed", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--modelscope_model_id",
        type=str,
        default=None,
        help="Model ID on ModelScope (https://www.modelscope.cn/). The model will be uploaded to ModelScope automatically if you provide a Model ID.",
    )
    parser.add_argument(
        "--modelscope_access_token",
        type=str,
        default=None,
        help="Access key on ModelScope (https://www.modelscope.cn/). Required if you want to upload the model to ModelScope.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training.",
    )
    return parser


def launch_training_task(model, args, logger=None):
    # dataset and data loader
    dataset = ControlledTextImageDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch * args.batch_size,
        height=args.height,
        width=args.width,
        center_crop=True,
        random_flip=False
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )

    # train
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger
    )
    trainer.fit(model=model, train_dataloaders=train_loader,ckpt_path=args.resume_from_checkpoint)

    # Upload models
    if args.modelscope_model_id is not None and args.modelscope_access_token is not None:
        print(f"Uploading models to modelscope. model_id: {args.modelscope_model_id} local_path: {trainer.log_dir}")
        with open(os.path.join(trainer.log_dir, "configuration.json"), "w", encoding="utf-8") as f:
            f.write('{"framework":"Pytorch","task":"text-to-image-synthesis"}\n')
        api = HubApi()
        api.login(args.modelscope_access_token)
        api.push_model(model_id=args.modelscope_model_id, model_dir=trainer.log_dir)
