import lightning as pl
from peft import LoraConfig, inject_adapter_in_model
import torch, os
from ..data.qwen_visual_to_image import QwenVisual2Image
from modelscope.hub.api import HubApi
from ..models.utils import load_state_dict
from diffsynth import ModelManager, FluxImagePipeline
from transformers import get_cosine_schedule_with_warmup



class FluxForQwen(pl.LightningModule):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="kaiming", pretrained_lora_path=None,
        state_dict_converter=None, quantize = None, in_channel=3584, out_channel=4096, expand_ratio=1, lr_warmup_steps=500,
    ):
        super().__init__()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.state_dict_converter = state_dict_converter
        self.lora_alpha = None
        self.lr_warmup_steps = lr_warmup_steps
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        if quantize is None:
            model_manager.load_models(pretrained_weights)
        else:
            model_manager.load_models(pretrained_weights[1:])
            model_manager.load_model(pretrained_weights[0], torch_dtype=quantize)
        if preset_lora_path is not None:
            model_manager.load_lora(preset_lora_path)

        self.pipe = FluxImagePipeline.from_model_manager(model_manager)

        if quantize is not None:
            self.pipe.dit.quantize()

        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.freeze_parameters()
        self.add_lora_to_model(self.pipe.denoising_model(), lora_rank=lora_rank, lora_alpha=lora_alpha, lora_target_modules=lora_target_modules, init_lora_weights=init_lora_weights)
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(in_channel, out_channel * expand_ratio),
            torch.nn.LayerNorm(out_channel * expand_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channel * expand_ratio, out_channel),
            torch.nn.LayerNorm(out_channel))



    def load_models(self):
        # This function is implemented in other modules
        self.pipe = None


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()


    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="gaussian", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")


    def training_step(self, batch, batch_idx):
        # Data
        embed, image = batch["embed"], batch["image"]

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt("", positive=True, clip_only=True)
        visual_emb = self.adapter(embed)
        prompt_emb['prompt_emb'] = visual_emb
        prompt_emb['text_ids'] = torch.zeros(visual_emb.shape[0], visual_emb.shape[1], 3).to(device=self.device, dtype=visual_emb.dtype)

        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents, guidance=3.5)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # Record log
        self.log("train_learning_rate", lr, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss


    def configure_optimizers(self):
        import itertools
        trainable_modules = itertools.chain(
            self.adapter.parameters(),
            filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        )
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
    
        # 获取全局总训练步数（自动适配多卡和梯度累积）
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.lr_warmup_steps  # 从命令行参数读取
        print('total_steps:', total_steps)

        # 使用 Hugging Face 的调度器
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # # 生成学习率数据
        # lrs = []
        # for step in range(total_steps):
        #     scheduler.step()
        #     lrs.append(optimizer.param_groups[0]["lr"])
        # import matplotlib.pyplot as plt

        # # 绘制曲线
        # plt.figure(figsize=(10, 5))
        # plt.plot(lrs)
        # plt.title("Learning Rate Schedule")
        # plt.xlabel("Training Steps")
        # plt.ylabel("Learning Rate")
        # plt.savefig('lr_schedule.png')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 按步更新学习率
                "frequency": 1,
            },
        }


    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        
        save_state_dict = self.adapter.state_dict()
        
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        if self.state_dict_converter is not None:
            lora_state_dict = self.state_dict_converter(lora_state_dict, alpha=self.lora_alpha)
        save_state_dict.update(lora_state_dict)

        checkpoint.update(save_state_dict)



def add_general_parsers(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="Whether to randomly flip images horizontally",
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
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
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
        "--lr_warmup_steps",
        type=int,
        default=1,
        help="lr_warmup_steps.",
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
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_wandb",
        default=True,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    return parser


def launch_training_task(model, args):
    print(args)
    # dataset and data loader
    dataset = QwenVisual2Image(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        random_flip=args.random_flip
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )
    # train
    if args.use_wandb:        
        from pytorch_lightning.loggers import WandbLogger

        wandb_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        wandb_config.update(vars(args))
        import os
        os.makedirs(os.path.join(args.output_path, "wandb"), exist_ok=True)
        wandb_logger = WandbLogger(
            project="diffsynth_studio",
            name="diffsynth_studio",
            config=wandb_config,
            save_dir=os.path.join(args.output_path, "wandb")
        )
        
        logger = wandb_logger
        print("Using WandbLogger")
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
        log_every_n_steps=5,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

    # Upload models
    if args.modelscope_model_id is not None and args.modelscope_access_token is not None:
        print(f"Uploading models to modelscope. model_id: {args.modelscope_model_id} local_path: {trainer.log_dir}")
        with open(os.path.join(trainer.log_dir, "configuration.json"), "w", encoding="utf-8") as f:
            f.write('{"framework":"Pytorch","task":"text-to-image-synthesis"}\n')
        api = HubApi()
        api.login(args.modelscope_access_token)
        api.push_model(model_id=args.modelscope_model_id, model_dir=trainer.log_dir)
