# torchrun --standalone --nproc_per_node=8 examples/ltx2/model_inference_parallel/LTX-2.5-T2AV-OneStage.py
import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig, CFGParallelConfig, FSDPConfig
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    tokenizer_config=ModelConfig(path="./models/DiffSynth-Studio/LTX-2.5-Repackage/tokenizer"),
    model_configs=[
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.5-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors"),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-bf16.safetensors"),
        # For lower VRAM and faster decoding, replace the line above with the conv vae decoder:
        # ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-audio-vae-bf16.safetensors"),
    ],
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses"), cfg=CFGParallelConfig(), fsdp=FSDPConfig()),
)
prompt = "A girl is very happy, she is speaking: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
negative_prompt = pipe.default_negative_prompt["LTX-2.5"]
height, width, num_frames = 512, 768, 121
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
    tile_size_in_frames=80,
)
if dist.get_rank() == 0:
    write_video_audio_ltx2(
        video=video,
        audio=audio,
        output_path="ltx2.5_onestage_t2av.mp4",
        fps=24,
        audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
    )
