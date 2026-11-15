"""Wan 2.2 TI2V 5B: text to video, or image to video with a start image.

Graph from ComfyUI's template video_wan2_2_5B_ti2v.json: UMT5 XXL fp8 text
encoder (CLIPLoader type wan), ModelSamplingSD3 shift 8, KSampler 20 steps
cfg 5 uni_pc/simple, Wan22ImageToVideoLatent, CreateVideo at 24 fps, SaveVideo.
"""

from __future__ import annotations

from . import Workflow

# Wan's own default negative prompt (from the template), in Chinese as trained.
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def build(p: dict) -> dict:
    frames = int(p["frames"])
    if (frames - 1) % 4:
        frames = ((frames - 1) // 4) * 4 + 1  # Wan needs 4n+1 frames
    latent_inputs = {"vae": ["3", 0], "width": p["width"], "height": p["height"], "length": frames, "batch_size": 1}
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": p.get("negative_prompt") or DEFAULT_NEGATIVE, "clip": ["2", 0]}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": p["shift"]}},
        "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": latent_inputs},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0],
            "seed": p["seed"], "steps": p["steps"], "cfg": p["cfg"], "sampler_name": "uni_pc",
            "scheduler": "simple", "denoise": 1}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": p["fps"]}},
        "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0], "filename_prefix": p["prefix"],
                                                     "format": "mp4", "format.codec": "h264"}},
    }
    start = p.get("images", {}).get("start_image")
    if start:
        g["12"] = {"class_type": "LoadImage", "inputs": {"image": start}}
        latent_inputs["start_image"] = ["12", 0]
    return g


WORKFLOW = Workflow(
    name="wan2.2-ti2v-5b",
    modality="video",
    description="Wan 2.2 TI2V 5B (Apache 2.0): text to video, or image to video with input_image; 24 fps, native 1280x704",
    models=[
        "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae/wan2.2_vae.safetensors",
    ],
    defaults={"width": 1280, "height": 704, "frames": 49, "fps": 24, "steps": 20, "cfg": 5.0, "shift": 8.0},
    limits={"width": (256, 1280), "height": (256, 1280), "frames": (5, 121), "fps": (8, 30), "steps": (4, 50)},
    build=build,
    image_slots=("start_image",),
    is_default=True,
)
