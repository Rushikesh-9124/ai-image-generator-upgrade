"""
model/pipeline_loader.py
========================
Rebuilt for v3:

txt2img:
  • RealVisXL V5.0  — photorealistic fine-tune of SDXL, best-in-class
    for real human faces, skin texture, and prompt adherence
  • SDXL Refiner   — final detail pass (pores, hair, fabric)
  • DPM++ 2M Karras scheduler — sharpest at 30-40 steps

img2img (REBUILT — was broken):
  • ControlNet Canny  — locks edge structure of source image
  • ControlNet Depth  — locks spatial depth/pose of source image
  • Both run together via StableDiffusionXLControlNetImg2ImgPipeline
  • This is why the old img2img produced irrelevant images: plain
    img2img at strength > 0.5 effectively ignores the source. ControlNet
    hard-constrains the spatial layout regardless of strength.

Routes / response format: UNCHANGED.
"""

import gc
import logging
from typing import Optional

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMScheduler,
    AutoencoderKL,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Model identifiers
# ─────────────────────────────────────────────────────────────────────────────

# RealVisXL V5.0 — photorealistic SDXL fine-tune
# Dramatically better than base SDXL for real human faces, accurate anatomy,
# and prompt adherence. Lightning variant available for 4-step inference.
REALVIS_ID      = "SG161222/RealVisXL_V5.0"

# SDXL Refiner — unchanged, adds fine detail in the last denoising steps
SDXL_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"

# fp16-fixed VAE — prevents colour saturation drift
SDXL_VAE_ID     = "madebyollin/sdxl-vae-fp16-fix"

# ControlNet models for structure-preserving img2img
# Canny: locks hard edges (hair, clothing outline, face structure)
# Depth: locks spatial depth map (pose, foreground/background separation)
CONTROLNET_CANNY_ID = "diffusers/controlnet-canny-sdxl-1.0"
CONTROLNET_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0-small"

# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler registry
# ─────────────────────────────────────────────────────────────────────────────

SCHEDULER_MAP = {
    "dpm++":    DPMSolverMultistepScheduler,
    "euler_a":  EulerAncestralDiscreteScheduler,
    "kdpm2_a":  KDPM2AncestralDiscreteScheduler,   # new: great for faces
    "ddim":     DDIMScheduler,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Singleton storage
# ─────────────────────────────────────────────────────────────────────────────

_txt2img_pipe:   Optional[StableDiffusionXLPipeline]                    = None
_refiner_pipe:   Optional[StableDiffusionXLImg2ImgPipeline]             = None
_controlnet_pipe: Optional[StableDiffusionXLControlNetImg2ImgPipeline]  = None

# ─────────────────────────────────────────────────────────────────────────────
#  Device / dtype helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_dtype() -> torch.dtype:
    return torch.float16 if get_device() == "cuda" else torch.float32

def pipelines_loaded() -> bool:
    return _txt2img_pipe is not None

# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_vae(dtype: torch.dtype) -> AutoencoderKL:
    logger.info(f"  Loading fixed VAE ({SDXL_VAE_ID}) ...")
    vae = AutoencoderKL.from_pretrained(SDXL_VAE_ID, torch_dtype=dtype)
    logger.info("  ✓ VAE loaded")
    return vae


def _configure_scheduler(pipe, name: str) -> None:
    cls = SCHEDULER_MAP.get(name, DPMSolverMultistepScheduler)
    kwargs = dict(use_karras_sigmas=True)
    if cls is DPMSolverMultistepScheduler:
        kwargs["algorithm_type"] = "dpmsolver++"
    pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
    logger.info(f"  ✓ Scheduler → {cls.__name__} (karras)")


def _optimise(pipe, device: str) -> None:
    """Apply all memory/speed optimisations in the correct order."""
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("    ✓ xformers enabled")
        except Exception:
            # PyTorch ≥ 2.0 SDPA is a solid fallback
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                pipe.unet.set_attn_processor(AttnProcessor2_0())
                logger.info("    ✓ PyTorch SDPA attention enabled")
            except Exception:
                logger.info("    ✗ Using default attention")


# ─────────────────────────────────────────────────────────────────────────────
#  Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pipelines(
    scheduler_name: str  = "dpm++",
    load_refiner:   bool = True,
    enable_safety:  bool = False,
) -> None:
    """
    Load all pipelines. Idempotent — safe to call multiple times.

    Load order (VRAM budget):
      1. Fixed VAE                    ~0.3 GB
      2. RealVisXL txt2img            ~6.5 GB  (fp16)
      3. ControlNet × 2               ~2.5 GB  (canny + depth, fp16)
      4. ControlNet img2img pipeline  ~0.1 GB  (shares UNet/VAE/encoders)
      5. SDXL Refiner                 ~5.5 GB  (fp16) — optional

    Minimum VRAM: 10 GB (without refiner), 14 GB (with refiner)
    With xformers: subtract ~2 GB each
    """
    global _txt2img_pipe, _refiner_pipe, _controlnet_pipe

    if _txt2img_pipe is not None:
        logger.info("Pipelines already loaded — skipping.")
        return

    device = get_device()
    dtype  = get_dtype()
    logger.info(f"Loading v3 pipelines | device={device} | dtype={dtype}")

    # ── 1. VAE ───────────────────────────────────────────────────────────────
    vae = _load_vae(dtype)

    # ── 2. RealVisXL txt2img ─────────────────────────────────────────────────
    logger.info(f"  Loading RealVisXL ({REALVIS_ID}) ...")
    safety_kwargs = {} if enable_safety else {
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    _txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
        REALVIS_ID,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
        **safety_kwargs,
    ).to(device)
    _configure_scheduler(_txt2img_pipe, scheduler_name)
    _optimise(_txt2img_pipe, device)
    logger.info("  ✓ RealVisXL txt2img ready")

    # ── 3. ControlNet models ─────────────────────────────────────────────────
    logger.info("  Loading ControlNet Canny + Depth ...")
    controlnet_canny = ControlNetModel.from_pretrained(
        CONTROLNET_CANNY_ID, torch_dtype=dtype, use_safetensors=True,
    ).to(device)
    controlnet_depth = ControlNetModel.from_pretrained(
        CONTROLNET_DEPTH_ID, torch_dtype=dtype, use_safetensors=True,
    ).to(device)
    logger.info("  ✓ ControlNet models loaded")

    # ── 4. ControlNet img2img pipeline (shares weights from txt2img) ─────────
    # This is the critical fix for img2img. StableDiffusionXLControlNetImg2ImgPipeline
    # takes both the init image AND the controlnet conditioning simultaneously,
    # so the model cannot drift away from the source structure even at high strength.
    logger.info("  Building ControlNet img2img pipeline (shared UNet/VAE/encoders) ...")
    _controlnet_pipe = StableDiffusionXLControlNetImg2ImgPipeline(
        vae=_txt2img_pipe.vae,
        text_encoder=_txt2img_pipe.text_encoder,
        text_encoder_2=_txt2img_pipe.text_encoder_2,
        tokenizer=_txt2img_pipe.tokenizer,
        tokenizer_2=_txt2img_pipe.tokenizer_2,
        unet=_txt2img_pipe.unet,
        controlnet=[controlnet_canny, controlnet_depth],
        scheduler=_txt2img_pipe.scheduler,
        image_processor=_txt2img_pipe.image_processor,
        requires_aesthetics_score=False,
        force_zeros_for_empty_prompt=True,
    ).to(device)
    _optimise(_controlnet_pipe, device)
    logger.info("  ✓ ControlNet img2img pipeline ready")

    # ── 5. Optional SDXL Refiner ─────────────────────────────────────────────
    if load_refiner:
        logger.info(f"  Loading SDXL Refiner ({SDXL_REFINER_ID}) ...")
        _refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            SDXL_REFINER_ID,
            vae=vae,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
            add_watermarker=False,
        ).to(device)
        _configure_scheduler(_refiner_pipe, scheduler_name)
        _optimise(_refiner_pipe, device)
        logger.info("  ✓ Refiner loaded")
    else:
        logger.info("  ✗ Refiner skipped (LOAD_REFINER=false)")

    logger.info("All v3 pipelines ready ✓")


def unload_pipelines() -> None:
    global _txt2img_pipe, _refiner_pipe, _controlnet_pipe
    for name, obj in [
        ("txt2img",    _txt2img_pipe),
        ("controlnet", _controlnet_pipe),
        ("refiner",    _refiner_pipe),
    ]:
        if obj is not None:
            del obj
            logger.info(f"  Released {name} pipeline")
    _txt2img_pipe = _refiner_pipe = _controlnet_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("All pipelines unloaded.")


# ─────────────────────────────────────────────────────────────────────────────
#  Public accessors
# ─────────────────────────────────────────────────────────────────────────────

def get_txt2img_pipe()    -> Optional[StableDiffusionXLPipeline]:
    return _txt2img_pipe

def get_controlnet_pipe() -> Optional[StableDiffusionXLControlNetImg2ImgPipeline]:
    return _controlnet_pipe

def get_refiner_pipe()    -> Optional[StableDiffusionXLImg2ImgPipeline]:
    return _refiner_pipe

# Keep old name for backward compat with any direct callers
def get_base_pipe()       -> Optional[StableDiffusionXLPipeline]:
    return _txt2img_pipe

def get_img2img_pipe()    -> Optional[StableDiffusionXLControlNetImg2ImgPipeline]:
    return _controlnet_pipe


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA management
# ─────────────────────────────────────────────────────────────────────────────

def load_lora_weights(
    lora_path:    str,
    lora_scale:   float = 0.9,
    adapter_name: str   = "default",
) -> None:
    if _txt2img_pipe is None:
        raise RuntimeError("Pipelines not loaded. Call load_pipelines() first.")
    for pipe_name, pipe in [("txt2img", _txt2img_pipe), ("controlnet", _controlnet_pipe)]:
        if pipe is not None:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
            logger.info(f"  ✓ LoRA applied to {pipe_name}")


def unload_lora_weights(adapter_name: str = "default") -> None:
    for pipe in [_txt2img_pipe, _controlnet_pipe]:
        if pipe is not None:
            pipe.delete_adapters([adapter_name])
    logger.info(f"LoRA '{adapter_name}' unloaded.")


def swap_scheduler(scheduler_name: str) -> None:
    for pipe in [_txt2img_pipe, _controlnet_pipe, _refiner_pipe]:
        if pipe is not None:
            _configure_scheduler(pipe, scheduler_name)
    logger.info(f"Scheduler swapped → {scheduler_name}")