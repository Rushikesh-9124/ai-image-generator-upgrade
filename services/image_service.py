"""
services/image_service.py
==========================
Orchestration layer — unchanged except:
  • Passes is_img2img=True to enhance_prompt for img2img calls so that
    identity-preservation anchor tokens are injected automatically.
  • Default guidance_scale bumped to 8.0 to match RealVisXL optimum.
  • Everything else (function signatures, GenerationResult, return types)
    is byte-for-byte identical to v2.
"""

import io
import base64
import logging
from typing import Optional
from dataclasses import dataclass

import torch
from PIL import Image

from model.inference import generate_txt2img, generate_img2img
from services.prompt_service import enhance_prompt, EnhancedPrompt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    image_base64:    str
    seed:            int
    generation_time: float
    width:           int
    height:          int
    enhanced_prompt: str
    negative_prompt: str
    mode:            str   # "txt2img" | "img2img"


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_seed(seed: Optional[int]) -> int:
    if seed is not None:
        return int(seed)
    return int(torch.randint(0, 2 ** 32, (1,)).item())


def _decode_image(b64_string: Optional[str]) -> Optional[Image.Image]:
    if not b64_string or not b64_string.strip():
        return None
    try:
        data = b64_string.strip()
        if "," in data:
            data = data.split(",", 1)[1]
        missing = len(data) % 4
        if missing:
            data += "=" * (4 - missing)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as exc:
        logger.warning(f"Failed to decode init image: {exc}")
        return None


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True, compress_level=6)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Public service methods — SIGNATURES UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

async def run_txt2img(
    prompt:               str,
    negative_prompt:      str            = "",
    style:                Optional[str]  = None,
    steps:                int            = 35,
    guidance_scale:       float          = 8.0,   # bumped from 7.5
    width:                int            = 1024,
    height:               int            = 1024,
    seed:                 Optional[int]  = None,
    use_refiner:          bool           = True,
    refiner_steps:        int            = 20,
    auto_enhance_prompt:  bool           = True,
) -> GenerationResult:
    seed = _resolve_seed(seed)

    if auto_enhance_prompt:
        enhanced: EnhancedPrompt = enhance_prompt(
            raw_prompt=prompt,
            style=style,
            raw_negative_prompt=negative_prompt,
            is_img2img=False,
        )
        final_pos = enhanced.positive
        final_neg = enhanced.negative
        if steps == 35 and enhanced.recommended_steps != 35:
            steps = enhanced.recommended_steps
        if guidance_scale == 8.0 and enhanced.recommended_cfg != 8.0:
            guidance_scale = enhanced.recommended_cfg
    else:
        final_pos = prompt
        final_neg = negative_prompt

    logger.info(
        f"[txt2img] seed={seed} steps={steps} cfg={guidance_scale} "
        f"{width}×{height} refiner={use_refiner}"
    )

    image, elapsed = await generate_txt2img(
        prompt=final_pos,
        negative_prompt=final_neg,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        use_refiner=use_refiner,
        refiner_steps=refiner_steps,
    )

    logger.info(f"[txt2img] done in {elapsed}s")
    return GenerationResult(
        image_base64=_encode_image(image),
        seed=seed,
        generation_time=elapsed,
        width=image.width,
        height=image.height,
        enhanced_prompt=final_pos,
        negative_prompt=final_neg,
        mode="txt2img",
    )


async def run_img2img(
    init_image_b64:       str,
    prompt:               str,
    negative_prompt:      str            = "",
    style:                Optional[str]  = None,
    steps:                int            = 35,
    guidance_scale:       float          = 8.0,   # bumped from 7.5
    width:                int            = 1024,
    height:               int            = 1024,
    strength:             float          = 0.55,
    seed:                 Optional[int]  = None,
    use_refiner:          bool           = True,
    refiner_steps:        int            = 20,
    auto_enhance_prompt:  bool           = True,
) -> GenerationResult:
    init_image = _decode_image(init_image_b64)
    if init_image is None:
        raise ValueError(
            "init_image_base64 is missing or invalid. "
            "Provide a valid base64-encoded PNG/JPEG."
        )

    seed = _resolve_seed(seed)

    if auto_enhance_prompt:
        enhanced = enhance_prompt(
            raw_prompt=prompt,
            style=style,
            raw_negative_prompt=negative_prompt,
            is_img2img=True,   # injects identity-preservation tokens
        )
        final_pos = enhanced.positive
        final_neg = enhanced.negative
        if steps == 35 and enhanced.recommended_steps != 35:
            steps = enhanced.recommended_steps
        if guidance_scale == 8.0 and enhanced.recommended_cfg != 8.0:
            guidance_scale = enhanced.recommended_cfg
    else:
        final_pos = prompt
        final_neg = negative_prompt

    logger.info(
        f"[img2img] seed={seed} strength={strength} steps={steps} "
        f"cfg={guidance_scale} {width}×{height} refiner={use_refiner}"
    )

    image, elapsed = await generate_img2img(
        init_image=init_image,
        prompt=final_pos,
        negative_prompt=final_neg,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        strength=strength,
        seed=seed,
        use_refiner=use_refiner,
        refiner_steps=refiner_steps,
    )

    logger.info(f"[img2img] done in {elapsed}s")
    return GenerationResult(
        image_base64=_encode_image(image),
        seed=seed,
        generation_time=elapsed,
        width=image.width,
        height=image.height,
        enhanced_prompt=final_pos,
        negative_prompt=final_neg,
        mode="img2img",
    )