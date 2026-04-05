"""
api/routes.py
=============
FastAPI router definitions.

Endpoints:
  GET  /health             — Service + GPU health check
  POST /generate           — txt2img or img2img (auto-detected)
  POST /generate/txt2img   — Explicit txt2img
  POST /generate/img2img   — Explicit img2img
  POST /enhance-prompt     — Prompt enrichment only (no image generated)
  GET  /styles             — Available style templates
  GET  /schedulers         — Available scheduler names
  POST /lora/load          — Load a LoRA checkpoint
  DELETE /lora/{name}      — Unload a LoRA by name
  POST /scheduler/swap     — Hot-swap scheduler

All generation endpoints return the same GenerateResponse shape,
keeping full backward-compatibility with the original app.py frontend.
"""

import logging
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from model.pipeline_loader import (
    pipelines_loaded,
    load_lora_weights,
    unload_lora_weights,
    swap_scheduler,
    SCHEDULER_MAP,
)
from services.image_service import run_txt2img, run_img2img, GenerationResult
from services.prompt_service import enhance_prompt, list_styles

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """
    Unified request schema — backward-compatible with original app.py.
    If init_image_base64 is provided the service auto-switches to img2img.
    """
    prompt:               str   = Field(..., min_length=1, max_length=2000)
    negative_prompt:      str   = Field(default="", max_length=2000)
    style:                Optional[str]   = Field(default=None)

    steps:                int   = Field(default=35,  ge=1,   le=150)
    guidance_scale:       float = Field(default=7.5, ge=1.0, le=20.0)
    width:                int   = Field(default=1024, ge=256, le=2048)
    height:               int   = Field(default=1024, ge=256, le=2048)
    seed:                 Optional[int]   = Field(default=None)

    # img2img fields
    init_image_base64:    Optional[str]   = Field(default=None)
    strength:             float = Field(default=0.55, ge=0.0, le=1.0)

    # quality controls
    use_refiner:          bool  = Field(default=True)
    refiner_steps:        int   = Field(default=20, ge=1, le=50)
    auto_enhance_prompt:  bool  = Field(default=True)


class GenerateResponse(BaseModel):
    """Backward-compatible with original GenerateResponse."""
    image_base64:     str
    seed:             int
    generation_time:  float
    width:            int
    height:           int
    # Extended fields (new — ignored by old clients)
    enhanced_prompt:  str = ""
    negative_prompt:  str = ""
    mode:             str = "txt2img"


class EnhancePromptRequest(BaseModel):
    prompt:               str           = Field(..., min_length=1, max_length=500)
    style:                Optional[str] = Field(default=None)
    negative_prompt:      str           = Field(default="")
    force_human_tokens:   bool          = Field(default=False)


class EnhancePromptResponse(BaseModel):
    enhanced_prompt:      str
    negative_prompt:      str
    original_prompt:      str
    style_applied:        Optional[str]
    is_human_subject:     bool
    recommended_steps:    int
    recommended_cfg:      float


class HealthResponse(BaseModel):
    status:           str
    device:           str
    model_loaded:     bool
    gpu_memory_used:  Optional[str]
    gpu_memory_total: Optional[str]
    gpu_utilization:  Optional[str]
    vram_free_gb:     Optional[float]


class LoRALoadRequest(BaseModel):
    lora_path:    str   = Field(..., description="HF repo ID or local path")
    lora_scale:   float = Field(default=0.9, ge=0.0, le=1.0)
    adapter_name: str   = Field(default="default")


class SwapSchedulerRequest(BaseModel):
    scheduler: str = Field(..., description="One of: dpm++, euler_a, ddim")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared guard
# ─────────────────────────────────────────────────────────────────────────────

def _require_model():
    if not pipelines_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")


def _result_to_response(r: GenerationResult) -> GenerateResponse:
    return GenerateResponse(
        image_base64=r.image_base64,
        seed=r.seed,
        generation_time=r.generation_time,
        width=r.width,
        height=r.height,
        enhanced_prompt=r.enhanced_prompt,
        negative_prompt=r.negative_prompt,
        mode=r.mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """GPU / model health check."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_used = gpu_total = gpu_util = vram_free = None

    if torch.cuda.is_available():
        props      = torch.cuda.get_device_properties(0)
        used_bytes = torch.cuda.memory_allocated(0)
        total_bytes = props.total_memory
        free_bytes  = total_bytes - used_bytes

        gpu_used  = f"{used_bytes  / 1024**3:.2f} GB"
        gpu_total = f"{total_bytes / 1024**3:.2f} GB"
        vram_free = round(free_bytes / 1024**3, 2)

        try:
            import subprocess, json
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            gpu_util = f"{result.stdout.strip()} %"
        except Exception:
            gpu_util = "unavailable"

    return HealthResponse(
        status="healthy",
        device=device,
        model_loaded=pipelines_loaded(),
        gpu_memory_used=gpu_used,
        gpu_memory_total=gpu_total,
        gpu_utilization=gpu_util,
        vram_free_gb=vram_free,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Generation endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate(request: GenerateRequest):
    """
    Unified generation endpoint.
    Auto-routes to txt2img or img2img based on presence of init_image_base64.
    Backward-compatible with the original /generate endpoint.
    """
    _require_model()
    try:
        if request.init_image_base64:
            result = await run_img2img(
                init_image_b64=request.init_image_base64,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                style=request.style,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                strength=request.strength,
                seed=request.seed,
                use_refiner=request.use_refiner,
                refiner_steps=request.refiner_steps,
                auto_enhance_prompt=request.auto_enhance_prompt,
            )
        else:
            result = await run_txt2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                style=request.style,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                seed=request.seed,
                use_refiner=request.use_refiner,
                refiner_steps=request.refiner_steps,
                auto_enhance_prompt=request.auto_enhance_prompt,
            )
        return _result_to_response(result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Generation runtime error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/generate/txt2img", response_model=GenerateResponse, tags=["generation"])
async def generate_txt2img_explicit(request: GenerateRequest):
    """Explicit txt2img endpoint (ignores init_image_base64 if provided)."""
    _require_model()
    try:
        result = await run_txt2img(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            style=request.style,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
            use_refiner=request.use_refiner,
            refiner_steps=request.refiner_steps,
            auto_enhance_prompt=request.auto_enhance_prompt,
        )
        return _result_to_response(result)
    except Exception as e:
        logger.error(f"txt2img failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/img2img", response_model=GenerateResponse, tags=["generation"])
async def generate_img2img_explicit(request: GenerateRequest):
    """
    Explicit img2img endpoint.
    Returns 422 if init_image_base64 is not provided.
    """
    _require_model()
    if not request.init_image_base64:
        raise HTTPException(
            status_code=422,
            detail="init_image_base64 is required for the img2img endpoint.",
        )
    try:
        result = await run_img2img(
            init_image_b64=request.init_image_base64,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            style=request.style,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            strength=request.strength,
            seed=request.seed,
            use_refiner=request.use_refiner,
            refiner_steps=request.refiner_steps,
            auto_enhance_prompt=request.auto_enhance_prompt,
        )
        return _result_to_response(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"img2img failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt enhancement
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/enhance-prompt", response_model=EnhancePromptResponse, tags=["prompts"])
async def enhance_prompt_endpoint(request: EnhancePromptRequest):
    """
    Enrich a user prompt with quality and style tokens.
    Does NOT generate an image — useful for previewing the full prompt before generation.
    """
    try:
        result = enhance_prompt(
            raw_prompt=request.prompt,
            style=request.style,
            raw_negative_prompt=request.negative_prompt,
            force_human_tokens=request.force_human_tokens,
        )
        return EnhancePromptResponse(
            enhanced_prompt=result.positive,
            negative_prompt=result.negative,
            original_prompt=request.prompt,
            style_applied=result.style_applied,
            is_human_subject=result.is_human_subject,
            recommended_steps=result.recommended_steps,
            recommended_cfg=result.recommended_cfg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/styles", tags=["metadata"])
async def get_styles():
    """List all available style templates with recommended parameters."""
    return {"styles": list_styles()}


@router.get("/schedulers", tags=["metadata"])
async def get_schedulers():
    """List available scheduler names."""
    return {
        "schedulers": [
            {"id": k, "class": v.__name__}
            for k, v in SCHEDULER_MAP.items()
        ],
        "default": "dpm++",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA management
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/lora/load", tags=["model"])
async def load_lora(request: LoRALoadRequest):
    """Load a LoRA checkpoint into the base and img2img pipelines."""
    _require_model()
    try:
        load_lora_weights(
            lora_path=request.lora_path,
            lora_scale=request.lora_scale,
            adapter_name=request.adapter_name,
        )
        return {"status": "ok", "adapter": request.adapter_name, "path": request.lora_path}
    except Exception as e:
        logger.error(f"LoRA load failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/lora/{adapter_name}", tags=["model"])
async def unload_lora(adapter_name: str):
    """Unload a LoRA adapter by name."""
    _require_model()
    try:
        unload_lora_weights(adapter_name)
        return {"status": "ok", "unloaded": adapter_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler hot-swap
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/scheduler/swap", tags=["model"])
async def swap_scheduler_endpoint(request: SwapSchedulerRequest):
    """Hot-swap the scheduler on all loaded pipelines without reloading models."""
    _require_model()
    if request.scheduler not in SCHEDULER_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown scheduler '{request.scheduler}'. "
                   f"Valid options: {list(SCHEDULER_MAP.keys())}",
        )
    try:
        swap_scheduler(request.scheduler)
        return {"status": "ok", "scheduler": request.scheduler}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))