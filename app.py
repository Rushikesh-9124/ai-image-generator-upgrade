"""
app.py
======
FastAPI application entrypoint.

Handles:
  - App factory + lifespan (startup/shutdown)
  - CORS middleware
  - Router registration
  - Startup config (refiner toggle, scheduler choice, LoRA paths)
  - Graceful shutdown with VRAM cleanup

Run with:
  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

NOTE: Always use --workers 1.
      The model is a GPU singleton — multiple workers would OOM.
"""

import os
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model.pipeline_loader import load_pipelines, unload_pipelines
from api.routes import router

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Startup configuration (override via environment variables)
# ─────────────────────────────────────────────────────────────────────────────
def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


LOAD_REFINER    = _env_bool("LOAD_REFINER", default=True)
ENABLE_SAFETY   = _env_bool("ENABLE_SAFETY_CHECKER", default=False)
SCHEDULER       = os.getenv("SCHEDULER", "dpm++")           # dpm++ | euler_a | ddim
LORA_PATH       = os.getenv("LORA_PATH", "")                # optional: HF repo or local
LORA_SCALE      = float(os.getenv("LORA_SCALE", "0.9"))


# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  AI Image Generation Service — starting up")
    logger.info(f"  Device  : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"  Refiner : {LOAD_REFINER}")
    logger.info(f"  Scheduler: {SCHEDULER}")
    logger.info(f"  Safety  : {ENABLE_SAFETY}")
    logger.info("=" * 60)

    load_pipelines(
        scheduler_name=SCHEDULER,
        load_refiner=LOAD_REFINER,
        enable_safety=ENABLE_SAFETY,
    )

    # Optional: load a default LoRA on startup
    if LORA_PATH:
        from model.pipeline_loader import load_lora_weights
        try:
            load_lora_weights(lora_path=LORA_PATH, lora_scale=LORA_SCALE)
        except Exception as exc:
            logger.warning(f"LoRA load failed (continuing without it): {exc}")

    logger.info("Service ready — accepting requests.")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing GPU memory ...")
    unload_pipelines()
    logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Image Generation Service",
    description=(
        "Production-grade SDXL-based image generation API. "
        "Supports txt2img, img2img, prompt enhancement, LoRA, and scheduler hot-swap."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes
app.include_router(router)