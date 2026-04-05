# AI Image Generation Service v2.0

Production-grade SDXL image generation API — rebuilt from scratch.

---

## Project Structure

```
ai_image_service/
│
├── app.py                         # FastAPI entrypoint, lifespan, CORS
│
├── model/
│   ├── pipeline_loader.py         # SDXL load, LoRA, scheduler, memory optimisation
│   └── inference.py               # txt2img + img2img sync/async inference
│
├── services/
│   ├── prompt_service.py          # Prompt enrichment, style templates, negative builder
│   └── image_service.py           # Orchestration: decode → enhance → infer → encode
│
├── api/
│   └── routes.py                  # All FastAPI endpoints + Pydantic schemas
│
├── utils/
│   └── gpu_monitor.py             # VRAM / GPU utilisation helpers
│
└── requirements.txt
```

---

## What Changed vs v1

| Area | v1 | v2 |
|---|---|---|
| Base model | SD 1.5 (512px) | SDXL 1.0 (1024px native) |
| Refiner | None | SDXL Refiner (2-pass) |
| VAE | Default (colour drift) | madebyollin/sdxl-vae-fp16-fix |
| Scheduler | PNDM (default) | DPM-Solver++ with Karras sigmas |
| img2img | Basic resize | Proper SDXL img2img pipeline, configurable strength |
| Prompt | Raw passthrough | Auto-enriched with quality + style tokens |
| LoRA | None | Hot-loadable, named adapters |
| Architecture | Monolithic app.py | Layered: model / services / api |
| Memory | Basic | xformers + attention slicing + VAE tiling |
| Endpoints | 3 | 10 (backward-compatible) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# Install torch for your CUDA version separately (see requirements.txt)

# 2. Run (always 1 worker — GPU singleton)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

# 3. Open API docs
open http://localhost:8000/docs
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOAD_REFINER` | `true` | Load SDXL Refiner for 2-pass generation |
| `ENABLE_SAFETY_CHECKER` | `false` | Enable/disable safety filter |
| `SCHEDULER` | `dpm++` | Scheduler: `dpm++`, `euler_a`, `ddim` |
| `LORA_PATH` | `` | HF repo ID or local path to load LoRA on startup |
| `LORA_SCALE` | `0.9` | LoRA blend weight (0.0–1.0) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

---

## API Reference

### `POST /generate`
Unified endpoint — auto-detects txt2img vs img2img.

```json
{
  "prompt": "a woman in a cafe, natural light",
  "negative_prompt": "",
  "style": "photorealistic",
  "steps": 35,
  "guidance_scale": 7.5,
  "width": 1024,
  "height": 1024,
  "seed": null,
  "init_image_base64": null,
  "strength": 0.55,
  "use_refiner": true,
  "refiner_steps": 20,
  "auto_enhance_prompt": true
}
```

Response:
```json
{
  "image_base64": "...",
  "seed": 1234567890,
  "generation_time": 4.23,
  "width": 1024,
  "height": 1024,
  "enhanced_prompt": "a woman in a cafe, natural light, photorealistic, ...",
  "negative_prompt": "blurry, low quality, ...",
  "mode": "txt2img"
}
```

### `POST /generate/txt2img`
Explicit txt2img (same schema, ignores `init_image_base64`).

### `POST /generate/img2img`
Explicit img2img — requires `init_image_base64`.

`strength` guide:
- `0.30–0.45` → subtle: lighting, texture, colour
- `0.45–0.65` → moderate: style transfer, outfit change
- `0.65–0.85` → heavy: background, mood, full repaint
- `0.85–1.00` → near txt2img (ignores original structure)

### `POST /enhance-prompt`
Preview the enriched prompt without generating an image.

```json
{
  "prompt": "a woman smiling",
  "style": "studio_portrait",
  "force_human_tokens": true
}
```

### `GET /styles`
List all style templates with recommended steps/CFG.

### `GET /schedulers`
List available scheduler names.

### `POST /lora/load`
Load a LoRA at runtime.

```json
{
  "lora_path": "TheLastBen/Papercut_SDXL",
  "lora_scale": 0.8,
  "adapter_name": "papercut"
}
```

### `DELETE /lora/{adapter_name}`
Unload a LoRA by name.

### `POST /scheduler/swap`
Hot-swap scheduler without reloading the model.

```json
{ "scheduler": "euler_a" }
```

### `GET /health`
GPU + model status.

---

## VRAM Requirements

| Configuration | Minimum VRAM |
|---|---|
| SDXL Base only (1024px, float16) | 8 GB |
| SDXL Base + Refiner | 12 GB |
| SDXL Base + Refiner + xformers | 10 GB |
| 512px output (base only) | 6 GB |

---

## Style Templates

| ID | Best for |
|---|---|
| `photorealistic` | People, portraits, real-world scenes |
| `cinematic` | Dramatic scenes, storytelling |
| `studio_portrait` | Professional headshots, clean backgrounds |
| `editorial` | Fashion, magazine-quality |
| `street_photography` | Candid, documentary |
| `oil_painting` | Artistic, painterly |
| `cyberpunk` | Sci-fi, neon, dystopian |
| `fantasy` | Magical worlds, creatures |

---

## Architecture Decisions

**Why SDXL over SD 1.5?**
SDXL was trained at 1024×1024 and produces dramatically better anatomy, skin texture, and detail for human subjects — the primary use case.

**Why the fixed VAE?**
The default SDXL VAE in float16 produces colour oversaturation. `madebyollin/sdxl-vae-fp16-fix` eliminates this with no speed cost.

**Why DPM-Solver++ with Karras sigmas?**
It converges in 20–30 steps to quality that PNDM needs 50+ steps to match. Karras sigmas further improve sharpness at low step counts.

**Why shared weights for img2img?**
`StableDiffusionXLImg2ImgPipeline` is instantiated pointing at the same VAE, text encoders, UNet, and scheduler as the base. Zero extra VRAM.

**Why the 2-pass refiner?**
The SDXL Refiner was trained specifically to add fine detail (pores, hair strands, fabric texture) in the final denoising steps. Handing off the last 20% of steps gives a measurable quality boost for portraits.

**Why singleton + run_in_executor?**
The model must live in one process (GPU singleton). `run_in_executor` offloads the blocking inference call to a thread pool so FastAPI's event loop stays unblocked for health checks and parallel requests.