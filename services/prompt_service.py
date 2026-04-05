"""
services/prompt_service.py
===========================
Rebuilt for v3.

Key changes over v2:
  • RealVisXL responds best to specific trigger tokens — added to BASE_QUALITY_TOKENS
  • Stronger human/portrait negative prompt (the main lever for realistic faces)
  • Compulsory anatomy negative tokens — extra fingers / fused hands wrecked realism
  • Guidance scale bumped to 8.0 for human subjects (better prompt adherence on RealVisXL)
  • New style: "hyperrealistic" — the highest-quality human portrait template
  • ControlNet img2img benefits from explicit edit-direction tokens in the prompt
    (e.g. "same pose, same person") — added via img2img-specific inject

Everything else (function signatures, return types, style IDs) is UNCHANGED.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Quality token banks
# ─────────────────────────────────────────────────────────────────────────────

# Applied to every positive prompt.
# "score_9, score_8_up" are RealVisXL trigger tokens — they shift the model
# towards its highest-quality training distribution.
BASE_QUALITY_TOKENS = [
    "score_9",
    "score_8_up",
    "photorealistic",
    "ultra detailed",
    "8k uhd",
    "sharp focus",
    "high dynamic range",
    "professional photography",
    "RAW photo",
]

# Applied when subject is human.
# These are the single biggest lever for realistic faces on RealVisXL.
HUMAN_QUALITY_TOKENS = [
    "realistic skin texture",
    "natural skin pores",
    "subsurface scattering",
    "detailed face",
    "symmetrical facial features",
    "accurate anatomy",
    "lifelike eyes",
    "catchlight in eyes",
    "natural hair strands",
    "real person",
]

# Core negative — applied to everything.
# The anatomy tokens are the most important for preventing the grotesque
# deformations that plague AI-generated hands and faces.
BASE_NEGATIVE_TOKENS = [
    "blurry", "out of focus", "low quality", "low resolution",
    "jpeg artifacts", "compression artifacts", "noise", "grain",
    "pixelated", "watermark", "signature", "logo", "text", "username",
    "deformed", "distorted", "disfigured", "malformed",
    "bad anatomy", "wrong anatomy",
    "extra limbs", "missing limbs", "extra arms", "extra legs",
    "extra fingers", "missing fingers", "fused fingers", "too many fingers",
    "mutated hands", "poorly drawn hands", "clawed hands",
    "duplicate body parts", "multiple heads", "clone",
    "cartoon", "anime", "illustration", "painting", "drawing", "sketch",
    "3d render", "cgi", "plastic", "doll", "mannequin",
    "unrealistic", "oversaturated", "ugly",
]

# Extra negative for human portraits — the sharpest quality lever for faces.
HUMAN_NEGATIVE_TOKENS = [
    "bad face", "distorted face", "unnatural face", "asymmetric eyes",
    "cross-eyed", "wall-eyed", "floating eyes", "misaligned eyes",
    "double chin (when not described)", "visible neck seam",
    "unnatural skin color", "orange skin", "grey skin", "yellow skin",
    "plastic skin", "over-smoothed skin", "airbrushed",
    "makeup smear", "clown makeup",
    "open mouth showing bad teeth", "unrealistic teeth",
    "bald patches (when not described)",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Style templates
# ─────────────────────────────────────────────────────────────────────────────

STYLE_TEMPLATES: dict[str, dict] = {

    # ── New in v3: highest-quality human portrait template ────────────────
    "hyperrealistic": {
        "positive": [
            "hyperrealistic portrait",
            "50mm f/1.4 prime lens",
            "golden hour natural light",
            "subtle rim lighting",
            "shallow depth of field",
            "bokeh background",
            "Hasselblad H6D",
            "color graded",
            "skin imperfections visible",
        ],
        "negative": ["CGI", "render", "illustration", "heavily retouched"],
        "steps": 40,
        "guidance_scale": 8.0,
    },

    "photorealistic": {
        "positive": [
            "DSLR photograph",
            "50mm lens",
            "f/1.8 aperture",
            "natural lighting",
            "cinematic lighting",
            "soft rim light",
            "shallow depth of field",
            "bokeh background",
        ],
        "negative": [],
        "steps": 35,
        "guidance_scale": 8.0,     # bumped from 7.5 for RealVisXL
    },

    "cinematic": {
        "positive": [
            "cinematic shot",
            "movie still",
            "35mm anamorphic lens",
            "dramatic lighting",
            "chiaroscuro",
            "deep shadows",
            "subtle lens flare",
            "film color grading",
        ],
        "negative": ["amateur", "snapshot"],
        "steps": 40,
        "guidance_scale": 8.0,
    },

    "studio_portrait": {
        "positive": [
            "professional studio portrait",
            "Profoto softbox lighting",
            "three-point lighting setup",
            "clean white seamless background",
            "85mm portrait lens",
            "catchlights in eyes",
            "editorial quality",
            "Vogue style",
        ],
        "negative": ["harsh shadows", "red-eye", "amateur"],
        "steps": 38,
        "guidance_scale": 8.5,
    },

    "editorial": {
        "positive": [
            "editorial fashion photography",
            "magazine cover quality",
            "high-key lighting",
            "professional model",
            "striking pose",
            "stylized composition",
        ],
        "negative": ["casual", "snapshot", "amateur"],
        "steps": 38,
        "guidance_scale": 8.0,
    },

    "street_photography": {
        "positive": [
            "candid street photography",
            "natural ambient light",
            "urban environment",
            "35mm prime lens",
            "documentary style",
            "authentic moment",
            "photojournalism",
        ],
        "negative": ["posed", "studio", "artificial light"],
        "steps": 30,
        "guidance_scale": 7.5,
    },

    "oil_painting": {
        "positive": [
            "oil painting",
            "impressionist brushwork",
            "textured canvas",
            "old masters technique",
            "rich colour palette",
            "painterly",
        ],
        "negative": ["photo", "photograph", "realistic", "CGI"],
        "steps": 35,
        "guidance_scale": 8.5,
    },

    "cyberpunk": {
        "positive": [
            "cyberpunk aesthetic",
            "neon lights",
            "dystopian future",
            "rain-slicked streets",
            "holographic displays",
            "techno-noir atmosphere",
        ],
        "negative": ["bright daylight", "natural", "pastoral"],
        "steps": 35,
        "guidance_scale": 8.0,
    },

    "fantasy": {
        "positive": [
            "epic fantasy setting",
            "magical atmosphere",
            "mystical lighting",
            "otherworldly environment",
            "concept art quality",
            "dramatic sky",
        ],
        "negative": ["mundane", "modern", "contemporary"],
        "steps": 40,
        "guidance_scale": 8.5,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_HUMAN_RE = re.compile(
    r"\b(person|woman|man|girl|boy|human|face|portrait|model|people|"
    r"female|male|lady|gentleman|child|baby|teenager|adult|elderly|"
    r"celebrity|actor|actress|athlete|student|professional|"
    r"nurse|doctor|engineer|soldier|chef|dancer|musician)\b",
    re.IGNORECASE,
)

def _is_human(prompt: str) -> bool:
    return bool(_HUMAN_RE.search(prompt))

def _dedup(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out:  list[str] = []
    for t in tokens:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(t.strip())
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  Public API — SIGNATURES UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnhancedPrompt:
    positive:          str
    negative:          str
    style_applied:     Optional[str]
    is_human_subject:  bool
    recommended_steps: int   = 35
    recommended_cfg:   float = 8.0   # bumped from 7.5 for RealVisXL


def enhance_prompt(
    raw_prompt:          str,
    style:               Optional[str] = None,
    raw_negative_prompt: str = "",
    force_human_tokens:  bool = False,
    is_img2img:          bool = False,  # new: inject structure-preserving tokens
) -> EnhancedPrompt:
    """
    Enrich a user prompt with quality + style tokens.

    Args:
        raw_prompt:           User's positive prompt
        style:                Optional style key
        raw_negative_prompt:  User negative (merged with defaults)
        force_human_tokens:   Always add human quality tokens
        is_img2img:           If True, add "same person, same pose" anchor tokens
    """
    raw_prompt = raw_prompt.strip()
    is_human   = force_human_tokens or _is_human(raw_prompt)

    # ── Positive ──────────────────────────────────────────────────────────────
    pos = [raw_prompt]
    pos.extend(BASE_QUALITY_TOKENS)
    if is_human:
        pos.extend(HUMAN_QUALITY_TOKENS)

    # Img2img anchor tokens — tell the model to preserve identity
    # These work synergistically with ControlNet conditioning
    if is_img2img:
        pos.extend([
            "same person",
            "same pose",
            "preserve facial features",
            "consistent identity",
        ])

    rec_steps = 35
    rec_cfg   = 8.0
    applied   = None

    if style and style in STYLE_TEMPLATES:
        tmpl      = STYLE_TEMPLATES[style]
        pos.extend(tmpl["positive"])
        rec_steps = tmpl.get("steps", rec_steps)
        rec_cfg   = tmpl.get("guidance_scale", rec_cfg)
        applied   = style

    # ── Negative ──────────────────────────────────────────────────────────────
    neg = list(BASE_NEGATIVE_TOKENS)
    if is_human:
        neg.extend(HUMAN_NEGATIVE_TOKENS)
    if style and style in STYLE_TEMPLATES:
        neg.extend(STYLE_TEMPLATES[style].get("negative", []))
    if raw_negative_prompt.strip():
        neg.extend(t.strip() for t in raw_negative_prompt.split(","))

    return EnhancedPrompt(
        positive=", ".join(_dedup(pos)),
        negative=", ".join(_dedup(neg)),
        style_applied=applied,
        is_human_subject=is_human,
        recommended_steps=rec_steps,
        recommended_cfg=rec_cfg,
    )


def list_styles() -> list[dict]:
    return [
        {
            "id":                key,
            "name":              key.replace("_", " ").title(),
            "steps":             tmpl.get("steps", 35),
            "guidance_scale":    tmpl.get("guidance_scale", 8.0),
            "preview_keywords":  tmpl["positive"][:3],
        }
        for key, tmpl in STYLE_TEMPLATES.items()
    ]


def build_negative_prompt(user_negative: str = "", is_human: bool = False) -> str:
    parts = list(BASE_NEGATIVE_TOKENS)
    if is_human:
        parts.extend(HUMAN_NEGATIVE_TOKENS)
    if user_negative.strip():
        parts.extend(t.strip() for t in user_negative.split(","))
    return ", ".join(_dedup(parts))