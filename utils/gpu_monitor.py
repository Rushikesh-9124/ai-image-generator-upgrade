"""
utils/gpu_monitor.py
====================
GPU memory and utilization monitoring helpers.
Used by the /health endpoint and can be called anywhere for diagnostics.
"""

import logging
from typing import Optional
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    available:      bool
    device_name:    Optional[str]
    total_gb:       Optional[float]
    used_gb:        Optional[float]
    free_gb:        Optional[float]
    used_pct:       Optional[float]
    utilization_pct: Optional[str]


def get_gpu_stats() -> GPUStats:
    """Return current GPU memory statistics."""
    if not torch.cuda.is_available():
        return GPUStats(
            available=False,
            device_name=None,
            total_gb=None,
            used_gb=None,
            free_gb=None,
            used_pct=None,
            utilization_pct=None,
        )

    props       = torch.cuda.get_device_properties(0)
    total_bytes = props.total_memory
    used_bytes  = torch.cuda.memory_allocated(0)
    free_bytes  = total_bytes - used_bytes

    total_gb = total_bytes / 1024 ** 3
    used_gb  = used_bytes  / 1024 ** 3
    free_gb  = free_bytes  / 1024 ** 3
    used_pct = (used_bytes / total_bytes) * 100

    util = None
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        util = f"{result.stdout.strip()} %"
    except Exception:
        util = "unavailable"

    return GPUStats(
        available=True,
        device_name=props.name,
        total_gb=round(total_gb, 2),
        used_gb=round(used_gb, 2),
        free_gb=round(free_gb, 2),
        used_pct=round(used_pct, 1),
        utilization_pct=util,
    )


def log_gpu_stats(prefix: str = "") -> None:
    """Log current GPU stats at INFO level — useful around inference calls."""
    s = get_gpu_stats()
    if not s.available:
        logger.info(f"{prefix}GPU: not available (CPU mode)")
        return
    logger.info(
        f"{prefix}GPU {s.device_name} | "
        f"VRAM: {s.used_gb:.1f}/{s.total_gb:.1f} GB "
        f"({s.used_pct:.0f}%) | util: {s.utilization_pct}"
    )


def warn_if_low_vram(threshold_gb: float = 2.0) -> None:
    """Emit a warning if free VRAM drops below threshold_gb."""
    s = get_gpu_stats()
    if s.available and s.free_gb is not None and s.free_gb < threshold_gb:
        logger.warning(
            f"Low VRAM warning: only {s.free_gb:.1f} GB free "
            f"(threshold={threshold_gb} GB). Consider reducing resolution or steps."
        )