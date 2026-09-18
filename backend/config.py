from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    model_path: Path = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "best.pt"))
    results_dir: Path = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "outputs" / "results"))
    frontend_dist: Path = Path(os.getenv("FRONTEND_DIST", PROJECT_ROOT / "frontend" / "dist"))
    max_upload_bytes: int = _env_int("MAX_UPLOAD_MB", 15) * 1024 * 1024
    max_image_pixels: int = _env_int("MAX_IMAGE_PIXELS", 50_000_000)
    max_ws_frame_bytes: int = _env_int("MAX_WS_FRAME_MB", 8) * 1024 * 1024
    result_retention_hours: int = _env_int("RESULT_RETENTION_HOURS", 24)
    default_confidence: float = _env_float("DEFAULT_CONFIDENCE", 0.35)
    default_iou: float = _env_float("DEFAULT_IOU", 0.45)
    allowed_origins: tuple[str, ...] = tuple(
        item.strip() for item in os.getenv("ALLOWED_ORIGINS", "*").split(",") if item.strip()
    )


settings = Settings()
