from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    detected: bool
    fire_count: int
    smoke_count: int
    image_width: int
    image_height: int
    inference_ms: float
    device: str
    result_image_url: str | None = None
    detections: list[Detection]


class ModelInfo(BaseModel):
    ready: bool
    model_path: str
    model_name: str
    device: str
    classes: dict[int, str]
    error: str | None = None
