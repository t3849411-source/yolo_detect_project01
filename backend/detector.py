from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


class DetectorUnavailableError(RuntimeError):
    pass


class InvalidImageError(ValueError):
    pass


@dataclass
class Prediction:
    width: int
    height: int
    inference_ms: float
    device: str
    detections: list[dict[str, Any]]
    image: Image.Image


def choose_device(requested: str | None = None) -> str:
    """CUDA가 가능하면 첫 GPU를, 그렇지 않으면 CPU를 선택한다."""
    if requested and requested.lower() not in {"", "auto"}:
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def decode_image(data: bytes, max_pixels: int = 50_000_000) -> Image.Image:
    if not data:
        raise InvalidImageError("빈 이미지입니다.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > max_pixels:
                raise InvalidImageError(f"이미지 해상도가 제한({max_pixels:,} pixels)을 초과합니다.")
            # 스마트폰 EXIF 방향을 실제 픽셀 방향에 반영하되 입력 바이트는 변경하지 않는다.
            return ImageOps.exif_transpose(source).convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("손상되었거나 지원하지 않는 이미지입니다.") from exc


class Detector:
    def __init__(
        self,
        model_path: Path,
        device: str | None = None,
        max_image_pixels: int = 50_000_000,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = choose_device(device)
        self.max_image_pixels = max_image_pixels
        self.model: Any | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def class_names(self) -> dict[int, str]:
        if not self.ready:
            return {0: "fire", 1: "smoke"}
        names = getattr(self.model, "names", {0: "fire", 1: "smoke"})
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
        return {int(key): str(value) for key, value in names.items()}

    def load(self) -> None:
        if self.ready:
            return
        if not self.model_path.is_file():
            self.error = f"모델 파일이 없습니다: {self.model_path}"
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.error = None
        except Exception as exc:
            self.error = f"모델 로딩 실패: {exc}"

    def info(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "device": self.device,
            "classes": self.class_names,
            "error": self.error,
        }

    def predict(self, data: bytes, confidence: float, iou: float) -> Prediction:
        if not self.ready:
            raise DetectorUnavailableError(self.error or "모델을 사용할 수 없습니다.")

        image = decode_image(data, self.max_image_pixels)
        started = time.perf_counter()
        # GPU 모델 호출은 스레드 안전성을 보장하지 않으므로 한 번에 하나만 실행한다.
        with self._lock:
            results = self.model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                device=self.device,
                verbose=False,
            )
        inference_ms = (time.perf_counter() - started) * 1000

        detections: list[dict[str, Any]] = []
        result = results[0]
        if getattr(result, "speed", None) and result.speed.get("inference") is not None:
            inference_ms = float(result.speed["inference"])
        for box in result.boxes:
            class_id = int(box.cls.item())
            coords = [float(value) for value in box.xyxy[0].tolist()]
            x1 = min(max(coords[0], 0.0), float(image.width))
            y1 = min(max(coords[1], 0.0), float(image.height))
            x2 = min(max(coords[2], 0.0), float(image.width))
            y2 = min(max(coords[3], 0.0), float(image.height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, str(class_id)),
                    "confidence": round(float(box.conf.item()), 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            )
        return Prediction(image.width, image.height, inference_ms, self.device, detections, image)


def draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for item in detections:
        color = "#ef4444" if item["class_name"].lower() == "fire" else "#f59e0b"
        box = (item["x1"], item["y1"], item["x2"], item["y2"])
        draw.rectangle(box, outline=color, width=max(3, round(min(image.size) / 240)))
        label = f'{item["class_name"]} {item["confidence"]:.0%}'
        left, top, right, bottom = draw.textbbox((item["x1"], item["y1"]), label, font=font)
        text_height = bottom - top
        label_top = max(0, item["y1"] - text_height - 8)
        draw.rectangle(
            (item["x1"], label_top, item["x1"] + (right - left) + 8, label_top + text_height + 8),
            fill=color,
        )
        draw.text((item["x1"] + 4, label_top + 4), label, fill="white", font=font)
    return output


def count_classes(detections: list[dict[str, Any]]) -> tuple[int, int]:
    fire = sum(item["class_name"].lower() == "fire" for item in detections)
    smoke = sum(item["class_name"].lower() == "smoke" for item in detections)
    return fire, smoke
