from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import Settings
from backend.detector import Prediction, decode_image
from backend.main import create_app


class FakeDetector:
    ready = True
    device = "cpu"
    error = None

    def info(self):
        return {"ready": True, "model_path": "fake.pt", "model_name": "fake.pt", "device": "cpu", "classes": {0: "fire", 1: "smoke"}, "error": None}

    def predict(self, data, confidence, iou):
        image = decode_image(data)
        return Prediction(image.width, image.height, 3.2, "cpu", [{"class_id": 0, "class_name": "fire", "confidence": 0.91, "x1": 1, "y1": 2, "x2": image.width - 1, "y2": image.height - 2}], image)


@pytest.fixture
def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "navy").save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path: Path):
    config = Settings(model_path=tmp_path / "missing.pt", results_dir=tmp_path / "results", frontend_dist=tmp_path / "dist", max_upload_bytes=1024 * 1024)
    with TestClient(create_app(config, FakeDetector())) as test_client:
        yield test_client
