from __future__ import annotations

import hashlib
import io
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from backend.detector import Detector, InvalidImageError, choose_device, decode_image


def test_missing_model(tmp_path):
    detector = Detector(tmp_path / "does-not-exist.pt")
    detector.load()
    assert not detector.ready
    assert "모델 파일이 없습니다" in detector.error


def test_model_loads_once_from_existing_path(tmp_path, monkeypatch):
    calls = []
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")

    class FakeYolo:
        names = {0: "fire", 1: "smoke"}

        def __init__(self, path):
            calls.append(path)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYolo))
    detector = Detector(model_path, "cpu")
    detector.load()
    detector.load()
    assert detector.ready
    assert calls == [str(model_path)]


def test_device_selection(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    assert choose_device() == "cuda:0"
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert choose_device() == "cpu"
    assert choose_device("1") == "1"


def test_decode_invalid():
    with pytest.raises(InvalidImageError):
        decode_image(b"broken")


def test_decode_applies_exif_without_mutating_input():
    buffer = io.BytesIO()
    image = Image.new("RGB", (20, 10), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(buffer, "JPEG", exif=exif)
    data = buffer.getvalue()
    before = hashlib.sha256(data).hexdigest()
    decoded = decode_image(data)
    assert decoded.size == (10, 20)
    assert hashlib.sha256(data).hexdigest() == before


def test_decode_rejects_excessive_pixel_count():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "black").save(buffer, "PNG")
    with pytest.raises(InvalidImageError, match="해상도"):
        decode_image(buffer.getvalue(), max_pixels=100)
