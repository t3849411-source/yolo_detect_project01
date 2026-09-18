from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.skipif(not Path("models/best.pt").is_file(), reason="학습된 models/best.pt가 필요합니다.")
def test_small_image_real_inference(tmp_path):
    from ultralytics import YOLO

    image = tmp_path / "small.jpg"
    Image.new("RGB", (64, 64), "black").save(image)
    result = YOLO("models/best.pt").predict(image, imgsz=64, device="cpu", verbose=False)
    assert len(result) == 1
