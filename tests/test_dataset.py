from __future__ import annotations

from pathlib import Path

from PIL import Image

from training.inspect_dataset import inspect
from training.prepare_dataset import split_groups


def test_inspector_detects_valid_yolo_and_duplicates(tmp_path: Path):
    Image.new("RGB", (20, 10), "red").save(tmp_path / "fire_001.jpg")
    (tmp_path / "fire_001.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (tmp_path / "copy.jpg").write_bytes((tmp_path / "fire_001.jpg").read_bytes())
    report = inspect(tmp_path, ["fire", "smoke"])
    assert report["image_count"] == 2
    assert report["bounding_boxes_present"] is True
    assert report["class_object_counts"]["0"] == 1
    assert len(report["duplicate_image_groups"]) == 1


def test_grouped_frames_never_leak(tmp_path: Path):
    images = [tmp_path / name for name in ("clip_frame001.jpg", "clip_frame002.jpg", "other001.jpg", "third001.jpg")]
    split = split_groups(images, 42)
    locations = {path.name: name for name, paths in split.items() for path in paths}
    assert locations["clip_frame001.jpg"] == locations["clip_frame002.jpg"]
