from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def images_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{number} 필드가 5개가 아닙니다.")
        rows.append((int(parts[0]), *(float(value) for value in parts[1:])))
    return rows


def parse_voc(path: Path) -> tuple[tuple[int, int], list[tuple[str, float, float, float, float]]]:
    root = ET.parse(path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    objects = []
    for node in root.findall("object"):
        name = (node.findtext("name") or "").strip().lower()
        box = node.find("bndbox")
        if box is None:
            continue
        objects.append(
            (
                name,
                float(box.findtext("xmin", "0")),
                float(box.findtext("ymin", "0")),
                float(box.findtext("xmax", "0")),
                float(box.findtext("ymax", "0")),
            )
        )
    return (width, height), objects


def coco_documents(root: Path) -> list[Path]:
    matches = []
    for path in root.rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and {"images", "annotations", "categories"} <= value.keys():
                matches.append(path)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
    return sorted(matches)


def find_sidecar(image: Path, candidates: Iterable[Path]) -> Path | None:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        by_stem[path.stem.lower()].append(path)
    matches = by_stem.get(image.stem.lower(), [])
    if not matches:
        return None
    # 같은 경로 깊이/부모 이름이 가까운 라벨을 우선한다.
    return min(matches, key=lambda path: abs(len(path.parts) - len(image.parts)))


def group_key(path: Path) -> str:
    """연속 프레임으로 보이는 파일은 같은 split에 넣기 위한 보수적 그룹 키."""
    stem = path.stem.lower()
    # Roboflow export hash는 같은 원본의 변형/중복을 서로 다른 파일처럼 보이게 하므로 제거한다.
    stem = re.sub(r"(?:_jpg)?\.rf\.[0-9a-f]+$", "", stem)
    key = re.sub(r"(?:[_-]?(?:frame|img|image))?[_-]?\d{3,}$", "", stem)
    return key or stem


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size
