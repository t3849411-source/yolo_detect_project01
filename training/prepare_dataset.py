from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from training.dataset_utils import coco_documents, group_key, image_size, images_under, parse_voc, parse_yolo
except ModuleNotFoundError:
    from dataset_utils import coco_documents, group_key, image_size, images_under, parse_voc, parse_yolo


CLASSES = ["fire", "smoke"]


def normalized_name(value: str) -> str:
    name = value.strip().lower()
    aliases = {"flame": "fire", "flames": "fire", "fires": "fire", "smokes": "smoke"}
    return aliases.get(name, name)


def split_groups(images: list[Path], seed: int) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[group_key(image)].append(image)
    values = list(groups.values())
    random.Random(seed).shuffle(values)
    targets = {"train": len(images) * 0.7, "val": len(images) * 0.2, "test": len(images) * 0.1}
    result = {name: [] for name in targets}
    for group in values:
        destination = max(targets, key=lambda key: targets[key] - len(result[key]))
        result[destination].extend(group)
    return result


def discover_format(raw: Path) -> str:
    if list(raw.rglob("*.txt")):
        for path in raw.rglob("*.txt"):
            try:
                if path.read_text(encoding="utf-8", errors="ignore").strip() and len(path.read_text().split()[0:5]) == 5:
                    parse_yolo(path)
                    return "yolo"
            except (ValueError, OSError):
                pass
    if list(raw.rglob("*.xml")):
        return "voc"
    if coco_documents(raw):
        return "coco"
    return "none"


def unique_output_name(image: Path, raw: Path) -> str:
    relative = image.relative_to(raw).with_suffix("")
    digest = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in image.stem)[:90]
    return f"{digest}_{safe_stem}{image.suffix.lower()}"


def convert_voc(image: Path, xml: Path) -> list[str]:
    (width, height), objects = parse_voc(xml)
    if not width or not height:
        width, height = image_size(image)
    lines = []
    for name, x1, y1, x2, y2 in objects:
        name = normalized_name(name)
        if name not in CLASSES:
            raise ValueError(f"지원하지 않는 클래스 '{name}': {xml}")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"유효하지 않은 박스: {xml}")
        lines.append(
            f"{CLASSES.index(name)} {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} "
            f"{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"
        )
    return lines


def coco_index(raw: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for document in coco_documents(raw):
        data = json.loads(document.read_text(encoding="utf-8"))
        categories = {item["id"]: normalized_name(item["name"]) for item in data["categories"]}
        images = {item["id"]: item for item in data["images"]}
        for annotation in data["annotations"]:
            item = images[annotation["image_id"]]
            name = categories[annotation["category_id"]]
            if name not in CLASSES:
                raise ValueError(f"지원하지 않는 COCO 클래스 '{name}'")
            x, y, width, height = map(float, annotation["bbox"])
            if width <= 0 or height <= 0:
                raise ValueError(f"COCO의 크기가 0인 박스: {document}")
            result[Path(item["file_name"]).stem.lower()].append(
                f"{CLASSES.index(name)} {(x+width/2)/item['width']:.6f} {(y+height/2)/item['height']:.6f} "
                f"{width/item['width']:.6f} {height/item['height']:.6f}"
            )
    return result


def parse_class_map(value: str) -> dict[int, int | None]:
    """예: 0:drop,1:fire,2:smoke를 대상 클래스 번호로 바꾼다."""
    result: dict[int, int | None] = {}
    for item in value.split(","):
        source, destination = item.split(":", 1)
        destination = destination.strip().lower()
        if destination == "drop":
            result[int(source)] = None
        elif destination in CLASSES:
            result[int(source)] = CLASSES.index(destination)
        else:
            raise ValueError(f"알 수 없는 대상 클래스: {destination}")
    return result


def clean_yolo_rows(rows, class_map: dict[int, int | None]):
    lines, rejected = [], 0
    for class_id, cx, cy, width, height in rows:
        target = class_map.get(class_id)
        if target is None:
            continue
        if width <= 0 or height <= 0:
            rejected += 1
            continue
        # 원본 경계가 약간 벗어난 경우 실제 이미지 영역으로 잘라 유효한 박스로 만든다.
        x1, y1 = max(0.0, cx - width / 2), max(0.0, cy - height / 2)
        x2, y2 = min(1.0, cx + width / 2), min(1.0, cy + height / 2)
        if x2 <= x1 or y2 <= y1:
            rejected += 1
            continue
        lines.append(f"{target} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
    return lines, rejected


def yolo_sidecar(image: Path) -> Path | None:
    if image.parent.name.lower() == "images":
        candidate = image.parent.parent / "labels" / image.with_suffix(".txt").name
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="원본 Annotation을 YOLO 형식으로 변환하고 분할")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--format", choices=["auto", "yolo", "voc", "coco"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo-class-map", default="0:fire,1:smoke", help="원본 ID 매핑. 예: 0:drop,1:fire,2:smoke")
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="기본값 copy는 학습 라이브러리의 이미지 자동 복구가 raw 원본을 변경하지 않게 한다.",
    )
    args = parser.parse_args()
    images = images_under(args.raw)
    if not images:
        print("원본 이미지가 없습니다.")
        return 2
    annotation_format = discover_format(args.raw) if args.format == "auto" else args.format
    if annotation_format == "none":
        print("Bounding Box Annotation이 없어 변환/학습을 중단합니다. CVAT 또는 Label Studio로 라벨링하세요.")
        return 3

    xml_by_stem = defaultdict(list)
    for path in args.raw.rglob("*.xml"):
        xml_by_stem[path.stem.lower()].append(path)
    coco = coco_index(args.raw) if annotation_format == "coco" else {}
    class_map = parse_class_map(args.yolo_class_map)

    records = []
    seen_hashes: dict[str, Path] = {}
    duplicate_images: list[dict[str, str]] = []
    rejected_boxes = 0
    for image in images:
        direct_yolo = yolo_sidecar(image)
        if direct_yolo and args.format in {"auto", "yolo"}:
            lines, rejected = clean_yolo_rows(parse_yolo(direct_yolo), class_map)
            rejected_boxes += rejected
        elif xml_by_stem[image.stem.lower()] and args.format in {"auto", "voc"}:
            lines = convert_voc(image, xml_by_stem[image.stem.lower()][0])
        elif args.format in {"auto", "coco"} and image.stem.lower() in coco:
            lines = coco[image.stem.lower()]
        else:
            continue
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in seen_hashes:
            duplicate_images.append({"kept": str(seen_hashes[digest].relative_to(args.raw)), "dropped": str(image.relative_to(args.raw))})
            continue
        seen_hashes[digest] = image
        # other 전용 이미지는 라벨을 비운 채 유용한 negative sample로 유지한다.
        records.append((image, lines))

    if not records:
        print("이미지와 연결되는 유효한 Bounding Box가 없습니다. 학습 데이터를 만들지 않았습니다.")
        return 3

    split_paths = split_groups([image for image, _ in records], args.seed)
    lines_by_image = {image: lines for image, lines in records}
    manifest = {}
    for split_name, split_images in split_paths.items():
        image_dir = args.output / "images" / split_name
        label_dir = args.output / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        manifest[split_name] = []
        for image in split_images:
            output_name = unique_output_name(image, args.raw)
            output_image = image_dir / output_name
            lines = lines_by_image[image]
            if output_image.exists():
                output_image.unlink()
            if args.copy_mode == "hardlink":
                try:
                    output_image.hardlink_to(image)
                except OSError:
                    shutil.copy2(image, output_image)
            else:
                shutil.copy2(image, output_image)
            (label_dir / Path(output_name).with_suffix(".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest[split_name].append(str(image.relative_to(args.raw)))

    config = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "fire", 1: "smoke"}}
    (args.output / "fire.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (args.output / "classes.txt").write_text("fire\nsmoke\n", encoding="utf-8")
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    group_memberships: dict[str, set[str]] = defaultdict(set)
    for split_name, split_images in split_paths.items():
        for image in split_images:
            group_memberships[group_key(image)].add(split_name)
    preparation_report = {
        "input_images": len(images),
        "prepared_images": len(records),
        "exact_duplicates_removed": len(duplicate_images),
        "rejected_invalid_boxes": rejected_boxes,
        "split_counts": {key: len(value) for key, value in split_paths.items()},
        "class_map": args.yolo_class_map,
        "cross_split_group_leaks": sum(len(memberships) > 1 for memberships in group_memberships.values()),
        "duplicates": duplicate_images,
    }
    (args.output / "preparation_report.json").write_text(json.dumps(preparation_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in preparation_report.items() if key != "duplicates"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
