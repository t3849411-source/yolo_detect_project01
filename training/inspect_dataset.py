from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import UnidentifiedImageError

try:
    from training.dataset_utils import coco_documents, group_key, image_size, images_under, parse_voc, parse_yolo, sha256
except ModuleNotFoundError:
    from dataset_utils import coco_documents, group_key, image_size, images_under, parse_voc, parse_yolo, sha256


def inspect(root: Path, class_names: list[str] | None = None, exclude_dirs: set[str] | None = None) -> dict:
    exclude_dirs = {item.lower() for item in (exclude_dirs or set())}

    def included(path: Path) -> bool:
        return not ({part.lower() for part in path.relative_to(root).parts} & exclude_dirs)

    images = [path for path in images_under(root) if included(path)]
    txts = [path for path in root.rglob("*.txt") if included(path) and path.name.lower() not in {"classes.txt", "readme.txt"}]
    xmls = [path for path in root.rglob("*.xml") if included(path)]
    cocos = [path for path in coco_documents(root) if included(path)]
    stem_images = defaultdict(list)
    for image in images:
        stem_images[image.stem.lower()].append(image)

    damaged, sizes, hashes = [], Counter(), defaultdict(list)
    for image in images:
        try:
            sizes["%dx%d" % image_size(image)] += 1
            hashes[sha256(image)].append(str(image.relative_to(root)))
        except (OSError, ValueError, UnidentifiedImageError):
            damaged.append(str(image.relative_to(root)))

    invalid_labels, missing_labels, orphan_labels = [], [], []
    class_objects, class_images = Counter(), Counter()
    labelled_stems = set()
    has_boxes = False

    for label in txts:
        if label.stem.lower() not in stem_images:
            orphan_labels.append(str(label.relative_to(root)))
        try:
            rows = parse_yolo(label)
            seen = set()
            for class_id, x, y, width, height in rows:
                has_boxes = True
                if class_id < 0 or (class_names and class_id >= len(class_names)):
                    invalid_labels.append(f"{label}: 잘못된 클래스 {class_id}")
                if any(value < 0 or value > 1 for value in (x, y, width, height)):
                    invalid_labels.append(f"{label}: 좌표 범위 오류")
                if width <= 0 or height <= 0:
                    invalid_labels.append(f"{label}: 크기가 0인 박스")
                class_objects[str(class_id)] += 1
                seen.add(str(class_id))
            for key in seen:
                class_images[key] += 1
            labelled_stems.add(label.stem.lower())
        except (ValueError, OSError) as exc:
            invalid_labels.append(str(exc))

    voc_classes = Counter()
    for label in xmls:
        if label.stem.lower() not in stem_images:
            orphan_labels.append(str(label.relative_to(root)))
        try:
            _, objects = parse_voc(label)
            seen = set()
            for name, x1, y1, x2, y2 in objects:
                has_boxes = True
                if not name:
                    invalid_labels.append(f"{label}: 빈 클래스 이름")
                if x2 <= x1 or y2 <= y1:
                    invalid_labels.append(f"{label}: 크기가 0인 박스")
                voc_classes[name] += 1
                seen.add(name)
            for name in seen:
                class_images[name] += 1
            labelled_stems.add(label.stem.lower())
        except (ET.ParseError, ValueError, OSError) as exc:  # type: ignore[name-defined]
            invalid_labels.append(f"{label}: {exc}")

    coco_summary = []
    for document in cocos:
        data = json.loads(document.read_text(encoding="utf-8"))
        categories = {item["id"]: str(item["name"]).lower() for item in data["categories"]}
        annotations = data["annotations"]
        has_boxes = has_boxes or any(len(item.get("bbox", [])) == 4 for item in annotations)
        counts = Counter(categories.get(item.get("category_id"), str(item.get("category_id"))) for item in annotations)
        coco_summary.append({"file": str(document.relative_to(root)), "classes": dict(counts)})

    for image in images:
        if image.stem.lower() not in labelled_stems and not cocos:
            missing_labels.append(str(image.relative_to(root)))

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    split_names = {"train", "val", "test"}

    def splits_for(paths):
        return {part.lower() for value in paths for part in Path(value).parts if part.lower() in split_names}

    cross_split_duplicates = [paths for paths in duplicate_groups if len(splits_for(paths)) > 1]
    groups = defaultdict(list)
    for image in images:
        groups[group_key(image)].append(str(image.relative_to(root)))
    cross_split_groups = [paths for paths in groups.values() if len(splits_for(paths)) > 1]
    license_files = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and (path.name.lower().startswith("license") or path.name.lower().startswith("readme"))
    ]
    result = {
        "root": str(root.resolve()),
        "image_count": len(images),
        "image_extensions": dict(Counter(path.suffix.lower() for path in images)),
        "yolo_label_count": len(txts),
        "voc_xml_count": len(xmls),
        "coco_json_count": len(cocos),
        "bounding_boxes_present": has_boxes,
        "classification_only": bool(images) and not has_boxes,
        "class_names": class_names or sorted(voc_classes),
        "class_image_counts": dict(class_images),
        "class_object_counts": dict(class_objects + voc_classes),
        "damaged_images": damaged,
        "unlabelled_images": missing_labels,
        "orphan_labels": orphan_labels,
        "invalid_labels": invalid_labels,
        "duplicate_image_groups": duplicate_groups,
        "cross_split_duplicate_groups": cross_split_duplicates,
        "cross_split_similar_frame_groups": cross_split_groups,
        "image_size_distribution": dict(sizes.most_common()),
        "class_imbalance_ratio": (
            round(max((class_objects + voc_classes).values()) / min((class_objects + voc_classes).values()), 3)
            if class_objects + voc_classes and min((class_objects + voc_classes).values()) > 0
            else None
        ),
        "coco": coco_summary,
        "license_or_readme_files": license_files,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/raw"))
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_inspection.json"))
    parser.add_argument("--exclude", nargs="*", default=[], help="검사에서 제외할 디렉터리 이름")
    args = parser.parse_args()
    if not args.root.is_dir():
        print(f"데이터셋 폴더가 없습니다: {args.root}")
        return 2
    result = inspect(args.root, args.classes, set(args.exclude))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["classification_only"]:
        print("경고: Bounding Box Annotation이 없습니다. 객체 탐지 학습을 진행할 수 없습니다.")
        return 3
    return 0 if result["image_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
