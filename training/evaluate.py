from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
EXAMPLE_LIMIT = 20


def iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def ground_truth(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        boxes.append(
            (
                int(cls),
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            )
        )
    return boxes


def match_boxes(truths, predictions, threshold: float = 0.5):
    matched_truth: set[int] = set()
    matched_pred: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for pi, prediction in enumerate(predictions):
        for ti, truth in enumerate(truths):
            if prediction[0] == truth[0]:
                candidates.append((iou(prediction[2:], truth[1:]), pi, ti))
    for overlap, pi, ti in sorted(candidates, reverse=True):
        if overlap < threshold:
            break
        if pi not in matched_pred and ti not in matched_truth:
            matched_pred.add(pi)
            matched_truth.add(ti)
    return matched_truth, matched_pred


def annotated_image(result, truths, names, status: str) -> Image.Image:
    rendered = Image.fromarray(result.plot()[..., ::-1]).convert("RGB")
    draw = ImageDraw.Draw(rendered)
    for cls, x1, y1, x2, y2 in truths:
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=3)
        draw.text((max(0, x1), max(0, y1 - 12)), f"GT {names[int(cls)]}", fill=(0, 255, 255))
    draw.rectangle((0, 0, rendered.width, 24), fill=(0, 0, 0))
    draw.text((8, 6), status, fill=(255, 255, 255))
    return rendered


def reset_example_directories(output: Path) -> dict[str, Path]:
    directories = {
        "samples": output / "samples",
        "false_positives": output / "false_positives",
        "false_negatives": output / "false_negatives",
        "undetected_negatives": output / "undetected_negatives",
    }
    for directory in directories.values():
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    return directories


def dataset_state(images: list[Path]) -> dict[str, tuple[int, int]]:
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in images}


def make_read_write_test_copy(images: list[Path], label_dir: Path, names: dict, root: Path) -> Path:
    image_copy = root / "images" / "test"
    label_copy = root / "labels" / "test"
    image_copy.mkdir(parents=True)
    label_copy.mkdir(parents=True)
    for image in images:
        shutil.copy2(image, image_copy / image.name)
        label = label_dir / image.with_suffix(".txt").name
        if label.is_file():
            shutil.copy2(label, label_copy / label.name)
    yaml_path = root / "fire-test-copy.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/test",
                "val": "images/test",
                "test": "images/test",
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained detector on the independent test split.")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/fire.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if not args.model.is_file() or not args.data.is_file():
        print("Model file or data YAML is missing.")
        return 2
    if not 0.0 <= args.conf <= 1.0:
        print("--conf must be between 0 and 1.")
        return 2

    import torch
    from ultralytics import YOLO

    device = "0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    base = Path(config["path"])
    if not base.is_absolute():
        base = (args.data.parent / base).resolve()
    test_dir = base / config["test"]
    label_dir = base / str(config["test"]).replace("images", "labels", 1)
    images = sorted(path for path in test_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"No test images found in {test_dir}")
        return 2

    source_state = dataset_state(images)
    print(f"Evaluating split=test with {len(images)} images on device={device}", flush=True)
    model = YOLO(str(args.model))
    with TemporaryDirectory(prefix="fire-evaluation-") as temporary:
        safe_yaml = make_read_write_test_copy(images, label_dir, config["names"], Path(temporary))
        metrics = model.val(
            data=str(safe_yaml),
            split="test",
            device=device,
            plots=True,
            project=str(args.output),
            name="metrics",
            exist_ok=True,
        )

    class_names = {int(key): value for key, value in model.names.items()}
    per_class = {}
    for index, name in class_names.items():
        per_class[name] = {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "ap50": float(metrics.box.ap50[index]),
            "ap50_95": float(metrics.box.maps[index]),
        }
    summary = {
        "evaluated_split": "test",
        "evaluated_images": len(images),
        "device": str(device),
        "confidence_for_error_examples": args.conf,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
        "per_class_map50_95": {name: values["ap50_95"] for name, values in per_class.items()},
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_unchanged": dataset_state(images) == source_state,
    }

    directories = reset_example_directories(args.output)
    selected = {key: 0 for key in directories}
    comparison_rows: list[dict] = []
    totals = {
        "ground_truth_objects": 0,
        "prediction_objects": 0,
        "false_positive_objects_iou50": 0,
        "false_negative_objects_iou50": 0,
        "false_positive_images": 0,
        "false_negative_images": 0,
        "negative_images": 0,
        "undetected_negative_images": 0,
    }

    for result in model.predict(str(test_dir), conf=args.conf, device=device, stream=True, verbose=False):
        image_path = Path(result.path)
        width, height = result.orig_shape[1], result.orig_shape[0]
        truths = ground_truth(label_dir / image_path.with_suffix(".txt").name, width, height)
        predictions = [
            (int(box.cls.item()), float(box.conf.item()), *map(float, box.xyxy[0].tolist()))
            for box in result.boxes
        ]
        matched_truth, matched_pred = match_boxes(truths, predictions)
        fp_count = len(predictions) - len(matched_pred)
        fn_count = len(truths) - len(matched_truth)
        totals["ground_truth_objects"] += len(truths)
        totals["prediction_objects"] += len(predictions)
        totals["false_positive_objects_iou50"] += fp_count
        totals["false_negative_objects_iou50"] += fn_count
        totals["false_positive_images"] += int(fp_count > 0)
        totals["false_negative_images"] += int(fn_count > 0)
        totals["negative_images"] += int(not truths)
        totals["undetected_negative_images"] += int(not truths and not predictions)

        status = f"GT={len(truths)} Pred={len(predictions)} FP={fp_count} FN={fn_count} @ IoU 0.50"
        rendered = None

        def save_example(category: str) -> None:
            nonlocal rendered
            if selected[category] >= EXAMPLE_LIMIT:
                return
            if rendered is None:
                rendered = annotated_image(result, truths, class_names, status)
            target = directories[category] / f"{selected[category] + 1:02d}_{image_path.stem}.jpg"
            rendered.save(target, quality=92)
            selected[category] += 1
            comparison_rows.append(
                {
                    "category": category,
                    "source": str(image_path),
                    "saved_as": str(target.relative_to(args.output)),
                    "ground_truth_count": len(truths),
                    "prediction_count": len(predictions),
                    "false_positive_count_iou50": fp_count,
                    "false_negative_count_iou50": fn_count,
                }
            )

        if predictions:
            save_example("samples")
        if fp_count:
            save_example("false_positives")
        if fn_count:
            save_example("false_negatives")
        if not truths and not predictions:
            save_example("undetected_negatives")

    summary["error_analysis"] = totals
    summary["saved_examples"] = selected
    summary["source_dataset_unchanged"] = dataset_state(images) == source_state
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "example_manifest.json").write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
