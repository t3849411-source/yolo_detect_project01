from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.detector import decode_image


SOURCES = {
    "fire": "0020918499_fire4_mp4-206_jpg_rf_08c07a1c98791d9cea24415c08a9dcf9",
    "smoke": "029a2819e8_fire8_mp4-150_jpg_rf_ebea2c1ac8aa1c0acc0fe65e2587a356",
    "negative": "002358930c_no_original_no_original_S3-N1725MN00036_jpg_ba9a961e-27ab-46dc-a10d-86be6aeffe32_jpg_e0d43",
}


def find_image(directory: Path, stem: str) -> Path:
    matches = [path for path in directory.glob(f"{stem}.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not matches:
        raise FileNotFoundError(f"Smoke-test source not found: {stem}")
    return matches[0]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_both_image(fire_path: Path, smoke_path: Path, target: Path) -> None:
    images = []
    for path in (fire_path, smoke_path):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((960, 720))
            images.append(image.copy())
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (sum(image.width for image in images), height), "black")
    x = 0
    for image in images:
        canvas.paste(image, (x, (height - image.height) // 2))
        x += image.width
    canvas.save(target, "JPEG", quality=94)


def save_large_phone_image(source_path: Path, target: Path) -> None:
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.fit(image, (3024, 4032), method=Image.Resampling.LANCZOS)
        image.save(target, "JPEG", quality=90)


def save_exif_image(source_path: Path, target: Path) -> None:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        exif = image.getexif()
        exif[274] = 6
        image.save(target, "JPEG", quality=92, exif=exif)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run representative CUDA smoke tests with models/best.pt.")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("data/images/test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model_smoke_test"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    args.output.mkdir(parents=True, exist_ok=True)
    inputs = args.output / "inputs"
    rendered_dir = args.output / "rendered"
    inputs.mkdir(exist_ok=True)
    rendered_dir.mkdir(exist_ok=True)

    source_paths = {name: find_image(args.images, stem) for name, stem in SOURCES.items()}
    source_hashes = {str(path): digest(path) for path in source_paths.values()}
    both_path = inputs / "fire_and_smoke.jpg"
    large_path = inputs / "large_smartphone_3024x4032.jpg"
    exif_path = inputs / "exif_orientation_6.jpg"
    save_both_image(source_paths["fire"], source_paths["smoke"], both_path)
    save_large_phone_image(source_paths["fire"], large_path)
    # 비정방형 원본을 사용해 Orientation=6 적용 후 너비/높이 교환까지 검증한다.
    save_exif_image(source_paths["negative"], exif_path)

    cases = {
        "fire": source_paths["fire"],
        "smoke": source_paths["smoke"],
        "fire_and_smoke": both_path,
        "negative": source_paths["negative"],
        "large_smartphone": large_path,
        "exif_orientation": exif_path,
    }
    input_hashes = {name: digest(path) for name, path in cases.items()}

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test but is unavailable.")
    model = YOLO(str(args.model))
    report = {
        "model": str(args.model.resolve()),
        "model_loaded": True,
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "confidence": args.conf,
        "cases": {},
    }

    for name, path in cases.items():
        image = decode_image(path.read_bytes())
        result = model.predict(image, conf=args.conf, device="0", verbose=False)[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(model.names[class_id]),
                    "confidence": float(box.conf.item()),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        rendered_path = rendered_dir / f"{name}.jpg"
        Image.fromarray(result.plot()[..., ::-1]).save(rendered_path, "JPEG", quality=92)
        valid_boxes = all(
            0 <= item["x1"] < item["x2"] <= image.width
            and 0 <= item["y1"] < item["y2"] <= image.height
            for item in detections
        )
        valid_classes = all(item["class_name"] in {"fire", "smoke"} for item in detections)
        valid_confidence = all(0 <= item["confidence"] <= 1 for item in detections)
        payload = {
            "source": str(path.resolve()),
            "image_width": image.width,
            "image_height": image.height,
            "device": str(result.boxes.xyxy.device),
            "inference_ms": float(result.speed.get("inference", 0.0)),
            "detections": detections,
            "checks": {
                "bounding_boxes_in_bounds": valid_boxes,
                "class_names_valid": valid_classes,
                "confidence_valid": valid_confidence,
                "result_image_saved": rendered_path.is_file(),
                "input_unchanged": digest(path) == input_hashes[name],
            },
        }
        json_path = args.output / f"{name}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report["cases"][name] = payload

    report["source_images_unchanged"] = all(digest(Path(path)) == value for path, value in source_hashes.items())
    report["all_checks_passed"] = report["source_images_unchanged"] and all(
        all(case["checks"].values()) and case["device"].startswith("cuda")
        for case in report["cases"].values()
    )
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_checks_passed": report["all_checks_passed"],
        "source_images_unchanged": report["source_images_unchanged"],
        "cases": {name: {"size": [case["image_width"], case["image_height"]], "detections": len(case["detections"]), "device": case["device"]} for name, case in report["cases"].items()},
    }, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
