from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 화염·연기 커스텀 학습")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", type=Path, default=Path("data/fire.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", nargs="?", const=True, default=False)
    parser.add_argument("--project", type=Path, default=Path("outputs/training"))
    parser.add_argument("--name", default="fire-smoke-yolov8n")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data = args.data.resolve()
    args.project = args.project.resolve()
    if not args.data.is_file():
        print(f"데이터 설정이 없습니다: {args.data}. Annotation 검사와 변환을 먼저 실행하세요.")
        return 2
    import torch
    from ultralytics import YOLO

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = "0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    resume = args.resume
    model_source = args.model
    if isinstance(resume, str):
        model_source = resume
        resume = True
    model = YOLO(model_source)
    train_args = {
        "data": str(args.data), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": device, "workers": args.workers, "patience": args.patience, "seed": 42,
        "pretrained": True, "project": str(args.project), "name": args.name, "exist_ok": True,
        "resume": resume, "plots": True, "cache": False, "amp": True,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 5.0,
        "translate": 0.1, "scale": 0.5, "fliplr": 0.5, "mosaic": 1.0, "close_mosaic": 10,
    }
    args.project.mkdir(parents=True, exist_ok=True)
    (args.project / f"{args.name}-settings.json").write_text(json.dumps(train_args, indent=2, default=str), encoding="utf-8")
    try:
        model.train(**train_args)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("CUDA 메모리 부족: --batch 값을 절반으로 낮추고 다시 실행하세요.")
        raise
    # Ultralytics 전역 설정과 관계없이 실제 trainer 저장 경로에서 가중치를 가져온다.
    run_dir = Path(model.trainer.save_dir).resolve() / "weights"
    models_dir = Path("models").resolve()
    models_dir.mkdir(exist_ok=True)
    for name in ("best.pt", "last.pt"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, models_dir / name)
    print(f"완료: {models_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
