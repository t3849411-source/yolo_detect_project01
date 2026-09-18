from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    if not args.model.is_file():
        print(f"모델 파일이 없습니다: {args.model}")
        return 2
    from ultralytics import YOLO

    YOLO(str(args.model)).predict(source=str(args.source), conf=args.conf, save=True, project="outputs", name="predictions", exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
