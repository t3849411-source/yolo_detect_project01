from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


SLUG = "dataclusterlabs/fire-and-smoke-dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle Fire and Smoke Dataset 다운로드")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", default=SLUG, help="Kaggle owner/dataset slug")
    args = parser.parse_args()

    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.is_file() and not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        print("Kaggle 인증이 없습니다. ~/.kaggle/kaggle.json을 만들고 chmod 600을 적용하세요.")
        return 2
    kaggle_executable = shutil.which("kaggle")
    sibling_cli = Path(sys.executable).with_name("kaggle")
    if kaggle_executable is None and sibling_cli.is_file():
        kaggle_executable = str(sibling_cli)
    if kaggle_executable is None:
        print("kaggle 명령을 찾을 수 없습니다. requirements.txt를 설치하세요.")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    command = [kaggle_executable, "datasets", "download", "-d", args.dataset, "-p", str(args.output), "--unzip"]
    print("실행:", " ".join(command))
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
