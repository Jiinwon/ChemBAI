#!/usr/bin/env python3
"""Convenience wrapper for running predictions locally.

This script mirrors the behaviour of ``run_pipeline.sh`` but does not
require a Bash environment. Use it to download the input template and
run predictions on a local machine.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if getattr(sys, "frozen", False):  # support PyInstaller
    REPO_DIR = Path(sys.executable).resolve().parent
else:
    REPO_DIR = Path(__file__).resolve().parent
MODEL_DIR = REPO_DIR / "ToxCast_model"
TEMPLATE_FILE = (
    REPO_DIR / "Template" / "template_for_predict(PROJECT_NAME)" / "prediction_input_template.xlsx"
)

def download_template(out_dir: str) -> None:
    """Copy the prediction template to ``out_dir``."""
    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / TEMPLATE_FILE.name
    shutil.copy(TEMPLATE_FILE, target)
    print(f"Template copied to {target}")

def run_prediction() -> None:
    """Generate fingerprints and run prediction as configured."""
    os.chdir(MODEL_DIR)
    import config

    config.validate_paths()
    subprocess.run([sys.executable, "-m", "toxcast_pkg.smiles2fing"], check=True)
    mode = config.OBJECTS[config.OBJECT]
    print(f"Running mode: {mode}")
    if mode == "training":
        subprocess.run(["bash", "ToxCast_model_training.sh"], check=True)
    elif mode == "prediction":
        subprocess.run([sys.executable, "-m", "prediction.Predict_data"], check=True)
    else:
        raise RuntimeError(f"Unknown mode: {mode}")

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local helper for predictions")
    sub = parser.add_subparsers(dest="command")
    dl = sub.add_parser("download-template", help="Copy input template")
    dl.add_argument("--out", default=".", help="Destination directory")
    sub.add_parser("predict", help="Run prediction pipeline")
    args = parser.parse_args(argv)

    if args.command == "download-template":
        download_template(args.out)
    elif args.command == "predict":
        run_prediction()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

