#!/usr/bin/env python3
"""Build a standalone executable for the GUI using PyInstaller."""
from __future__ import annotations


import os

import subprocess
from pathlib import Path

ENTRY = "run_local_gui.py"
DIST = Path("Release")

APP_NAME = "ChemBAI_Predictor"



def main() -> None:
    DIST.mkdir(exist_ok=True)
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",

        "--windowed",
        "--name",
        APP_NAME,
        "--distpath",
        str(DIST),
        "--add-data",
        "Template{}Template".format(os.pathsep),
        "--add-data",
        "ToxCast_model{}ToxCast_model".format(os.pathsep),

        ENTRY,
    ]
    subprocess.check_call(cmd)
    print(f"Executable written to {DIST}")


if __name__ == "__main__":
    main()
