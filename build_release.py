#!/usr/bin/env python3
"""Build a standalone executable for the GUI using PyInstaller."""
from __future__ import annotations

import subprocess
from pathlib import Path

ENTRY = "run_local_gui.py"
DIST = Path("Release")


def main() -> None:
    DIST.mkdir(exist_ok=True)
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--distpath",
        str(DIST),
        ENTRY,
    ]
    subprocess.check_call(cmd)
    print(f"Executable written to {DIST}")


if __name__ == "__main__":
    main()
