#!/usr/bin/env python3
"""Submit DoA computations in chunks via sbatch.

This script looks for Excel files under ``ToxCast_model/experiments``. For each
file it extracts assay names from the ``data`` sheet starting from the third
column, splits them into groups of at most 10 and submits jobs according to the
GPU scheduling policy described in the documentation.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterable, List

try:
    import openpyxl
except ImportError:  # allow execution even if module is missing during packaging
    openpyxl = None

EXPERIMENTS_DIR = Path("ToxCast_model/experiments")
CHUNK_SIZE = 10
NODE_ORDER = ["gpu6", "gpu1", "gpu2", "gpu3", "gpu4", "gpu5"]


def iter_excel_files() -> Iterable[Path]:
    for path in EXPERIMENTS_DIR.glob("*/*.xlsx"):
        if "results" in path.parts:
            continue
        yield path


def read_assays(xlsx: Path) -> List[str]:
    if openpyxl is None:
        raise RuntimeError("openpyxl is required to parse excel files")
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb["data"]
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    wb.close()
    return [h for h in headers[2:] if h]


def chunks(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def submit_once(node: str, assays: List[str], project: str) -> int | None:
    env = f"ASSAY_NAMES={','.join(assays)},PROJECT_NAME={project}"
    cmd = ["sbatch", f"--nodelist={node}", f"--export={env}", "run_pipeline.sh"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        return None
    for token in res.stdout.split():
        if token.isdigit():
            return int(token)
    return None


def job_state(job_id: int) -> str:
    out = subprocess.check_output(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
    return out.decode().strip()


def wait_until_not_pd(job_id: int, delay: float = 1.0, max_checks: int = 30) -> bool:
    for _ in range(max_checks):
        if job_state(job_id) != "PD":
            return True
        time.sleep(delay)
    return False


def cancel(job_id: int) -> None:
    subprocess.run(["scancel", str(job_id)])


def submit_with_policy(assays: List[str], project: str) -> None:
    for node in NODE_ORDER:
        job = submit_once(node, assays, project)
        if job is None:
            continue
        if wait_until_not_pd(job):
            print(f"job {job} running on {node}")
            return
        cancel(job)
    # fallback: submit to gpu1 and wait
    job = submit_once("gpu1", assays, project)
    while job_state(job) == "PD":
        time.sleep(30)
    print(f"job {job} started on gpu1 after waiting")


def main() -> None:
    for xlsx in iter_excel_files():
        project = xlsx.parent.name
        assays = read_assays(xlsx)
        for chunk in chunks(assays, CHUNK_SIZE):
            submit_with_policy(chunk, project)


if __name__ == "__main__":
    main()
