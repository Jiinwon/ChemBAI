#!/usr/bin/env python
"""Submit VERSION=3 training jobs based on ``training_config.yaml``."""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import importlib.util
import itertools
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import yaml


@dataclasses.dataclass
class SeedEntry:
    name: str
    path: Path


@dataclasses.dataclass
class Combo:
    assay: str
    model: str
    fingerprint: str

    @property
    def label(self) -> str:
        return f"{self.assay}_{self.model}_{self.fingerprint}"


class SubmissionError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config_module(path: Path):
    spec = importlib.util.spec_from_file_location("toxcast_config", path)
    if spec is None or spec.loader is None:
        raise SubmissionError(f"Unable to import config module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_training_v3(config_module) -> None:
    objects = getattr(config_module, "OBJECTS", None)
    object_idx = getattr(config_module, "OBJECT", None)
    if not objects or object_idx is None:
        raise SubmissionError("config.py must define OBJECTS and OBJECT")
    try:
        mode = objects[object_idx]
    except Exception as exc:  # pragma: no cover - defensive
        raise SubmissionError("Unable to resolve OBJECT from config.py") from exc
    if mode != "training":
        raise SubmissionError(
            f"config.py specifies OBJECT={object_idx} -> {mode!r}. Training submissions require 'training'."
        )
    version = getattr(config_module, "VERSION", None)
    if version != 3:
        raise SubmissionError(
            f"VERSION={version!r} in config.py. This launcher only supports VERSION=3."
        )


def resolve_path(base: Path | None, default: Path) -> Path:
    return (base if base else default).expanduser().resolve()


def discover_seeds(data_dir: Path, include: Sequence[str], pattern: str) -> list[SeedEntry]:
    if include:
        seeds = []
        for name in include:
            seed_path = data_dir / name
            if not seed_path.is_dir():
                raise SubmissionError(f"Configured seed directory missing: {seed_path}")
            seeds.append(SeedEntry(name=name, path=seed_path.resolve()))
        return seeds

    seed_paths = sorted(p for p in data_dir.glob(pattern) if p.is_dir())
    if not seed_paths:
        raise SubmissionError(f"No seed directories found in {data_dir} (pattern={pattern!r})")
    return [SeedEntry(name=p.name, path=p.resolve()) for p in seed_paths]


def resolve_seed_train_csv(seed: SeedEntry) -> Path:
    candidates = [
        seed.path / "train_df.csv",
        seed.path / "train" / "train_df.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise SubmissionError(f"Missing train_df.csv for seed {seed.name} ({seed.path})")


def discover_assays(train_csv: Path, model_dir: Path) -> list[str]:
    sys.path.insert(0, str(model_dir))
    try:
        from toxcast_pkg.v3_data import get_assay_names_from_csv
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SubmissionError(
            "toxcast_pkg.v3_data could not be imported. Ensure PYTHONPATH includes the model directory."
        ) from exc
    assay_names = list(get_assay_names_from_csv(str(train_csv)))
    if not assay_names:
        raise SubmissionError(f"No assays discovered in {train_csv}")
    return assay_names


def sanitise_job_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:200]


def write_seed_file(seeds: Iterable[SeedEntry], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for seed in seeds:
            fh.write(f"{seed.path}|{seed.name}\n")


def count_queue(user: str, max_queue: int | None) -> int:
    if max_queue is None:
        return 0
    try:
        cmd = ["squeue", "-u", user, "-h"]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return 0
    if result.returncode != 0:
        return 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return len(lines)


def wait_for_queue(throttle_cfg: dict) -> None:
    max_queue = throttle_cfg.get("max_queue")
    if not max_queue:
        return
    poll = throttle_cfg.get("poll_interval_sec", 30)
    user = getpass.getuser()
    while True:
        queued = count_queue(user, max_queue)
        if queued < max_queue:
            return
        print(
            f"[throttle] Current queue length {queued} exceeds limit {max_queue}. Sleeping {poll}s...",
            file=sys.stderr,
        )
        time.sleep(poll)


def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def submit_job(
    worker_script: Path,
    combo: Combo,
    env: dict,
    slurm_cfg: dict,
    log_dir: Path,
) -> str:
    export_parts = ["ALL"]
    for key, value in env.items():
        export_parts.append(f"{key}={value}")
    export_arg = ",".join(export_parts)

    partitions = slurm_cfg.get("partitions") or [None]
    gres = slurm_cfg.get("gres")
    cpus = slurm_cfg.get("cpus_per_task")
    mem = slurm_cfg.get("mem")
    time_limit = slurm_cfg.get("time")

    job_name = sanitise_job_name(env.get("JOB_LABEL", combo.label))
    stdout_path = log_dir / f"{job_name}_%j.out"
    stderr_path = log_dir / f"{job_name}_%j.err"

    for idx, partition in enumerate(partitions):
        cmd = ["sbatch", "--parsable"]
        if partition:
            cmd.append(f"--partition={partition}")
        if gres:
            cmd.append(f"--gres={gres}")
        if cpus:
            cmd.append(f"--cpus-per-task={cpus}")
        if mem:
            cmd.append(f"--mem={mem}")
        if time_limit:
            cmd.append(f"--time={time_limit}")
        cmd.extend(
            [
                f"--job-name={job_name}",
                f"--output={stdout_path}",
                f"--error={stderr_path}",
                f"--export={export_arg}",
                str(worker_script),
            ]
        )
        result = run_command(cmd)
        if result.returncode == 0 and result.stdout.strip():
            job_id = result.stdout.strip().splitlines()[0]
            print(f"Submitted {combo.label} on partition {partition or 'default'} as JobID {job_id}")
            return job_id

        stderr = result.stderr.strip()
        failure_reason = stderr or result.stdout.strip() or "unknown error"
        print(
            f"Failed to submit {combo.label} on partition {partition or 'default'}: {failure_reason}",
            file=sys.stderr,
        )
        if idx == len(partitions) - 1:
            raise SubmissionError(f"All partition submissions failed for {combo.label}")
        time.sleep(2)

    raise SubmissionError(f"Unable to submit job for {combo.label}")


def submit_summary_job(
    script_dir: Path,
    project_dir: Path,
    model_dir: Path,
    run_subdir: str,
    base_model_dir: Path,
    slurm_cfg: dict,
    summary_cfg: dict,
    log_dir: Path,
    dependencies: Sequence[str],
) -> None:
    if not summary_cfg.get("enabled", False):
        return

    summary_script = script_dir / summary_cfg.get("script", "run_training.sh")
    if not summary_script.exists():
        raise SubmissionError(f"Summary script not found: {summary_script}")

    partition = summary_cfg.get("partition") or (slurm_cfg.get("partitions") or [None])[0]
    gres = summary_cfg.get("gres") or slurm_cfg.get("gres")
    cpus = summary_cfg.get("cpus_per_task") or slurm_cfg.get("cpus_per_task")
    mem = summary_cfg.get("mem") or slurm_cfg.get("mem")
    time_limit = summary_cfg.get("time") or slurm_cfg.get("time")
    dependency_type = "afterany" if summary_cfg.get("depends_on_fail") else "afterok"

    export_arg = ",".join(
        [
            "ALL",
            f"SLURM_JOB_MODE=summary",
            f"PROJECT_DIR={project_dir}",
            f"MODEL_DIR={model_dir}",
            f"RUN_SUBDIR={run_subdir}",
            f"BASE_MODEL_DIR={base_model_dir}",
        ]
    )

    job_name = sanitise_job_name(f"summary_{project_dir.name}")
    stdout_path = log_dir / f"{job_name}_%j.out"
    stderr_path = log_dir / f"{job_name}_%j.err"

    dependency = f"{dependency_type}:" + ":".join(dependencies)
    cmd = ["sbatch", "--parsable", f"--dependency={dependency}"]
    if partition:
        cmd.append(f"--partition={partition}")
    if gres:
        cmd.append(f"--gres={gres}")
    if cpus:
        cmd.append(f"--cpus-per-task={cpus}")
    if mem:
        cmd.append(f"--mem={mem}")
    if time_limit:
        cmd.append(f"--time={time_limit}")
    cmd.extend(
        [
            f"--job-name={job_name}",
            f"--output={stdout_path}",
            f"--error={stderr_path}",
            f"--export={export_arg}",
            str(summary_script),
            str(project_dir),
        ]
    )

    result = run_command(cmd)
    if result.returncode != 0 or not result.stdout.strip():
        stderr = result.stderr.strip()
        failure_reason = stderr or result.stdout.strip() or "unknown error"
        raise SubmissionError(f"Failed to submit summary job: {failure_reason}")

    job_id = result.stdout.strip().splitlines()[0]
    print(f"Scheduled summary job {job_id} ({job_name}) with dependency on {len(dependencies)} jobs")


def build_env(
    combo: Combo,
    project_dir: Path,
    model_dir: Path,
    run_subdir: str,
    logs_dir: Path,
    results_dir: Path,
    seed_file: Path,
    env_cfg: dict,
    random_state: int | None,
) -> dict:
    env = {
        "PROJECT_DIR": str(project_dir),
        "MODEL_DIR": str(model_dir),
        "RUN_SUBDIR": run_subdir,
        "ASSAY_NAME": combo.assay,
        "MODEL_NAME": combo.model,
        "FINGERPRINT_NAME": combo.fingerprint,
        "SEED_FILE": str(seed_file),
        "LOGS_DIR": str(logs_dir),
        "RESULTS_DIR": str(results_dir),
        "PYTHON_BIN": env_cfg.get("python", "python"),
        "PYTHONPATH_BASE": str(model_dir),
        "JOB_LABEL": sanitise_job_name(f"{project_dir.name}_{combo.label}"),
    }
    module_cfg = env_cfg.get("module", {})
    env["MODULE_INIT"] = module_cfg.get("init", "")
    env["MODULE_PURGE"] = "1" if module_cfg.get("purge", True) else "0"
    env["ENV_MODULES"] = ",".join(module_cfg.get("load", []))
    conda_cfg = env_cfg.get("conda", {})
    env["CONDA_SETUP"] = conda_cfg.get("setup", "")
    env["CONDA_ENV"] = conda_cfg.get("env", "")
    if random_state is not None:
        env["RANDOM_STATE"] = str(random_state)
    return env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit v3 training jobs from YAML configuration")
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("training_config.yaml"),
        type=Path,
        help="Path to training configuration YAML file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without calling sbatch")
    args = parser.parse_args(argv)

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.exists():
        raise SubmissionError(f"Configuration file not found: {cfg_path}")

    config = load_yaml(cfg_path) or {}

    project_cfg = config.get("project", {})
    config_path = Path(project_cfg.get("config_path", "../ToxCast_model/config.py")).expanduser()
    if not config_path.is_file():
        raise SubmissionError(f"config.py not found at {config_path}")

    config_module = load_config_module(config_path)
    ensure_training_v3(config_module)

    base_dir = resolve_path(project_cfg.get("base_dir"), getattr(config_module, "BASE_DIR"))
    data_dir = resolve_path(project_cfg.get("data_dir"), getattr(config_module, "DATA_DIR"))
    logs_dir = resolve_path(project_cfg.get("logs_dir"), getattr(config_module, "LOGS_DIR", base_dir / "logs"))
    results_dir = resolve_path(project_cfg.get("results_dir"), getattr(config_module, "RESULTS_DIR", base_dir / "results"))

    seed_cfg = project_cfg.get("seeds", {})
    include_seeds = seed_cfg.get("include", []) or []
    pattern = seed_cfg.get("pattern", "seed_*")
    seeds = discover_seeds(data_dir, include_seeds, pattern)

    worker_seed_file = logs_dir / "seed_list.txt"
    worker_seed_file.parent.mkdir(parents=True, exist_ok=True)
    write_seed_file(seeds, worker_seed_file)

    model_dir = Path(getattr(config_module, "ROOT_DIR"))
    run_subdir = config.get("training", {}).get("run_subdir", "run_v3")

    assays_cfg = config.get("assays", {})
    assays = assays_cfg.get("include") or []
    if not assays:
        first_train_csv = resolve_seed_train_csv(seeds[0])
        assays = discover_assays(first_train_csv, model_dir)

    training_cfg = config.get("training", {})
    models = training_cfg.get("models") or list(getattr(config_module, "MODELS"))
    fingerprints = training_cfg.get("fingerprints") or list(getattr(config_module, "FINGERPRINTS"))
    random_state = training_cfg.get("random_state")

    combos = [Combo(assay=a, model=m, fingerprint=f) for a, m, f in itertools.product(assays, models, fingerprints)]

    slurm_cfg = config.get("slurm", {})
    max_jobs = slurm_cfg.get("max_jobs")
    if max_jobs and len(combos) > max_jobs:
        raise SubmissionError(
            f"Configuration would submit {len(combos)} jobs which exceeds max_jobs={max_jobs}. "
            "Adjust the assay/model/fingerprint lists or increase the limit."
        )

    worker_script = Path(__file__).with_name("train_combo_worker.sh")
    if not worker_script.exists():
        raise SubmissionError(f"Worker script missing: {worker_script}")

    slurm_log_dir = config.get("logging", {}).get("slurm_dir")
    if slurm_log_dir:
        slurm_log_dir = Path(slurm_log_dir).expanduser().resolve()
    else:
        slurm_log_dir = logs_dir / "slurm"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = config.get("environment", {})

    if args.dry_run or training_cfg.get("dry_run", False):
        print("[dry-run] The following jobs would be submitted:")
        for combo in combos:
            env = build_env(combo, base_dir, model_dir, run_subdir, logs_dir, results_dir, worker_seed_file, env_cfg, random_state)
            print(f"  - {combo.label} -> sbatch with env {env}")
        return 0

    throttle_cfg = slurm_cfg.get("throttle", {})
    job_ids: list[str] = []
    for combo in combos:
        wait_for_queue(throttle_cfg)
        env = build_env(combo, base_dir, model_dir, run_subdir, logs_dir, results_dir, worker_seed_file, env_cfg, random_state)
        job_id = submit_job(worker_script, combo, env, slurm_cfg, slurm_log_dir)
        job_ids.append(job_id)

    summary_cfg = training_cfg.get("summary", {})
    if job_ids:
        submit_summary_job(
            script_dir=worker_script.parent,
            project_dir=base_dir,
            model_dir=model_dir,
            run_subdir=run_subdir,
            base_model_dir=model_dir,
            slurm_cfg=slurm_cfg,
            summary_cfg=summary_cfg,
            log_dir=slurm_log_dir,
            dependencies=job_ids,
        )

    print(f"Submitted {len(job_ids)} training jobs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SubmissionError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
