#!/usr/bin/env python
"""Submit VERSION=3 training jobs with user-wide throttling & submit-limit aware retry."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import itertools
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Iterable, Sequence

import yaml


# =========================
# Data classes
# =========================

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


def combo_missing_seeds(
    combo: Combo, seeds: Sequence[SeedEntry], results_dir: Path
) -> list[SeedEntry]:
    """Return the seeds that still require training for *combo*."""

    if not results_dir.exists():
        return list(seeds)

    missing: list[SeedEntry] = []
    for seed in seeds:
        model_path = (
            results_dir
            / seed.name
            / combo.label
            / "model.joblib"
        )
        if not model_path.is_file():
            missing.append(seed)
    return missing


@dataclasses.dataclass
class GpuSubmitContext:
    script: Path
    work_dir: Path
    gpu_count: int | None
    cpus: int | None
    memory: str | None
    time_limit: str | None
    compiler_module: str
    cuda_module: str
    extra_modules: tuple[str, ...]
    poll_interval: int | None
    poll_timeout: int | None


class SubmissionError(RuntimeError):
    pass


# =========================
# Utils
# =========================

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
    except Exception as exc:
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
    except ImportError as exc:
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


def normalise_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        if val in {"0", "false", "no", "off"}:
            return False
    return default


def coerce_optional_int(value, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SubmissionError(f"Invalid integer for {name}: {value!r}") from exc


def parse_gpu_count(gres: str | None) -> int | None:
    if not gres:
        return None
    match = re.search(r"gpu(?::[^:]+)?(?::(\d+))?", gres)
    if not match:
        return None
    count = match.group(1)
    if not count:
        return 1
    try:
        value = int(count)
    except ValueError:
        return None
    return value if value > 0 else None


def normalise_module_config(module_cfg: dict) -> tuple[str, str, tuple[str, ...]]:
    loads = module_cfg.get("load") or []
    compiler = module_cfg.get("compiler") or module_cfg.get("compiler_module")
    if not compiler:
        for mod in loads:
            mod_l = mod.lower()
            if mod_l.startswith("gnu") or mod_l.startswith("gcc"):
                compiler = mod
                break
    if not compiler:
        compiler = "gnu12/12.3.0"

    cuda = module_cfg.get("cuda") or module_cfg.get("cuda_module")
    if not cuda:
        for mod in loads:
            if "cuda" in mod.lower():
                cuda = mod
                break
    if not cuda:
        cuda = "cuda/12.1.1"

    extra = module_cfg.get("extra")
    if extra is None:
        extra_list = [mod for mod in loads if mod not in {compiler, cuda}]
    else:
        extra_list = list(extra)
    return compiler, cuda, tuple(extra_list)


# =========================
# Throttle helpers
# =========================

SUBMIT_LIMIT_PAT = re.compile(
    r"(AssocMaxSubmitJobLimit|job submit limit|violates accounting/QOS policy)", re.IGNORECASE
)

def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def count_queue(user: str) -> int:
    """
    사용자 전체 큐(PENDING+RUNNING) 잡 개수를 squeue로 카운트.
    """
    try:
        # %T=state. 기본 squeue는 PD/R만, -h로 헤더 제거
        cmd = ["squeue", "-u", user, "-h", "-o", "%i"]
        result = run_command(cmd)
        if result.returncode != 0:
            return 0
        # 한 줄당 하나의 잡
        return sum(1 for line in result.stdout.splitlines() if line.strip())
    except FileNotFoundError:
        return 0


def wait_for_queue(user: str, max_queue: int, poll_interval: int) -> None:
    """
    사용자 전체 잡 수가 max_queue 미만이 될 때까지 대기.
    """
    while True:
        queued = count_queue(user)
        if queued < max_queue:
            return
        print(
            f"[throttle] user={user} queue={queued} >= {max_queue}; sleep {poll_interval}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(poll_interval)


# =========================
# Slurm submission
# =========================

def is_submit_limit_error(out: str, err: str) -> bool:
    msg = f"{out}\n{err}"
    return bool(SUBMIT_LIMIT_PAT.search(msg))


def submit_job_once(
    worker_script: Path,
    combo: Combo,
    seed: SeedEntry,
    env: dict,
    gpu_ctx: GpuSubmitContext,
) -> tuple[str | None, bool, str]:
    """
    gpu_submit.sh를 통해 한 번 제출 시도.
    return: (job_id or None, submit_limit_hit, failure_reason)
    """

    if not gpu_ctx.script.exists():
        raise SubmissionError(f"GPU submit script not found: {gpu_ctx.script}")

    job_name = sanitise_job_name(env.get("JOB_LABEL", combo.label))
    cmd: list[str] = [str(gpu_ctx.script), "--job-name", job_name]

    if gpu_ctx.gpu_count:
        cmd.extend(["--gpu-count", str(gpu_ctx.gpu_count)])
    if gpu_ctx.cpus:
        cmd.extend(["--cpus", str(gpu_ctx.cpus)])
    if gpu_ctx.memory:
        cmd.extend(["--mem", gpu_ctx.memory])
    if gpu_ctx.time_limit:
        cmd.extend(["--time", gpu_ctx.time_limit])
    if gpu_ctx.work_dir:
        cmd.extend(["--workdir", str(gpu_ctx.work_dir)])
    if gpu_ctx.compiler_module:
        cmd.extend(["--compiler-module", gpu_ctx.compiler_module])
    if gpu_ctx.cuda_module:
        cmd.extend(["--cuda-module", gpu_ctx.cuda_module])
    for module_name in gpu_ctx.extra_modules:
        if module_name:
            cmd.extend(["--module", module_name])
    if gpu_ctx.poll_interval is not None:
        cmd.extend(["--poll-interval", str(gpu_ctx.poll_interval)])
    if gpu_ctx.poll_timeout is not None:
        cmd.extend(["--poll-timeout", str(gpu_ctx.poll_timeout)])

    cmd.extend([
        "--assay",
        combo.assay,
        "--model",
        combo.model,
        "--mf",
        combo.fingerprint,
        "--label",
        job_name,
        "--seed",
        seed.name,
        "--",
    ])

    env_pairs = [f"{key}={value}" for key, value in sorted((k, str(v)) for k, v in env.items())]
    cmd.append("env")
    cmd.extend(env_pairs)
    cmd.append(str(worker_script))

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (result.stdout or "")
    err = (result.stderr or "")

    if out:
        print(out, end="", flush=True)
    if err:
        print(err, file=sys.stderr, end="", flush=True)

    if result.returncode == 0:
        match = re.search(r"Final allocated job ID: (\S+)", out)
        if match:
            job_id = match.group(1)
            print(f"Submitted {combo.label} as JobID {job_id}", flush=True)
            return job_id, False, ""
        failure_reason = "submit succeeded but job id missing"
        print(f"[error] {failure_reason} for {combo.label}", file=sys.stderr, flush=True)
        return None, False, failure_reason

    if is_submit_limit_error(out, err):
        print(f"[throttle] submit-limit hit for {combo.label}; deferring...", file=sys.stderr, flush=True)
        return None, True, (err or out or "submit-limit")

    failure_reason = (err or out or "unknown error").strip()
    print(f"[error] Failed to submit {combo.label}: {failure_reason}", file=sys.stderr, flush=True)
    return None, False, failure_reason


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
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0 or not out:
        raise SubmissionError(f"Failed to submit summary job: {err or out or 'unknown error'}")
    job_id = out.splitlines()[0]
    print(f"Scheduled summary job {job_id} ({job_name}) with dependency on {len(dependencies)} jobs", flush=True)


def build_env(
    combo: Combo,
    seed: SeedEntry,
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
        "JOB_LABEL": sanitise_job_name(f"{project_dir.name}_{combo.label}_{seed.name}"),
        "TARGET_SEED_NAME": seed.name,
        "TARGET_SEED_DIR": str(seed.path),
    }
    module_cfg = env_cfg.get("module", {})
    env["MODULE_INIT"] = module_cfg.get("init", "")
    env["MODULE_PURGE"] = "1" if module_cfg.get("purge", True) else "0"
    env["ENV_MODULES"] = ",".join(module_cfg.get("load", []))
    #conda_cfg = env_cfg.get("conda", {})
    #env["CONDA_SETUP"] = conda_cfg.get("setup", "")
    #env["CONDA_ENV"] = conda_cfg.get("env", "")
    env["USE_GPU"] = "1" if normalise_bool(env_cfg.get("use_gpu", True), True) else "0"
    if random_state is not None:
        env["RANDOM_STATE"] = str(random_state)
    return env


# =========================
# Main
# =========================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit v3 training jobs (user-wide throttled, submit-limit aware)")
    parser.add_argument("--config", default=Path(__file__).with_name("training_config.yaml"), type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--user", default="won0316", help="User name for squeue counting (default: won0316)")
    parser.add_argument("--max-queue", type=int, default=20, help="Max concurrent jobs for the user (default 20)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds (default 30)")
    parser.add_argument("--post-submit-sleep", type=int, default=3, help="Sleep seconds after successful sbatch (default 3)")
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

    seed_file_dir = logs_dir / "seed_lists"
    seed_file_dir.mkdir(parents=True, exist_ok=True)
    seed_file_map: dict[str, Path] = {}
    for seed in seeds:
        seed_file_path = seed_file_dir / f"{seed.name}.txt"
        write_seed_file([seed], seed_file_path)
        seed_file_map[seed.name] = seed_file_path

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

    pending_tasks: list[tuple[Combo, SeedEntry]] = []
    skipped_combos: list[Combo] = []
    for combo in combos:
        missing_seeds = combo_missing_seeds(combo, seeds, results_dir)
        if not missing_seeds:
            skipped_combos.append(combo)
        else:
            for seed in missing_seeds:
                pending_tasks.append((combo, seed))

    if skipped_combos:
        skipped_labels = ", ".join(c.label for c in skipped_combos[:10])
        more = len(skipped_combos) - 10
        if more > 0:
            skipped_labels += f", ... (+{more} more)"
        print(
            f"[skip] {len(skipped_combos)} combos already have model.joblib artifacts. "
            f"Skipping: {skipped_labels}",
            flush=True,
        )

    if not pending_tasks:
        print("[info] All combos already trained. No submissions required.", flush=True)
        return 0

    # 로그 디렉토리
    worker_script = Path(__file__).with_name("train_combo_worker.sh")
    if not worker_script.exists():
        raise SubmissionError(f"Worker script missing: {worker_script}")

    slurm_cfg = config.get("slurm", {})
    slurm_log_dir = config.get("logging", {}).get("slurm_dir")
    if slurm_log_dir:
        slurm_log_dir = Path(slurm_log_dir).expanduser().resolve()
    else:
        slurm_log_dir = logs_dir / "slurm"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = config.get("environment", {})
    module_cfg = env_cfg.get("module", {})
    compiler_module, cuda_module, extra_modules = normalise_module_config(module_cfg)

    if args.max_queue == parser.get_default("max_queue") and slurm_cfg.get("max_jobs"):
        args.max_queue = int(slurm_cfg["max_jobs"])

    throttle_cfg = slurm_cfg.get("throttle", {})
    if args.max_queue == parser.get_default("max_queue") and throttle_cfg.get("max_queue"):
        args.max_queue = int(throttle_cfg["max_queue"])
    if args.poll_interval == parser.get_default("poll_interval") and throttle_cfg.get("poll_interval_sec"):
        args.poll_interval = int(throttle_cfg["poll_interval_sec"])

    gpu_submit_cfg = slurm_cfg.get("gpu_submit", {})
    poll_interval_raw = gpu_submit_cfg.get("poll_interval")
    if poll_interval_raw is None:
        poll_interval_raw = gpu_submit_cfg.get("poll_interval_sec")
    poll_timeout_raw = gpu_submit_cfg.get("poll_timeout")
    if poll_timeout_raw is None:
        poll_timeout_raw = gpu_submit_cfg.get("poll_timeout_sec")

    gpu_poll_interval = coerce_optional_int(poll_interval_raw, "gpu_submit.poll_interval")
    gpu_poll_timeout = coerce_optional_int(poll_timeout_raw, "gpu_submit.poll_timeout")

    gpu_submit_script = worker_script.with_name("gpu_submit.sh")
    gpu_count = parse_gpu_count(slurm_cfg.get("gres")) or 1
    gpu_ctx = GpuSubmitContext(
        script=gpu_submit_script,
        work_dir=base_dir,
        gpu_count=gpu_count,
        cpus=slurm_cfg.get("cpus_per_task"),
        memory=slurm_cfg.get("mem"),
        time_limit=slurm_cfg.get("time"),
        compiler_module=compiler_module,
        cuda_module=cuda_module,
        extra_modules=extra_modules,
        poll_interval=gpu_poll_interval,
        poll_timeout=gpu_poll_timeout,
    )

    total_to_submit = len(pending_tasks)

    if args.dry_run or training_cfg.get("dry_run", False):
        print("[dry-run] user-wide throttled submission")
        print(
            "  user="
            f"{args.user} seeds={total_to_submit} max_queue={args.max_queue} poll={args.poll_interval}s"
        )
        if skipped_combos:
            print(f"  skipping={len(skipped_combos)} (artifacts exist)")
        for combo, seed in pending_tasks:
            job_label = sanitise_job_name(
                f"{base_dir.name}_{combo.label}_{seed.name}"
            )
            print(f"  - {combo.label} seed={seed.name} JOB_LABEL={job_label}")
        return 0

    job_ids: list[str] = []
    for idx, (combo, seed) in enumerate(pending_tasks, 1):
        # 1) 사용자 전체 큐 길이로 먼저 스로틀
        wait_for_queue(user=args.user, max_queue=args.max_queue, poll_interval=args.poll_interval)

        # 2) 제출 시도; submit-limit이면 대기 후 재시도
        while True:
            seed_file = seed_file_map.get(seed.name)
            if seed_file is None:
                raise SubmissionError(f"Seed file missing for {seed.name}")
            env = build_env(
                combo,
                seed,
                base_dir,
                model_dir,
                run_subdir,
                logs_dir,
                results_dir,
                seed_file,
                env_cfg,
                random_state,
            )
            job_id, submit_limited, reason = submit_job_once(worker_script, combo, seed, env, gpu_ctx)
            if submit_limited:
                # 사용자의 타 잡/어카운트 제한 포함. 일정 시간 대기 후 다시 squeue 기반 스로틀부터 반복.
                time.sleep(args.poll_interval)
                wait_for_queue(user=args.user, max_queue=args.max_queue, poll_interval=args.poll_interval)
                continue
            if job_id is None:
                # 기타 실패: 스킵(필요하면 raise로 바꿔도 됨)
                break
            job_ids.append(job_id)
            # 제출 직후 squeue 반영 지연 완화
            time.sleep(args.post_submit_sleep)
            break

        if idx % 10 == 0 or idx == total_to_submit:
            print(
                f"[progress] Submitted {len(job_ids)}/{total_to_submit} seed jobs (scope=user)",
                flush=True,
            )

    # Summary job
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

    print(
        f"Submitted {len(job_ids)} training seed jobs (throttled user={args.user}, max_queue={args.max_queue}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SubmissionError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
