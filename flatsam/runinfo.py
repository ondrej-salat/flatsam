"""Run-environment record (GPU, node, Slurm job, software) stored next to timing results.

Used by the weekend full-run scripts so every per-sequence timing file says what hardware it was measured on.
"""
import datetime
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _nvidia_smi(field: str) -> Optional[str]:
    """First GPU's value of an ``nvidia-smi --query-gpu`` field, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20,
        ).stdout.strip().splitlines()
        return out[0].strip() if out else None
    except Exception:
        return None


def code_commit(repo_root: Path) -> str:
    """Commit of the code that ran: ``WEEKEND_COMMIT`` env, a ``GIT_COMMIT`` file in the repo root, or ``git rev-parse``."""
    if os.environ.get("WEEKEND_COMMIT"):
        return os.environ["WEEKEND_COMMIT"]
    f = repo_root / "GIT_COMMIT"
    if f.is_file():
        return f.read_text().strip()
    try:
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Hardware and job description of the current process (call after CUDA is usable)."""
    cuda = torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0) if cuda else None
    return {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "user": os.environ.get("USER"),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gb": round(props.total_memory / 1024 ** 3, 1) if props else None,
        "gpu_count_visible": torch.cuda.device_count() if cuda else 0,
        "gpu_driver": _nvidia_smi("driver_version"),
        "gpu_clock_max_mhz": _nvidia_smi("clocks.max.sm"),
        "cpus_allocated": os.environ.get("SLURM_CPUS_PER_TASK"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "code_commit": code_commit(repo_root) if repo_root else None,
    }


def sections(profiler) -> Dict[str, Dict[str, float]]:
    """Profiler sections as ``{path: {calls, total_s}}`` (insertion order)."""
    return {p: {"calls": profiler._stats[p].n_calls, "total_s": round(profiler._stats[p].total_s, 4)} for p in profiler._order}
