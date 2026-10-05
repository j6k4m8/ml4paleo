"""
Work out what this machine can run: CPUs, memory, and GPUs.
"""

import os
import pathlib
import subprocess

from ml4paleo.protocol import WorkerCaps

from . import __version__
from .handlers import HANDLERS


def gpu_memory_gb() -> list[float]:
    """
    Return the memory of each NVIDIA GPU, in GB, or [] if there are none (or
    `nvidia-smi` isn't available).
    """
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    sizes = []
    for line in output.splitlines():
        try:
            sizes.append(round(float(line.strip()) / 1024, 1))  # MiB to GiB
        except ValueError:
            continue
    return sizes


def _cpus() -> int:
    # The CPUs this process may use, where the platform can tell (Linux).
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


# Where a container's memory limit shows up (cgroup v2, then v1).
CGROUP_MEMORY_LIMITS = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)


def _memory_gb() -> float:
    """
    The memory this process may use: the machine's, or its container's limit
    if that is lower.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        total = 0
    for path in CGROUP_MEMORY_LIMITS:
        try:
            limit = int(pathlib.Path(path).read_text().strip())
        except (OSError, ValueError):  # missing, or "max" (no limit)
            continue
        total = min(total, limit) if total else limit
    return round(total / 1024**3, 1)


def detect(labels: list[str] | None = None, slots: int = 1) -> WorkerCaps:
    gpus = gpu_memory_gb()
    all_labels = set(labels or [])
    if gpus:
        all_labels.add("gpu")
    return WorkerCaps(
        version=__version__,
        kinds=sorted(HANDLERS),
        labels=sorted(all_labels),
        vram_gb=max(gpus, default=0),
        cpus=_cpus(),
        memory_gb=_memory_gb(),
        slots=slots,
    )
