"""Keep the pipeline a polite guest on a shared laptop: RAM guard + low priority."""

from __future__ import annotations

import os
import time

import psutil


def lower_priority() -> None:
    """Run this process below normal priority so interactive apps stay responsive."""
    try:
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:
        pass


def wait_for_ram(min_free_gb: float = None, poll_s: float = 5.0, verbose: bool = True) -> None:
    """Block until at least ``min_free_gb`` of RAM is available (never crash the machine)."""
    need = float(min_free_gb if min_free_gb is not None else os.environ.get("ER_MIN_FREE_GB", "2.5"))
    warned = False
    while psutil.virtual_memory().available / 2**30 < need:
        if verbose and not warned:
            print(f"  [ram-guard] < {need:.1f} GB free - pausing until memory frees up", flush=True)
            warned = True
        time.sleep(poll_s)


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 2**30
