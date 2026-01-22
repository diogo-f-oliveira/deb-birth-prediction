from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Union


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_run_outdir(model_name: str, base: Union[str, Path] = ".") -> Path:
    """Create and return a run directory named "<timestamp>_<model_name>" under "<base>/results/runs".

    - timestamp is generated from now and formatted as "%Y-%m-%dT%H-%M-%S" (safe for filenames).
    - base may be a string or Path; defaults to current directory.
    - The directory is created (parents=True).
    """
    base_path = Path(base) if base else Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = base_path / "results" / "runs" / f"{timestamp}_{model_name}"
    return ensure_outdir(run_dir)
