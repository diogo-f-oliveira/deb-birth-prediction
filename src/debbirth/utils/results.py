from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Union


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_run_outdir(model_name: str, base: Union[str, Path] = "results/runs") -> Path:
    """Create and return a run directory named "<timestamp>_<model_name>" under "<base>/results/runs".

    - timestamp is generated from now and formatted as "%Y-%m-%dT%H-%M-%S" (safe for filenames).
    - base may be a string or Path; defaults to current directory.
    - The directory is created (parents=True).
    """
    base_path = Path(base) if base else Path.cwd()
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = base_path / f"{timestamp}_{model_name}"
    return ensure_outdir(run_dir)


def save_figure_in_formats(
        fig,
        outdir: Union[str, Path],
        filename: str,
        formats: list[str] = ("png", "pdf"),
        dpi: int = 300,
        verbose: bool = True,
) -> None:
    """Save a matplotlib figure in multiple formats.

    Args:
        fig: matplotlib Figure object.
        outdir: directory or Path where files will be saved.
        filename: base filename (without extension).
        formats: list/tuple of formats (extensions) to save, e.g. ("png", "pdf").
        dpi: resolution in dots per inch for raster formats (default 300).
        verbose: if True, print the path of each saved file after saving.
    """
    outpath = Path(outdir) / filename
    for fmt in formats:
        saved_path = outpath.with_suffix(f".{fmt}")
        fig.savefig(saved_path, dpi=dpi)
        if verbose:
            try:
                print(f"Saved figure: {saved_path.resolve()}")
            except Exception:
                # fallback to printing the relative path if resolve() fails
                print(f"Saved figure: {saved_path}")
