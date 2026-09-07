from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Union

from .paths import REPO_ROOT, resolve_repo_path


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_run_outdir(model_name: str, base: Union[str, Path] = "results/runs") -> Path:
    """Create and return "<base>/<timestamp>_<model_name>".

    - timestamp includes microseconds to distinguish successive small runs.
    - base is repository-relative unless absolute; defaults to results/runs.
    - The directory is created (parents=True).
    """
    base_path = resolve_repo_path(base) if base else REPO_ROOT
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    run_dir = base_path / f"{timestamp}_{model_name}"
    return ensure_outdir(run_dir)


def resolve_run_config(cfg, model_name):
    """Called only when saving a run; the caller's config remains unchanged."""
    from dataclasses import replace
    outdir = cfg.outdir if cfg.outdir is not None else create_run_outdir(model_name)
    outdir = ensure_outdir(resolve_repo_path(outdir))
    return replace(cfg, outdir=outdir)


def save_run_metadata(outdir, cfg, data_metadata=None):
    """Record source identity, dependency versions, and code used for this run."""
    import json
    import platform
    import subprocess
    from hashlib import sha256
    from importlib.metadata import version, PackageNotFoundError
    from .config import config_dict

    metadata = dict(data_metadata or {})
    metadata["config"] = config_dict(cfg)
    metadata["python"] = platform.python_version()
    metadata["dependencies"] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "gplearn"):
        try:
            metadata["dependencies"][package] = version(package)
        except PackageNotFoundError:
            metadata["dependencies"][package] = None
    try:
        metadata["code_revision"] = subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["code_revision"] = None
    # Source hashes also identify uncommitted implementations accurately.
    metadata["source_code_hashes"] = {
        path.relative_to(REPO_ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted((REPO_ROOT / "src/debbirth").rglob("*.py"))}
    (Path(outdir) / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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
