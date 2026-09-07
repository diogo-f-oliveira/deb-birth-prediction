"""Pure path resolution shared by configs and data loaders."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def portable_path(path):
    if path is None:
        return None
    resolved = resolve_repo_path(path)
    return resolved.relative_to(REPO_ROOT).as_posix() if resolved.is_relative_to(REPO_ROOT) else str(resolved)
