"""NumPy/pandas preparation shared by families; no training dependencies."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import DatasetSpec, FULL_PAR_COLS

SOURCE_FILE = "__source_file"
SOURCE_ROW = "__source_row"


def validate_numeric(values, columns, *, positive=False):
    values = np.asarray(values, dtype=float)
    bad = ~np.isfinite(values)
    if positive:
        bad |= values <= 0
    if bad.any():
        locations = [(int(row), str(columns[col])) for row, col in np.argwhere(bad)[:10]]
        domain = "finite positive" if positive else "finite"
        raise ValueError(f"Expected {domain} inputs; invalid (row position, column): {locations}")


def normalized_parameters(frame: pd.DataFrame, *, include_nu_b=True) -> pd.DataFrame:
    """Transform physical parameters without dropping/clipping rows.

    Boundary callers need only log_nu_b, not an exponentiated extreme maturity.
    Computing nu_b in log space avoids intermediate overflow of f**3.
    """
    missing = sorted(set(FULL_PAR_COLS) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing physical parameters: {missing}")
    physical = frame.loc[:, list(FULL_PAR_COLS)].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    validate_numeric(physical, FULL_PAR_COLS, positive=True)
    g, k, v_Hb, f = physical.T
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        gamma = g / f
        log_nu_b = np.log(v_Hb) - 3 * np.log(f)
        out = pd.DataFrame({"gamma": gamma, "k": k, "log_nu_b": log_nu_b}, index=frame.index)
        if include_nu_b:
            out["nu_b"] = np.exp(log_nu_b)
    positive_cols = ["gamma", "k"] + (["nu_b"] if include_nu_b else [])
    validate_numeric(out[positive_cols], positive_cols, positive=True)
    validate_numeric(out[["log_nu_b"]], ["log_nu_b"])
    return out


def prepare_features(frame: pd.DataFrame, spec: DatasetSpec):
    """Return (learned feature frame, separate log maturity or None).

    Accepts physical columns, never already-normalized input. No learned
    standardization is applied here.
    """
    if spec.formulation == "full_par":
        features = frame.loc[:, spec.feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        validate_numeric(features, spec.feature_cols)
        return features, None
    normalized = normalized_parameters(frame, include_nu_b=spec.formulation == "normalized")
    if spec.include_x_b:
        normalized["x_b"] = normalized.gamma / (1.0 + normalized.gamma)
    features = normalized.loc[:, spec.feature_cols].copy()
    validate_numeric(features, spec.feature_cols, positive=True)
    offset = normalized.log_nu_b.to_numpy(copy=True) if spec.formulation == "boundary" else None
    return features, offset


@dataclass(frozen=True)
class PreparedSplit:
    features: np.ndarray
    feature_names: tuple[str, ...]
    labels: np.ndarray
    source_rows: pd.DataFrame
    diagnostics: pd.DataFrame
    log_nu_b: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.features)
        if self.features.shape != (n, len(self.feature_names)) or self.labels.shape != (n,):
            raise ValueError("Prepared features/labels have inconsistent shapes.")
        if len(self.source_rows) != n or len(self.diagnostics) != n:
            raise ValueError("Prepared provenance/diagnostics must align with features.")
        if self.log_nu_b is not None and self.log_nu_b.shape != (n,):
            raise ValueError("Prepared log_nu_b must have one value per row.")

    def select(self, selection):
        """Apply positional indices, a slice, or a boolean mask to all fields."""
        indices = np.atleast_1d(np.arange(len(self.features))[selection])
        return PreparedSplit(
            self.features[indices], self.feature_names, self.labels[indices],
            self.source_rows.iloc[indices].reset_index(drop=True),
            self.diagnostics.iloc[indices].reset_index(drop=True),
            None if self.log_nu_b is None else self.log_nu_b[indices],
        )


def prepare_split(frame: pd.DataFrame, spec: DatasetSpec, *, source_name="in_memory") -> PreparedSplit:
    missing = sorted(spec.required_cols - set(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    features, offset = prepare_features(frame, spec)
    labels = frame[spec.target_col]
    if labels.isna().any() or not labels.isin([0, 1, False, True]).all():
        raise ValueError(f"{spec.target_col} must contain nonmissing binary labels (0/1 or boolean).")
    has_source = SOURCE_FILE in frame and SOURCE_ROW in frame
    source_rows = (frame[[SOURCE_FILE, SOURCE_ROW]].copy() if has_source else
                   pd.DataFrame({SOURCE_FILE: source_name, SOURCE_ROW: np.arange(len(frame))}))
    diagnostics = frame.drop(columns=list(set(spec.required_cols) | {SOURCE_FILE, SOURCE_ROW}), errors="ignore")
    return PreparedSplit(features.to_numpy(copy=True), tuple(spec.feature_cols),
                         labels.to_numpy(dtype=int), source_rows.reset_index(drop=True),
                         diagnostics.reset_index(drop=True), offset)
