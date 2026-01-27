from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

TARGET_COL: str = "reached_birth"  # boolean label: 1 if birth reached, else 0

ID_COLS: tuple[str, ...] = (
    'generator_species',  # The species used to generate the parameters by adding noise.
    'point_id',  # The identifier of the parameter set generated for the species
)

BIRTH_SIMULATION_COLS: tuple[str, ...] = (
    'lb',  # The scaled length at birth $l_b = L_b / z$ (-).
    'tb',  # The age at birth $t_b$ (d). It is the age at birth divided by $\dot{k}_M$.
    'lb<f',  # Whether the scaled length at birth is less than the functional response (True/False).
    'k*vHb<c',  # Condition for maturity maintenance at birth (True/False).
    'reached_birth',  # Whether the simulated organism reached birth (True/False).
)

LOG_COLS: tuple[str, ...] = (
    'success',  # Whether the DEB model simulation ran successfully to completion (True/False)
    'execution_time',  # The time taken to run the DEB model simulation (seconds)
    'error_message',  # The message describing the error encountered during the DEB model simulation
    'error_type',  # The type of error encountered during the DEB model simulation
)

PARAMETER_COLS: tuple[str, ...] = (
    'z',  # The zoom factor $z$ (-)
    'v',  # The energy conductance $\dot{v}$ (cm/d)
    'kap',  # The allocation fraction to soma $\kappa$ (-)
    'E_G',  # The specific cost for structure $[E_G]$ (J/cm³)
    'p_M',  # The volume-specific somatic maintenance costs $[\dot{p}_M]$ (J/d.cm³)
    'E_Hb',  # The maturity at birth $E_H^b$ (J)
    'k_J',  # The maturity maintenance rate coefficient $\dot{k}_J$ (1/d)
    'f',  # The scaled functional response $f$ (-)
)

COMPOUND_COLS: tuple[str, ...] = (
    'E_m',  # The maximum reserve density $[E_m] = \{\dot{p}_{Am}\} / \dot{v}$ (J/cm³)
    'k_M',  # The somatic maintenance rate coefficient $\dot{k}_M = [\dot{p}_M] / [E_G]$ (1/d)
    'p_Am',  # The maximum specific assimilation rate $\{\dot{p}_{Am}\} = z [\dot{p}_M] / \kappa$ (J/d.cm²)
    'g',  # The energy investment ratio $g = [E_G] / (\kappa [E_m])$ (-)
    'k',  # The maintenance ratio $k = \dot{k}_J / \dot{k}_M$ (-)
    'v_Hb',  # The scaled maturity at birth $v_H^b = E_H^b \dot{k}_M / ( \{\dot{p}_{Am}\} L_m^3)$ (-)
)

DIMENSIONLESS_PARAMETER_COLS: tuple[str, ...] = (
    'kap',
    'f',
)

DIMENSIONLESS_COMPOUND_COLS: tuple[str, ...] = (
    'g',
    'k',
    'v_Hb',
)

# Convenience "named feature sets" used by config files / CLI
FEATURE_SETS: Mapping[str, Sequence[str]] = {
    "dimensionless": DIMENSIONLESS_COMPOUND_COLS + ('f',),
    "dimensionless_plus_kap": DIMENSIONLESS_COMPOUND_COLS + DIMENSIONLESS_PARAMETER_COLS,
    "parameters": PARAMETER_COLS,
    "compound_parameters": COMPOUND_COLS,
    "all": PARAMETER_COLS + COMPOUND_COLS,
}

DTYPES: dict[str, str] = {
    TARGET_COL: "boolean",
    # If you have IDs:
    # "species": "string",
}

# Any column listed here must be numeric if present (helps catch parsing issues).
NUMERIC_COLS: tuple[str, ...] = (
    *PARAMETER_COLS,
    *COMPOUND_COLS,
)


# ----------------------------
# Helpers
# ----------------------------

def get_feature_columns(feature_set: str = "dimensionless") -> list[str]:
    """
    Return the feature column names for a given feature_set key.
    """
    if feature_set not in FEATURE_SETS:
        valid = ", ".join(sorted(FEATURE_SETS.keys()))
        raise ValueError(f"Unknown feature_set='{feature_set}'. Valid: {valid}")
    return list(FEATURE_SETS[feature_set])


def required_columns(feature_set: str = "dimensionless") -> set[str]:
    """
    Columns required to train a model: features + target.
    """
    return set(get_feature_columns(feature_set)) | {TARGET_COL}


def validate_required_columns(df: pd.DataFrame, feature_set: str = "dimensionless") -> None:
    """
    Raise if required columns are missing.
    """
    req = required_columns(feature_set)
    missing = sorted(req - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce known dtypes and numeric columns (best-effort).
    """
    out: pd.DataFrame = df.copy()

    # Coerce target to boolean if present
    if TARGET_COL in out.columns:
        out[TARGET_COL] = out[TARGET_COL].astype("boolean")

    # Coerce numeric columns
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


@dataclass(frozen=True)
class DatasetSpec:
    feature_set: str = "dimensionless"
    target_col: str = TARGET_COL

    @property
    def feature_cols(self) -> list[str]:
        cols = get_feature_columns(self.feature_set)
        return cols

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def required_cols(self) -> set[str]:
        return required_columns(self.feature_set)
