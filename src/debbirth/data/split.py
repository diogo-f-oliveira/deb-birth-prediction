from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupSplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_groups: Tuple[str, ...]
    val_groups: Tuple[str, ...]
    test_groups: Tuple[str, ...]


def group_split_by_generator_species(
        df: pd.DataFrame,
        *,
        group_col: str = "generator_species",
        label_col: Optional[str] = "reached_birth",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        n_rate_bins: int = 20,
        stratify_groups: bool = True,
) -> GroupSplitResult:
    """
    Split df into train/val/test so that generator_species groups do not overlap.

    If stratify_groups=True and label_col is provided, groups are binned by their
    within-group positive rate to keep splits roughly balanced at the group level.
    This is useful when some generator species are 'always feasible' or 'never feasible'.

    Returns:
        GroupSplitResult with dataframes and the group lists per split.
    """
    # Validate inputs
    for col in [group_col] + ([label_col] if label_col else []):
        if col is not None and col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df.")

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1. Got {total}.")

    groups = df[group_col].astype(str)
    unique_groups = np.array(sorted(groups.unique()), dtype=str)
    rng = np.random.default_rng(seed)

    # helper function to allocate groups within each stratum ---
    def _allocate(group_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gl = group_list.copy()
        rng.shuffle(gl)
        n = len(gl)

        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # ensure all groups assigned; adjust by remainder
        n_test = n - n_train - n_val
        if n_test < 0:
            # in edge cases due to rounding, fix by moving from train to test
            n_train = n - n_val

        train_g = gl[:n_train]
        val_g = gl[n_train:n_train + n_val]
        test_g = gl[n_train + n_val:]
        return train_g, val_g, test_g

    # Non-stratified group split
    if (not stratify_groups) or (label_col is None):
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train_groups = shuffled[:n_train]
        val_groups = shuffled[n_train:n_train + n_val]
        test_groups = shuffled[n_train + n_val:]

    # Stratified group split
    else:
        # Compute per-group positive rate
        tmp = df[[group_col, label_col]].copy()
        tmp[group_col] = tmp[group_col].astype(str)

        grp = tmp.groupby(group_col)[label_col]
        grp_count = grp.size().rename("n")
        grp_pos = grp.sum().rename("pos")
        grp_rate = (grp_pos / grp_count).rename("rate")

        stats = pd.concat([grp_count, grp_pos, grp_rate], axis=1).reset_index()
        stats[group_col] = stats[group_col].astype(str)

        # Create strata:
        # - exact 0.0 and 1.0 rates are their own strata
        # - remaining rates binned into quantiles
        stats["stratum"] = "mid"
        stats.loc[stats["rate"] <= 0.0 + 1e-12, "stratum"] = "all_zero"
        stats.loc[stats["rate"] >= 1.0 - 1e-12, "stratum"] = "all_one"

        mid_mask = stats["stratum"].eq("mid")
        mid_rates = stats.loc[mid_mask, "rate"]

        if len(mid_rates) > 0:
            # qcut can drop bins if many duplicates; handle gracefully
            try:
                bins = pd.qcut(mid_rates, q=min(n_rate_bins, len(mid_rates)), duplicates="drop")
                stats.loc[mid_mask, "stratum"] = bins.astype(str).values
            except ValueError:
                # fallback: single bin
                stats.loc[mid_mask, "stratum"] = "mid_single"

        # Allocate groups per stratum and then merge
        train_groups_list = []
        val_groups_list = []
        test_groups_list = []

        for _, sub in stats.groupby("stratum"):
            g = sub[group_col].to_numpy(dtype=str)
            tr, va, te = _allocate(g)
            train_groups_list.append(tr)
            val_groups_list.append(va)
            test_groups_list.append(te)

        train_groups = np.concatenate(train_groups_list) if train_groups_list else np.array([], dtype=str)
        val_groups = np.concatenate(val_groups_list) if val_groups_list else np.array([], dtype=str)
        test_groups = np.concatenate(test_groups_list) if test_groups_list else np.array([], dtype=str)

        # Final shuffle inside each split for aesthetics (not required)
        rng.shuffle(train_groups)
        rng.shuffle(val_groups)
        rng.shuffle(test_groups)

        # Guard: ensure disjointness (should always hold)
        inter_tv = set(train_groups).intersection(val_groups)
        inter_tt = set(train_groups).intersection(test_groups)
        inter_vt = set(val_groups).intersection(test_groups)
        if inter_tv or inter_tt or inter_vt:
            raise RuntimeError("Group overlap detected between splits (unexpected).")

    # Build splits
    train_mask = groups.isin(train_groups)
    val_mask = groups.isin(val_groups)
    test_mask = groups.isin(test_groups)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    return GroupSplitResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_groups=tuple(map(str, train_groups.tolist())),
        val_groups=tuple(map(str, val_groups.tolist())),
        test_groups=tuple(map(str, test_groups.tolist())),
    )
