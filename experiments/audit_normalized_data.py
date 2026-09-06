"""Read-only T02 data audit; writes derived summaries/figures, never input CSVs.

Run from the repository root in conda debbirth:
    python experiments/audit_normalized_data.py
This is an exploratory audit, not the production formulation implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RAW_NAME = "sample4D_lhs_N_200000_g_1em3_1e2_k_m4_1_f_0p1_1_ddec_1_getlb2.csv"
PARAMS = ["g", "k", "v_Hb", "f"]
NORMALIZED = ["gamma", "k", "nu_b"]
BOOLS = ["reached_birth", "success", "lb<f", "k*vHb<c"]


def load(path):
    df = pd.read_csv(path, float_precision="round_trip")
    for col in BOOLS:
        converted = df[col].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if converted.isna().any():
            raise ValueError(f"Unexpected/missing boolean in {path}: {col}")
        df[col] = converted.astype(bool)
    return df


def positive_finite(df):
    x = df[PARAMS].to_numpy(float)
    return np.isfinite(x).all(axis=1) & (x > 0).all(axis=1)


def normalize(df):
    result = df.copy()
    valid = positive_finite(df)
    result[["gamma", "nu_b", "log_nu_b"]] = np.nan
    x = df.loc[valid]
    result.loc[valid, "gamma"] = x.g / x.f
    result.loc[valid, "nu_b"] = x.v_Hb / x.f**3
    result.loc[valid, "log_nu_b"] = np.log(x.v_Hb) - 3*np.log(x.f)
    return result


def duplicate_summary(df, columns):
    grouped = df.groupby(columns, dropna=False).agg(
        n=("reached_birth", "size"), labels=("reached_birth", "nunique"),
        splits=("split", "nunique")
    )
    duplicates = grouped[grouped.n > 1]
    return {
        "unique_points": len(grouped), "duplicate_groups": len(duplicates),
        "excess_rows": int((duplicates.n - 1).sum()),
        "conflicting_label_groups": int((duplicates.labels > 1).sum()),
        "cross_split_groups": int((duplicates.splits > 1).sum()),
    }


def match_rows(source, target, description):
    """Match in original log coordinates, allowing CSV float-rounding differences."""
    distances, indices = cKDTree(np.log(target[PARAMS])).query(
        np.log(source[PARAMS]), k=1, p=np.inf
    )
    tolerance = 1e-10
    if np.any(distances > tolerance) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"No unique mapping for {description}; max distance={distances.max()}")
    agreement = {
        col: int(np.sum(source[col].to_numpy() != target.iloc[indices][col].to_numpy()))
        for col in BOOLS
    }
    if any(agreement.values()):
        raise ValueError(f"Diagnostic mismatch in {description}: {agreement}")
    return indices, {
        "matched_rows": len(indices), "unique_target_rows": len(np.unique(indices)),
        "max_log_coordinate_distance": float(distances.max()),
        "log_coordinate_tolerance": tolerance, "boolean_mismatches": agreement,
    }


def write_plots(df, outdir):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, x, y, xlabel, ylabel in [
        (axes[0], np.log10(df.gamma), np.log10(df.k), r"$\log_{10}(\gamma)$", r"$\log_{10}(k)$"),
        (axes[1], np.log10(df.k), df.log_nu_b/np.log(10), r"$\log_{10}(k)$", r"$\log_{10}(\nu_b)$"),
    ]:
        hist = ax.hist2d(x, y, bins=60, norm=LogNorm(), cmap="viridis", cmin=1)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(hist[3], ax=ax, label="Observations per bin")
    fig.suptitle(f"Normalized sampling coverage | {len(df):,} observations")
    fig.savefig(outdir / "coverage.png", dpi=170)
    plt.close(fig)

    centers = [0.03, 0.3, 3.0]
    halfwidth = 0.05
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
    counts = []
    for ax, center in zip(axes, centers):
        slab = df.loc[np.abs(np.log10(df.k/center)) <= halfwidth]
        categories = [
            (~slab.success, "Solver unsuccessful", "#b65c7b"),
            (slab.success & ~slab.reached_birth, "Solved, labeled infeasible", "#e39736"),
            (slab.reached_birth, "Labeled feasible", "#218b87"),
        ]
        for mask, label, color in categories:
            ax.scatter(slab.loc[mask, "gamma"], slab.loc[mask, "nu_b"], s=7,
                       alpha=0.5, color=color, label=label, rasterized=True)
        ax.set(xscale="log", yscale="log", xlabel=r"$\gamma=g/f$",
               ylabel=r"$\nu_b=v_H^b/f^3$", title=f"k near {center:g} | n={len(slab):,}")
        counts.append({"center_k": center, "log10_halfwidth": halfwidth, "rows": len(slab)})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, markerscale=2)
    fig.suptitle("Observed outcomes in narrow k bands (not exact fixed-k slices)")
    fig.savefig(outdir / "outcome_slices.png", dpi=170)
    plt.close(fig)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/runs/normalized_data_audit")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    if outdir.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Audit output must not be inside the source data directory")
    paths = {s: ROOT / "data/processed" / f"{s}.csv" for s in ["train", "val", "test"]}
    paths["processed"] = ROOT / "data/processed/deb_reach_birth.csv"
    paths["raw"] = ROOT / "data/raw" / RAW_NAME
    sources = {name: load(path) for name, path in paths.items()}
    raw = sources["raw"]
    preprocess_mask = (raw.execution_time > 0) & np.isfinite(raw[PARAMS]).all(axis=1)
    eligible_raw = raw.loc[preprocess_mask].copy()
    processed = sources["processed"]
    splits = pd.concat([
        normalize(sources[name]).assign(split=name, split_row=np.arange(len(sources[name])))
        for name in ["train", "val", "test"]
    ], ignore_index=True)
    if not positive_finite(splits).all():
        raise ValueError("Invalid split parameters; inspect before normalized neighbor analysis")
    processed_to_raw, raw_match = match_rows(processed, eligible_raw, "processed to eligible raw")
    split_to_processed, split_match = match_rows(splits, processed, "splits to processed")
    if len(processed) != len(eligible_raw) or len(splits) != len(processed):
        raise ValueError("Source mapping is incomplete")
    mapped_raw = eligible_raw.iloc[processed_to_raw[split_to_processed]]
    splits["raw_row"] = mapped_raw["Row"].to_numpy()
    splits["raw_record_1based"] = mapped_raw.index.to_numpy() + 1
    splits["processed_record_1based"] = split_to_processed + 1
    outdir.mkdir(parents=True, exist_ok=True)
    raw.loc[~preprocess_mask].to_csv(outdir / "excluded_raw_rows.csv", index=False)
    # Derived provenance stays separate from the supplied splits.
    splits[["split", "split_row", "raw_row", "raw_record_1based", "processed_record_1based",
            *PARAMS, "gamma", "nu_b", "log_nu_b"]].to_csv(outdir / "row_provenance.csv", index=False)

    summaries, ranges, regime_rows = [], [], []
    for name in ["train", "val", "test", "all"]:
        df = splits if name == "all" else splits.loc[splits.split == name]
        necessary_fail = np.log(df.k) + df.log_nu_b >= 0
        finite_lb = np.isfinite(df.lb)
        computed_growth = df.lb < df.f
        computed_maturity = df.k * df.v_Hb < df.f/(df.g + df.f) * df.lb**2 * (df.g + df.lb)
        summaries.append({
            "split": name, "rows": len(df), "feasible": int(df.reached_birth.sum()),
            "feasible_fraction": float(df.reached_birth.mean()),
            "invalid_parameter_rows": int((~positive_finite(df)).sum()),
            "nonpositive_execution_time": int((df.execution_time <= 0).sum()),
            "unsuccessful": int((~df.success).sum()),
            "timeouts": int(df.error_type.eq("time_limit_reached").sum()),
            "solved_infeasible": int((df.success & ~df.reached_birth).sum()),
            "feasible_but_unsuccessful": int((df.reached_birth & ~df.success).sum()),
            "label_condition_mismatches": int((df.reached_birth != (df["lb<f"] & df["k*vHb<c"])).sum()),
            "finite_lb_rows": int(finite_lb.sum()),
            "unsuccessful_with_finite_lb": int((~df.success & finite_lb).sum()),
            "growth_flag_mismatches_at_finite_lb": int((finite_lb & (computed_growth != df["lb<f"])).sum()),
            "maturity_flag_mismatches_at_finite_lb": int((finite_lb & (computed_maturity != df["k*vHb<c"])).sum()),
            "growth_flag_mismatches_at_successful_lb": int((df.success & finite_lb & (computed_growth != df["lb<f"])).sum()),
            "maturity_flag_mismatches_at_successful_lb": int((df.success & finite_lb & (computed_maturity != df["k*vHb<c"])).sum()),
            "timeout_zero_placeholders": int((df.error_type.eq("time_limit_reached") & df.lb.eq(0) & df.tb.eq(0)).sum()),
            "timeout_message_without_timeout_type": int((df.error_message.eq("Maximum execution time exceeded") & ~df.error_type.eq("time_limit_reached")).sum()),
            "necessary_bound_violations": int(necessary_fail.sum()),
            "feasible_violating_necessary_bound": int((df.reached_birth & necessary_fail).sum()),
            "unsuccessful_inside_necessary_bound": int((~df.success & ~necessary_fail).sum()),
            **duplicate_summary(df, NORMALIZED),
        })
        for col in [*PARAMS, "gamma", "nu_b", "log_nu_b"]:
            ranges.append({"split": name, "variable": col, "min": float(df[col].min()),
                           "max": float(df[col].max()), "median": float(df[col].median())})
        for regime, mask in [("k<1", df.k < 1), ("k=1", df.k == 1), ("k>1", df.k > 1)]:
            sub = df.loc[mask]
            regime_rows.append({"split": name, "regime": regime, "rows": len(sub),
                                "feasible": int(sub.reached_birth.sum()),
                                "unsuccessful": int((~sub.success).sum())})
    pd.DataFrame(summaries).to_csv(outdir / "split_summary.csv", index=False)
    pd.DataFrame(ranges).to_csv(outdir / "ranges.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(outdir / "regimes.csv", index=False)
    diagnostics = splits.groupby(["split", "success", "reached_birth", "error_type"], dropna=False).size().rename("rows")
    diagnostics.to_csv(outdir / "diagnostics.csv")
    splits.groupby(["error_type", "error_message"], dropna=False).size().rename("rows").to_csv(outdir / "error_diagnostics.csv")

    logs = np.log(splits[NORMALIZED].to_numpy())
    tree = cKDTree(logs)
    near = []
    for relative_tolerance in [1e-8, 1e-3]:
        pairs = tree.query_pairs(np.log1p(relative_tolerance), p=np.inf, output_type="ndarray")
        left, right = pairs[:, 0], pairs[:, 1]
        cross = splits.split.to_numpy()[left] != splits.split.to_numpy()[right]
        conflict = splits.reached_birth.to_numpy()[left] != splits.reached_birth.to_numpy()[right]
        near.append({"max_coordinate_ratio_minus_one": relative_tolerance, "pairs": len(pairs),
                     "cross_split_pairs": int(cross.sum()), "different_label_pairs": int(conflict.sum())})
        if len(pairs):
            pd.DataFrame({"left_combined_row": left, "right_combined_row": right,
                          "cross_split": cross, "different_label": conflict}).to_csv(
                              outdir / f"near_pairs_{relative_tolerance:g}.csv", index=False)
    train = splits.loc[splits.split == "train"]
    train_logs = np.log(train[NORMALIZED].to_numpy())
    train_tree = cKDTree(train_logs)
    nearest = {}
    for name in ["val", "test"]:
        query = np.log(splits.loc[splits.split == name, NORMALIZED].to_numpy())
        distance, _ = train_tree.query(query, p=np.inf)
        nearest[name] = {
            "max_coordinate_ratio_minus_one_quantiles": dict(zip(
                ["min", "q25", "median", "q75", "max"], np.expm1(np.quantile(distance, [0, .25, .5, .75, 1])).tolist())),
            "outside_training_coordinate_ranges": int(((query < train_logs.min(axis=0)) | (query > train_logs.max(axis=0))).any(axis=1).sum()),
        }
    gamma_edges = np.linspace(-3, 3, 31)
    k_edges = np.linspace(-4, 1, 26)
    histogram, _, _ = np.histogram2d(np.log10(train.gamma), np.log10(train.k), bins=[gamma_edges, k_edges])
    coverage = {"train_gamma_k_grid_shape": list(histogram.shape), "empty_cells": int((histogram == 0).sum()),
                "cells_with_fewer_than_20_train_rows": int((histogram < 20).sum()),
                "gamma_log10_edges": gamma_edges.tolist(), "k_log10_edges": k_edges.tolist()}
    summary = {
        "python": platform.python_version(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "matplotlib"]},
        "source_files": {name: {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(sources[name])} for name, path in paths.items()},
        "raw_excluded_rows": int((~preprocess_mask).sum()),
        "raw_nonpositive_execution_time": int((raw.execution_time <= 0).sum()),
        "raw_invalid_parameter_rows": int((~positive_finite(raw)).sum()),
        "processed_to_raw": raw_match, "split_to_processed": split_match,
        "original_duplicates": duplicate_summary(splits, PARAMS),
        "normalized_duplicates": duplicate_summary(splits, NORMALIZED),
        "near_pairs": near, "nearest_training_points": nearest, "coverage_grid": coverage,
        "split_summaries": summaries, "regimes": regime_rows,
        "plot_slabs": write_plots(splits, outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps({"near_pairs": near, "nearest_training_points": nearest,
                      "raw_match": raw_match, "coverage_grid": {k: v for k, v in coverage.items() if "edges" not in k}}, indent=2))
    print(f"Audit outputs: {outdir}")


if __name__ == "__main__":
    main()
