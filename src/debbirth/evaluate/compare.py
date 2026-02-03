from typing import Dict, List, Optional
import pandas as pd

from .metrics import BinaryMetrics, METRIC_LABELS

def compile_metrics(metrics_dict: Dict[str, BinaryMetrics]) -> pd.DataFrame:
    """Compile a dictionary of BinaryMetrics into a DataFrame."""
    results_df = pd.DataFrame(index=list(metrics_dict.keys()),
                              columns=list(vars(BinaryMetrics.empty()).keys()))
    for model_name, metrics in metrics_dict.items():
        results_df.loc[model_name] = pd.Series(vars(metrics))
    return results_df

def format_metrics_latex_table(
    results_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    num_format: str = "{:.4f}",
) -> str:
    """
    Render results_df as a LaTeX tabular using booktabs (\toprule, \midrule, \bottomrule).

    Args:
      results_df: DataFrame produced by compile_metrics (index = model names).
      columns: optional list of column names to include (default: all columns from results_df).
      num_format: format string for numeric values, e.g. "{:.3f}".

    Returns:
      A string containing only the LaTeX tabular environment.
    """
    if columns is not None:
        df = results_df.loc[:, columns]
    else:
        df = results_df.copy()

    num_cols = len(df.columns)

    lines = []
    # inline the alignment spec (no separate 'align' variable)
    lines.append(rf"\begin{{tabular}}{{l{('r' * num_cols)}}}")
    lines.append(r"\toprule")

    # header row: empty first column (models) then formatted column names
    header_cells = ["Model"] + [ METRIC_LABELS.get(c, str(c)) for c in df.columns ]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # body rows (no escaping)
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for col in df.columns:
            v = row[col]
            if pd.isna(v):
                rendered = ""
            else:
                fv = float(v)
                rendered = num_format.format(fv)
            cells.append(rendered)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)

# New function: produce a two-line header LaTeX table for per-class metrics
def format_per_class_metrics_latex_table(
    results_df: pd.DataFrame,
    num_format: str = "{:.4f}",
    extra_metrics: Optional[List[str]] = None,
) -> str:
    """
    Render a LaTeX tabular where each row is a model.
    - Column order: precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg,
      then for each extra metric: {name} (single-valued column).
    - extra_metrics: list of metric names; for each name the function expects a single column
      with that exact name in results_df (no _pos/_neg suffix).
    Header layout:
      First header row: \multirow{2}{*}{Model} | \multicolumn{3}{c}{Birth is reached} | \multicolumn{3}{c}{Birth is not reached} | \multirow{2}{*}{Extra1} ...
      Second header row: empty | Precision | Recall | F1 | Precision | Recall | F1 | (empty cells for extras)
    Returns a tabular string (uses METRIC_LABELS for column headers).
    """
    df = results_df.copy()
    if extra_metrics is None:
        extra_metrics = []

    # desired order: pos metrics grouped, then neg metrics grouped
    base_cols = [
        "precision_pos", "recall_pos", "f1_pos",
        "precision_neg", "recall_neg", "f1_neg",
    ]

    ordered_cols = base_cols + list(extra_metrics)

    # validate columns exist
    missing = [c for c in ordered_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(set(missing))}")

    # alignment: one 'l' for Model, then one 'r' per data column
    num_data_cols = len(ordered_cols)
    align = rf"l{('r' * num_data_cols)}"

    lines = []
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")

    # First header row: Model multirow + two multicolumn groups + multirow extras
    first_header = []
    first_header.append(r"\multirow{2}{*}{Model}")
    first_header.append(r"\multicolumn{3}{c}{Birth is reached}")
    first_header.append(r"\multicolumn{3}{c}{Birth is not reached}")
    for name in extra_metrics:
        label = METRIC_LABELS.get(name, name)
        # ensure proper TeX escaping is left to the caller if needed
        first_header.append(r"\multirow{2}{*}{" + str(label) + "}")
    lines.append(" & ".join(first_header) + r" \\")
    # Second header row: empty for multirow Model, then metric labels, then empty for each extra metric
    second_header = []
    second_header.append("")  # placeholder for the multirow Model cell
    # metric labels (use generic metric names)
    second_header.extend([
        METRIC_LABELS.get("precision", "Precision"),
        METRIC_LABELS.get("recall", "Recall"),
        METRIC_LABELS.get("f1", "F1-score"),
        METRIC_LABELS.get("precision", "Precision"),
        METRIC_LABELS.get("recall", "Recall"),
        METRIC_LABELS.get("f1", "F1-score"),
    ])
    # placeholders for extras (their header is in the multirow above)
    second_header.extend([""] * len(extra_metrics))
    lines.append(" & ".join(second_header) + r" \\")
    lines.append(r"\midrule")

    # body rows: one row per model, values in ordered_cols
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for col in ordered_cols:
            v = row[col]
            if pd.isna(v):
                rendered = ""
            else:
                rendered = num_format.format(float(v))
            cells.append(rendered)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)
