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
) -> str:
    """
    Render a LaTeX tabular with two header lines:
      - First line: "Model" then multicolumn groups "Birth is reached" (positive) and
        "Birth is not reached" (negative).
      - Second line: metric names (Precision, Recall, F1-score) taken from METRIC_LABELS.
    Expects the dataframe to contain these columns:
      precision_pos, recall_pos, f1_pos, precision_neg, recall_neg, f1_neg
    Rows are printed for each dataframe index (model name).
    """
    df = results_df.copy()

    pos_metrics = ["precision_pos", "recall_pos", "f1_pos"]
    neg_metrics = ["precision_neg", "recall_neg", "f1_neg"]
    required = pos_metrics + neg_metrics
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required per-class columns: {missing}")

    lines = []
    # alignment: model column + 6 numeric columns
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")

    # First header row: group titles
    lines.append(
        "Model & "
        r"\multicolumn{3}{c}{Birth is reached} & "
        r"\multicolumn{3}{c}{Birth is not reached} \\"
    )
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")

    # Second header row: metric labels from METRIC_LABELS
    header_metrics = [METRIC_LABELS.get(c, c) for c in pos_metrics + neg_metrics]
    lines.append(" & ".join([""] + header_metrics) + r" \\")
    lines.append(r"\midrule")

    # Body: print rows in grouped order (pos then neg)
    ordered_cols = pos_metrics + neg_metrics
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for col in ordered_cols:
            v = row.get(col, None)
            if pd.isna(v):
                rendered = ""
            else:
                rendered = num_format.format(float(v))
            cells.append(rendered)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)
