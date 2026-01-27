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

def results_df_to_latex(
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
    header_cells = [""] + [ METRIC_LABELS.get(c, str(c)) for c in df.columns ]
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
