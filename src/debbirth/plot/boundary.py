from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

LABEL_TO_LATEX = {
    'g': r'g',
    'k': r'k',
    'v_Hb': r'v_H^b',
    'f': r'f',
}


def plot_decision_mesh(
        df: pd.DataFrame,
        decision_col: str,
        x_col: str = 'g',
        y_col: str = 'v_Hb',
        logx: bool = True,
        logy: bool = True,
        shading: str = "auto",
        legend_loc: str = "lower left",
        bound_linewidth: float = 1,
        neg_color: str = "#E0F3F8",
        pos_color: str = "#FEE8C8",
):
    """
    Plot a decision map on a meshgrid using columns x_col (x-axis), y_col (y-axis)
    and decision_col (binary or numeric value to plot).

    Draw horizontal lines at f**3 / k and f**k using f and k taken from df.iloc[0].
    Returns (fig, ax).
      neg_color, pos_color: colors used for negative (<=0.5) and positive (>0.5)
      mesh cells respectively. Accepts any Matplotlib color spec.
    """

    # build sorted unique axis values
    x_vals = np.sort(df[x_col].unique())
    y_vals = np.sort(df[y_col].unique())

    # pivot decision values into a 2D array matching y (rows) x (cols)
    Z = (
        df.pivot(index=y_col, columns=x_col, values=decision_col)
        .reindex(index=y_vals, columns=x_vals)
        .to_numpy()
    )

    # mask invalid cells so they are not colored
    Zm = np.ma.masked_invalid(Z)

    # coordinate arrays for plotting
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # create axis
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

    # create a 2-color colormap (neg_color for <=0.5, pos_color for >0.5)
    cmap = ListedColormap([neg_color, pos_color])
    norm = BoundaryNorm([-np.inf, 0.5, np.inf], cmap.N)

    # use provided colormap for the mesh so model lines stand out
    pcm = ax.pcolormesh(X_grid, Y_grid, Zm, shading=shading, cmap=cmap, norm=norm, alpha=0.9)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    # Get f and k from the dataframe
    f = float(df.iloc[0]["f"])
    k = float(df.iloc[0]["k"])

    # Label axes and add horizontal lines
    ax.set_xlabel(f"${LABEL_TO_LATEX[x_col]}$", fontsize=12)
    ax.set_ylabel(f"${LABEL_TO_LATEX[y_col]}$", fontsize=12)
    ax.set_title(f"${LABEL_TO_LATEX['k']}={k:.1f}$, ${LABEL_TO_LATEX['f']}={f:.1f}$")

    # Get first two colors from Set2 colormap for horizontal lines
    if k == 1:
        ax.axhline(y=f ** 3 / k, color="#4D4D4D", linestyle="--",
                   linewidth=bound_linewidth, label=rf"$v_H^b = f^3$", zorder=3)
    else:
        ax.axhline(y=f ** 3 / k, color="#4D4D4D", linestyle="--",
                   linewidth=bound_linewidth, label=rf"$v_H^b = f^3/k$", zorder=3)
        if k > 1:
            ax.axhline(y=(f / k) ** 3, color="#8C8C8C", linestyle="-.",
                       linewidth=bound_linewidth, label=rf"$v_H^b = f^3/k^3$", zorder=3)
        if k < 1:
            ax.axhline(y=f ** 3, color="#8C8C8C", linestyle="-.",
                       linewidth=bound_linewidth, label=rf"$v_H^b = f^3$", zorder=3)

    # create legend entries for the decision regions using the provided colors
    pos_patch = Patch(facecolor=pos_color, edgecolor="none", label="Birth is reached")
    neg_patch = Patch(facecolor=neg_color, edgecolor="none", label="Birth is not reached")

    # collect existing handles/labels (from the horizontal lines) and prepend the region patches
    base_handles, base_labels = ax.get_legend_handles_labels()
    region_handles = [pos_patch, neg_patch]
    region_labels = [h.get_label() for h in region_handles]
    all_handles = region_handles + base_handles
    all_labels = region_labels + base_labels
    if all_handles:
        ax.legend(all_handles, all_labels, loc=legend_loc)

    # expose region legend entries to callers so higher-level functions can include them
    ax._region_legend_handles = region_handles
    ax._region_legend_labels = region_labels

    return fig, ax


MODEL_COLORS = [plt.get_cmap("tab10").colors[i] for i in range(10)]  # first 10 colors from 'tab' cmap


def plot_decision_mesh_with_models(
        df: pd.DataFrame,
        model_cols,
        decision_col: str,
        x_col: str = 'g',
        y_col: str = 'v_Hb',
        logx: bool = True,
        logy: bool = True,
        shading: str = "auto",
        contour_level: float = 0.5,
        linewidth: float = 1.5,  # thicker model lines
        linestyle: str = "-",
        legend_loc: str = "lower left",
        model_labels=None,
        model_colors: List[str] = None,
        model_zorder: int = 4,
        model_alpha: float = 0.9,
        neg_color: str = "#E0F3F8",
        pos_color: str = "#FEE8C8",
):
    """
    Draw the base decision mesh (via plot_decision_mesh) and overlay model decision
    boundaries. Each model's decision column in `model_cols` is expected to contain
    binary labels or probabilities; the contour at `contour_level` is used as the boundary.

    Parameters
    - df: DataFrame containing x_col, y_col, decision_col and model prediction columns.
    - model_cols: list of column names (strings) with model predictions to plot boundaries for.
    - model_labels: optional list of labels (strings) for the models to appear in the legend.
    - x_col, y_col: names of the grid axis columns (defaults match plot_decision_mesh).
    - decision_col: column used to draw the base pcolormesh.
    - contour_level: threshold for boundary (default 0.5).
    - cmap: matplotlib colormap name for cycling model colors.
    Returns (fig, ax).
    """
    # validate model_labels if provided
    if model_labels is not None:
        if len(model_labels) != len(model_cols):
            raise ValueError("model_labels must be the same length as model_cols")
        labels = list(model_labels)
    else:
        labels = [str(c) for c in model_cols]

    # draw base mesh using the same x_col/y_col; pass the mesh colors through
    fig, ax = plot_decision_mesh(
        df,
        decision_col=decision_col,
        x_col=x_col,
        y_col=y_col,
        logx=logx,
        logy=logy,
        shading=shading,
        legend_loc=legend_loc,
        neg_color=neg_color,
        pos_color=pos_color,
    )

    # build sorted unique axis values (must match the mesh used by plot_decision_mesh)
    x_vals = np.sort(df[x_col].unique())
    y_vals = np.sort(df[y_col].unique())
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # prepare distinct colors by sampling the chosen cmap evenly
    if model_colors is None:
        model_colors = MODEL_COLORS[:len(model_cols)]

    # collect explicit legend handles for model boundaries
    model_handles = []
    model_labels_used = []

    # overlay each model's decision boundary
    for i, col in enumerate(model_cols):
        # pivot model predictions to grid shape; reindex to ensure same ordering
        Z = (
            df.pivot(index=y_col, columns=x_col, values=col)
            .reindex(index=y_vals, columns=x_vals)
            .to_numpy()
        )
        # mask invalid cells so contouring skips them
        Zm = np.ma.masked_invalid(Z)

        # contour at the given decision level
        try:
            cs = ax.contour(
                X_grid, Y_grid, Zm,
                levels=[contour_level],
                colors=[model_colors[i]],
                linewidths=linewidth,
                linestyles=linestyle,
                zorder=model_zorder,
                alpha=model_alpha,
            )
            # create an explicit Line2D handle matching the contour appearance
            handle = Line2D([0], [0], color=model_colors[i], linewidth=linewidth, linestyle=linestyle)
            model_handles.append(handle)
            model_labels_used.append(labels[i])
        except Exception:
            # skip contours that can't be computed (e.g., constant Z or fully masked)
            continue

    # recreate legend: include region patches (if present), existing handles (hlines), and explicit model handles
    base_handles, base_labels = ax.get_legend_handles_labels()

    # get region handles/labels exposed by plot_decision_mesh (if available)
    region_handles = getattr(ax, "_region_legend_handles", [])
    region_labels = getattr(ax, "_region_legend_labels", [])

    all_handles = list(region_handles) + list(base_handles) + model_handles
    all_labels = list(region_labels) + list(base_labels) + model_labels_used
    if all_handles:
        ax.legend(all_handles, all_labels, loc=legend_loc)

    return fig, ax
