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
    'gamma': r'\gamma',
    'nu_b': r'\nu_b',
    'log_nu_b': r'\log \nu_b',
    'x_b': r'x_b',
    'log_psi': r'F = \log \Psi',
    'psi': r'\Psi',
}
REFERENCE_STYLE = {"primary": dict(color="#4D4D4D", linestyle="--"), "secondary": dict(color="#8C8C8C", linestyle="-.")}


def slice_value(df: pd.DataFrame, col: str):
    """The single value of `col` in a plotted slice; raise if it varies or is absent."""
    if col not in df:
        raise ValueError(f"Slice annotation needs column {col!r}.")
    values = np.unique(df[col].to_numpy())
    if len(values) != 1:
        raise ValueError(f"Expected one {col} value in this slice, found {len(values)}; "
                         "pass reference_lines=False or plot one slice at a time.")
    return float(values[0])


def reference_bounds(df: pd.DataFrame, y_col: str):
    """Analytical screening lines in the plotted maturity coordinate, with title text.

    Original v_Hb axes at fixed (k, f): v_Hb = f^3/k, plus (f/k)^3 for k > 1 or f^3
    for k < 1. Normalized nu_b axes at fixed k (f-free): nu_b = 1/k, plus 1/k^3 or 1.
    Other maturity coordinates get no reference lines.
    """
    if y_col == "v_Hb":
        k, f = slice_value(df, "k"), slice_value(df, "f")
        title = f"${LABEL_TO_LATEX['k']}={k:.1f}$, ${LABEL_TO_LATEX['f']}={f:.1f}$"
        primary = (f ** 3 / k, r"$v_H^b = f^3$" if k == 1 else r"$v_H^b = f^3/k$")
        secondary = ((f / k) ** 3, r"$v_H^b = f^3/k^3$") if k > 1 else (f ** 3, r"$v_H^b = f^3$") if k < 1 else None
    elif y_col == "nu_b":
        k = slice_value(df, "k")
        title = f"${LABEL_TO_LATEX['k']}={k:.1f}$ (normalized, $f$-free)"
        primary = (1 / k, r"$\nu_b = 1$" if k == 1 else r"$\nu_b = 1/k$")
        secondary = (1 / k ** 3, r"$\nu_b = 1/k^3$") if k > 1 else (1.0, r"$\nu_b = 1$") if k < 1 else None
    else:
        return [], None
    lines = [(primary[0], primary[1], "primary")] + ([(secondary[0], secondary[1], "secondary")] if secondary else [])
    return lines, title


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
        reference_lines: bool = True,
):
    """
    Plot a decision map on a meshgrid using columns x_col (x-axis), y_col (y-axis)
    and decision_col (binary or numeric value to plot).

    Axes may be original (g, v_Hb) or normalized (gamma, nu_b) coordinates. With
    reference_lines, the analytical screening lines of `reference_bounds` are drawn;
    their slice values (k, and f for original axes) must be single-valued in df.
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

    # Label axes; reference lines/title come from the verified slice values.
    ax.set_xlabel(f"${LABEL_TO_LATEX.get(x_col, x_col)}$", fontsize=12)
    ax.set_ylabel(f"${LABEL_TO_LATEX.get(y_col, y_col)}$", fontsize=12)
    if reference_lines:
        lines, title = reference_bounds(df, y_col)
        if title:
            ax.set_title(title)
        for y, label, style in lines:
            ax.axhline(y=y, linewidth=bound_linewidth, label=label, zorder=3, **REFERENCE_STYLE[style])

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


def draw_boundary(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        values_col: str,
        contour_level: float = 0.5,
        ax: plt.Axes = None,
        color: str = "k",
        linewidth: float = 1.5,
        linestyle: str = "-",
        zorder: int = 4,
        alpha: float = 0.9,
):
    """
    Draw a single decision boundary (contour at `contour_level`) for values_col on
    the grid defined by x_col and y_col present in df.

    Required args: df, x_col, y_col, values_col.
    Optional styling args: color, linewidth, linestyle, zorder, alpha.

    Returns (contour_set, line_handle) or (None, None) if contour could not be computed.
    """
    # build sorted unique axis values
    x_vals = np.sort(df[x_col].unique())
    y_vals = np.sort(df[y_col].unique())

    # pivot values into grid shape and reindex to ensure ordering matches mesh
    Z = (
        df.pivot(index=y_col, columns=x_col, values=values_col)
        .reindex(index=y_vals, columns=x_vals)
        .to_numpy()
    )
    Zm = np.ma.masked_invalid(Z)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    if ax is None:
        fig, ax = plt.subplots()
    try:
        cs = ax.contour(
            X_grid,
            Y_grid,
            Zm,
            levels=[contour_level],
            colors=[color],
            linewidths=linewidth,
            linestyles=linestyle,
            zorder=zorder,
            alpha=alpha,
        )
        handle = Line2D([0], [0], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        return cs, handle
    except Exception:
        # return None on any contour failure (constant data, fully masked, etc.)
        return None, None


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
        reference_lines: bool = True,
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
        reference_lines=reference_lines,
    )

    # (keep grid computation if callers expect it elsewhere)
    x_vals = np.sort(df[x_col].unique())
    y_vals = np.sort(df[y_col].unique())
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # prepare distinct colors by sampling the chosen cmap evenly
    if model_colors is None:
        model_colors = MODEL_COLORS[:len(model_cols)]

    # collect explicit legend handles for model boundaries
    model_handles = []
    model_labels_used = []

    # overlay each model's decision boundary using the shared helper
    for i, col in enumerate(model_cols):
        cs, handle = draw_boundary(
            df=df,
            x_col=x_col,
            y_col=y_col,
            values_col=col,
            contour_level=contour_level,
            ax=ax,
            color=model_colors[i],
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=model_zorder,
            alpha=model_alpha,
        )
        if handle is not None:
            model_handles.append(handle)
            model_labels_used.append(labels[i])

    # expose model contour handles so later overlays can rebuild the full legend
    ax._model_legend_handles = model_handles
    ax._model_legend_labels = model_labels_used
    update_legend(ax, loc=legend_loc)

    return fig, ax


def update_legend(ax: plt.Axes, **legend_kwargs):
    """Rebuild the legend: region patches, labelled lines/curves, then model contours.

    Call after adding overlays such as `draw_critical_curve`; a plain ax.legend()
    would drop the region patches and contour handles.
    """
    base_handles, base_labels = ax.get_legend_handles_labels()
    handles = list(getattr(ax, "_region_legend_handles", [])) + list(base_handles) \
        + list(getattr(ax, "_model_legend_handles", []))
    labels = list(getattr(ax, "_region_legend_labels", [])) + list(base_labels) \
        + list(getattr(ax, "_model_legend_labels", []))
    if handles:
        ax.legend(handles, labels, **legend_kwargs)


def slice_grid(x_col: str, x_values, y_col: str, y_values, **fixed) -> pd.DataFrame:
    """Mesh of two coordinates plus fixed values, with original parameters added.

    Coordinates may be original (g, k, v_Hb, f) or normalized (gamma, nu_b, k).
    Normalized slices are f-free, so g = gamma * f and v_Hb = nu_b * f^3 with
    f = 1 unless given; predictors then receive equivalent original parameters.
    """
    X, Y = np.meshgrid(np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float))
    df = pd.DataFrame({x_col: X.ravel(), y_col: Y.ravel()})
    for name, value in fixed.items():
        df[name] = float(value)
    if {"gamma", "nu_b"} & set(df):
        if "f" not in df:
            df["f"] = 1.0
        if "gamma" in df and "g" not in df:
            df["g"] = df["gamma"] * df["f"]
        if "nu_b" in df and "v_Hb" not in df:
            df["v_Hb"] = df["nu_b"] * df["f"] ** 3
    missing = [c for c in ("g", "k", "v_Hb", "f") if c not in df]
    if missing:
        raise ValueError(f"Slice does not determine original parameters {missing}; fix them explicitly.")
    return df


def draw_critical_curve(ax: plt.Axes, x, critical, *, label: str, color: str = "k", linewidth: float = 1.5,
                        linestyle: str = "-", zorder: int = 5, alpha: float = 0.9):
    """Plot a boundary model's critical maturity against an x coordinate.

    In normalized axes pass Psi(gamma, k); in original axes pass f^3 * Psi(g/f, k)
    (e.g. predictor.critical_maturity(..., normalized=False)). Feasible points lie
    strictly below the curve. Returns the line handle.
    """
    (line,) = ax.plot(x, critical, color=color, linewidth=linewidth, linestyle=linestyle,
                      zorder=zorder, alpha=alpha, label=label)
    return line


def plot_critical_surface(
        df: pd.DataFrame,
        value_col: str = "log_psi",
        x_col: str = "gamma",
        y_col: str = "k",
        logx: bool = True,
        logy: bool = True,
        title: str = None,
        cmap: str = "RdBu_r",
):
    """Map F = log(Psi) over (gamma, k) with its zero contour and the k = 1 reference.

    Analytically Psi(gamma, 1) = 1, i.e. F = 0 on k = 1; the dotted line marks where
    the learned zero contour should lie. Colors are centred at F = 0. Returns (fig, ax).
    """
    from matplotlib.colors import TwoSlopeNorm

    x_vals = np.sort(df[x_col].unique())
    y_vals = np.sort(df[y_col].unique())
    Z = df.pivot(index=y_col, columns=x_col, values=value_col).reindex(index=y_vals, columns=x_vals).to_numpy()
    Zm = np.ma.masked_invalid(Z)
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    limit = float(np.nanmax(np.abs(Z))) or 1.0
    pcm = ax.pcolormesh(*np.meshgrid(x_vals, y_vals), Zm, shading="auto", cmap=cmap,
                        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    fig.colorbar(pcm, ax=ax, label=f"${LABEL_TO_LATEX.get(value_col, value_col)}$")
    handles = []
    if np.nanmin(Z) < 0 < np.nanmax(Z):
        ax.contour(x_vals, y_vals, Zm, levels=[0.0], colors="k", linewidths=1.2)
        handles.append(Line2D([0], [0], color="k", linewidth=1.2, label=r"learned $F = 0$"))
    if y_col == "k" and y_vals.min() <= 1 <= y_vals.max():
        ax.axhline(1.0, color="#4D4D4D", linestyle=":", linewidth=1.2)
        handles.append(Line2D([0], [0], color="#4D4D4D", linestyle=":", linewidth=1.2,
                              label=r"$k = 1$ (analytical $F = 0$)"))
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(f"${LABEL_TO_LATEX.get(x_col, x_col)}$", fontsize=12)
    ax.set_ylabel(f"${LABEL_TO_LATEX.get(y_col, y_col)}$", fontsize=12)
    if title:
        ax.set_title(title)
    if handles:
        ax.legend(handles=handles, loc="lower left")
    return fig, ax
