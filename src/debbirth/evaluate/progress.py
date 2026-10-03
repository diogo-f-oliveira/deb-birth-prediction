"""Per-step validation progress shared by GP and NN training (T08D).

One step is one GP generation or one NN epoch. `step` counts completed
steps, so for GP it is gplearn's generation index + 1 and the last step equals
the configured number of generations/epochs. Each logged step is printed and
appended (and flushed) to `progress.csv` in the run directory, so the file can
be followed live and an interrupted run keeps its completed rows.

`val_bce` is the unweighted mean BCE of the classification logit on the
validation rows, computed in float64 identically for every family and
formulation (boundary logit: (F - log nu_b)/T). Metric columns are the
BinaryMetrics fields of the same predictions. This module imports no model
backend.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter

import numpy as np

from .metrics import BinaryMetrics

PROGRESS_FILENAME = "progress.csv"
BASE_COLUMNS = ("step_kind", "step", "elapsed_s", "val_eval_s", "val_bce")
METRIC_COLUMNS = tuple(f.name for f in fields(BinaryMetrics))


def bce_from_logits(y, logits) -> float:
    """Unweighted mean BCE of sigmoid(logits); stable for any logit, inf allowed."""
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    if y.shape != z.shape or y.ndim != 1 or not y.size:
        raise ValueError("Labels and logits must be aligned nonempty vectors.")
    # -log sigmoid(z) for positives, -log sigmoid(-z) for negatives.
    with np.errstate(over="ignore", invalid="ignore"):
        losses = np.where(y == 1, np.logaddexp(0.0, -z), np.logaddexp(0.0, z))
    return float(np.mean(losses))


def check_progress_every(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("progress_every must be a nonnegative integer (0 disables progress logging).")
    return value


class ProgressLogger:
    """Print and append one validation row per logged step.

    `extra_columns` follow the shared columns (family-specific values such as
    program length or NN training loss); `print_extras` selects those shown
    on the printed line. With `path=None` rows are printed only. Logged steps
    are every `every`-th step and always the final one.
    """

    def __init__(self, *, step_kind, total_steps, every=1, path=None, extra_columns=(), print_extras=()):
        self.step_kind, self.total_steps = step_kind, int(total_steps)
        self.every = check_progress_every(every)
        self.extra_columns, self.print_extras = tuple(extra_columns), tuple(print_extras)
        self.columns = BASE_COLUMNS + METRIC_COLUMNS + self.extra_columns
        self.path = None if path is None else Path(path)
        self._fh = self._writer = None
        self._start = perf_counter()
        if self.enabled and self.path is not None:
            self._fh = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.columns)
            self._writer.writeheader()
            self._fh.flush()
            print(f"Validation progress: {self.path}")

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def due(self, step: int) -> bool:
        return self.enabled and (step % self.every == 0 or step == self.total_steps)

    def elapsed(self) -> float:
        return perf_counter() - self._start

    def log(self, step, *, metrics: BinaryMetrics | None, val_bce, val_eval_s, **extras):
        """Write one row. metrics=None records a step whose validation outputs were nonfinite."""
        if set(extras) != set(self.extra_columns):
            raise ValueError("Progress extras do not match the declared columns.")
        values = asdict(metrics) if metrics is not None else dict.fromkeys(METRIC_COLUMNS, float("nan"))
        row = {"step_kind": self.step_kind, "step": int(step), "elapsed_s": round(self.elapsed(), 3),
               "val_eval_s": round(float(val_eval_s), 4), "val_bce": float(val_bce), **values, **extras}
        if self._writer is not None:
            self._writer.writerow(row)
            self._fh.flush()
        print(self._line(row, metrics is None))
        return row

    def _line(self, row, nonfinite):
        width = len(str(self.total_steps))
        head = f"[{self.step_kind} {row['step']:>{width}}/{self.total_steps}]"
        if nonfinite:
            body = "nonfinite validation outputs (metrics NaN)"
        else:
            body = (f"val_bce={_fmt(row['val_bce'])} f1_macro={row['f1_macro']:.4f} mcc={row['mcc']:.4f} "
                    f"f1_pos={row['f1_pos']:.4f} f1_neg={row['f1_neg']:.4f}")
        extras = " ".join(f"{name}={_fmt(row[name])}" for name in self.print_extras)
        return f"{head} {body}" + (f" | {extras}" if extras else "") + f" | {row['elapsed_s']:.1f}s"

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _fmt(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    return f"{value:.4f}" if math.isfinite(value) else str(value)
