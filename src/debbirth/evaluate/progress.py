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
BASE_COLUMNS = ("step_kind", "step", "elapsed_s", "val_eval_s", "train_loss", "val_bce")
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


# Printed table: (header, row key, width, best-so-far direction or None) per column, in groups.
_LOSS_COLUMNS = (("train", "train_loss", 7, None), ("val", "val_bce", 7, "min"))
_METRIC_PRINT = (("f1_macro", "f1_macro", 8, "max"), ("mcc", "mcc", 7, None),
                 ("f1_pos", "f1_pos", 7, None), ("f1_neg", "f1_neg", 7, None))
_HEADER_EVERY = 25  # reprint the header so long runs stay readable


class ProgressLogger:
    """Print and append one validation row per logged step.

    The printout is an aligned table: a banner, a header (repeated every
    25 rows), one row per logged step with `*` marking a new best validation
    loss or macro-F1, elapsed time and ETA, and a summary when training ends.
    `extra_columns` follow the shared CSV columns; `print_extras` maps those
    shown in the table to (header, width). With `path=None` rows are printed
    only. Logged steps are every `every`-th step and always the final one.
    """

    def __init__(self, *, step_kind, total_steps, every=1, path=None, title=None, extra_columns=(),
                 print_extras=None):
        self.step_kind, self.total_steps = step_kind, int(total_steps)
        self.every = check_progress_every(every)
        self.extra_columns = tuple(extra_columns)
        self.print_extras = dict(print_extras or {})
        self.columns = BASE_COLUMNS + METRIC_COLUMNS + self.extra_columns
        self.path = None if path is None else Path(path)
        self._fh = self._writer = None
        self._printed = 0
        self._best = {}  # row key -> (value, step)
        self._last_step = None
        self._start = perf_counter()
        if not self.enabled:
            return
        if title:
            print(title)
        if self.path is not None:
            self._fh = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.columns)
            self._writer.writeheader()
            self._fh.flush()
            print(f"progress: {_display_path(self.path)}")

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def due(self, step: int) -> bool:
        return self.enabled and (step % self.every == 0 or step == self.total_steps)

    def elapsed(self) -> float:
        return perf_counter() - self._start

    def log(self, step, *, train_loss, metrics: BinaryMetrics | None, val_bce, val_eval_s, **extras):
        """Write one row. metrics=None records a step whose validation outputs were nonfinite."""
        if set(extras) != set(self.extra_columns):
            raise ValueError("Progress extras do not match the declared columns.")
        values = asdict(metrics) if metrics is not None else dict.fromkeys(METRIC_COLUMNS, float("nan"))
        row = {"step_kind": self.step_kind, "step": int(step), "elapsed_s": round(self.elapsed(), 3),
               "val_eval_s": round(float(val_eval_s), 4), "train_loss": float(train_loss), "val_bce": float(val_bce),
               **values, **extras}
        if self._writer is not None:
            self._writer.writerow(row)
            self._fh.flush()
        if self._printed % _HEADER_EVERY == 0:
            print(("\n" if self._printed else "") + self._header())
        print(self._row(row, nonfinite=metrics is None))
        self._printed += 1
        self._last_step = int(step)
        return row

    # ----- table layout -----
    def _groups(self):
        groups = [((self._step_label(), "step", self._step_width(), None),), _LOSS_COLUMNS, _METRIC_PRINT]
        if self.print_extras:
            groups.append(tuple((head, key, width, None) for key, (head, width) in self.print_extras.items()))
        return groups

    def _step_label(self):
        return "gen" if self.step_kind == "generation" else self.step_kind

    def _step_width(self):
        return max(len(self._step_label()), len(str(self.total_steps)))

    @staticmethod
    def _join(groups):
        return " | ".join(" ".join(cells) for cells in groups)

    def _header(self):
        groups = [[head.rjust(width) + ("" if key == "step" else " ") for head, key, width, _ in group]
                  for group in self._groups()]
        groups.append(["time".rjust(7), "eta".rjust(7)])
        line = self._join(groups)
        rule = "-+-".join("-" * len(" ".join(cells)) for cells in groups)
        return f"{line}\n{rule}"

    def _row(self, row, *, nonfinite):
        cells = []
        for group in self._groups():
            cells.append([self._cell(row, key, width, direction) for _, key, width, direction in group])
        if nonfinite:
            # Keep the step, losses (val is nan) and extras; replace the metric cells with a note.
            note = "nonfinite validation outputs".ljust(len(" ".join(cells[2])))
            cells[2] = [note]
        elapsed = row["elapsed_s"]
        remaining = elapsed / row["step"] * (self.total_steps - row["step"])
        cells.append([_clock(elapsed).rjust(7), _clock(remaining).rjust(7)])
        return self._join(cells)

    def _cell(self, row, key, width, direction):
        value = row[key]
        if key == "step":
            return str(value).rjust(width)
        mark = " "
        if direction is not None and _finite(value):
            # Compare at display precision, so a marked value is visibly better.
            best, shown = self._best.get(key), round(float(value), 4)
            improved = best is None or (shown < round(best[0], 4) if direction == "min" else shown > round(best[0], 4))
            if improved:
                mark = "*" if best is not None else " "  # the first row is not marked
                self._best[key] = (value, row["step"])
        return _num(value).rjust(width) + mark

    # ----- end of run -----
    def summary(self):
        if not self._printed:
            return None
        parts = [f"best {name} {_num(value)} at {self._step_label()} {step}"
                 for name, (value, step) in (("val", self._best.get("val_bce", (float("nan"), "-"))),
                                             ("f1_macro", self._best.get("f1_macro", (float("nan"), "-"))))]
        return " | ".join(parts) + f" | {_clock(self.elapsed())} total  (* = new best)"

    def close(self, completed=True):
        if self._fh is not None:
            self._fh.close()
            self._fh = self._writer = None
        if completed and self.enabled:
            text = self.summary()
            if text:
                print(text)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        self.close(completed=exc_type is None)


def _finite(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(value)


def _num(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.4f}" if abs(value) < 100 else f"{value:.4g}"


def _clock(seconds):
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _display_path(path):
    try:
        from ..utils.paths import portable_path
        return portable_path(path)
    except (ValueError, OSError):
        return str(path)
