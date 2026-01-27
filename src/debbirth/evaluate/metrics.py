from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Optional, Any, Dict

import numpy as np
import torch


def extract_pos_proba(proba: np.ndarray) -> np.ndarray:
    """Return positive-class probability as shape [N].

    Accepts:
    - [N] (already positive probs)
    - [N, 1] (treated as positive probs)
    - [N, 2] (sklearn convention: column 1 is positive)
    """
    if proba.ndim == 1:
        return proba
    if proba.ndim == 2 and proba.shape[1] == 1:
        return proba[:, 0]
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    raise ValueError(f"Unexpected predict_proba shape: {proba.shape}")


def compute_pos_weight(y_train: torch.Tensor) -> torch.Tensor:
    # pos_weight = (#neg / #pos) for BCEWithLogitsLoss
    n_pos = torch.sum(y_train == 1).float()
    n_neg = torch.sum(y_train == 0).float()
    if n_pos <= 0:
        return torch.tensor(1.0)
    return n_neg / n_pos


@dataclass(frozen=True)
class BinaryMetrics:
    # Per-class metrics
    precision_pos: float
    recall_pos: float
    f1_pos: float

    precision_neg: float
    recall_neg: float
    f1_neg: float

    # Macro averages
    precision_macro: float
    recall_macro: float
    f1_macro: float

    # Ranking / probabilistic metrics (positive-class)
    auroc: float
    avg_precision: float

    # macro averaged ranking metrics (average of per-class AUROC/AP)
    auroc_macro: float
    avg_precision_macro: float

    # confusion counts
    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def empty(cls) -> BinaryMetrics:
        nan = float("nan")
        return cls(
            precision_pos=nan,
            recall_pos=nan,
            f1_pos=nan,
            precision_neg=nan,
            recall_neg=nan,
            f1_neg=nan,
            precision_macro=nan,
            recall_macro=nan,
            f1_macro=nan,
            auroc=nan,
            avg_precision=nan,
            auroc_macro=nan,
            avg_precision_macro=nan,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
        )

    def confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix in sklearn convention:
        [[tn, fp],
         [fn, tp]]
        """
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=int)

    def __str__(self) -> str:
        """Readable string with floats at 4 decimal places and integer confusion counts."""
        return (
            f"accuracy: {self.accuracy:.4f}\n"
            f"precision (pos): {self.precision_pos:.4f}\n"
            f"recall    (pos): {self.recall_pos:.4f}\n"
            f"f1        (pos): {self.f1_pos:.4f}\n"
            f"precision (neg): {self.precision_neg:.4f}\n"
            f"recall    (neg): {self.recall_neg:.4f}\n"
            f"f1        (neg): {self.f1_neg:.4f}\n"
            f"precision (macro): {self.precision_macro:.4f}\n"
            f"recall    (macro): {self.recall_macro:.4f}\n"
            f"f1        (macro): {self.f1_macro:.4f}\n"
            f"auroc (pos): {self.auroc:.4f}\n"
            f"avg_precision (pos): {self.avg_precision:.4f}\n"
            f"auroc (macro): {self.auroc_macro:.4f}\n"
            f"avg_precision (macro): {self.avg_precision_macro:.4f}\n"
            f"tp: {int(self.tp)}  fp: {int(self.fp)}  tn: {int(self.tn)}  fn: {int(self.fn)}"
        )

    def save_json(self, path) -> None:
        """Save this BinaryMetrics to a JSON file (creates parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    # New: load metrics from JSON file. Returns BinaryMetrics or EpochBinaryMetrics or None on failure.
    @classmethod
    def load_json(cls, path: Any) -> BinaryMetrics:
        """
        Load metrics saved with save_json. If the JSON contains an 'epoch' key an
        EpochBinaryMetrics will be returned; otherwise a BinaryMetrics.
        Returns None if the file does not exist or cannot be parsed.
        """
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls(**raw)


# New subclass that extends BinaryMetrics with epoch/train_loss/val_loss
@dataclass(frozen=True)
class EpochBinaryMetrics(BinaryMetrics):
    epoch: int
    train_loss: float
    val_loss: float

    @classmethod
    def from_binary_metrics(cls, bm: BinaryMetrics, *, epoch: int, train_loss: float,
                            val_loss: float) -> "EpochBinaryMetrics":
        """Create an EpochBinaryMetrics from an existing BinaryMetrics plus epoch/loss info."""
        # Convert base dataclass to dict then pass through to subclass constructor
        base = asdict(bm)
        # asdict includes all BinaryMetrics fields; extend with epoch/loss
        base.update({"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})
        return cls(**base)
