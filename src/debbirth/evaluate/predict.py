from __future__ import annotations

from typing import Any, Tuple, Callable, Optional

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from ..utils.numpy import convert_to_numpy
from .metrics import extract_pos_proba, BinaryMetrics
from ..utils.pytorch import collect_xy_from_dataloader


def _safe_pair_mean(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float((a + b) / 2.0)


def evaluate_binary_classifier(
        model: Any,
        X: Any,
        y: Any,
        pos_label: int = 1,
) -> BinaryMetrics:
    """Generic evaluation for binary classifiers.

    Requirements on `model`:
      - predict_proba(X) and predict(X)

    Args:
        model: object with sklearn-like predict_proba/predict methods.
        X: inputs compatible with the model (np.ndarray, torch.Tensor, list, etc.).
        y: true labels (array-like), will be binarized by threshold 0.5 for metrics.
        pos_label: positive label used by sklearn.

    Returns:
        BinaryMetrics (contains per-class and macro metrics)
    """
    # If possible, check early for empty dataset without forcing conversion
    try:
        if len(X) == 0:
            return BinaryMetrics.empty()
    except Exception:
        # len() not supported; proceed and let model handle it
        pass

    # Call model with the original X (do not convert before prediction)
    proba_raw = model.predict_proba(X)
    pred_raw = model.predict(X)

    # Convert model outputs and ground truth to numpy for sklearn
    proba_np = convert_to_numpy(proba_raw)
    y_prob = extract_pos_proba(np.asarray(proba_np)).astype(np.float64)

    y_pred = convert_to_numpy(pred_raw).astype(np.int32)

    y_true = convert_to_numpy(y)
    y_true = (y_true >= 0.5).astype(np.int32)

    # If after conversion we find no samples, return empty
    if y_true.size == 0:
        return BinaryMetrics.empty()

    # Core classification metrics
    acc = float(accuracy_score(y_true, y_pred))

    # Per-class precision/recall/f1 (order: negative(0), positive(1))
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    precision_neg, precision_pos = float(prec_arr[0]), float(prec_arr[1])
    recall_neg, recall_pos = float(rec_arr[0]), float(rec_arr[1])
    f1_neg, f1_pos = float(f1_arr[0]), float(f1_arr[1])

    # Keep legacy names as positive-class aliases
    precision_pos_alias = precision_pos
    recall_pos_alias = recall_pos
    f1_pos_alias = f1_pos

    # Macro averaged classification metrics (use sklearn helpers / fallback)
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix with fixed label order (0,1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm = [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    # Ranking metrics for positive and negative class
    try:
        auroc_pos = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc_pos = float("nan")

    try:
        ap_pos = float(average_precision_score(y_true, y_prob))
    except Exception:
        ap_pos = float("nan")

    # For negative class, invert labels and probabilities
    try:
        y_true_neg = (y_true == 0).astype(int)
        y_prob_neg = 1.0 - y_prob
        auroc_neg = float(roc_auc_score(y_true_neg, y_prob_neg))
    except Exception:
        auroc_neg = float("nan")

    try:
        ap_neg = float(average_precision_score(y_true_neg, y_prob_neg))
    except Exception:
        ap_neg = float("nan")

    # Macro averaged ranking metrics : mean of pos and neg (NaN if either is NaN)
    auroc_macro = _safe_pair_mean(auroc_pos, auroc_neg)
    ap_macro = _safe_pair_mean(ap_pos, ap_neg)

    return BinaryMetrics(
        accuracy=acc,
        precision=precision_pos_alias,
        recall=recall_pos_alias,
        f1=f1_pos_alias,
        precision_pos=precision_pos,
        recall_pos=recall_pos,
        f1_pos=f1_pos,
        precision_neg=precision_neg,
        recall_neg=recall_neg,
        f1_neg=f1_neg,
        precision_macro=prec_macro,
        recall_macro=rec_macro,
        f1_macro=f1_macro,
        auroc=auroc_pos,
        avg_precision=ap_pos,
        auroc_macro=auroc_macro,
        avg_precision_macro=ap_macro,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


# new helper: run evaluation loop and compute average loss using provided loss function
@torch.no_grad()
def compute_loss_from_dataloader(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: Optional[torch.device] = None,
) -> float:
    """Compute average loss over dataloader using loss_fn.

    loss_fn is expected to return a scalar tensor representing the batch loss (typically the mean for that batch).
    The function accumulates weighted by batch size to return an overall average.
    """
    model.eval()
    # infer device from model if not provided
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

    total_loss = 0.0
    total_samples = 0

    for xb, yb in dataloader:
        if isinstance(xb, torch.Tensor):
            xb = xb.to(device)
        if isinstance(yb, torch.Tensor):
            yb = yb.to(device)

        logits = model(xb)
        batch_loss = loss_fn(logits, yb)
        # ensure floating scalar
        if isinstance(batch_loss, torch.Tensor):
            batch_loss_val = float(batch_loss.detach().cpu())
        else:
            batch_loss_val = float(batch_loss)

        # determine batch size
        if isinstance(yb, torch.Tensor):
            bsz = int(yb.size(0))
        else:
            bsz = len(yb)

        total_loss += batch_loss_val * bsz
        total_samples += bsz

    return float(total_loss / total_samples) if total_samples > 0 else float("nan")


@torch.no_grad()
def evaluate_pytorch_binary_classifier(
        model: Any,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        pos_label: int = 1,
        device: Optional[torch.device] = None,
) -> Tuple[BinaryMetrics, float]:
    """PyTorch-specific wrapper: compute loss over dataloader and call the generic evaluator.

    Returns:
        (BinaryMetrics, loss)
    """
    loss = compute_loss_from_dataloader(model, dataloader, loss_fn, device=device)
    X, y = collect_xy_from_dataloader(dataloader)
    metrics = evaluate_binary_classifier(model, X, y, pos_label=pos_label)
    return metrics, loss
