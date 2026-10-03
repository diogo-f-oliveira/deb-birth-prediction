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
    matthews_corrcoef,
    log_loss,
    brier_score_loss,
)

from ..utils.numpy import convert_to_numpy
from .metrics import extract_pos_proba, BinaryMetrics
from ..formulations import output_to_logit, boundary_margin


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

    return metrics_from_predictions(y_true, y_prob, y_pred=y_pred)


def metrics_from_predictions(y, probabilities, *, y_pred):
    """Metrics without model execution; callers supply their explicit decision.

    Probability scores are unweighted. log_loss uses sklearn's numerical
    clipping; training/validation BCE is separately computed from raw logits.
    """
    y_true = np.asarray(convert_to_numpy(y))
    y_prob = extract_pos_proba(np.asarray(convert_to_numpy(probabilities))).astype(np.float64)
    y_pred = np.asarray(convert_to_numpy(y_pred))
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or y_prob.shape != y_true.shape:
        raise ValueError("Labels, probabilities and decisions must be aligned vectors.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Labels and decisions must be binary.")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")

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
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        log_loss=float(log_loss(y_true, y_prob, labels=[0, 1])),
        brier_score=float(brier_score_loss(y_true, y_prob)),
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
        *, formulation: Optional[str] = None, temperature: Optional[float] = None,
        return_bce: bool = False,
):
    """Collect loss, probabilities and decisions in one batched forward pass.

    Supports legacy tuple batches and aligned dictionary batches. Boundary
    decisions use the margin sign, never rounded sigmoid probabilities.
    loss_fn must return a batch mean (the NN trainer uses BCEWithLogitsLoss).
    Returns (metrics, loss), or (metrics, loss, bce) with return_bce, where
    bce is the unweighted float64 logit BCE shared with GP progress (T08D).
    """
    model.eval()
    if formulation is None:
        formulation = "boundary" if hasattr(model, "temperature") else "full_par"
    if temperature is None:
        temperature = getattr(model, "temperature", 1.0)
    device = device or next(model.parameters()).device
    labels, probabilities, decisions, all_logits = [], [], [], []
    total_loss, count = 0.0, 0
    for batch in dataloader:
        logits, y, margin = logits_from_batch(model, batch, device=device,
                                              formulation=formulation, temperature=temperature)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite validation loss/logits.")
        p = torch.sigmoid(logits)
        pred = margin > 0 if formulation == "boundary" else p >= model.threshold
        labels.append(y.cpu().numpy())
        probabilities.append(p.cpu().numpy())
        decisions.append(pred.cpu().numpy())
        if return_bce:
            all_logits.append(logits.cpu().numpy())
        total_loss += float(loss) * len(y)
        count += len(y)
    if not count:
        return (BinaryMetrics.empty(), float("nan")) + ((float("nan"),) if return_bce else ())
    y_true = np.concatenate(labels)
    metrics = metrics_from_predictions(y_true, np.concatenate(probabilities), y_pred=np.concatenate(decisions))
    if return_bce:
        from .progress import bce_from_logits
        return metrics, total_loss / count, bce_from_logits(y_true, np.concatenate(all_logits))
    return metrics, total_loss / count


def logits_from_batch(model, batch, *, device, formulation, temperature):
    """Return differentiable logits, labels and optional untempered margin."""
    if isinstance(batch, dict):
        x, y = batch["x"], batch["y"]
        offset = batch.get("log_nu_b")
    else:
        x, y = batch
        offset = None
    x, y = x.to(device), y.to(device)
    if offset is not None:
        offset = offset.to(device)
    learned = model(x)
    logits = output_to_logit(learned, formulation=formulation, log_nu_b=offset, temperature=temperature)
    margin = boundary_margin(learned, offset) if formulation == "boundary" else None
    return logits, y, margin
