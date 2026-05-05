from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd



def _to_numpy(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def _validate_same_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes do not match: y_true {y_true.shape}, y_pred {y_pred.shape}")


def _drop_nan_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _prepare_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    _validate_same_length(y_true, y_pred)
    y_true, y_pred = _drop_nan_pairs(y_true, y_pred)
    return y_true, y_pred


def _tail_mask(y_true: np.ndarray, q: float) -> np.ndarray:
    """
    Bottom-q tail based on realized y_true.
    Example:
      q=0.10 -> bottom 10% days
      q=0.05 -> bottom 5% days
    """
    if not (0 < q < 1):
        raise ValueError(f"q must be in (0, 1), got {q}")
    thr = np.quantile(y_true, q)
    return y_true <= thr


# =========================
# Basic losses
# =========================

def mse(y_true, y_pred) -> float:
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    val = mse(y_true, y_pred)
    return float(np.sqrt(val)) if np.isfinite(val) else float("nan")


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def huber_loss_array(y_true, y_pred, delta: float = 2.0) -> np.ndarray:
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    err = y_true - y_pred
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    return 0.5 * quad**2 + delta * lin


def huber(y_true, y_pred, delta: float = 2.0) -> float:
    losses = huber_loss_array(y_true, y_pred, delta=delta)
    if len(losses) == 0:
        return float("nan")
    return float(np.mean(losses))


# =========================
# Directional metrics
# =========================

def sign_accuracy(y_true, y_pred, zero_tolerance: float = 0.0) -> float:
    """
    Fraction of times prediction and realization have the same sign.

    If zero_tolerance > 0, values with |x| <= zero_tolerance are treated as 0.
    """
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")

    def signed(x):
        out = np.sign(x)
        if zero_tolerance > 0:
            out[np.abs(x) <= zero_tolerance] = 0.0
        return out

    s_true = signed(y_true)
    s_pred = signed(y_pred)
    return float(np.mean(s_true == s_pred))


# =========================
# Tail metrics
# =========================

def tail_rmse(y_true, y_pred, q: float = 0.10) -> float:
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    mask = _tail_mask(y_true, q=q)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def tail_mae(y_true, y_pred, q: float = 0.10) -> float:
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    mask = _tail_mask(y_true, q=q)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def negative_day_sign_accuracy(y_true, y_pred) -> float:
    """
    Accuracy of sign prediction only on realized negative days.
    """
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    mask = y_true < 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


# =========================
# Downside event metrics
# =========================

def downside_hit_rate(y_true, y_pred, q: float = 0.10) -> float:
    """
    Of the truly bad tail days (bottom q by y_true), how many did the model
    also flag as tail days by prediction?
    """
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")

    true_thr = np.quantile(y_true, q)
    pred_thr = np.quantile(y_pred, q)

    true_event = y_true <= true_thr
    pred_event = y_pred <= pred_thr

    if true_event.sum() == 0:
        return float("nan")

    return float(np.mean(pred_event[true_event]))


def downside_precision(y_true, y_pred, q: float = 0.10) -> float:
    """
    Of the days the model flagged as tail days, how many were truly tail days?
    """
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")

    true_thr = np.quantile(y_true, q)
    pred_thr = np.quantile(y_pred, q)

    true_event = y_true <= true_thr
    pred_event = y_pred <= pred_thr

    if pred_event.sum() == 0:
        return float("nan")

    return float(np.mean(true_event[pred_event]))


def downside_f1(y_true, y_pred, q: float = 0.10) -> float:
    """
    F1 score for tail-event detection based on quantile thresholds.
    """
    rec = downside_hit_rate(y_true, y_pred, q=q)
    prec = downside_precision(y_true, y_pred, q=q)

    if not np.isfinite(rec) or not np.isfinite(prec) or (prec + rec) == 0:
        return float("nan")

    return float(2 * prec * rec / (prec + rec))


# =========================
# Relative / scale-aware metrics
# =========================

def mase(y_true, y_pred, insample_y: Optional[np.ndarray] = None) -> float:
    """
    Mean Absolute Scaled Error.

    If insample_y is provided, scale by mean absolute 1-step naive error on insample series.
    If not provided, scale by mean absolute 1-step naive error on y_true itself.
    """
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")

    if insample_y is None:
        base = y_true
    else:
        base = _to_numpy(insample_y)
        base = base[np.isfinite(base)]

    if len(base) < 2:
        return float("nan")

    naive_denom = np.mean(np.abs(np.diff(base)))
    if naive_denom <= 0 or not np.isfinite(naive_denom):
        return float("nan")

    return float(np.mean(np.abs(y_true - y_pred)) / naive_denom)


# =========================
# Main combined function
# =========================

def compute_all_metrics(
    y_true,
    y_pred,
    delta: float = 2.0,
    tail_qs: tuple[float, ...] = (0.10, 0.05),
    zero_tolerance: float = 0.0,
    insample_y: Optional[np.ndarray] = None,
) -> dict:
    y_true_arr, y_pred_arr = _prepare_arrays(y_true, y_pred)

    out = {
        "n": int(len(y_true_arr)),
        "mse": mse(y_true_arr, y_pred_arr),
        "rmse": rmse(y_true_arr, y_pred_arr),
        "mae": mae(y_true_arr, y_pred_arr),
        "huber": huber(y_true_arr, y_pred_arr, delta=delta),
        "sign_accuracy": sign_accuracy(y_true_arr, y_pred_arr, zero_tolerance=zero_tolerance),
        "negative_day_sign_accuracy": negative_day_sign_accuracy(y_true_arr, y_pred_arr),
        "mase": mase(y_true_arr, y_pred_arr, insample_y=insample_y),
    }

    for q in tail_qs:
        q_name = int(round(q * 100))
        out[f"tail_rmse_{q_name}"] = tail_rmse(y_true_arr, y_pred_arr, q=q)
        out[f"tail_mae_{q_name}"] = tail_mae(y_true_arr, y_pred_arr, q=q)
        out[f"downside_hit_rate_{q_name}"] = downside_hit_rate(y_true_arr, y_pred_arr, q=q)
        out[f"downside_precision_{q_name}"] = downside_precision(y_true_arr, y_pred_arr, q=q)
        out[f"downside_f1_{q_name}"] = downside_f1(y_true_arr, y_pred_arr, q=q)

    return out


# =========================
# DataFrame helpers
# =========================

def compute_metrics_from_dataframe(
    df: pd.DataFrame,
    label_col: str = "label",
    prediction_col: str = "prediction",
    delta: float = 2.0,
    tail_qs: tuple[float, ...] = (0.10, 0.05),
    zero_tolerance: float = 0.0,
    insample_y: Optional[np.ndarray] = None,
) -> dict:
    if label_col not in df.columns or prediction_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain columns '{label_col}' and '{prediction_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    return compute_all_metrics(
        y_true=df[label_col].values,
        y_pred=df[prediction_col].values,
        delta=delta,
        tail_qs=tail_qs,
        zero_tolerance=zero_tolerance,
        insample_y=insample_y,
    )


def compute_metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "label",
    prediction_col: str = "prediction",
    delta: float = 2.0,
    tail_qs: tuple[float, ...] = (0.10, 0.05),
    zero_tolerance: float = 0.0,
    insample_y: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Example:
      - by ticker
      - by horizon_step
      - by model_name
    """
    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' not in DataFrame columns: {list(df.columns)}")

    rows = []
    for g, part in df.groupby(group_col):
        metrics = compute_metrics_from_dataframe(
            part,
            label_col=label_col,
            prediction_col=prediction_col,
            delta=delta,
            tail_qs=tail_qs,
            zero_tolerance=zero_tolerance,
            insample_y=insample_y,
        )
        metrics[group_col] = g
        rows.append(metrics)

    return pd.DataFrame(rows)
