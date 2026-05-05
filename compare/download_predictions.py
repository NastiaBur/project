from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


AggregationMode = Literal["mean", "first", 'last']


def _validate_required_columns(df: pd.DataFrame, required: list[str], source_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _coerce_datetime_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col])
    return out


def _aggregate_predictions(
    df: pd.DataFrame,
    mode: AggregationMode = "mean",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate predictions when there are repeated rows for the same date/ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least date, ticker, label, prediction.
    mode : {"mean", "first"}
        "mean"  -> average label/prediction over duplicates
        "first" -> keep first row in sorted order
    group_cols : list[str] or None
        Columns defining uniqueness. Default: ["date", "ticker"] if ticker exists,
        otherwise ["date"].
    """
    if group_cols is None:
        group_cols = ["date", "ticker"] if "ticker" in df.columns else ["date"]

    sort_cols = [c for c in ["date", "ticker", "horizon_step", "forecast_origin_date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    if mode == "mean":
        numeric_cols = [c for c in ["label", "prediction"] if c in df.columns]
        keep_first_cols = [c for c in df.columns if c not in group_cols + numeric_cols]

        agg_spec = {c: "mean" for c in numeric_cols}
        for c in keep_first_cols:
            agg_spec[c] = "first"

        out = df.groupby(group_cols, as_index=False).agg(agg_spec)
        return out

    if mode == "first":
        out = df.drop_duplicates(subset=group_cols, keep="first").reset_index(drop=True)
        return out
    if mode == "last":
        out = df.drop_duplicates(subset=group_cols, keep="last").reset_index(drop=True)
        return out

    raise ValueError(f"Unknown aggregation mode: {mode}")


def _normalize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize canonical column order where possible.
    """
    preferred_order = [
        "date",
        "ticker",
        "label",
        "prediction",
        "horizon_step",
        "forecast_origin_date",
        "window_start_date",
        "eval_start_date",
        "source_row",
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    return df[cols].copy()



def load_npz_predictions_raw(npz_path: str | os.PathLike) -> pd.DataFrame:
    """
    Load prediction results from .npz into a long-format DataFrame.

    Expected possible fields in npz:
      - predictions
      - labels
      - dates
      - tickers (optional)
      - forecast_origin_dates (optional) - day, when the prediction first made, because there is some history before the forecast. (It's last day, after which forecasting begins)
      - window_start_dates (optional)
      - eval_start_dates (optional)

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with one row per predicted point:
          date, ticker, label, prediction, horizon_step, ...
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    if "predictions" not in data or "labels" not in data:
        raise ValueError(f"{npz_path} must contain 'predictions' and 'labels' arrays.")

    preds = np.asarray(data["predictions"])
    labels = np.asarray(data["labels"])

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    n_samples, horizon = preds.shape

    if "dates" not in data:
        raise ValueError(f"{npz_path} must contain 'dates'.")

    raw_dates = np.asarray(data["dates"], dtype=object)
    if raw_dates.ndim == 2:
        dates = raw_dates
    elif raw_dates.size == n_samples * horizon:
        dates = raw_dates.reshape(n_samples, horizon)
    elif raw_dates.size == n_samples and horizon == 1:
        dates = raw_dates.reshape(n_samples, 1)
    else:
        raise ValueError(
            f"Cannot align 'dates' with predictions shape {preds.shape}. "
            f"dates shape/size: {raw_dates.shape} / {raw_dates.size}"
        )

    dates = np.asarray(pd.to_datetime(np.asarray(dates).ravel())).reshape(n_samples, horizon)

    # Tickers
    tickers = None
    if "tickers" in data:
        raw_tickers = np.asarray(data["tickers"], dtype=object)
        if raw_tickers.ndim == 2:
            tickers = raw_tickers
        elif raw_tickers.size == n_samples:
            tickers = np.repeat(raw_tickers.reshape(-1, 1), horizon, axis=1)
        elif raw_tickers.size == n_samples * horizon:
            tickers = raw_tickers.reshape(n_samples, horizon)
        else:
            raise ValueError(
                f"Cannot align 'tickers' with predictions shape {preds.shape}. "
                f"tickers shape/size: {raw_tickers.shape} / {raw_tickers.size}"
            )

    # Optional 1D metadata
    forecast_origin_dates = None
    if "forecast_origin_dates" in data:
        arr = np.asarray(data["forecast_origin_dates"], dtype=object)
        if arr.size != n_samples:
            raise ValueError(f"'forecast_origin_dates' must have length {n_samples}, got {arr.size}")
        forecast_origin_dates = pd.to_datetime(arr)

    window_start_dates = None
    if "window_start_dates" in data:
        arr = np.asarray(data["window_start_dates"], dtype=object)
        if arr.size != n_samples:
            raise ValueError(f"'window_start_dates' must have length {n_samples}, got {arr.size}")
        window_start_dates = pd.to_datetime(arr)

    eval_start_dates = None
    if "eval_start_dates" in data:
        arr = np.asarray(data["eval_start_dates"], dtype=object)
        if arr.size != n_samples:
            raise ValueError(f"'eval_start_dates' must have length {n_samples}, got {arr.size}")
        eval_start_dates = pd.to_datetime(arr)

    rows = []
    for i in range(n_samples):
        for h in range(horizon):
            row = {
                "date": pd.to_datetime(dates[i, h]),
                "label": float(labels[i, h]),
                "prediction": float(preds[i, h]),
                "horizon_step": h + 1,
                "source_row": i,
            }

            if tickers is not None:
                row["ticker"] = tickers[i, h]

            if forecast_origin_dates is not None:
                row["forecast_origin_date"] = forecast_origin_dates[i]

            if window_start_dates is not None:
                row["window_start_date"] = window_start_dates[i]

            if eval_start_dates is not None:
                row["eval_start_date"] = eval_start_dates[i]

            rows.append(row)

    df = pd.DataFrame(rows)

    if "ticker" not in df.columns:
        df["ticker"] = "ALL"

    return _normalize_output_columns(df)


def load_npz_predictions_mean(npz_path: str | os.PathLike) -> pd.DataFrame:
    """
    Load .npz predictions and average rows with identical date+ticker.
    """
    df = load_npz_predictions_raw(npz_path)
    df = _aggregate_predictions(df, mode="mean", group_cols=["date", "ticker"])
    return _normalize_output_columns(df)


def load_npz_predictions_first(npz_path: str | os.PathLike) -> pd.DataFrame:
    """
    Load .npz predictions and keep the first row for each identical date+ticker.
    """
    df = load_npz_predictions_raw(npz_path)
    df = _aggregate_predictions(df, mode="first", group_cols=["date", "ticker"])
    return _normalize_output_columns(df)

def load_npz_predictions_last(npz_path: str | os.PathLike) -> pd.DataFrame:
    """
    Load .npz predictions and keep the last (freshest) row for each identical date+ticker.
    """
    df = load_npz_predictions_raw(npz_path)
    df = _aggregate_predictions(df, mode="last", group_cols=["date", "ticker"])
    return _normalize_output_columns(df)


# =========================
# CSV loading
# =========================

def load_csv_predictions_raw(
    csv_path: str | os.PathLike,
    date_col: str = "date",
    label_col: str = "y_true",
    prediction_col: str = "y_pred",
    ticker_col: Optional[str] = "ticker",
) -> pd.DataFrame:
    """
    Load prediction CSV into the same canonical format.

    Expected default columns:
      - date
      - y_true
      - y_pred
      - ticker (optional)

    Optionally also keeps:
      - horizon_step
      - forecast_origin_date
      - window_start_date
      - eval_start_date
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = [date_col, label_col, prediction_col]
    if ticker_col is not None and ticker_col in df.columns:
        pass

    _validate_required_columns(df, required, source_name=str(csv_path))

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "label": df[label_col].astype(float),
        "prediction": df[prediction_col].astype(float),
    })

    if ticker_col is not None and ticker_col in df.columns:
        out["ticker"] = df[ticker_col].astype(object)
    else:
        out["ticker"] = "ALL"

    # carry optional useful metadata if present
    for src, dst in [
        ("horizon_step", "horizon_step"),
        ("forecast_origin_date", "forecast_origin_date"),
        ("window_start_date", "window_start_date"),
        ("eval_start_date", "eval_start_date"),
    ]:
        if src in df.columns:
            if "date" in src:
                out[dst] = pd.to_datetime(df[src])
            else:
                out[dst] = df[src]

    return _normalize_output_columns(out)


def load_csv_predictions_mean(
    csv_path: str | os.PathLike,
    date_col: str = "date",
    label_col: str = "y_true",
    prediction_col: str = "y_pred",
    ticker_col: Optional[str] = "ticker",
) -> pd.DataFrame:
    """
    Load CSV predictions and average rows with identical date+ticker.
    """
    df = load_csv_predictions_raw(
        csv_path=csv_path,
        date_col=date_col,
        label_col=label_col,
        prediction_col=prediction_col,
        ticker_col=ticker_col,
    )
    df = _aggregate_predictions(df, mode="mean", group_cols=["date", "ticker"])
    return _normalize_output_columns(df)


def load_csv_predictions_first(
    csv_path: str | os.PathLike,
    date_col: str = "date",
    label_col: str = "y_true",
    prediction_col: str = "y_pred",
    ticker_col: Optional[str] = "ticker",
) -> pd.DataFrame:
    """
    Load CSV predictions and keep the first row with identical date+ticker.
    """
    df = load_csv_predictions_raw(
        csv_path=csv_path,
        date_col=date_col,
        label_col=label_col,
        prediction_col=prediction_col,
        ticker_col=ticker_col,
    )
    df = _aggregate_predictions(df, mode="first", group_cols=["date", "ticker"])
    return _normalize_output_columns(df)

def load_csv_predictions_last(
    csv_path: str | os.PathLike,
    date_col: str = "date",
    label_col: str = "y_true",
    prediction_col: str = "y_pred",
    ticker_col: Optional[str] = "ticker",
) -> pd.DataFrame:
    """
    Load CSV predictions and keep the last row with identical date+ticker.
    """
    df = load_csv_predictions_raw(
        csv_path=csv_path,
        date_col=date_col,
        label_col=label_col,
        prediction_col=prediction_col,
        ticker_col=ticker_col,
    )
    df = _aggregate_predictions(df, mode="last", group_cols=["date", "ticker"])
    return _normalize_output_columns(df)


# =========================
# Universal loader
# =========================

def load_predictions(
    path: str | os.PathLike,
    duplicate_mode: AggregationMode = "mean",
    date_col: str = "date",
    label_col: str = "y_true",
    prediction_col: str = "y_pred",
    ticker_col: Optional[str] = "ticker",
) -> pd.DataFrame:
    """
    Universal loader for .npz or .csv.

    Parameters
    ----------
    path : str or PathLike
        File path to predictions.
    duplicate_mode : {"mean", "first"}
        How to resolve duplicate date+ticker rows.
    date_col, label_col, prediction_col, ticker_col
        Used only for CSV inputs.

    Returns
    -------
    pd.DataFrame
        Canonical dataframe with columns such as:
          date, ticker, label, prediction, ...
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        if duplicate_mode == "mean":
            return load_npz_predictions_mean(path)
        if duplicate_mode == "first":
            return load_npz_predictions_first(path)
        if duplicate_mode == "last":
            return load_npz_predictions_last(path)
        raise ValueError(f"Unknown duplicate_mode={duplicate_mode}")

    if suffix == ".csv":
        if duplicate_mode == "mean":
            return load_csv_predictions_mean(
                path,
                date_col=date_col,
                label_col=label_col,
                prediction_col=prediction_col,
                ticker_col=ticker_col,
            )
        if duplicate_mode == "first":
            return load_csv_predictions_first(
                path,
                date_col=date_col,
                label_col=label_col,
                prediction_col=prediction_col,
                ticker_col=ticker_col,
            )
        if duplicate_mode == "last":
            return load_csv_predictions_last(
                path,
                date_col=date_col,
                label_col=label_col,
                prediction_col=prediction_col,
                ticker_col=ticker_col,
            )
        raise ValueError(f"Unknown duplicate_mode={duplicate_mode}")

    raise ValueError(f"Unsupported file type: {path.suffix}. Expected .npz or .csv")


# =========================
# Convenience filters
# =========================

def filter_predictions(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    horizon_step: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simple filtering helper for downstream plotting/metrics.
    """
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])

    if ticker is not None:
        out = out[out["ticker"] == ticker]

    if start_date is not None:
        out = out[out["date"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        out = out[out["date"] <= pd.to_datetime(end_date)]

    if horizon_step is not None and "horizon_step" in out.columns:
        out = out[out["horizon_step"] == horizon_step]

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# =========================
# Alignment helper
# =========================

def align_two_prediction_frames(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    suffixes: tuple[str, str] = ("_1", "_2"),
) -> pd.DataFrame:
    """
    Align two canonical prediction dataframes on date+ticker.

    Returns dataframe with:
      date, ticker,
      label_1, prediction_1,
      label_2, prediction_2,
      ...
    """
    left = df_left.copy()
    right = df_right.copy()

    _validate_required_columns(left, ["date", "ticker", "label", "prediction"], "df_left")
    _validate_required_columns(right, ["date", "ticker", "label", "prediction"], "df_right")

    merge_cols = ["date", "ticker"]

    out = left.merge(
        right,
        on=merge_cols,
        how=how,
        suffixes=suffixes,
    )

    return out.sort_values(merge_cols).reset_index(drop=True)
