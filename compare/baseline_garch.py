from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Literal, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from arch import arch_model
from tqdm.auto import tqdm


AggregateMode = Literal["none", "mean", "latest", "first"]


# =========================
# Metrics
# =========================

def huber_loss(y_true, y_pred, delta=2.0):
    err = y_true - y_pred
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    return 0.5 * quad**2 + delta * lin


def compute_metrics(y_true, y_pred, delta=2.0):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "huber": np.nan,
            "sign_accuracy": np.nan,
        }

    err = y_true - y_pred
    mse = np.mean(err ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(err))
    huber = np.mean(huber_loss(y_true, y_pred, delta=delta))
    sign_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {
        "n": int(len(y_true)),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "huber": float(huber),
        "sign_accuracy": float(sign_acc),
    }


# =========================
# Helpers
# =========================

def infer_date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in {"date", "datetime", "timestamp"}:
            return c
    raise ValueError("Could not find a date column in CSV.")


def infer_ticker_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "market_ret_1d",
        "fed_funds", "rate_10y", "rate_2y", "yc_slope_10y2y",
        "cpi", "unemp", "indpro", "vix", "baa_aaa_spread",
        "dbaa", "daaa", "is_eval_target",
    }
    return [c for c in df.columns if c not in exclude]


# =========================
# Core fit + forecast
# =========================

def fit_garch_and_forecast(
    train_values: np.ndarray,
    horizon: int = 1,
    mean_type: str = "AR",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> tuple[np.ndarray, object]:
    """
    Fit one GARCH model and forecast `horizon` mean returns.
    Returns:
      pred, fitted_result
    """
    train_values = np.asarray(train_values, dtype=float).reshape(-1)

    if len(train_values) < 10:
        raise ValueError("train_values too short for GARCH fit.")

    scale = 100.0
    train_scaled = train_values * scale

    if mean_type.upper() == "AR":
        am = arch_model(
            train_scaled,
            mean="AR",
            lags=1,
            vol="GARCH",
            p=p,
            q=q,
            dist=dist,
            rescale=False,
        )
    elif mean_type.upper() == "CONSTANT":
        am = arch_model(
            train_scaled,
            mean="Constant",
            vol="GARCH",
            p=p,
            q=q,
            dist=dist,
            rescale=False,
        )
    else:
        raise ValueError(f"Unsupported mean_type={mean_type}. Use 'AR' or 'Constant'.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)

    fcast = res.forecast(horizon=horizon, reindex=False)
    pred_scaled = np.asarray(fcast.mean.values[-1], dtype=float).reshape(-1)

    if len(pred_scaled) != horizon:
        raise ValueError(f"Expected horizon={horizon}, got len(pred_scaled)={len(pred_scaled)}")

    pred = pred_scaled / scale
    return pred, res


def forecast_from_existing_result(res, horizon: int) -> np.ndarray:
    """
    Approximate forecast using already fitted result.
    This does NOT refit on the newly extended history.
    Useful only when refit_every > 1.
    """
    fcast = res.forecast(horizon=horizon, reindex=False)
    pred_scaled = np.asarray(fcast.mean.values[-1], dtype=float).reshape(-1)
    pred = pred_scaled / 100.0
    return pred


# =========================
# Per-ticker worker
# =========================

def _run_garch_for_one_ticker(
    ticker: str,
    ticker_series_values: np.ndarray,
    all_dates: np.ndarray,
    eval_mask: np.ndarray,
    warmup: int,
    horizon: int,
    step: int,
    mean_type: str,
    p: int,
    q: int,
    dist: str,
    only_eval_targets: bool,
    refit_every: int,
) -> pd.DataFrame:
    """
    Sequential GARCH forecasts for one ticker.
    Can be called inside a subprocess.
    """
    values = np.asarray(ticker_series_values, dtype=float)
    dates = pd.to_datetime(all_dates)

    rows = []
    cached_res = None
    cached_pred = None
    fit_counter_since_refit = 0

    origins = range(warmup, len(values) - horizon + 1, step)

    for t in origins:
        if only_eval_targets:
            block_mask = eval_mask[t:t + horizon]
            if len(block_mask) < horizon or not np.all(block_mask == 1):
                continue

        target_true = values[t:t + horizon]
        target_dates = dates[t:t + horizon]

        if not np.all(np.isfinite(target_true)):
            continue

        train = values[:t]
        train = train[np.isfinite(train)]
        if len(train) < warmup:
            continue

        try:
            need_refit = (
                cached_res is None
                or refit_every <= 1
                or fit_counter_since_refit >= refit_every
            )

            if need_refit:
                pred, cached_res = fit_garch_and_forecast(
                    train_values=train,
                    horizon=horizon,
                    mean_type=mean_type,
                    p=p,
                    q=q,
                    dist=dist,
                )
                cached_pred = pred
                fit_counter_since_refit = 0
            else:
                # approximate mode
                pred = forecast_from_existing_result(cached_res, horizon=horizon)
                cached_pred = pred

            fit_counter_since_refit += 1

            forecast_origin_date = dates[t - 1]

            for h in range(horizon):
                rows.append({
                    "ticker": ticker,
                    "forecast_origin_date": pd.to_datetime(forecast_origin_date),
                    "date": pd.to_datetime(target_dates[h]),
                    "horizon_step": h + 1,
                    "y_true": float(target_true[h]),
                    "y_pred": float(pred[h]),
                })

        except Exception:
            continue

    if len(rows) == 0:
        return pd.DataFrame(
            columns=["ticker", "forecast_origin_date", "date", "horizon_step", "y_true", "y_pred"]
        )

    return pd.DataFrame(rows)


# =========================
# Aggregation
# =========================

def aggregate_garch_predictions(
    pred_df: pd.DataFrame,
    mode: AggregateMode = "none",
) -> pd.DataFrame:
    if mode == "none":
        return pred_df.copy().reset_index(drop=True)

    required = ["ticker", "forecast_origin_date", "date", "y_true", "y_pred"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"pred_df missing columns for aggregation: {missing}")

    df = pred_df.copy()
    df["forecast_origin_date"] = pd.to_datetime(df["forecast_origin_date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date", "forecast_origin_date", "horizon_step"]).reset_index(drop=True)

    group_cols = ["ticker", "date"]

    if mode == "mean":
        out = df.groupby(group_cols, as_index=False).agg({
            "y_true": "mean",
            "y_pred": "mean",
            "forecast_origin_date": "first",
            "horizon_step": "first",
        })
        return out.sort_values(group_cols).reset_index(drop=True)

    if mode == "latest":
        idx = df.groupby(group_cols)["forecast_origin_date"].idxmax()
        out = df.loc[idx].sort_values(group_cols).reset_index(drop=True)
        return out

    if mode == "first":
        out = df.drop_duplicates(subset=group_cols, keep="first").reset_index(drop=True)
        return out

    raise ValueError(f"Unknown mode={mode}")


# =========================
# Main function for notebook use
# =========================

def run_garch_baseline(
    csv_path: str | Path,
    warmup: int = 256,
    horizon: int = 64,
    step: int = 64,
    mean_type: str = "AR",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    tickers_subset: Optional[list[str]] = None,
    only_eval_targets: bool = True,
    aggregate_mode: AggregateMode = "none",
    save_path: Optional[str | Path] = None,
    n_jobs: int = 1,
    refit_every: int = 1,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Practical GARCH baseline runner.

    Parameters
    ----------
    refit_every : int
        1 = exact refit on every forecast origin
        >1 = approximate speed-up, refit less often
    n_jobs : int
        Number of parallel worker processes across tickers.
    """
    df = pd.read_csv(csv_path)
    date_col = infer_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    if only_eval_targets and "is_eval_target" in df.columns:
        eval_mask = df["is_eval_target"].astype(int).values
    else:
        eval_mask = np.ones(len(df), dtype=int)

    candidate_cols = infer_ticker_columns(df)
    if tickers_subset is not None:
        tickers_subset = set(tickers_subset)
        candidate_cols = [c for c in candidate_cols if c in tickers_subset]

    all_dates = df.index.values

    tasks = []
    for ticker in candidate_cols:
        values = df[ticker].astype(float).values
        tasks.append((ticker, values))

    all_parts = []

    if n_jobs == 1:
        iterator = tasks
        if show_progress:
            iterator = tqdm(tasks, desc="GARCH by ticker")

        for ticker, values in iterator:
            part = _run_garch_for_one_ticker(
                ticker=ticker,
                ticker_series_values=values,
                all_dates=all_dates,
                eval_mask=eval_mask,
                warmup=warmup,
                horizon=horizon,
                step=step,
                mean_type=mean_type,
                p=p,
                q=q,
                dist=dist,
                only_eval_targets=only_eval_targets,
                refit_every=refit_every,
            )
            if len(part):
                all_parts.append(part)

    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = []
            for ticker, values in tasks:
                fut = ex.submit(
                    _run_garch_for_one_ticker,
                    ticker,
                    values,
                    all_dates,
                    eval_mask,
                    warmup,
                    horizon,
                    step,
                    mean_type,
                    p,
                    q,
                    dist,
                    only_eval_targets,
                    refit_every,
                )
                futures.append(fut)

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="GARCH by ticker")

            for fut in iterator:
                part = fut.result()
                if len(part):
                    all_parts.append(part)

    if len(all_parts) == 0:
        raw_df = pd.DataFrame(
            columns=["ticker", "forecast_origin_date", "date", "horizon_step", "y_true", "y_pred"]
        )
    else:
        raw_df = pd.concat(all_parts, axis=0, ignore_index=True)
        raw_df = raw_df.sort_values(
            ["ticker", "forecast_origin_date", "horizon_step"]
        ).reset_index(drop=True)

    out_df = aggregate_garch_predictions(raw_df, mode=aggregate_mode)

    metrics = compute_metrics(
        y_true=out_df["y_true"].values if len(out_df) else np.array([]),
        y_pred=out_df["y_pred"].values if len(out_df) else np.array([]),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(save_path, index=False)

    return out_df, metrics
