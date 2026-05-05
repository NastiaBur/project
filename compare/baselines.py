from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ProcessPoolExecutor, as_completed


ModelName = Literal[
    "ols",
    "ridge",
    "pcr",
    "pls",
    "rf",
    "predict_zero",
    "predict_previous",
    "linear_market",
]

AggregateMode = Literal["none", "mean", "latest", "first"]


# =========================
# Helpers
# =========================

def infer_date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in {"date", "datetime", "timestamp"}:
            return c
    raise ValueError("Could not find a date column in CSV.")


def infer_ticker_columns(df: pd.DataFrame, date_col: Optional[str] = None) -> list[str]:
    exclude = {
        "market_ret_1d",
        "fed_funds", "rate_10y", "rate_2y", "yc_slope_10y2y",
        "cpi", "unemp", "indpro", "vix", "baa_aaa_spread",
        "dbaa", "daaa", "is_eval_target",
    }
    if date_col is not None:
        exclude.add(date_col)

    ticker_cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        # Ticker return columns should be numeric and not datetimelike.
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            ticker_cols.append(c)
    return ticker_cols


def infer_macro_columns(df: pd.DataFrame) -> list[str]:
    possible = [
        "fed_funds", "rate_10y", "rate_2y", "yc_slope_10y2y",
        "cpi", "unemp", "indpro", "vix", "baa_aaa_spread",
        "dbaa", "daaa",
    ]
    return [c for c in possible if c in df.columns]


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

    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    huber = np.mean(0.5 * quad**2 + delta * lin)

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
# Aggregation
# =========================

def aggregate_baseline_predictions(
    pred_df: pd.DataFrame,
    mode: AggregateMode = "none",
) -> pd.DataFrame:
    """
    Aggregate overlapping predictions by ticker + date.

    none   : keep all rows
    mean   : average y_true/y_pred over same ticker+date
    latest : keep row with latest forecast_origin_date for same ticker+date
    first  : keep first row after sorting
    """
    if mode == "none":
        return pred_df.copy().reset_index(drop=True)

    required = ["ticker", "forecast_origin_date", "date", "y_true", "y_pred"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing columns for aggregation: {missing}")

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

    raise ValueError(f"Unknown aggregation mode={mode}")


# =========================
# Supervised sample builder
# =========================

def build_supervised_panel_samples(
    panel_df: pd.DataFrame,
    context_length: int,
    horizon: int,
    include_market: bool = True,
    include_macro: bool = True,
    only_eval_targets_for_prediction: bool = True,
):
    """
    Build pooled supervised samples across all tickers.

    Features:
      - past `context_length` firm returns
      - optional last available market_ret_1d at forecast origin
      - optional last available macro snapshot at forecast origin

    Targets:
      - next `horizon` firm returns

    Returns
    -------
    dict with:
      X, Y, meta_df
    """
    df = panel_df.copy()
    date_col = infer_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    ticker_cols = infer_ticker_columns(df, date_col=date_col)
    macro_cols = infer_macro_columns(df)

    use_eval_mask = "is_eval_target" in df.columns
    eval_mask = df["is_eval_target"].astype(int).values if use_eval_mask else np.ones(len(df), dtype=int)

    market_values = df["market_ret_1d"].values if ("market_ret_1d" in df.columns and include_market) else None
    macro_values = df[macro_cols].values if (len(macro_cols) > 0 and include_macro) else None
    all_dates = pd.to_datetime(df[date_col]).values

    X_list = []
    Y_list = []
    meta_rows = []

    for ticker in ticker_cols:
        y = df[ticker].astype(float).values

        for origin_idx in range(context_length, len(df) - horizon + 1):
            # context is [origin_idx-context_length, ..., origin_idx-1]
            ctx = y[origin_idx - context_length:origin_idx]
            tgt = y[origin_idx:origin_idx + horizon]

            if not np.all(np.isfinite(ctx)) or not np.all(np.isfinite(tgt)):
                continue

            feats = [ctx.astype(float)]

            # snapshot at forecast origin date = origin_idx - 1
            snap_idx = origin_idx - 1

            if market_values is not None:
                mv = market_values[snap_idx]
                if not np.isfinite(mv):
                    continue
                feats.append(np.array([mv], dtype=float))

            if macro_values is not None:
                macro_snap = macro_values[snap_idx]
                if not np.all(np.isfinite(macro_snap)):
                    continue
                feats.append(macro_snap.astype(float))

            x_vec = np.concatenate(feats, axis=0)

            block_eval = True
            if use_eval_mask and only_eval_targets_for_prediction:
                block_eval = bool(np.all(eval_mask[origin_idx:origin_idx + horizon] == 1))

            meta_rows.append({
                "ticker": ticker,
                "forecast_origin_date": pd.to_datetime(all_dates[origin_idx - 1]),
                "origin_idx": origin_idx,
                "target_end_idx": origin_idx + horizon - 1,
                "is_prediction_window": block_eval,
                "target_dates": list(pd.to_datetime(all_dates[origin_idx:origin_idx + horizon])),
            })
            X_list.append(x_vec)
            Y_list.append(tgt.astype(float))

    if len(X_list) == 0:
        raise ValueError("No supervised samples could be built.")

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    meta_df = pd.DataFrame(meta_rows)

    return {
        "X": X,
        "Y": Y,
        "meta": meta_df,
    }


# =========================
# Model factories
# =========================

def _make_estimator(
    model_name: ModelName,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    ridge_alpha: float = 1.0,
    pcr_n_components: int = 10,
    pls_n_components: int = 5,
    rf_n_estimators: int = 200,
    rf_max_depth: Optional[int] = None,
    rf_min_samples_leaf: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
):
    n_samples, n_features = X_train.shape
    n_targets = Y_train.shape[1]

    if model_name == "ols":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    if model_name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ridge_alpha)),
        ])

    if model_name == "pcr":
        n_comp = max(1, min(pcr_n_components, n_features, n_samples - 1))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=random_state)),
            ("model", LinearRegression()),
        ])

    if model_name == "pls":
        n_comp = max(1, min(pls_n_components, n_features, n_targets, n_samples - 1))
        return PLSRegression(n_components=n_comp, scale=True)

    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    raise ValueError(f"Unsupported model_name={model_name}")


# =========================
# Simple baselines
# =========================

def _predict_zero(X_eval: np.ndarray, horizon: int) -> np.ndarray:
    return np.zeros((X_eval.shape[0], horizon), dtype=float)


def _predict_previous(X_eval: np.ndarray, horizon: int) -> np.ndarray:
    # last lag is the most recent firm return
    last_ret = X_eval[:, -1].reshape(-1, 1)
    return np.repeat(last_ret, horizon, axis=1)


def _fit_linear_market_baseline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    market_feature_position: int,
) -> LinearRegression:
    """
    Uses only one market feature from X as predictor.
    Multi-output linear regression.
    """
    if market_feature_position < 0 or market_feature_position >= X_train.shape[1]:
        raise ValueError("market_feature_position out of bounds")

    model = LinearRegression()
    model.fit(X_train[:, [market_feature_position]], Y_train)
    return model


def _ensure_2d_target_array(arr: np.ndarray, horizon: int, name: str) -> np.ndarray:
    """
    Normalize target/prediction arrays to shape (n_samples, horizon).
    """
    out = np.asarray(arr)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    elif out.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape={out.shape}")

    if out.shape[1] != horizon:
        raise ValueError(
            f"{name} second dim must match horizon={horizon}, got shape={out.shape}"
        )
    return out


# =========================
# Main baseline runner
# =========================

def run_panel_baseline(
    csv_path: str | Path,
    model_name: ModelName,
    *,
    context_length: int = 256,
    horizon: int = 64,
    step: int = 64,
    include_market: bool = True,
    include_macro: bool = True,
    only_eval_targets: bool = True,
    aggregate_mode: AggregateMode = "none",
    save_path: Optional[str | Path] = None,
    show_progress: bool = True,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
    pcr_n_components: int = 10,
    pls_n_components: int = 5,
    rf_n_estimators: int = 200,
    rf_max_depth: Optional[int] = None,
    rf_min_samples_leaf: int = 1,
    rf_n_jobs: int = -1,
    n_jobs_outer: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """
    Pooled baseline over all tickers.

    n_jobs_outer:
      parallelism across forecast origins.
      If >1, recommended to set rf_n_jobs=1 for RF to avoid nested parallelism.
    """
    panel_df = pd.read_csv(csv_path)
    date_col = infer_date_column(panel_df)
    panel_df[date_col] = pd.to_datetime(panel_df[date_col])

    built = build_supervised_panel_samples(
        panel_df=panel_df,
        context_length=context_length,
        horizon=horizon,
        include_market=include_market,
        include_macro=include_macro,
        only_eval_targets_for_prediction=only_eval_targets,
    )

    X = built["X"]
    Y = built["Y"]
    meta = built["meta"].copy()

    pred_meta = meta[meta["is_prediction_window"]].copy()
    if len(pred_meta) == 0:
        raise ValueError("No evaluation windows available for prediction.")

    unique_origins = sorted(pred_meta["origin_idx"].unique())
    if step > 1:
        unique_origins = unique_origins[::step]

    all_rows = []

    # avoid nested parallelism for RF
    effective_rf_n_jobs = rf_n_jobs
    if model_name == "rf" and n_jobs_outer > 1:
        effective_rf_n_jobs = 1

    if n_jobs_outer == 1:
        iterator = unique_origins
        if show_progress:
            iterator = tqdm(unique_origins, desc=f"Baseline {model_name}")

        for origin_idx in iterator:
            rows = _run_one_origin_baseline(
                origin_idx=origin_idx,
                model_name=model_name,
                X=X,
                Y=Y,
                meta=meta,
                pred_meta=pred_meta,
                horizon=horizon,
                context_length=context_length,
                include_market=include_market,
                random_state=random_state,
                ridge_alpha=ridge_alpha,
                pcr_n_components=pcr_n_components,
                pls_n_components=pls_n_components,
                rf_n_estimators=rf_n_estimators,
                rf_max_depth=rf_max_depth,
                rf_min_samples_leaf=rf_min_samples_leaf,
                rf_n_jobs=effective_rf_n_jobs,
            )
            all_rows.extend(rows)

    else:
        with ProcessPoolExecutor(max_workers=n_jobs_outer) as ex:
            futures = [
                ex.submit(
                    _run_one_origin_baseline,
                    origin_idx,
                    model_name,
                    X,
                    Y,
                    meta,
                    pred_meta,
                    horizon,
                    context_length,
                    include_market,
                    random_state,
                    ridge_alpha,
                    pcr_n_components,
                    pls_n_components,
                    rf_n_estimators,
                    rf_max_depth,
                    rf_min_samples_leaf,
                    effective_rf_n_jobs,
                )
                for origin_idx in unique_origins
            ]

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=f"Baseline {model_name}")

            for fut in iterator:
                rows = fut.result()
                all_rows.extend(rows)

    if len(all_rows) == 0:
        out_df = pd.DataFrame(
            columns=["ticker", "forecast_origin_date", "date", "horizon_step", "y_true", "y_pred"]
        )
    else:
        out_df = pd.DataFrame(all_rows).sort_values(
            ["ticker", "forecast_origin_date", "horizon_step"]
        ).reset_index(drop=True)

    out_df = aggregate_baseline_predictions(out_df, mode=aggregate_mode)
    metrics = compute_metrics(out_df["y_true"].values, out_df["y_pred"].values)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(save_path, index=False)

    return out_df, metrics

def _run_one_origin_baseline(
    origin_idx: int,
    model_name: ModelName,
    X: np.ndarray,
    Y: np.ndarray,
    meta: pd.DataFrame,
    pred_meta: pd.DataFrame,
    horizon: int,
    context_length: int,
    include_market: bool,
    random_state: int,
    ridge_alpha: float,
    pcr_n_components: int,
    pls_n_components: int,
    rf_n_estimators: int,
    rf_max_depth: Optional[int],
    rf_min_samples_leaf: int,
    rf_n_jobs: int,
):
    rows = []

    eval_mask = pred_meta["origin_idx"] == origin_idx
    eval_idx = pred_meta.index[eval_mask].to_numpy()

    train_idx = meta.index[meta["target_end_idx"] < origin_idx].to_numpy()
    if len(train_idx) == 0 or len(eval_idx) == 0:
        return rows

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_eval = X[eval_idx]
    Y_eval = Y[eval_idx]
    Y_eval = _ensure_2d_target_array(Y_eval, horizon=horizon, name="Y_eval")

    market_feature_position = context_length if include_market else None

    if model_name in {"predict_zero", "predict_previous"}:
        if model_name == "predict_zero":
            Y_pred = _predict_zero(X_eval, horizon=horizon)
        else:
            Y_pred = _predict_previous(X_eval, horizon=horizon)

    elif model_name == "linear_market":
        if market_feature_position is None:
            raise ValueError("linear_market baseline requires include_market=True.")
        model = _fit_linear_market_baseline(
            X_train=X_train,
            Y_train=Y_train,
            market_feature_position=market_feature_position,
        )
        Y_pred = model.predict(X_eval[:, [market_feature_position]])

    else:
        model = _make_estimator(
            model_name=model_name,
            X_train=X_train,
            Y_train=Y_train,
            ridge_alpha=ridge_alpha,
            pcr_n_components=pcr_n_components,
            pls_n_components=pls_n_components,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state,
            n_jobs=rf_n_jobs,
        )
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_eval)

    Y_pred = _ensure_2d_target_array(Y_pred, horizon=horizon, name="Y_pred")

    eval_block = pred_meta.loc[eval_idx]
    for j, (_, row_meta) in enumerate(eval_block.iterrows()):
        target_dates = row_meta["target_dates"]
        ticker = row_meta["ticker"]
        forecast_origin_date = row_meta["forecast_origin_date"]

        for h in range(horizon):
            rows.append({
                "ticker": ticker,
                "forecast_origin_date": pd.to_datetime(forecast_origin_date),
                "date": pd.to_datetime(target_dates[h]),
                "horizon_step": h + 1,
                "y_true": float(Y_eval[j, h]),
                "y_pred": float(Y_pred[j, h]),
            })

    return rows