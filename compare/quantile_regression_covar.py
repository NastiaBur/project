from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import statsmodels.api as sm


InstitutionMode = Literal["true", "pred"]


# =========================================================
# Helpers
# =========================================================

def _infer_first_existing(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {what}. Tried columns: {candidates}")


def _load_panel_df(panel: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(panel, (str, Path)):
        x = pd.read_csv(panel)
    else:
        x = panel.copy()

    date_col = _infer_first_existing(x, ["date", "datetime", "timestamp", 'Date'], "date column")
    x[date_col] = pd.to_datetime(x[date_col])
    x = x.rename(columns={date_col: "date"}).sort_values("date").reset_index(drop=True)
    return x


def _standardize_prediction_df(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize input predictions to columns:
      date, ticker(optional), true, pred
    """
    x = df.copy()

    date_col = _infer_first_existing(x, ["date", "datetime", "timestamp", 'Date'], "date column")
    true_col = _infer_first_existing(x, ["y_true", "true", "label", "labels"], "true-value column")
    pred_col = _infer_first_existing(
        x, ["y_pred", "pred", "pred_mean", "prediction", "predictions"], "prediction column"
    )

    x[date_col] = pd.to_datetime(x[date_col])

    out = pd.DataFrame({
        "date": x[date_col],
        "true": pd.to_numeric(x[true_col], errors="coerce"),
        "pred": pd.to_numeric(x[pred_col], errors="coerce"),
    })

    if "ticker" in x.columns:
        out["ticker"] = x["ticker"].astype(str)

    if ticker is not None:
        if "ticker" not in out.columns:
            raise ValueError("Ticker filter requested but dataframe has no ticker column.")
        out = out[out["ticker"] == ticker].copy()

    out = out.dropna(subset=["date", "true", "pred"]).sort_values("date").reset_index(drop=True)
    return out


def _aggregate_same_day_predictions(
    df: pd.DataFrame,
    how: str = "mean",
) -> pd.DataFrame:
    x = df.copy()

    group_cols = ["date"]
    if "ticker" in x.columns:
        group_cols = ["ticker", "date"]

    x = x.sort_values(group_cols).reset_index(drop=True)

    if how == "mean":
        agg = (
            x.groupby(group_cols, as_index=False)[["true", "pred"]]
            .mean()
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        return agg

    if how == "first":
        return x.drop_duplicates(subset=group_cols, keep="first").reset_index(drop=True)

    raise ValueError(f"Unknown aggregation mode: {how}")


def _default_control_cols(panel_df: pd.DataFrame) -> list[str]:
    candidates = [
        "market_ret_1d",   # keep if you want richer system state; can remove manually
        "fed_funds",
        "rate_10y",
        "rate_2y",
        "yc_slope_10y2y",
        "cpi",
        "unemp",
        "indpro",
        "vix",
        "baa_aaa_spread",
        "dbaa",
        "daaa",
    ]
    out = [c for c in candidates if c in panel_df.columns]
    # In system equation market_ret_1d is dependent variable, not control.
    if "market_ret_1d" in out:
        out.remove("market_ret_1d")
    return out


def _fit_quantreg(X: pd.DataFrame, y: pd.Series, q: float):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.QuantReg(y, Xc)
    res = model.fit(q=q)
    return res


def _predict_quantreg(res, X: pd.DataFrame) -> np.ndarray:
    Xc = sm.add_constant(X, has_constant="add")
    return np.asarray(res.predict(Xc), dtype=float)

def _predict_quantreg_with_ci(
    res,
    X: pd.DataFrame,
    z_value: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pred, se, lower_ci, upper_ci

    CI is coefficient-based:
      pred +/- z * sqrt(x' Cov(beta) x)
    """
    Xc = sm.add_constant(X, has_constant="add")
    Xv = np.asarray(Xc, dtype=float)

    pred = np.asarray(res.predict(Xc), dtype=float)

    cov = res.cov_params()
    if isinstance(cov, pd.DataFrame):
        cov = cov.values
    cov = np.asarray(cov, dtype=float)

    # diag(X Cov X')
    var_pred = np.einsum("ij,jk,ik->i", Xv, cov, Xv)
    var_pred = np.maximum(var_pred, 0.0)
    se = np.sqrt(var_pred)

    lower = pred - z_value * se
    upper = pred + z_value * se
    return pred, se, lower, upper


def _linear_combo_ci_from_same_model(
    res,
    X_left: pd.DataFrame,
    X_right: pd.DataFrame,
    z_value: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For quantities of the form:
      diff = x_left' beta - x_right' beta = (x_left - x_right)' beta

    Returns:
      diff, se, lower_ci, upper_ci
    """
    Xl = sm.add_constant(X_left, has_constant="add")
    Xr = sm.add_constant(X_right, has_constant="add")

    Xlv = np.asarray(Xl, dtype=float)
    Xrv = np.asarray(Xr, dtype=float)
    Xdv = Xlv - Xrv

    pred_left = np.asarray(res.predict(Xl), dtype=float)
    pred_right = np.asarray(res.predict(Xr), dtype=float)
    diff = pred_left - pred_right

    cov = res.cov_params()
    if isinstance(cov, pd.DataFrame):
        cov = cov.values
    cov = np.asarray(cov, dtype=float)

    var_diff = np.einsum("ij,jk,ik->i", Xdv, cov, Xdv)
    var_diff = np.maximum(var_diff, 0.0)
    se = np.sqrt(var_diff)

    lower = diff - z_value * se
    upper = diff + z_value * se
    return diff, se, lower, upper


# =========================================================
# Main computation
# =========================================================

def compute_quantile_regression_covar(
    pred_df: pd.DataFrame,
    panel: Union[str, Path, pd.DataFrame],
    ticker: str,
    *,
    q: float = 0.05,
    median_q: float = 0.50,
    institution_mode: InstitutionMode = "pred",
    control_cols: Optional[list[str]] = None,
    aggregate_same_date: str = "mean",
    warmup: int = 252,
    step: int = 1,
    show_progress: bool = True,
    z_value: float = 1.96,
) -> pd.DataFrame:
    """
    Two-step QR CoVaR in the spirit of Adrian & Brunnermeier.

    Step 1: Institution VaR / median
      firm_return_t ~ controls_t    via QuantReg(q) and QuantReg(0.5)

    Step 2: System tail regression
      market_return_t ~ institution_state_t + controls_t   via QuantReg(q)

    Then:
      CoVaR_distress_t = a_q + b_q * VaR_i_t + g_q' controls_t
      CoVaR_median_t   = a_q + b_q * Median_i_t + g_q' controls_t
      DeltaCoVaR_t     = CoVaR_distress_t - CoVaR_median_t

    institution_mode:
      - "true": use true firm return in the system equation
      - "pred": use predicted firm return in the system equation
    """
    if not (0 < q < 1):
        raise ValueError(f"q must be in (0,1), got {q}")

    x_pred = _standardize_prediction_df(pred_df, ticker=ticker)
    x_pred = _aggregate_same_day_predictions(x_pred, how=aggregate_same_date)

    panel_df = _load_panel_df(panel)

    if ticker not in panel_df.columns:
        raise ValueError(f"Ticker {ticker} not found in panel.")
    if "market_ret_1d" not in panel_df.columns:
        raise ValueError("panel must contain 'market_ret_1d'.")

    if control_cols is None:
        control_cols = _default_control_cols(panel_df)

    keep_cols = ["date", "market_ret_1d", ticker] + control_cols
    keep_cols = [c for c in keep_cols if c in panel_df.columns]
    panel_small = panel_df[keep_cols].copy()

    merged = x_pred.merge(panel_small, on="date", how="inner")
    merged = merged.rename(columns={
        ticker: "firm_true_panel",
        "market_ret_1d": "system_ret",
    })

    # consistency: use true from pred_df as firm true if available
    # fallback to panel firm return if there is any discrepancy
    if "true" not in merged.columns:
        merged["true"] = merged["firm_true_panel"]

    # institution state used in step 2
    if institution_mode == "pred":
        merged["institution_state"] = merged["pred"]
    elif institution_mode == "true":
        merged["institution_state"] = merged["true"]
    else:
        raise ValueError(f"Unknown institution_mode={institution_mode}")

    needed = ["date", "true", "pred", "institution_state", "system_ret"] + control_cols
    merged = merged[needed].copy()

    # drop rows with any NaNs in required fields
    merged = merged.dropna().sort_values("date").reset_index(drop=True)

    if len(merged) <= warmup:
        raise ValueError(
            f"Not enough rows after merge. Need > warmup={warmup}, got {len(merged)}."
        )

    rows = []
    iterator = range(warmup, len(merged), step)
    if show_progress:
        iterator = tqdm(iterator, desc=f"QR-CoVaR {ticker}")

    for t in iterator:
        train = merged.iloc[:t].copy()
        test_row = merged.iloc[[t]].copy()

        # -------- Step 1: institution VaR / Median from controls --------
        X_i_train = train[control_cols]
        y_i_train = train["true"]

        try:
            res_i_var = _fit_quantreg(X_i_train, y_i_train, q=q)
            res_i_med = _fit_quantreg(X_i_train, y_i_train, q=median_q)
        except Exception:
            continue

        X_i_test = test_row[control_cols]

        try:
            var_i_pred, var_i_se, var_i_lower, var_i_upper = _predict_quantreg_with_ci(
                res_i_var, X_i_test, z_value=z_value
            )
            med_i_pred, med_i_se, med_i_lower, med_i_upper = _predict_quantreg_with_ci(
                res_i_med, X_i_test, z_value=z_value
            )
        except Exception:
            continue

        var_i_t = float(var_i_pred[0])
        med_i_t = float(med_i_pred[0])

        # -------- Step 2: system QR on institution state + controls --------
        sys_features = ["institution_state"] + control_cols
        X_s_train = train[sys_features]
        y_s_train = train["system_ret"]

        try:
            res_s = _fit_quantreg(X_s_train, y_s_train, q=q)
        except Exception:
            continue

        # distress state = VaR_i_t
        X_s_distress = test_row[control_cols].copy()
        X_s_distress.insert(0, "institution_state", var_i_t)

        # median state = Median_i_t
        X_s_median = test_row[control_cols].copy()
        X_s_median.insert(0, "institution_state", med_i_t)

        try:
            covar_pred, covar_se, covar_lower, covar_upper = _predict_quantreg_with_ci(
                res_s, X_s_distress, z_value=z_value
            )
            covar_med_pred, covar_med_se, covar_med_lower, covar_med_upper = _predict_quantreg_with_ci(
                res_s, X_s_median, z_value=z_value
            )

            delta_pred, delta_se, delta_lower, delta_upper = _linear_combo_ci_from_same_model(
                res_s, X_s_distress, X_s_median, z_value=z_value
            )
        except Exception:
            continue

        covar_distress_t = float(covar_pred[0])
        covar_median_t = float(covar_med_pred[0])
        delta_covar_t = float(delta_pred[0])

        rows.append({
            "date": pd.to_datetime(test_row["date"].iloc[0]),
            "firm_true": float(test_row["true"].iloc[0]),
            "firm_pred": float(test_row["pred"].iloc[0]),
            "institution_state": float(test_row["institution_state"].iloc[0]),
            "system_ret": float(test_row["system_ret"].iloc[0]),

            "var_qr": var_i_t,
            "var_qr_se": float(var_i_se[0]),
            "var_qr_lower_ci": float(var_i_lower[0]),
            "var_qr_upper_ci": float(var_i_upper[0]),

            "median_qr": med_i_t,
            "median_qr_se": float(med_i_se[0]),
            "median_qr_lower_ci": float(med_i_lower[0]),
            "median_qr_upper_ci": float(med_i_upper[0]),

            "covar_qr": covar_distress_t,
            "covar_qr_se": float(covar_se[0]),
            "covar_qr_lower_ci": float(covar_lower[0]),
            "covar_qr_upper_ci": float(covar_upper[0]),

            "covar_median_qr": covar_median_t,
            "covar_median_qr_se": float(covar_med_se[0]),
            "covar_median_qr_lower_ci": float(covar_med_lower[0]),
            "covar_median_qr_upper_ci": float(covar_med_upper[0]),

            "delta_covar_qr": delta_covar_t,
            "delta_covar_qr_se": float(delta_se[0]),
            "delta_covar_qr_lower_ci": float(delta_lower[0]),
            "delta_covar_qr_upper_ci": float(delta_upper[0]),

            "q": q,
            "institution_mode": institution_mode,
            "ticker": ticker,
        })

        rows.append({
            "date": pd.to_datetime(test_row["date"].iloc[0]),
            "firm_true": float(test_row["true"].iloc[0]),
            "firm_pred": float(test_row["pred"].iloc[0]),
            "institution_state": float(test_row["institution_state"].iloc[0]),
            "system_ret": float(test_row["system_ret"].iloc[0]),
            "var_qr": var_i_t,
            "median_qr": med_i_t,
            "covar_qr": covar_distress_t,
            "covar_median_qr": covar_median_t,
            "delta_covar_qr": delta_covar_t,
            "q": q,
            "institution_mode": institution_mode,
            "ticker": ticker,
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def compute_reference_quantile_regression_covar(
    pred_df: pd.DataFrame,
    panel: Union[str, Path, pd.DataFrame],
    ticker: str,
    *,
    q: float = 0.05,
    median_q: float = 0.50,
    control_cols: Optional[list[str]] = None,
    aggregate_same_date: str = "mean",
    warmup: int = 252,
    step: int = 1,
    show_progress: bool = True,
    z_value: float = 1.96,
) -> pd.DataFrame:
    """
    Convenience wrapper:
    builds reference CoVaR / Delta-CoVaR using TRUE institution returns.
    """
    return compute_quantile_regression_covar(
        pred_df=pred_df,
        panel=panel,
        ticker=ticker,
        q=q,
        median_q=median_q,
        institution_mode="true",
        control_cols=control_cols,
        aggregate_same_date=aggregate_same_date,
        warmup=warmup,
        step=step,
        show_progress=show_progress,
        z_value=z_value,
    )


def _safe_rmse(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def _rolling_quantile_band(
    center: pd.Series,
    residual: pd.Series,
    window: int = 60,
    min_periods: int = 20,
    alpha: float = 0.10,
) -> tuple[pd.Series, pd.Series]:
    """
    Build an empirical uncertainty band around `center`
    using rolling quantiles of `residual`.

    Band:
      lower = center + rolling_quantile(residual, alpha/2)
      upper = center + rolling_quantile(residual, 1 - alpha/2)

    Example:
      alpha=0.10 -> central 90% band
    """
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0

    low_resid = residual.rolling(window=window, min_periods=min_periods).quantile(q_low)
    high_resid = residual.rolling(window=window, min_periods=min_periods).quantile(q_high)

    # fallback for early rows
    if low_resid.notna().sum() > 0:
        low_resid = low_resid.fillna(low_resid.dropna().iloc[0])
    else:
        low_resid = pd.Series(np.full(len(residual), np.nanquantile(residual, q_low)), index=residual.index)

    if high_resid.notna().sum() > 0:
        high_resid = high_resid.fillna(high_resid.dropna().iloc[0])
    else:
        high_resid = pd.Series(np.full(len(residual), np.nanquantile(residual, q_high)), index=residual.index)

    lower = center + low_resid
    upper = center + high_resid
    return lower, upper


def evaluate_covar_tracking(
    predicted_qr_df: pd.DataFrame,
    reference_qr_df: pd.DataFrame,
    *,
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Tracking metrics between predicted/reference CoVaR series.

    Expected columns in both dfs:
      date, covar_qr, delta_covar_qr
    Optional:
      ticker

    Returns:
      {
        "n": ...,
        "covar_rmse": ...,
        "covar_corr": ...,
        "delta_covar_rmse": ...,
        "delta_covar_corr": ...
      }
    """
    pred = predicted_qr_df.copy()
    ref = reference_qr_df.copy()

    if ticker is not None:
        if "ticker" in pred.columns:
            pred = pred[pred["ticker"] == ticker].copy()
        if "ticker" in ref.columns:
            ref = ref[ref["ticker"] == ticker].copy()

    pred["date"] = pd.to_datetime(pred["date"])
    ref["date"] = pd.to_datetime(ref["date"])

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        pred = pred[pred["date"] >= start_ts].copy()
        ref = ref[ref["date"] >= start_ts].copy()

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        pred = pred[pred["date"] <= end_ts].copy()
        ref = ref[ref["date"] <= end_ts].copy()

    need_cols = ["date", "covar_qr", "delta_covar_qr"]
    for c in need_cols:
        if c not in pred.columns:
            raise ValueError(f"predicted_qr_df missing column: {c}")
        if c not in ref.columns:
            raise ValueError(f"reference_qr_df missing column: {c}")

    merged = pred[["date", "covar_qr", "delta_covar_qr"]].merge(
        ref[["date", "covar_qr", "delta_covar_qr"]],
        on="date",
        how="inner",
        suffixes=("_pred", "_ref"),
    )

    covar_pred = merged["covar_qr_pred"].to_numpy(dtype=float)
    covar_ref = merged["covar_qr_ref"].to_numpy(dtype=float)

    delta_pred = merged["delta_covar_qr_pred"].to_numpy(dtype=float)
    delta_ref = merged["delta_covar_qr_ref"].to_numpy(dtype=float)

    out = {
        "n": int(len(merged)),
        "covar_rmse": _safe_rmse(covar_pred, covar_ref),
        "covar_corr": _safe_corr(covar_pred, covar_ref),
        "delta_covar_rmse": _safe_rmse(delta_pred, delta_ref),
        "delta_covar_corr": _safe_corr(delta_pred, delta_ref),
    }
    return out

def add_covar_uncertainty_bands(
    qr_df: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: int = 20,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """
    Add empirical uncertainty bands for CoVaR based on rolling residuals:

      residual_covar = system_ret - covar_qr

    Produces:
      covar_band_lower
      covar_band_upper
    """
    x = qr_df.copy().sort_values("date").reset_index(drop=True)

    required = {"system_ret", "covar_qr"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"Missing columns for CoVaR band: {missing}")

    resid = x["system_ret"] - x["covar_qr"]
    lower, upper = _rolling_quantile_band(
        center=x["covar_qr"],
        residual=resid,
        window=window,
        min_periods=min_periods,
        alpha=alpha,
    )

    x["covar_tracking_resid"] = resid
    x["covar_band_lower"] = lower
    x["covar_band_upper"] = upper
    return x

def add_delta_covar_uncertainty_bands(
    qr_df: pd.DataFrame,
    reference_qr_df: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: int = 20,
    alpha: float = 0.10,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add empirical uncertainty bands for Delta-CoVaR based on rolling residuals
    versus a reference Delta-CoVaR series.

      residual_delta = delta_covar_qr_pred - delta_covar_qr_ref

    Produces:
      delta_covar_band_lower
      delta_covar_band_upper
    """
    pred = qr_df.copy()
    ref = reference_qr_df.copy()

    if ticker is not None:
        if "ticker" in pred.columns:
            pred = pred[pred["ticker"] == ticker].copy()
        if "ticker" in ref.columns:
            ref = ref[ref["ticker"] == ticker].copy()

    pred["date"] = pd.to_datetime(pred["date"])
    ref["date"] = pd.to_datetime(ref["date"])

    merged = pred.merge(
        ref[["date", "delta_covar_qr"]],
        on="date",
        how="left",
        suffixes=("", "_ref"),
    ).sort_values("date").reset_index(drop=True)

    if "delta_covar_qr" not in merged.columns or "delta_covar_qr_ref" not in merged.columns:
        raise ValueError("Could not align predicted and reference delta_covar_qr.")

    resid = merged["delta_covar_qr"] - merged["delta_covar_qr_ref"]

    lower, upper = _rolling_quantile_band(
        center=merged["delta_covar_qr"],
        residual=resid,
        window=window,
        min_periods=min_periods,
        alpha=alpha,
    )

    merged["delta_covar_tracking_resid"] = resid
    merged["delta_covar_band_lower"] = lower
    merged["delta_covar_band_upper"] = upper
    return merged


# =========================================================
# Plotting
# =========================================================

def plot_qr_var(
    qr_df_1: pd.DataFrame,
    qr_df_2: Optional[pd.DataFrame] = None,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_confidence_band: bool = False,
    band_alpha: float = 0.18,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (14, 6),
):
    x1 = qr_df_1.copy()
    if ticker is not None and "ticker" in x1.columns:
        x1 = x1[x1["ticker"] == ticker].copy()

    if qr_df_2 is not None:
        x2 = qr_df_2.copy()
        if ticker is not None and "ticker" in x2.columns:
            x2 = x2[x2["ticker"] == ticker].copy()
    else:
        x2 = None

    for x in [x1, x2]:
        if x is not None:
            x["date"] = pd.to_datetime(x["date"])

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        x1 = x1[x1["date"] >= start_ts].copy()
        if x2 is not None:
            x2 = x2[x2["date"] >= start_ts].copy()

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        x1 = x1[x1["date"] <= end_ts].copy()
        if x2 is not None:
            x2 = x2[x2["date"] <= end_ts].copy()

    plt.figure(figsize=figsize)
    plt.plot(x1["date"], x1["firm_true"], label="True firm return", linewidth=1.2)
    plt.plot(x1["date"], x1["var_qr"], label=f"{model_1_name}: QR-VaR", linewidth=1.4, linestyle="--")

    if show_confidence_band and {"var_qr_lower_ci", "var_qr_upper_ci"}.issubset(x1.columns):
        plt.fill_between(
            x1["date"],
            x1["var_qr_lower_ci"],
            x1["var_qr_upper_ci"],
            alpha=band_alpha,
            label=f"{model_1_name}: 95% CI",
        )

    if x2 is not None:
        plt.plot(x2["date"], x2["var_qr"], label=f"{model_2_name}: QR-VaR", linewidth=1.4, linestyle='dotted')

        if show_confidence_band and {"var_qr_lower_ci", "var_qr_upper_ci"}.issubset(x2.columns):
            plt.fill_between(
                x2["date"],
                x2["var_qr_lower_ci"],
                x2["var_qr_upper_ci"],
                alpha=band_alpha,
                label=f"{model_2_name}: 95% CI",
            )
    title = "Quantile-regression VaR" 
    if ticker is not None:
        title += f" | {ticker}"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("VaR")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None: 
        save_path = Path(save_path) 
        save_path.parent.mkdir(parents=True, exist_ok=True) 
        plt.savefig(save_path, dpi=150, bbox_inches="tight") 
        print(f"Saved plot to: {save_path}") 
    else: 
        plt.show()


def plot_qr_covar(
    qr_df_1: pd.DataFrame,
    qr_df_2: Optional[pd.DataFrame] = None,
    # model_1_name: str = "Model 1",
    # model_2_name: str = "Model 2",
    # ticker: Optional[str] = None,
    # start_date: Optional[str] = None,
    # end_date: Optional[str] = None,
    # show_confidence_band: bool = False,
    # band_alpha: float = 0.18,
    # save_path: Optional[Union[str, Path]] = None,
    # figsize: tuple[int, int] = (14, 6),
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_confidence_band: bool = False,
    band_alpha: float = 0.18,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (14, 6),
):
    x1 = qr_df_1.copy()
    if ticker is not None and "ticker" in x1.columns:
        x1 = x1[x1["ticker"] == ticker].copy()

    if qr_df_2 is not None:
        x2 = qr_df_2.copy()
        if ticker is not None and "ticker" in x2.columns:
            x2 = x2[x2["ticker"] == ticker].copy()
    else:
        x2 = None

    for x in [x1, x2]:
        if x is not None:
            x["date"] = pd.to_datetime(x["date"])

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        x1 = x1[x1["date"] >= start_ts].copy()
        if x2 is not None:
            x2 = x2[x2["date"] >= start_ts].copy()

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        x1 = x1[x1["date"] <= end_ts].copy()
        if x2 is not None:
            x2 = x2[x2["date"] <= end_ts].copy()

    plt.figure(figsize=figsize)
    plt.plot(x1["date"], x1["system_ret"], label="True system return", linewidth=1.2)
    plt.plot(x1["date"], x1["covar_qr"], label=f"{model_1_name}: CoVaR", linewidth=1.4)

    if show_confidence_band and {"covar_band_lower", "covar_band_upper"}.issubset(x1.columns):
        plt.fill_between(
            x1["date"],
            x1["covar_band_lower"],
            x1["covar_band_upper"],
            alpha=band_alpha,
            label=f"{model_1_name}: uncertainty band",
        )

    if x2 is not None:
        plt.plot(x2["date"], x2["covar_qr"], label=f"{model_2_name}: CoVaR", linewidth=1.4)

        if show_confidence_band and {"covar_band_lower", "covar_band_upper"}.issubset(x2.columns):
            plt.fill_between(
                x2["date"],
                x2["covar_band_lower"],
                x2["covar_band_upper"],
                alpha=band_alpha,
                label=f"{model_2_name}: uncertainty band",
            )

    title = "Quantile-regression CoVaR" 
    if ticker is not None:
        title += f" | {ticker}"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("CoVaR")
    plt.ylim(-1.3, 1.3)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None: 
        save_path = Path(save_path) 
        save_path.parent.mkdir(parents=True, exist_ok=True) 
        plt.savefig(save_path, dpi=150, bbox_inches="tight") 
        print(f"Saved plot to: {save_path}") 
    else: 
        plt.show()


def plot_qr_delta_covar(
    qr_df_1: pd.DataFrame,
    qr_df_2: Optional[pd.DataFrame] = None,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    show_confidence_band: bool = False,
    band_alpha: float = 0.18,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (14, 5),
):
    x1 = qr_df_1.copy()
    if ticker is not None and "ticker" in x1.columns:
        x1 = x1[x1["ticker"] == ticker].copy()

    if qr_df_2 is not None:
        x2 = qr_df_2.copy()
        if ticker is not None and "ticker" in x2.columns:
            x2 = x2[x2["ticker"] == ticker].copy()
    else:
        x2 = None

    for x in [x1, x2]:
        if x is not None:
            x["date"] = pd.to_datetime(x["date"])

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        x1 = x1[x1["date"] >= start_ts].copy()
        if x2 is not None:
            x2 = x2[x2["date"] >= start_ts].copy()

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        x1 = x1[x1["date"] <= end_ts].copy()
        if x2 is not None:
            x2 = x2[x2["date"] <= end_ts].copy()

    plt.figure(figsize=figsize)
    plt.plot(x1["date"], x1["delta_covar_qr"], label=f"{model_1_name}: ΔCoVaR", linewidth=1.4)

    if show_confidence_band and {"delta_covar_band_lower", "delta_covar_band_upper"}.issubset(x1.columns):
        plt.fill_between(
            x1["date"],
            x1["delta_covar_band_lower"],
            x1["delta_covar_band_upper"],
            alpha=band_alpha,
            label=f"{model_1_name}: uncertainty band",
        )

    if x2 is not None:
        plt.plot(x2["date"], x2["delta_covar_qr"], label=f"{model_2_name}: ΔCoVaR", linewidth=1.4)

        if show_confidence_band and {"delta_covar_band_lower", "delta_covar_band_upper"}.issubset(x2.columns):
            plt.fill_between(
                x2["date"],
                x2["delta_covar_band_lower"],
                x2["delta_covar_band_upper"],
                alpha=band_alpha,
                label=f"{model_2_name}: uncertainty band",
            )
    title = "Quantile-regression ΔCoVaR" 
    if ticker is not None:
        title += f" | {ticker}"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("ΔCoVaR")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None: 
        save_path = Path(save_path) 
        save_path.parent.mkdir(parents=True, exist_ok=True) 
        plt.savefig(save_path, dpi=150, bbox_inches="tight") 
        print(f"Saved plot to: {save_path}") 
    else: 
        plt.show()