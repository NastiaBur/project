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
        var_i_t = float(_predict_quantreg(res_i_var, X_i_test)[0])
        med_i_t = float(_predict_quantreg(res_i_med, X_i_test)[0])

        # -------- Step 2: system QR on institution state + controls --------
        sys_features = ["institution_state"] + control_cols
        X_s_train = train[sys_features]
        y_s_train = train["system_ret"]

        try:
            res_s = _fit_quantreg(X_s_train, y_s_train, q=q)
        except Exception:
            continue

        # Build two counterfactual rows:
        # distress state = VaR_i_t
        # median state   = Median_i_t
        X_s_distress = test_row[control_cols].copy()
        X_s_distress.insert(0, "institution_state", var_i_t)

        X_s_median = test_row[control_cols].copy()
        X_s_median.insert(0, "institution_state", med_i_t)

        try:
            covar_distress_t = float(_predict_quantreg(res_s, X_s_distress)[0])
            covar_median_t = float(_predict_quantreg(res_s, X_s_median)[0])
        except Exception:
            continue

        delta_covar_t = covar_distress_t - covar_median_t

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

# =========================================================
# Plotting
# =========================================================

def plot_qr_var(
    qr_df_1: pd.DataFrame,
    qr_df_2: Optional[pd.DataFrame] = None,
    *,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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
    # plt.plot(x1["date"], x1["firm_pred"], label=f"{model_1_name}: mean prediction", linewidth=1.2)
    plt.plot(x1["date"], x1["var_qr"], label=f"{model_1_name}: QR-VaR", linewidth=1.4, linestyle="--")

    if x2 is not None:
        # plt.plot(x2["date"], x2["firm_pred"], label=f"{model_2_name}: mean prediction", linewidth=1.2)
        plt.plot(x2["date"], x2["var_qr"], label=f"{model_2_name}: QR-VaR", linewidth=1.4, linestyle='dotted')

    title = "Quantile-regression VaR"
    if ticker is not None:
        title += f" | {ticker}"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return / VaR threshold")
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
    *,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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
    # plt.plot(x1["date"], x1["covar_median_qr"], label=f"{model_1_name}: CoVaR median-state", linewidth=1.2, linestyle="--")

    if x2 is not None:
        plt.plot(x2["date"], x2["covar_qr"], label=f"{model_2_name}: CoVaR", linewidth=1.4)
        # plt.plot(x2["date"], x2["covar_median_qr"], label=f"{model_2_name}: CoVaR median-state", linewidth=1.2, linestyle="--")

    title = "Quantile-regression CoVaR"
    if ticker is not None:
        title += f" | {ticker}"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("System return / CoVaR threshold")
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
    *,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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

    if x2 is not None:
        plt.plot(x2["date"], x2["delta_covar_qr"], label=f"{model_2_name}: ΔCoVaR", linewidth=1.4)

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
