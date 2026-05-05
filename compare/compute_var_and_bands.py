from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _infer_first_existing(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {what}. Tried columns: {candidates}")


def _z_value(alpha: float) -> float:
    """
    Standard normal quantile for lower alpha-tail.
    Example:
      alpha=0.05 -> about -1.64485
    """
    return NormalDist().inv_cdf(alpha)


def _standardize_prediction_df(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize input dataframe to columns:
      date, ticker(optional), true, pred

    Accepted target/prediction column names:
      true: y_true / true / label / labels
      pred: y_pred / pred / pred_mean / prediction / predictions
    """
    x = df.copy()

    date_col = _infer_first_existing(x, ["date", "datetime", "timestamp"], "date column")
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
            raise ValueError("ticker was requested, but input dataframe has no 'ticker' column.")
        out = out[out["ticker"] == ticker].copy()

    out = out.dropna(subset=["date", "true", "pred"]).sort_values("date").reset_index(drop=True)

    return out


def _aggregate_same_day_predictions(
    df: pd.DataFrame,
    how: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate repeated predictions for the same date.

    how:
      - "mean": average pred/true across duplicates
      - "first": keep first row for each date
    """
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


# =========================================================
# Core calculations
# =========================================================

@dataclass
class BandConfig:
    alpha: float = 0.05
    sigma_window: int = 60
    sigma_min_periods: int = 20
    aggregate_same_date: str = "mean"   # "mean" or "first"
    clip_var_at_zero: bool = False


def compute_var_and_bands(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    config: BandConfig = BandConfig(),
) -> pd.DataFrame:
    """
    Compute:
      - pred_mean
      - residual
      - sigma_hat (rolling std of residuals)
      - lower_alpha band
      - var_line

    Formulas:
      residual_t = true_t - pred_t
      sigma_hat_t = rolling_std(residual_t)
      lower_alpha_t = pred_t + z_alpha * sigma_hat_t
      var_line_t = - lower_alpha_t

    Notes:
    - This is a Gaussian / residual-based downside band,
      NOT a true quantile-regression estimate.
    """
    x = _standardize_prediction_df(df, ticker=ticker)
    x = _aggregate_same_day_predictions(x, how=config.aggregate_same_date)

    z_alpha = _z_value(config.alpha)

    x["residual"] = x["true"] - x["pred"]
    x["sigma_hat"] = (
        x["residual"]
        .rolling(window=config.sigma_window, min_periods=config.sigma_min_periods)
        .std()
    )

    # fallback for earliest dates
    if x["sigma_hat"].notna().sum() > 0:
        first_valid_sigma = x["sigma_hat"].dropna().iloc[0]
        x["sigma_hat"] = x["sigma_hat"].fillna(first_valid_sigma)
    else:
        # if series too short or all NaN residual std
        fallback_sigma = float(np.nanstd(x["residual"].values))
        if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
            fallback_sigma = 1e-8
        x["sigma_hat"] = fallback_sigma

    x["pred_mean"] = x["pred"]
    x["lower_band"] = x["pred_mean"] + z_alpha * x["sigma_hat"]
    x["var_line"] = -x["lower_band"]

    if config.clip_var_at_zero:
        x["var_line"] = x["var_line"].clip(lower=0.0)

    x["alpha"] = config.alpha
    x["z_alpha"] = z_alpha

    return x


# =========================================================
# Plotting
# =========================================================

def plot_true_mean_and_band(
    df_model_1: pd.DataFrame,
    df_model_2: Optional[pd.DataFrame] = None,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    alpha: float = 0.05,
    sigma_window: int = 60,
    sigma_min_periods: int = 20,
    aggregate_same_date: str = "mean",
    show_var_line: bool = False,
    var_on_secondary_axis: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 7),
):
    """
    Plot:
      - true return
      - mean prediction
      - lower alpha-band
      - optional VaR line

    Supports either one or two model dataframes.
    """
    cfg = BandConfig(
        alpha=alpha,
        sigma_window=sigma_window,
        sigma_min_periods=sigma_min_periods,
        aggregate_same_date=aggregate_same_date,
    )

    m1 = compute_var_and_bands(df_model_1, ticker=ticker, config=cfg)

    if df_model_2 is not None:
        m2 = compute_var_and_bands(df_model_2, ticker=ticker, config=cfg)
    else:
        m2 = None

    # Apply display-range filtering AFTER sigma calculation
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        m1 = m1[m1["date"] >= start_ts].copy()
        if m2 is not None:
            m2 = m2[m2["date"] >= start_ts].copy()

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        m1 = m1[m1["date"] <= end_ts].copy()
        if m2 is not None:
            m2 = m2[m2["date"] <= end_ts].copy()

    if len(m1) == 0:
        raise ValueError("No rows left for model 1 after filtering.")

    fig, ax = plt.subplots(figsize=figsize)

    # True returns
    ax.plot(
        m1["date"],
        m1["true"],
        label="True return",
        linewidth=1.2,
    )

    # Model 1
    ax.plot(
        m1["date"],
        m1["pred_mean"],
        label=f"{model_1_name}: mean prediction",
        linewidth=1.4,
    )
    ax.plot(
        m1["date"],
        m1["lower_band"],
        linestyle="--",
        linewidth=1.2,
        label=f"{model_1_name}: lower {int(alpha * 100)}% bound",
    )

    # Model 2
    if m2 is not None:
        ax.plot(
            m2["date"],
            m2["pred_mean"],
            label=f"{model_2_name}: mean prediction",
            linewidth=1.4,
        )
        ax.plot(
            m2["date"],
            m2["lower_band"],
            linestyle="--",
            linewidth=1.2,
            label=f"{model_2_name}: lower {int(alpha * 100)}% bound",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    title = f"True return vs prediction bands"
    if ticker is not None:
        title += f" | {ticker}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    # Optional VaR line
    if show_var_line:
        if var_on_secondary_axis:
            ax2 = ax.twinx()
            ax2.plot(
                m1["date"],
                m1["var_line"],
                linestyle=":",
                linewidth=1.3,
                label=f"{model_1_name}: VaR",
            )
            if m2 is not None:
                ax2.plot(
                    m2["date"],
                    m2["var_line"],
                    linestyle=":",
                    linewidth=1.3,
                    label=f"{model_2_name}: VaR",
                )
            ax2.set_ylabel("VaR")
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        else:
            ax.plot(
                m1["date"],
                m1["var_line"],
                linestyle=":",
                linewidth=1.3,
                label=f"{model_1_name}: VaR",
            )
            if m2 is not None:
                ax.plot(
                    m2["date"],
                    m2["var_line"],
                    linestyle=":",
                    linewidth=1.3,
                    label=f"{model_2_name}: VaR",
                )
            ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()

    return (m1, m2) if m2 is not None else m1


def plot_var_only(
    df_model_1: pd.DataFrame,
    df_model_2: Optional[pd.DataFrame] = None,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    alpha: float = 0.05,
    sigma_window: int = 60,
    sigma_min_periods: int = 20,
    aggregate_same_date: str = "mean",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 5),
):
    """
    Plot only the VaR-style line(s).
    """
    cfg = BandConfig(
        alpha=alpha,
        sigma_window=sigma_window,
        sigma_min_periods=sigma_min_periods,
        aggregate_same_date=aggregate_same_date,
    )

    m1 = compute_var_and_bands(df_model_1, ticker=ticker, config=cfg)
    m2 = compute_var_and_bands(df_model_2, ticker=ticker, config=cfg) if df_model_2 is not None else None

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        m1 = m1[m1["date"] >= start_ts].copy()
        if m2 is not None:
            m2 = m2[m2["date"] >= start_ts].copy()

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        m1 = m1[m1["date"] <= end_ts].copy()
        if m2 is not None:
            m2 = m2[m2["date"] <= end_ts].copy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        m1["date"],
        m1["var_line"],
        label=f"{model_1_name}: VaR",
        linewidth=1.4,
    )

    if m2 is not None:
        ax.plot(
            m2["date"],
            m2["var_line"],
            label=f"{model_2_name}: VaR",
            linewidth=1.4,
        )

    title = f"VaR lines"
    if ticker is not None:
        title += f" | {ticker}"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("VaR")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()

    return (m1, m2) if m2 is not None else m1