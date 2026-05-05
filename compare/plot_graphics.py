from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Union, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AggregateMode = Literal["mean", "first", "last"]

Z_SCORE = {
    0.10: -1.2815515655446004,
    0.05: -1.6448536269514729,
    0.01: -2.3263478740408408,
}


def _infer_date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in {"date", "datetime", "timestamp"}:
            return c
    raise ValueError("Could not find date column.")


def _infer_true_column(df: pd.DataFrame) -> str:
    candidates = ["true", "y_true", "label", "labels", "actual", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer true/label column. Existing columns: {list(df.columns)}"
    )


def _infer_pred_column(df: pd.DataFrame) -> str:
    candidates = ["pred", "prediction", "predictions", "y_pred", "pred_mean"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer prediction column. Existing columns: {list(df.columns)}"
    )


def _infer_ticker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["ticker", "symbol", "asset"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_panel_df(panel: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(panel, (str, Path)):
        panel_df = pd.read_csv(panel)
    else:
        panel_df = panel.copy()

    date_col = _infer_date_column(panel_df)
    panel_df[date_col] = pd.to_datetime(panel_df[date_col])
    panel_df = panel_df.sort_values(date_col).rename(columns={date_col: "date"})
    return panel_df


def _standardize_prediction_df(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    aggregate_mode: AggregateMode = "mean",
) -> pd.DataFrame:
    """
    Standardizes prediction dataframe into columns:
      date, ticker (optional), true, pred
    """
    x = df.copy()

    date_col = _infer_date_column(x)
    true_col = _infer_true_column(x)
    pred_col = _infer_pred_column(x)
    ticker_col = _infer_ticker_column(x)

    x[date_col] = pd.to_datetime(x[date_col])

    rename_map = {
        date_col: "date",
        true_col: "true",
        pred_col: "pred",
    }
    if ticker_col is not None:
        rename_map[ticker_col] = "ticker"

    x = x.rename(columns=rename_map)

    keep_cols = ["date", "true", "pred"]
    if "ticker" in x.columns:
        keep_cols.append("ticker")
    x = x[keep_cols].copy()

    if ticker is not None:
        if "ticker" not in x.columns:
            raise ValueError("Ticker filter requested, but dataframe has no ticker column.")
        x = x[x["ticker"] == ticker].copy()

    if start_date is not None:
        x = x[x["date"] >= pd.to_datetime(start_date)].copy()
    if end_date is not None:
        x = x[x["date"] <= pd.to_datetime(end_date)].copy()

    if len(x) == 0:
        raise ValueError("No rows left after filtering.")

    # aggregate by date
    if aggregate_mode == "mean":
        agg = x.groupby("date", as_index=False)[["true", "pred"]].mean()
    elif aggregate_mode == "first":
        x = x.sort_values("date").drop_duplicates(subset=["date"], keep="first")
        agg = x[["date", "true", "pred"]].reset_index(drop=True)
    elif aggregate_mode == "last":
        x = x.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        agg = x[["date", "true", "pred"]].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown aggregate_mode={aggregate_mode}")

    agg = agg.sort_values("date").reset_index(drop=True)
    return agg


def _align_model_frames(
    df_1: pd.DataFrame,
    df_2: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Align by date.
    Returns columns:
      date, true, pred_1 [, pred_2]
    """
    out = df_1.rename(columns={"pred": "pred_1"}).copy()

    if df_2 is None:
        return out

    right = df_2.rename(columns={"true": "true_2", "pred": "pred_2"}).copy()
    merged = out.merge(right, on="date", how="inner")

    # keep true from first dataframe
    cols = ["date", "true", "pred_1", "pred_2"]
    merged = merged[cols].copy()
    return merged


# =========================================================
# Risk proxy helpers
# =========================================================

def compute_rolling_vol(series: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).std()


def compute_downside_beta(
    firm_ret: pd.Series,
    market_ret: pd.Series,
    window: int = 60,
    min_periods: int = 20,
) -> pd.Series:
    df = pd.DataFrame({"firm": firm_ret, "market": market_ret}).dropna()

    betas = []
    out_index = []

    for i in range(len(df)):
        start = max(0, i - window + 1)
        w = df.iloc[start:i + 1].copy()
        w = w[w["market"] < 0]

        out_index.append(df.index[i])

        if len(w) < min_periods:
            betas.append(np.nan)
            continue

        var_m = w["market"].var()
        if pd.isna(var_m) or var_m <= 1e-12:
            betas.append(np.nan)
            continue

        cov_im = w["firm"].cov(w["market"])
        betas.append(cov_im / var_m)

    return pd.Series(betas, index=out_index)


def build_risk_proxy_df(
    pred_df: pd.DataFrame,
    panel: Union[str, Path, pd.DataFrame],
    ticker: str,
    q: float = 0.05,
    vol_window: int = 60,
    beta_window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    pred_df must contain at least:
      date, true, pred
    panel must contain:
      date, market_ret_1d, <ticker column>
    """
    if q not in Z_SCORE:
        raise ValueError(f"Supported q values: {list(Z_SCORE.keys())}")

    z_q = Z_SCORE[q]
    panel_df = _load_panel_df(panel)

    if ticker not in panel_df.columns:
        raise ValueError(f"{ticker} not found in panel dataframe.")
    if "market_ret_1d" not in panel_df.columns:
        raise ValueError("market_ret_1d not found in panel dataframe.")

    firm_ret = panel_df.set_index("date")[ticker].astype(float)
    market_ret = panel_df.set_index("date")["market_ret_1d"].astype(float)

    firm_sigma = compute_rolling_vol(firm_ret, window=vol_window, min_periods=min_periods)
    market_sigma = compute_rolling_vol(market_ret, window=vol_window, min_periods=min_periods)
    downside_beta = compute_downside_beta(
        firm_ret=firm_ret,
        market_ret=market_ret,
        window=beta_window,
        min_periods=min_periods,
    )

    aux = pd.DataFrame({
        "date": firm_ret.index,
        "firm_ret_panel": firm_ret.values,
        "market_ret": market_ret.values,
        "firm_sigma": firm_sigma.values,
        "market_sigma": market_sigma.values,
        "downside_beta": downside_beta.reindex(firm_ret.index).values,
    })

    x = pred_df.copy()
    x["date"] = pd.to_datetime(x["date"])

    risk_df = x.merge(aux, on="date", how="left")
    risk_df = risk_df.sort_values("date").reset_index(drop=True)

    risk_df["firm_sigma"] = risk_df["firm_sigma"].ffill()
    risk_df["market_sigma"] = risk_df["market_sigma"].ffill()
    risk_df["downside_beta"] = risk_df["downside_beta"].ffill()

    # VaR proxy
    risk_df["true_var_proxy"] = risk_df["true"] + z_q * risk_df["firm_sigma"]
    risk_df["pred_var_proxy"] = risk_df["pred"] + z_q * risk_df["firm_sigma"]
    risk_df["market_var_proxy"] = z_q * risk_df["market_sigma"]

    # CoVaR proxy
    risk_df["true_covar_proxy"] = risk_df["true"] + risk_df["downside_beta"] * risk_df["market_var_proxy"]
    risk_df["pred_covar_proxy"] = risk_df["pred"] + risk_df["downside_beta"] * risk_df["market_var_proxy"]

    # Delta-CoVaR proxy
    risk_df["true_delta_covar_proxy"] = risk_df["true_covar_proxy"] - risk_df["true_var_proxy"]
    risk_df["pred_delta_covar_proxy"] = risk_df["pred_covar_proxy"] - risk_df["pred_var_proxy"]

    return risk_df


# =========================================================
# Plot 1: Returns
# =========================================================

def plot_true_vs_predicted_returns(
    df_1: pd.DataFrame,
    df_2: Optional[pd.DataFrame] = None,
    *,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    aggregate_mode: AggregateMode = "last",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    x1 = _standardize_prediction_df(
        df_1,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        aggregate_mode=aggregate_mode,
    )

    x2 = None
    if df_2 is not None:
        x2 = _standardize_prediction_df(
            df_2,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            aggregate_mode=aggregate_mode,
        )

    plot_df = _align_model_frames(x1, x2)

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df["date"], plot_df["true"], label="True returns", linewidth=1.4)
    plt.plot(plot_df["date"], plot_df["pred_1"], label=f"{model_1_name} predictions", linewidth=1.2)

    if df_2 is not None:
        plt.plot(plot_df["date"], plot_df["pred_2"], label=f"{model_2_name} predictions", linewidth=1.2)

    if title is None:
        title = "True returns vs predicted returns"
        if ticker is not None:
            title += f" | {ticker}"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


# =========================================================
# Plot 2: CoVaR
# =========================================================

def plot_covar_true_vs_predicted(
    df_1: pd.DataFrame,
    panel: Union[str, Path, pd.DataFrame],
    df_2: Optional[pd.DataFrame] = None,
    *,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: str,
    q: float = 0.05,
    vol_window: int = 60,
    beta_window: int = 60,
    min_periods: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    aggregate_mode: AggregateMode = "last",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    x1 = _standardize_prediction_df(
        df_1,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        aggregate_mode=aggregate_mode,
    )
    risk_1 = build_risk_proxy_df(
        pred_df=x1,
        panel=panel,
        ticker=ticker,
        q=q,
        vol_window=vol_window,
        beta_window=beta_window,
        min_periods=min_periods,
    )

    risk_2 = None
    if df_2 is not None:
        x2 = _standardize_prediction_df(
            df_2,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            aggregate_mode=aggregate_mode,
        )
        risk_2 = build_risk_proxy_df(
            pred_df=x2,
            panel=panel,
            ticker=ticker,
            q=q,
            vol_window=vol_window,
            beta_window=beta_window,
            min_periods=min_periods,
        )

    plt.figure(figsize=(14, 6))
    plt.plot(risk_1["date"], risk_1["true_covar_proxy"], label="CoVaR from true returns", linewidth=1.4)
    plt.plot(risk_1["date"], risk_1["pred_covar_proxy"], label=f"{model_1_name} CoVaR", linewidth=1.2)

    if risk_2 is not None:
        risk_merged = risk_1[["date"]].merge(
            risk_2[["date", "pred_covar_proxy"]],
            on="date",
            how="inner",
            suffixes=("", "_2"),
        )
        plt.plot(risk_merged["date"], risk_merged["pred_covar_proxy"], label=f"{model_2_name} CoVaR", linewidth=1.2)

    if title is None:
        title = f"CoVaR from true returns vs predicted returns | {ticker}"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("CoVaR proxy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


# =========================================================
# Plot 3: Delta-CoVaR
# =========================================================

def plot_delta_covar_true_vs_predicted(
    df_1: pd.DataFrame,
    panel: Union[str, Path, pd.DataFrame],
    df_2: Optional[pd.DataFrame] = None,
    *,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: str,
    q: float = 0.05,
    vol_window: int = 60,
    beta_window: int = 60,
    min_periods: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    aggregate_mode: AggregateMode = "last",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    x1 = _standardize_prediction_df(
        df_1,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        aggregate_mode=aggregate_mode,
    )
    risk_1 = build_risk_proxy_df(
        pred_df=x1,
        panel=panel,
        ticker=ticker,
        q=q,
        vol_window=vol_window,
        beta_window=beta_window,
        min_periods=min_periods,
    )

    risk_2 = None
    if df_2 is not None:
        x2 = _standardize_prediction_df(
            df_2,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            aggregate_mode=aggregate_mode,
        )
        risk_2 = build_risk_proxy_df(
            pred_df=x2,
            panel=panel,
            ticker=ticker,
            q=q,
            vol_window=vol_window,
            beta_window=beta_window,
            min_periods=min_periods,
        )

    plt.figure(figsize=(14, 6))
    plt.plot(
        risk_1["date"],
        risk_1["true_delta_covar_proxy"],
        label="ΔCoVaR from true returns",
        linewidth=1.4,
    )
    plt.plot(
        risk_1["date"],
        risk_1["pred_delta_covar_proxy"],
        label=f"{model_1_name} ΔCoVaR",
        linewidth=1.2,
    )

    if risk_2 is not None:
        risk_merged = risk_1[["date"]].merge(
            risk_2[["date", "pred_delta_covar_proxy"]],
            on="date",
            how="inner",
            suffixes=("", "_2"),
        )
        plt.plot(
            risk_merged["date"],
            risk_merged["pred_delta_covar_proxy"],
            label=f"{model_2_name} ΔCoVaR",
            linewidth=1.2,
        )

    if title is None:
        title = f"ΔCoVaR from true returns vs predicted returns | {ticker}"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("ΔCoVaR proxy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_requested_graphs(
    graph_types: Sequence[str],
    df_1,
    panel=None,
    df_2=None,
    *,
    model_1_name: str = "Model 1",
    model_2_name: str = "Model 2",
    ticker: Optional[str] = None,
    q: float = 0.05,
    vol_window: int = 60,
    beta_window: int = 60,
    min_periods: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    aggregate_mode: str = "last",
    save_dir: Optional[Union[str, Path]] = None,
    file_prefix: Optional[str] = None,
):
    """
    graph_types can contain:
      - "returns"
      - "covar"
      - "delta_covar"

    Parameters
    ----------
    df_1 : pd.DataFrame
        First model dataframe.
    panel : str | Path | pd.DataFrame | None
        Panel data source with date, market_ret_1d and ticker column.
        Required for "covar" and "delta_covar".
    df_2 : pd.DataFrame | None
        Optional second model dataframe.
    save_dir : str | Path | None
        If provided, saves each plot into this directory.
    file_prefix : str | None
        Optional prefix for filenames.
    """
    allowed = {"returns", "covar", "delta_covar"}
    bad = [g for g in graph_types if g not in allowed]
    if bad:
        raise ValueError(f"Unknown graph_types: {bad}. Allowed: {sorted(allowed)}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if file_prefix is None:
        parts = []
        if ticker is not None:
            parts.append(str(ticker))
        if start_date is not None and end_date is not None:
            parts.append(f"{start_date}_{end_date}")
        file_prefix = "_".join(parts) if parts else "plot"

    def _make_path(name: str):
        if save_dir is None:
            return None
        return save_dir / f"{file_prefix}_{name}.png"

    if "returns" in graph_types:
        plot_true_vs_predicted_returns(
            df_1=df_1,
            df_2=df_2,
            model_1_name=model_1_name,
            model_2_name=model_2_name,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            aggregate_mode=aggregate_mode,
            save_path=_make_path("returns"),
        )

    if "covar" in graph_types:
        if panel is None:
            raise ValueError("panel must be provided for graph type 'covar'.")
        if ticker is None:
            raise ValueError("ticker must be provided for graph type 'covar'.")

        plot_covar_true_vs_predicted(
            df_1=df_1,
            df_2=df_2,
            panel=panel,
            model_1_name=model_1_name,
            model_2_name=model_2_name,
            ticker=ticker,
            q=q,
            vol_window=vol_window,
            beta_window=beta_window,
            min_periods=min_periods,
            start_date=start_date,
            end_date=end_date,
            aggregate_mode=aggregate_mode,
            save_path=_make_path("covar"),
        )

    if "delta_covar" in graph_types:
        if panel is None:
            raise ValueError("panel must be provided for graph type 'delta_covar'.")
        if ticker is None:
            raise ValueError("ticker must be provided for graph type 'delta_covar'.")

        plot_delta_covar_true_vs_predicted(
            df_1=df_1,
            df_2=df_2,
            panel=panel,
            model_1_name=model_1_name,
            model_2_name=model_2_name,
            ticker=ticker,
            q=q,
            vol_window=vol_window,
            beta_window=beta_window,
            min_periods=min_periods,
            start_date=start_date,
            end_date=end_date,
            aggregate_mode=aggregate_mode,
            save_path=_make_path("delta_covar"),
        )