"""R5 report exhibits and fund fact sheets from verified precomputed results.

This module is deliberately downstream-only.  It reads the required R2/R3 CSV
artifacts, validates their internal arithmetic and schemas, and renders factual
report exhibits.  It does not load raw data, fit a model, change a portfolio,
score sentiment, or create economic interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd

from src import portfolios, sentiment


SAMPLE_END = pd.Timestamp("2023-12-31")
FIGURE_DPI = 180
SOURCE_R2 = (
    "Verified R2 precomputed walk-forward out-of-sample results in "
    "results/data/fund_returns.csv, results/data/fund_weights.csv, and "
    "results/tables/performance_metrics.csv."
)
SOURCE_R3 = (
    "Verified R3 precomputed VADER sector index in "
    "results/data/sector_sentiment_index.csv."
)

PERFORMANCE_COMPARISON_COLUMNS = [
    "fund_name",
    "asset_family",
    "portfolio_method",
    "sample_period",
    "observations",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "final_growth_of_1",
    "annualisation_days_per_year",
    "risk_free_rate_annual",
    "transaction_cost_rate",
    "result_basis",
    "source",
]

FACT_SHEET_SUMMARY_COLUMNS = [
    "fund_name",
    "asset_family",
    "portfolio_method",
    "sample_period",
    "observations",
    "final_growth_of_1",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "latest_rebalance_date",
    "current_holdings_count",
    "current_equity_target_weight",
    "current_cryptocurrency_target_weight",
    "current_target_weight_sum",
    "annualisation_days_per_year",
    "risk_free_rate_annual",
    "transaction_cost_rate",
    "current_target_holdings_source",
]

FACT_SHEET_HOLDING_COLUMNS = [
    "fund_name",
    "portfolio_method",
    "latest_rebalance_date",
    "holding_rank",
    "ticker",
    "holding_asset_class",
    "target_weight",
]

MANIFEST_COLUMNS = [
    "exhibit_id",
    "artifact_type",
    "file_path",
    "title",
    "sample_period",
    "source",
    "technical_caption",
    "student_interpretation_status",
    "final_figure_acceptance_status",
]


@dataclass(frozen=True)
class R5Tables:
    """Validated factual tables used by the R5 report exhibits."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    performance_metrics: pd.DataFrame
    sector_sentiment_index: pd.DataFrame
    performance_comparison: pd.DataFrame
    fact_sheet_summary: pd.DataFrame
    fact_sheet_holdings: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _normalise_inputs(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    performance_metrics: pd.DataFrame,
    sector_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(fund_returns, portfolios.FUND_RETURN_COLUMNS, "fund returns")
    _require_columns(fund_weights, portfolios.FUND_WEIGHT_COLUMNS, "fund weights")
    _require_columns(
        performance_metrics,
        portfolios.PERFORMANCE_METRIC_COLUMNS,
        "performance metrics",
    )
    _require_columns(
        sector_index,
        sentiment.SECTOR_SENTIMENT_COLUMNS,
        "sector sentiment index",
    )

    returns = fund_returns.loc[:, portfolios.FUND_RETURN_COLUMNS].copy()
    weights = fund_weights.loc[:, portfolios.FUND_WEIGHT_COLUMNS].copy()
    metrics = performance_metrics.loc[:, portfolios.PERFORMANCE_METRIC_COLUMNS].copy()
    sectors = sector_index.loc[:, sentiment.SECTOR_SENTIMENT_COLUMNS].copy()

    for column in ("date", "active_target_rebalance_date"):
        returns[column] = pd.to_datetime(returns[column], errors="raise").dt.normalize()
    for column in ("rebalance_date", "estimation_start_date", "estimation_end_date"):
        weights[column] = pd.to_datetime(weights[column], errors="raise").dt.normalize()
    for column in ("evaluation_start_date", "evaluation_end_date", "latest_rebalance_date"):
        metrics[column] = pd.to_datetime(metrics[column], errors="raise").dt.normalize()
    sectors["date"] = pd.to_datetime(sectors["date"], errors="raise").dt.normalize()

    numeric_return_columns = ["daily_return", "growth_of_1", "drawdown"]
    returns[numeric_return_columns] = returns[numeric_return_columns].apply(
        pd.to_numeric, errors="raise"
    )
    weights["target_weight"] = pd.to_numeric(weights["target_weight"], errors="raise")
    metric_columns = [
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "final_growth_of_1",
        "annualisation_days_per_year",
        "risk_free_rate_annual",
        "transaction_cost_rate",
        "observations",
    ]
    metrics[metric_columns] = metrics[metric_columns].apply(pd.to_numeric, errors="raise")
    sectors["sentiment_value"] = pd.to_numeric(
        sectors["sentiment_value"], errors="coerce"
    )
    return returns, weights, metrics, sectors


def _validate_verified_results(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    metrics: pd.DataFrame,
    sectors: pd.DataFrame,
) -> None:
    if returns.empty or weights.empty or metrics.empty or sectors.empty:
        raise ValueError("R5 requires non-empty verified R2/R3 results")
    if returns.duplicated(["fund_name", "date"]).any():
        raise ValueError("fund returns contain duplicate fund/date rows")
    if weights.duplicated(["fund_name", "rebalance_date", "ticker"]).any():
        raise ValueError("fund weights contain duplicate fund/rebalance/ticker rows")
    if sectors.duplicated(["date", "sector"]).any():
        raise ValueError("sector sentiment contains duplicate date/sector rows")
    if max(returns["date"].max(), weights["rebalance_date"].max(), sectors["date"].max()) > SAMPLE_END:
        raise ValueError("verified results exceed the approved 2023-12-31 sample cap")

    fund_sets = [
        set(returns["fund_name"]),
        set(weights["fund_name"]),
        set(metrics["fund_name"]),
    ]
    if not (fund_sets[0] == fund_sets[1] == fund_sets[2]):
        raise ValueError("fund names do not reconcile across R2 artifacts")
    if metrics["fund_name"].duplicated().any():
        raise ValueError("performance metrics must contain one row per fund")

    if not np.isfinite(returns[["daily_return", "growth_of_1", "drawdown"]]).all(axis=None):
        raise ValueError("fund return paths must be finite")
    if (returns["daily_return"] <= -1.0).any():
        raise ValueError("fund daily returns must exceed -100%")
    if not np.isfinite(weights["target_weight"]).all() or (weights["target_weight"] < 0.0).any():
        raise ValueError("target weights must be finite and long-only")

    target_sums = weights.groupby(["fund_name", "rebalance_date"])["target_weight"].sum()
    if not np.allclose(target_sums.to_numpy(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("each target portfolio must sum to one")
    if (weights["estimation_end_date"] >= weights["rebalance_date"]).any():
        raise ValueError("an estimation window reaches its rebalance date")

    metric_names = [
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "final_growth_of_1",
    ]
    for _, metric_row in metrics.iterrows():
        fund_name = metric_row["fund_name"]
        path = returns.loc[returns["fund_name"].eq(fund_name)].sort_values("date")
        if path.empty:
            raise ValueError(f"no return path exists for {fund_name}")
        growth = (1.0 + path["daily_return"]).cumprod()
        drawdown = growth / growth.cummax().clip(lower=1.0) - 1.0
        if not np.allclose(path["growth_of_1"], growth, atol=1e-12, rtol=1e-12):
            raise ValueError(f"saved growth arithmetic is inconsistent for {fund_name}")
        if not np.allclose(path["drawdown"], drawdown, atol=1e-12, rtol=1e-12):
            raise ValueError(f"saved drawdown arithmetic is inconsistent for {fund_name}")

        recomputed = portfolios.performance_metrics(
            path["daily_return"],
            periods_per_year=int(metric_row["annualisation_days_per_year"]),
            risk_free_rate_annual=float(metric_row["risk_free_rate_annual"]),
        )
        for name in metric_names:
            if not np.isclose(
                float(metric_row[name]), recomputed[name], atol=1e-12, rtol=1e-12
            ):
                raise ValueError(f"saved {name} is inconsistent for {fund_name}")
        if int(metric_row["observations"]) != len(path):
            raise ValueError(f"saved observation count is inconsistent for {fund_name}")
        if pd.Timestamp(metric_row["evaluation_start_date"]) != path["date"].min():
            raise ValueError(f"saved evaluation start is inconsistent for {fund_name}")
        if pd.Timestamp(metric_row["evaluation_end_date"]) != path["date"].max():
            raise ValueError(f"saved evaluation end is inconsistent for {fund_name}")

        latest = weights.loc[weights["fund_name"].eq(fund_name), "rebalance_date"].max()
        if pd.Timestamp(metric_row["latest_rebalance_date"]) != latest:
            raise ValueError(f"latest rebalance date is inconsistent for {fund_name}")

    observed_values = sectors["sentiment_value"].dropna()
    if not np.isfinite(observed_values).all() or not observed_values.between(-1.0, 1.0).all():
        raise ValueError("sector sentiment values must be finite and within [-1, 1]")
    observed_flag = sectors["sentiment_observed"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if observed_flag.isna().any():
        raise ValueError("sentiment_observed must contain Boolean values")
    if not sectors.loc[~observed_flag, "sentiment_value"].isna().all():
        raise ValueError("unobserved sector-days must retain missing sentiment")


def build_r5_tables(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    performance_metrics: pd.DataFrame,
    sector_index: pd.DataFrame,
) -> R5Tables:
    """Validate verified R2/R3 artifacts and build factual R5 tables."""
    returns, weights, metrics, sectors = _normalise_inputs(
        fund_returns, fund_weights, performance_metrics, sector_index
    )
    _validate_verified_results(returns, weights, metrics, sectors)

    comparison = metrics.assign(
        sample_period=metrics["evaluation_start_date"].dt.strftime("%Y-%m-%d")
        + " to "
        + metrics["evaluation_end_date"].dt.strftime("%Y-%m-%d"),
        result_basis="Gross of transaction costs; zero management fee",
        source=SOURCE_R2,
    ).loc[:, PERFORMANCE_COMPARISON_COLUMNS]

    holding_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for _, metric_row in metrics.sort_values(["asset_family", "portfolio_method"]).iterrows():
        fund_name = metric_row["fund_name"]
        latest_date = pd.Timestamp(metric_row["latest_rebalance_date"])
        latest = weights.loc[
            weights["fund_name"].eq(fund_name)
            & weights["rebalance_date"].eq(latest_date)
        ].copy()
        latest = latest.sort_values(["target_weight", "ticker"], ascending=[False, True])
        latest["holding_rank"] = np.arange(1, len(latest) + 1)
        latest["latest_rebalance_date"] = latest_date
        holding_frames.append(
            latest.assign(portfolio_method=metric_row["portfolio_method"])[
                [
                    "fund_name",
                    "portfolio_method",
                    "latest_rebalance_date",
                    "holding_rank",
                    "ticker",
                    "holding_asset_class",
                    "target_weight",
                ]
            ]
        )
        by_class = latest.groupby("holding_asset_class")["target_weight"].sum()
        summary_rows.append(
            {
                "fund_name": fund_name,
                "asset_family": metric_row["asset_family"],
                "portfolio_method": metric_row["portfolio_method"],
                "sample_period": metric_row["evaluation_period"],
                "observations": int(metric_row["observations"]),
                "final_growth_of_1": metric_row["final_growth_of_1"],
                "annualised_return": metric_row["annualised_return"],
                "annualised_volatility": metric_row["annualised_volatility"],
                "sharpe_ratio": metric_row["sharpe_ratio"],
                "maximum_drawdown": metric_row["maximum_drawdown"],
                "latest_rebalance_date": latest_date,
                "current_holdings_count": len(latest),
                "current_equity_target_weight": float(by_class.get("equity", 0.0)),
                "current_cryptocurrency_target_weight": float(
                    by_class.get("cryptocurrency", 0.0)
                ),
                "current_target_weight_sum": float(latest["target_weight"].sum()),
                "annualisation_days_per_year": int(
                    metric_row["annualisation_days_per_year"]
                ),
                "risk_free_rate_annual": metric_row["risk_free_rate_annual"],
                "transaction_cost_rate": metric_row["transaction_cost_rate"],
                "current_target_holdings_source": (
                    "results/data/fund_weights.csv; exact latest target weights"
                ),
            }
        )

    holdings = pd.concat(holding_frames, ignore_index=True).loc[
        :, FACT_SHEET_HOLDING_COLUMNS
    ]
    summary = pd.DataFrame(summary_rows, columns=FACT_SHEET_SUMMARY_COLUMNS)
    return R5Tables(returns, weights, metrics, sectors, comparison, summary, holdings)


def read_precomputed_r5_inputs(repository_root: str | Path) -> R5Tables:
    """Read only the four required verified R2/R3 CSVs and build R5 tables."""
    root = Path(repository_root)
    return build_r5_tables(
        pd.read_csv(root / "results" / "data" / "fund_returns.csv"),
        pd.read_csv(root / "results" / "data" / "fund_weights.csv"),
        pd.read_csv(root / "results" / "tables" / "performance_metrics.csv"),
        pd.read_csv(root / "results" / "data" / "sector_sentiment_index.csv"),
    )


def _figure_context():
    import matplotlib as mpl

    return mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.parse_math": False,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig, output_path: Path) -> None:
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        metadata={"Software": "matplotlib"},
    )
    import matplotlib.pyplot as plt

    plt.close(fig)


def _caption(fig, text: str, *, y: float = 0.012) -> None:
    fig.text(
        0.5,
        y,
        textwrap.fill(text, width=155),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#52606D",
    )


def _fund_colours(fund_names: list[str]) -> dict[str, str]:
    palette = ["#0B5CAD", "#D97706", "#2F855A", "#8B5CF6", "#C53030"]
    return {name: palette[index % len(palette)] for index, name in enumerate(fund_names)}


def _set_strict_date_limits(ax, dates: pd.Series | pd.Index) -> None:
    """Restrict a date axis to observed data so no post-sample tick is shown."""
    observed = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    if observed.empty:
        raise ValueError("date-axis limits require at least one observed date")
    ax.set_xlim(observed.min(), observed.max())
    ax.margins(x=0.0)


def _plot_growth(tables: R5Tables, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    funds = list(tables.performance_metrics["fund_name"])
    colours = _fund_colours(funds)
    start = tables.fund_returns["date"].min().date()
    end = tables.fund_returns["date"].max().date()
    with _figure_context():
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for fund in funds:
            frame = tables.fund_returns.loc[
                tables.fund_returns["fund_name"].eq(fund)
            ].sort_values("date")
            ax.plot(frame["date"], frame["growth_of_1"], label=fund, color=colours[fund], linewidth=2)
        ax.axhline(1.0, color="#7B8794", linewidth=0.8, linestyle="--")
        ax.set_title(f"Growth of $1 Across Combined Fund Methods\nHistorical OOS sample: {start} to {end}")
        ax.set_xlabel("Observed equity trading date")
        ax.set_ylabel("Portfolio value from $1 initial investment ($)")
        _set_strict_date_limits(ax, tables.fund_returns["date"])
        ax.legend(loc="best")
        _caption(
            fig,
            f"Source: {SOURCE_R2} Values are gross of transaction costs and fees; daily returns are compounded geometrically.",
        )
        fig.tight_layout(rect=(0, 0.065, 1, 1))
        _save_figure(fig, output_path)


def _plot_drawdown(tables: R5Tables, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    funds = list(tables.performance_metrics["fund_name"])
    colours = _fund_colours(funds)
    start = tables.fund_returns["date"].min().date()
    end = tables.fund_returns["date"].max().date()
    with _figure_context():
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for fund in funds:
            frame = tables.fund_returns.loc[
                tables.fund_returns["fund_name"].eq(fund)
            ].sort_values("date")
            ax.plot(frame["date"], frame["drawdown"] * 100.0, label=fund, color=colours[fund], linewidth=1.8)
        ax.axhline(0.0, color="#7B8794", linewidth=0.8)
        ax.set_title(f"Drawdown Across Combined Fund Methods\nHistorical OOS sample: {start} to {end}")
        ax.set_xlabel("Observed equity trading date")
        ax.set_ylabel("Drawdown from prior wealth peak (%)")
        _set_strict_date_limits(ax, tables.fund_returns["date"])
        ax.legend(loc="lower left")
        _caption(
            fig,
            f"Source: {SOURCE_R2} Drawdown equals current growth of $1 divided by its running peak (including the initial $1), minus one.",
        )
        fig.tight_layout(rect=(0, 0.065, 1, 1))
        _save_figure(fig, output_path)


def _plot_weights(tables: R5Tables, output_path: Path) -> str:
    import matplotlib.pyplot as plt

    candidates = tables.fund_weights.loc[
        tables.fund_weights["portfolio_method"].eq(portfolios.RISK_PARITY),
        "fund_name",
    ]
    fund = candidates.iloc[0] if not candidates.empty else tables.fund_weights["fund_name"].iloc[0]
    frame = tables.fund_weights.loc[tables.fund_weights["fund_name"].eq(fund)].copy()
    means = frame.groupby("ticker")["target_weight"].mean().sort_values(ascending=False)
    top_tickers = list(means.head(10).index)
    frame["display_holding"] = np.where(frame["ticker"].isin(top_tickers), frame["ticker"], "Other holdings")
    pivot = frame.pivot_table(
        index="rebalance_date",
        columns="display_holding",
        values="target_weight",
        aggfunc="sum",
    ).sort_index()
    order = top_tickers + ["Other holdings"]
    pivot = pivot.reindex(columns=[name for name in order if name in pivot.columns]).fillna(0.0)
    start, end = pivot.index.min().date(), pivot.index.max().date()
    colours = list(plt.get_cmap("tab20").colors[: len(top_tickers)]) + ["#CBD2D9"]
    with _figure_context():
        fig, ax = plt.subplots(figsize=(11, 6.2))
        ax.stackplot(
            pivot.index,
            *(pivot[column].to_numpy() * 100.0 for column in pivot.columns),
            labels=list(pivot.columns),
            colors=colours[: len(pivot.columns)],
            alpha=0.92,
        )
        ax.set_ylim(0, 100)
        ax.set_title(f"Target Portfolio Weights Over Time — {fund}\nMonthly targets: {start} to {end}")
        ax.set_xlabel("Target rebalance date")
        ax.set_ylabel("Target portfolio weight (%)")
        _set_strict_date_limits(ax, pivot.index)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
        _caption(
            fig,
            f"Source: {SOURCE_R2} The ten tickers with the highest mean target weight are shown separately; all other exact target holdings are aggregated as Other holdings. Targets are monthly and are not daily drifted weights.",
        )
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        _save_figure(fig, output_path)
    return fund


def _plot_risk_return(tables: R5Tables, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = tables.performance_metrics
    funds = list(metrics["fund_name"])
    colours = _fund_colours(funds)
    periods = metrics["evaluation_period"].unique()
    sample = periods[0] if len(periods) == 1 else "fund-specific periods in performance table"
    with _figure_context():
        fig, ax = plt.subplots(figsize=(9, 6.0))
        for _, row in metrics.iterrows():
            x = float(row["annualised_volatility"]) * 100.0
            y = float(row["annualised_return"]) * 100.0
            ax.scatter(x, y, s=130, color=colours[row["fund_name"]], edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(
                f"{row['fund_name']}\nSharpe {float(row['sharpe_ratio']):.3f}",
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8.5,
            )
        ax.set_title(f"Annualised Return and Risk Across Combined Fund Methods\nHistorical OOS sample: {sample}")
        ax.set_xlabel("Annualised volatility (%)")
        ax.set_ylabel("Geometric annualised return (%)")
        _caption(
            fig,
            f"Source: {SOURCE_R2} Volatility and Sharpe use 252 observations per year and a zero annual risk-free rate; returns are gross of transaction costs and fees.",
        )
        fig.tight_layout(rect=(0, 0.075, 1, 1))
        _save_figure(fig, output_path)


def _plot_sector_sentiment(tables: R5Tables, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = tables.sector_sentiment_index
    sectors = sorted(frame["sector"].unique())
    start, end = frame["date"].min().date(), frame["date"].max().date()
    with _figure_context():
        fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex=True, sharey=True)
        for ax, sector_name in zip(axes.flat, sectors, strict=False):
            sector_frame = frame.loc[frame["sector"].eq(sector_name)].sort_values("date")
            ax.plot(sector_frame["date"], sector_frame["sentiment_value"], color="#0B5CAD", linewidth=0.85)
            ax.axhline(0.0, color="#7B8794", linewidth=0.6)
            ax.set_title(sector_name, fontsize=10)
            ax.set_ylim(-1.02, 1.02)
            _set_strict_date_limits(ax, frame["date"])
        for ax in axes.flat[len(sectors) :]:
            ax.set_visible(False)
        fig.suptitle(
            "Equity Sector Headline-Sentiment Index\n"
            f"Observed equity trading dates: {start} to {end}",
            y=0.995,
        )
        fig.supxlabel("Observed equity trading date", y=0.07)
        fig.supylabel("VADER compound sector index (range -1 to +1)", x=0.03)
        _caption(
            fig,
            f"Source: {SOURCE_R3} Each value is the equal-weighted mean of observed ticker-day headline means within the sector. Sector-days without headlines remain missing and appear as gaps.",
            y=0.015,
        )
        fig.tight_layout(rect=(0.05, 0.085, 1, 0.965), h_pad=1.2)
        _save_figure(fig, output_path)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _holding_table_cells(holdings: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    ordered = holdings.sort_values(["holding_rank", "ticker"])
    boundaries = np.linspace(0, len(ordered), 4, dtype=int)
    blocks = [
        ordered.iloc[boundaries[index] : boundaries[index + 1]].reset_index(drop=True)
        for index in range(3)
    ]
    row_count = max(len(block) for block in blocks)
    headers: list[str] = []
    for _ in blocks:
        headers.extend(["Ticker", "Weight"])
    rows: list[list[str]] = []
    for row_number in range(row_count):
        row: list[str] = []
        for block in blocks:
            if row_number < len(block):
                holding = block.iloc[row_number]
                row.extend([str(holding["ticker"]), f"{float(holding['target_weight']) * 100.0:.3f}%"])
            else:
                row.extend(["", ""])
        rows.append(row)
    return headers, rows


def _plot_fact_sheet(
    tables: R5Tables,
    summary_row: pd.Series,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fund = summary_row["fund_name"]
    path = tables.fund_returns.loc[tables.fund_returns["fund_name"].eq(fund)].sort_values("date")
    holdings = tables.fact_sheet_holdings.loc[tables.fact_sheet_holdings["fund_name"].eq(fund)]
    headers, cells = _holding_table_cells(holdings)
    sample = summary_row["sample_period"]
    latest = pd.Timestamp(summary_row["latest_rebalance_date"]).date()
    labels = [
        ("Growth of $1", f"${float(summary_row['final_growth_of_1']):.3f}"),
        ("Annualised return", f"{float(summary_row['annualised_return']) * 100.0:.2f}%"),
        ("Annualised volatility", f"{float(summary_row['annualised_volatility']) * 100.0:.2f}%"),
        ("Sharpe ratio", f"{float(summary_row['sharpe_ratio']):.3f}"),
        ("Maximum drawdown", f"{float(summary_row['maximum_drawdown']) * 100.0:.2f}%"),
    ]
    with _figure_context():
        fig = plt.figure(figsize=(16, 9))
        grid = fig.add_gridspec(2, 5, height_ratios=[0.9, 5.2], hspace=0.28, wspace=0.30)
        for index, (label, value) in enumerate(labels):
            tile = fig.add_subplot(grid[0, index])
            tile.set_facecolor("#F0F4F8")
            tile.text(0.5, 0.62, value, ha="center", va="center", fontsize=16, color="#102A43", weight="bold")
            tile.text(0.5, 0.22, label, ha="center", va="center", fontsize=8.5, color="#52606D")
            tile.set_xticks([])
            tile.set_yticks([])
            for spine in tile.spines.values():
                spine.set_visible(False)

        growth_ax = fig.add_subplot(grid[1, :3])
        growth_ax.plot(path["date"], path["growth_of_1"], color="#0B5CAD", linewidth=2.0)
        growth_ax.axhline(1.0, color="#7B8794", linewidth=0.8, linestyle="--")
        growth_ax.set_title("Historical growth of $1")
        growth_ax.set_xlabel("Observed equity trading date")
        growth_ax.set_ylabel("Portfolio value from $1 initial investment ($)")
        _set_strict_date_limits(growth_ax, path["date"])

        table_ax = fig.add_subplot(grid[1, 3:])
        table_ax.axis("off")
        table_ax.set_title(
            f"Current target holdings — {latest}\n"
            f"{len(holdings)} holdings; total target weight {holdings['target_weight'].sum() * 100.0:.3f}%",
            fontsize=11,
            pad=8,
        )
        table = table_ax.table(
            cellText=cells,
            colLabels=headers,
            loc="center",
            cellLoc="right",
            colLoc="center",
            colWidths=[0.13, 0.16] * 3,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.1)
        table.scale(1.0, 1.22)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#D9E2EC")
            if row == 0:
                cell.set_facecolor("#D9EAF7")
                cell.set_text_props(weight="bold", color="#102A43")
            elif column % 2 == 0:
                cell.set_text_props(ha="left")

        fig.suptitle(
            f"Fund Fact Sheet — {fund}\nHistorical walk-forward OOS sample: {sample}",
            fontsize=16,
            y=0.985,
        )
        _caption(
            fig,
            f"Source: {SOURCE_R2} Metrics use 252 observations per year and a zero annual risk-free rate. Results are gross of transaction costs and fees. Holdings are the exact latest monthly target weights; displayed percentages are rounded to three decimals and exact weights are in results/tables/fund_fact_sheet_holdings.csv.",
            y=0.008,
        )
        fig.subplots_adjust(
            left=0.045,
            right=0.965,
            bottom=0.105,
            top=0.865,
            hspace=0.34,
            wspace=0.34,
        )
        _save_figure(fig, output_path)


def _manifest_rows(tables: R5Tables, weights_fund: str) -> list[dict[str, object]]:
    return_periods = tables.performance_metrics["evaluation_period"].unique()
    return_sample = return_periods[0] if len(return_periods) == 1 else "Fund-specific; see performance table"
    sentiment_sample = (
        f"{tables.sector_sentiment_index['date'].min().date()} to "
        f"{tables.sector_sentiment_index['date'].max().date()}"
    )
    weight_frame = tables.fund_weights.loc[tables.fund_weights["fund_name"].eq(weights_fund)]
    weights_sample = f"{weight_frame['rebalance_date'].min().date()} to {weight_frame['rebalance_date'].max().date()}"
    pending = "PENDING - student must write and verify interpretation"
    acceptance = "PENDING - student visual review and final acceptance"
    rows = [
        ("performance_comparison", "table", "results/tables/performance_comparison.csv", "Performance Comparison Across Combined Fund Methods", return_sample, SOURCE_R2, "Annualised return, annualised volatility, Sharpe ratio, maximum drawdown, and terminal growth of $1 for each verified R2 combined fund."),
        ("growth_of_1", "figure", "results/figures/growth_of_1_comparison.png", "Growth of $1 Across Combined Fund Methods", return_sample, SOURCE_R2, "Daily OOS fund returns compounded geometrically from an initial value of $1."),
        ("drawdown", "figure", "results/figures/drawdown_comparison.png", "Drawdown Across Combined Fund Methods", return_sample, SOURCE_R2, "Drawdown from the running wealth peak, including the initial $1 peak."),
        ("portfolio_weights", "figure", "results/figures/portfolio_weights_over_time_risk_parity.png", f"Target Portfolio Weights Over Time - {weights_fund}", weights_sample, SOURCE_R2, "Monthly target weights for the ten tickers with the highest mean target weight; remaining exact holdings are aggregated in the figure."),
        ("risk_return", "figure", "results/figures/risk_return_comparison.png", "Annualised Return and Risk Across Combined Fund Methods", return_sample, SOURCE_R2, "Geometric annualised return plotted against annualised volatility; point labels report Sharpe ratios."),
        ("sector_sentiment", "figure", "results/figures/sector_sentiment_index.png", "Equity Sector Headline-Sentiment Index", sentiment_sample, SOURCE_R3, "Equal-weighted mean of observed ticker-day VADER compound headline scores by equity sector; missing sector-days remain gaps."),
    ]
    for _, row in tables.fact_sheet_summary.iterrows():
        path = f"results/figures/fund_fact_sheet_{_slug(row['fund_name'])}.png"
        rows.append((f"fact_sheet_{_slug(row['fund_name'])}", "fact sheet", path, f"Fund Fact Sheet - {row['fund_name']}", row["sample_period"], SOURCE_R2, "One-page factual sheet containing growth of $1, annualised return, annualised volatility, Sharpe ratio, maximum drawdown, and all latest target holdings."))
    return [
        {
            "exhibit_id": exhibit_id,
            "artifact_type": artifact_type,
            "file_path": file_path,
            "title": title,
            "sample_period": sample_period,
            "source": source,
            "technical_caption": caption,
            "student_interpretation_status": pending,
            "final_figure_acceptance_status": acceptance,
        }
        for exhibit_id, artifact_type, file_path, title, sample_period, source, caption in rows
    ]


def _fusion_manifest_rows(root: Path) -> list[dict[str, object]]:
    table_path = root / "results" / "tables" / "sentiment_fusion_comparison.csv"
    figure_path = root / "results" / "figures" / "sentiment_fusion_comparison.png"
    if not table_path.exists() or not figure_path.exists():
        return []
    comparison = pd.read_csv(table_path)
    _require_columns(comparison, ["evaluation_period"], "R4 fusion comparison")
    periods = comparison["evaluation_period"].dropna().astype(str).unique()
    sample = periods[0] if len(periods) == 1 else "Variant-specific; see R4 table"
    shared = {
        "sample_period": sample,
        "source": "Verified R4 precomputed equity sentiment-fusion comparison.",
        "student_interpretation_status": "PENDING - student must write and verify interpretation",
        "final_figure_acceptance_status": "PENDING - student visual review and final acceptance",
    }
    return [
        {
            "exhibit_id": "sentiment_fusion_table",
            "artifact_type": "table",
            "file_path": "results/tables/sentiment_fusion_comparison.csv",
            "title": "Equity Equal Weight Before and After Sentiment Fusion",
            "technical_caption": "Verified R4 base-versus-fusion metrics on an identical historical OOS sample.",
            **shared,
        },
        {
            "exhibit_id": "sentiment_fusion_figure",
            "artifact_type": "figure",
            "file_path": "results/figures/sentiment_fusion_comparison.png",
            "title": "Equity Equal Weight Before and After Sentiment Fusion",
            "technical_caption": "Verified R4 growth and drawdown paths for the base and approved one-day-lagged sentiment variant.",
            **shared,
        },
    ]


def write_r5_artifacts(
    tables: R5Tables,
    repository_root: str | Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Write report-ready R5 tables, figures, fact sheets, and manifest."""
    root = Path(repository_root)
    table_directory = root / "results" / "tables"
    figure_directory = root / "results" / "figures"
    table_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    comparison_path = table_directory / "performance_comparison.csv"
    summary_path = table_directory / "fund_fact_sheet_summary.csv"
    holdings_path = table_directory / "fund_fact_sheet_holdings.csv"
    growth_path = figure_directory / "growth_of_1_comparison.png"
    drawdown_path = figure_directory / "drawdown_comparison.png"
    weights_path = figure_directory / "portfolio_weights_over_time_risk_parity.png"
    risk_return_path = figure_directory / "risk_return_comparison.png"
    sentiment_path = figure_directory / "sector_sentiment_index.png"

    tables.performance_comparison.to_csv(comparison_path, index=False, date_format="%Y-%m-%d")
    tables.fact_sheet_summary.to_csv(summary_path, index=False, date_format="%Y-%m-%d")
    tables.fact_sheet_holdings.to_csv(holdings_path, index=False, date_format="%Y-%m-%d")
    _plot_growth(tables, growth_path)
    _plot_drawdown(tables, drawdown_path)
    weights_fund = _plot_weights(tables, weights_path)
    _plot_risk_return(tables, risk_return_path)
    _plot_sector_sentiment(tables, sentiment_path)

    fact_sheet_paths: list[Path] = []
    for _, summary_row in tables.fact_sheet_summary.iterrows():
        path = figure_directory / f"fund_fact_sheet_{_slug(summary_row['fund_name'])}.png"
        _plot_fact_sheet(tables, summary_row, path)
        fact_sheet_paths.append(path)

    manifest = pd.DataFrame(
        _manifest_rows(tables, weights_fund) + _fusion_manifest_rows(root),
        columns=MANIFEST_COLUMNS,
    )
    manifest_path = table_directory / "report_exhibit_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    paths = [
        comparison_path,
        summary_path,
        holdings_path,
        manifest_path,
        growth_path,
        drawdown_path,
        weights_path,
        risk_return_path,
        sentiment_path,
        *fact_sheet_paths,
    ]
    return paths, manifest


def write_r5_date_axis_figures(
    tables: R5Tables,
    repository_root: str | Path,
) -> list[Path]:
    """Regenerate only the six R5 figures with observed-date x-axes."""
    figure_directory = Path(repository_root) / "results" / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    growth_path = figure_directory / "growth_of_1_comparison.png"
    drawdown_path = figure_directory / "drawdown_comparison.png"
    weights_path = figure_directory / "portfolio_weights_over_time_risk_parity.png"
    sentiment_path = figure_directory / "sector_sentiment_index.png"
    _plot_growth(tables, growth_path)
    _plot_drawdown(tables, drawdown_path)
    _plot_weights(tables, weights_path)
    _plot_sector_sentiment(tables, sentiment_path)

    fact_sheet_paths: list[Path] = []
    for _, summary_row in tables.fact_sheet_summary.iterrows():
        path = (
            figure_directory
            / f"fund_fact_sheet_{_slug(summary_row['fund_name'])}.png"
        )
        _plot_fact_sheet(tables, summary_row, path)
        fact_sheet_paths.append(path)
    return [
        drawdown_path,
        *fact_sheet_paths,
        growth_path,
        weights_path,
        sentiment_path,
    ]


def build_and_write_r5_from_precomputed_results(
    repository_root: str | Path,
) -> tuple[R5Tables, list[Path], pd.DataFrame]:
    """Validate the saved verified R2/R3 results and produce all R5 artifacts."""
    tables = read_precomputed_r5_inputs(repository_root)
    paths, manifest = write_r5_artifacts(tables, repository_root)
    return tables, paths, manifest
