"""Strict results-only data access for the deployed Streamlit app.

This module deliberately knows nothing about raw course data, portfolio fitting,
headline scoring, or the Part B build pipeline.  It validates and reads only the
committed, precomputed artifacts under ``results/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


MAX_APPROVED_DATE = pd.Timestamp("2023-12-31")

APP_CSV_PATHS = {
    "fund_returns": "results/data/fund_returns.csv",
    "fund_weights": "results/data/fund_weights.csv",
    "sector_sentiment": "results/data/sector_sentiment_index.csv",
    "fusion_returns": "results/data/sentiment_fusion_returns.csv",
    "fusion_weights": "results/data/sentiment_fusion_weights.csv",
    "performance_metrics": "results/tables/performance_metrics.csv",
    "performance_comparison": "results/tables/performance_comparison.csv",
    "fact_sheet_summary": "results/tables/fund_fact_sheet_summary.csv",
    "fact_sheet_holdings": "results/tables/fund_fact_sheet_holdings.csv",
    "fusion_comparison": "results/tables/sentiment_fusion_comparison.csv",
    "fusion_turnover": "results/tables/sentiment_fusion_turnover.csv",
    "fusion_cost_check": "results/tables/sentiment_fusion_transaction_cost_check.csv",
    "fusion_sensitivity": "results/tables/sentiment_multiplier_sensitivity.csv",
}

APP_FIGURE_PATHS = {
    "growth": "results/figures/growth_of_1_comparison.png",
    "drawdown": "results/figures/drawdown_comparison.png",
    "risk_return": "results/figures/risk_return_comparison.png",
    "weights": "results/figures/portfolio_weights_over_time_risk_parity.png",
    "sentiment": "results/figures/sector_sentiment_index.png",
    "fact_sheet_equal_weight": "results/figures/fund_fact_sheet_combined_equal_weight.png",
    "fact_sheet_risk_parity": "results/figures/fund_fact_sheet_combined_risk_parity.png",
}

REQUIRED_COLUMNS = {
    "fund_returns": {
        "date", "fund_name", "daily_return", "growth_of_1", "drawdown",
    },
    "fund_weights": {
        "rebalance_date", "fund_name", "ticker", "holding_asset_class",
        "target_weight", "estimation_end_date",
    },
    "sector_sentiment": {
        "date", "sector", "sentiment_value", "headline_count",
        "ticker_with_news_count", "sentiment_observed",
    },
    "fusion_returns": {
        "date", "strategy_name", "variant", "daily_return", "growth_of_1",
        "drawdown",
    },
    "fusion_weights": {
        "rebalance_date", "strategy_name", "variant", "ticker", "sector",
        "holding_asset_class", "target_weight", "sentiment_observation_date",
    },
    "performance_metrics": {
        "fund_name", "evaluation_start_date", "evaluation_end_date",
        "annualised_return", "annualised_volatility", "sharpe_ratio",
        "maximum_drawdown", "final_growth_of_1", "annualisation_days_per_year",
        "risk_free_rate_annual", "transaction_cost_rate", "rebalance_rule",
        "constraints",
    },
    "performance_comparison": {
        "fund_name", "sample_period", "annualised_return",
        "annualised_volatility", "sharpe_ratio", "maximum_drawdown",
        "final_growth_of_1",
    },
    "fact_sheet_summary": {
        "fund_name", "sample_period", "final_growth_of_1", "annualised_return",
        "annualised_volatility", "sharpe_ratio", "maximum_drawdown",
        "latest_rebalance_date", "current_holdings_count",
        "current_equity_target_weight", "current_cryptocurrency_target_weight",
    },
    "fact_sheet_holdings": {
        "fund_name", "latest_rebalance_date", "holding_rank", "ticker",
        "holding_asset_class", "target_weight",
    },
    "fusion_comparison": {
        "strategy_name", "variant", "evaluation_start_date", "evaluation_end_date",
        "annualised_return", "annualised_volatility", "sharpe_ratio",
        "maximum_drawdown", "final_growth_of_1", "total_one_way_turnover",
        "sentiment_lag_observed_trading_days", "sentiment_missing_policy",
        "tilt_rule",
    },
    "fusion_turnover": {
        "rebalance_date", "strategy_name", "variant", "initial_establishment",
        "one_way_turnover",
    },
    "fusion_cost_check": {
        "strategy_name", "variant", "cost_rate_bps_one_way",
        "net_annualised_return", "net_sharpe_ratio", "net_final_growth_of_1",
    },
    "fusion_sensitivity": {
        "strategy_name", "tilt_strength", "approved_design", "annualised_return",
        "annualised_volatility", "sharpe_ratio", "maximum_drawdown",
        "final_growth_of_1", "total_one_way_turnover",
    },
}

DATE_COLUMNS = {
    "fund_returns": ("date", "active_target_rebalance_date"),
    "fund_weights": ("rebalance_date", "estimation_start_date", "estimation_end_date"),
    "sector_sentiment": ("date",),
    "fusion_returns": ("date", "active_target_rebalance_date"),
    "fusion_weights": (
        "rebalance_date", "sentiment_observation_date", "estimation_start_date",
        "estimation_end_date",
    ),
    "performance_metrics": (
        "evaluation_start_date", "evaluation_end_date", "latest_rebalance_date",
    ),
    "fact_sheet_summary": ("latest_rebalance_date",),
    "fact_sheet_holdings": ("latest_rebalance_date",),
    "fusion_comparison": ("evaluation_start_date", "evaluation_end_date"),
    "fusion_turnover": ("rebalance_date",),
}


def _results_path(repository_root: Path, relative_path: str) -> Path:
    """Resolve and constrain an artifact path to the repository results folder."""
    results_root = (repository_root / "results").resolve()
    path = (repository_root / relative_path).resolve()
    if path != results_root and results_root not in path.parents:
        raise ValueError(f"App artifact is outside results/: {relative_path}")
    return path


def _validate_date_cap(frame: pd.DataFrame, label: str) -> None:
    for column in DATE_COLUMNS.get(label, ()):
        if column not in frame.columns:
            continue
        observed = frame[column].dropna()
        if not observed.empty and observed.max() > MAX_APPROVED_DATE:
            raise ValueError(
                f"{label}.{column} exceeds the approved 2023-12-31 sample cap"
            )


def load_app_data(repository_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load and validate every precomputed CSV required by the app."""
    root = Path(repository_root)
    loaded: dict[str, pd.DataFrame] = {}
    for label, relative_path in APP_CSV_PATHS.items():
        path = _results_path(root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Required precomputed app artifact is missing: {relative_path}")
        frame = pd.read_csv(path, parse_dates=list(DATE_COLUMNS.get(label, ())))
        missing = REQUIRED_COLUMNS[label].difference(frame.columns)
        if missing:
            raise ValueError(f"{relative_path} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{relative_path} is empty")
        _validate_date_cap(frame, label)
        loaded[label] = frame

    for relative_path in APP_FIGURE_PATHS.values():
        path = _results_path(root, relative_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required precomputed app figure is missing: {relative_path}")

    fund_names = set(loaded["performance_metrics"]["fund_name"])
    if fund_names != set(loaded["fund_returns"]["fund_name"]):
        raise ValueError("Fund names do not match between returns and performance metrics")
    if fund_names != set(loaded["fact_sheet_summary"]["fund_name"]):
        raise ValueError("Fund names do not match between metrics and fact sheets")
    if loaded["fund_returns"].duplicated(["date", "fund_name"]).any():
        raise ValueError("fund_returns.csv contains duplicate fund-date rows")
    if loaded["sector_sentiment"].duplicated(["date", "sector"]).any():
        raise ValueError("sector_sentiment_index.csv contains duplicate date-sector rows")
    if loaded["fusion_returns"].duplicated(["date", "variant"]).any():
        raise ValueError("sentiment_fusion_returns.csv contains duplicate date-variant rows")
    return loaded


def figure_paths(repository_root: str | Path) -> dict[str, Path]:
    """Return validated absolute paths for the app's precomputed figures."""
    root = Path(repository_root)
    return {
        label: _results_path(root, relative_path)
        for label, relative_path in APP_FIGURE_PATHS.items()
    }


def historical_allocation_scenarios(
    fund_returns: pd.DataFrame,
    investment_amount: float,
    custom_weights: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build historical buy-and-hold fund-allocation paths from saved growth data.

    Fund-level allocations are fixed at the start of the saved OOS sample.  There
    is no rebalancing between funds.  Each underlying fund retains its own saved
    monthly target-reset convention.
    """
    if not pd.notna(investment_amount) or investment_amount <= 0:
        raise ValueError("investment_amount must be positive and finite")
    pivot = (
        fund_returns.pivot(index="date", columns="fund_name", values="growth_of_1")
        .sort_index()
    )
    if pivot.isna().any().any():
        raise ValueError("Completed funds do not share a complete historical sample")
    funds = list(pivot.columns)
    if set(custom_weights) != set(funds):
        raise ValueError("Custom allocation must specify every completed fund exactly once")
    weights = pd.Series(custom_weights, dtype=float).reindex(funds)
    if not weights.map(pd.notna).all() or (weights < 0).any():
        raise ValueError("Custom allocation weights must be finite and non-negative")
    if abs(float(weights.sum()) - 1.0) > 1e-9:
        raise ValueError("Custom allocation weights must sum to one")

    scenarios: dict[str, pd.Series] = {
        f"100% {fund}": pivot[fund] * investment_amount for fund in funds
    }
    scenarios["Custom allocation"] = pivot.mul(weights, axis="columns").sum(axis=1) * investment_amount
    paths = pd.DataFrame(scenarios, index=pivot.index)
    summary = pd.DataFrame(
        {
            "Scenario": paths.columns,
            "Starting amount": investment_amount,
            "Historical ending value": paths.iloc[-1].to_numpy(),
        }
    )
    summary["Historical change"] = summary["Historical ending value"] - investment_amount
    summary["Historical return"] = summary["Historical ending value"] / investment_amount - 1.0
    return paths, summary
