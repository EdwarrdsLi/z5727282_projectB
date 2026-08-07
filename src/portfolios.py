"""Transparent portfolio construction and walk-forward OOS backtesting.

R2 offers two combined equity-and-cryptocurrency funds: Equal Weight and
long-only equal-risk-contribution Risk Parity.  Target weights are formed from
returns strictly before each rebalance date, reset on the first observed equity
trading day of each month, and allowed to drift between rebalances.  The module
does not use sentiment, forecasts, forward-filled data, or post-2023 evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src import features


SAMPLE_END = pd.Timestamp("2023-12-31")
DEFAULT_ESTIMATION_WINDOW = 252
DEFAULT_PERIODS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_TRANSACTION_COST_RATE = 0.0
DEFAULT_COVARIANCE_DIAGONAL_SHRINKAGE = 0.05

EQUAL_WEIGHT = "Equal Weight"
RISK_PARITY = "Risk Parity"
SUPPORTED_METHODS = (EQUAL_WEIGHT, RISK_PARITY)

FUND_RETURN_COLUMNS = [
    "date",
    "fund_name",
    "asset_family",
    "portfolio_method",
    "daily_return",
    "growth_of_1",
    "drawdown",
    "active_target_rebalance_date",
]

FUND_WEIGHT_COLUMNS = [
    "rebalance_date",
    "fund_name",
    "asset_family",
    "portfolio_method",
    "ticker",
    "holding_asset_class",
    "target_weight",
    "estimation_start_date",
    "estimation_end_date",
    "estimation_observations",
]

PERFORMANCE_METRIC_COLUMNS = [
    "fund_name",
    "asset_family",
    "portfolio_method",
    "evaluation_period",
    "evaluation_start_date",
    "evaluation_end_date",
    "observations",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "final_growth_of_1",
    "current_target_holdings_source",
    "latest_rebalance_date",
    "annualisation_days_per_year",
    "risk_free_rate_annual",
    "transaction_cost_rate",
    "estimation_window_type",
    "estimation_window_observations",
    "rebalance_rule",
    "constraints",
]


@dataclass(frozen=True)
class BacktestResult:
    """One fund's daily OOS path, monthly target weights, and metrics."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    metrics: dict[str, object]


@dataclass(frozen=True)
class R2Artifacts:
    """The three required R2 output tables."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    performance_metrics: pd.DataFrame


def build_combined_return_matrix(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return the R1 combined panel as a ticker matrix plus class metadata.

    :func:`src.features.build_combined_returns_panel` calculates adjusted-close
    returns within the two native calendars before aligning already-calculated
    cryptocurrency returns to observed equity dates.  This wrapper only removes
    the explicit column prefixes; it never fills a price or return.
    """
    combined = features.build_combined_returns_panel(equity_prices, crypto_prices)
    if combined.empty:
        raise ValueError("combined return panel must not be empty")
    if combined["date"].max() > SAMPLE_END:
        raise ValueError("combined return panel exceeds 2023-12-31")

    equity_prefix = "equity_return__"
    crypto_prefix = "crypto_return__"
    return_columns = [column for column in combined.columns if column != "date"]
    renamed: dict[str, str] = {}
    asset_classes: dict[str, str] = {}
    for column in return_columns:
        if column.startswith(equity_prefix):
            ticker = column.removeprefix(equity_prefix)
            asset_class = "equity"
        elif column.startswith(crypto_prefix):
            ticker = column.removeprefix(crypto_prefix)
            asset_class = "cryptocurrency"
        else:
            raise ValueError(f"unexpected combined-return column: {column}")
        if ticker in asset_classes:
            raise ValueError(f"ticker appears in both asset families: {ticker}")
        renamed[column] = ticker
        asset_classes[ticker] = asset_class

    matrix = combined.set_index("date").rename(columns=renamed)
    matrix = matrix.loc[:, sorted(matrix.columns, key=str)]
    asset_classes = {ticker: asset_classes[ticker] for ticker in matrix.columns}
    return matrix, asset_classes


def equal_weight_weights(estimation_returns: pd.DataFrame) -> pd.Series:
    """Return long-only equal weights for all columns in the window."""
    if estimation_returns.shape[1] == 0:
        raise ValueError("at least one asset is required")
    weight = 1.0 / estimation_returns.shape[1]
    return pd.Series(weight, index=estimation_returns.columns, dtype=float)


def risk_parity_weights(estimation_returns: pd.DataFrame) -> pd.Series:
    """Estimate long-only equal-risk-contribution target weights.

    The convex log-barrier formulation solves
    ``0.5 * x' Sigma x - sum_i b_i log(x_i)`` with equal risk budgets ``b_i``.
    Normalising the positive solution to sum to one preserves equal proportional
    risk contributions.  Five-percent diagonal covariance shrinkage makes the
    rolling estimate positive definite and reduces numerical instability.
    Coordinate-descent convergence, constraints, and achieved risk contributions
    are all checked rather than relying on an external solver success flag.
    """
    clean = _validate_estimation_window(estimation_returns)
    covariance = clean.cov(ddof=1).to_numpy(dtype=float)
    n_assets = covariance.shape[0]
    diagonal = np.diag(covariance)
    if not np.isfinite(covariance).all() or np.any(diagonal <= 0.0):
        raise ValueError("risk parity requires a finite covariance with positive variances")

    variance_scale = float(diagonal.mean())
    shrinkage = DEFAULT_COVARIANCE_DIAGONAL_SHRINKAGE
    regularised = (
        (1.0 - shrinkage) * covariance
        + shrinkage * np.diag(diagonal)
    )
    scaled = regularised / variance_scale
    budgets = np.full(n_assets, 1.0 / n_assets)
    raw = np.sqrt(budgets / np.diag(scaled))
    converged = False
    for _ in range(10_000):
        previous = raw.copy()
        for i in range(n_assets):
            variance = scaled[i, i]
            cross_term = float(scaled[i] @ raw - variance * raw[i])
            discriminant = np.sqrt(cross_term**2 + 4.0 * variance * budgets[i])
            # Use the algebraically equivalent stable root when cancellation is
            # possible in ``-cross_term + discriminant``.
            if cross_term >= 0.0:
                raw[i] = 2.0 * budgets[i] / (discriminant + cross_term)
            else:
                raw[i] = (discriminant - cross_term) / (2.0 * variance)
        relative_change = float(
            np.max(np.abs(raw - previous) / np.maximum(np.abs(previous), 1e-15))
        )
        if relative_change < 1e-12:
            converged = True
            break
    if not converged:
        raise RuntimeError("risk-parity coordinate descent did not converge")
    if not np.isfinite(raw).all() or np.any(raw <= 0.0):
        raise RuntimeError("risk-parity optimisation returned invalid positive weights")

    weights = raw / raw.sum()
    portfolio_variance = float(weights @ regularised @ weights)
    contributions = weights * (regularised @ weights) / portfolio_variance
    max_budget_error = float(np.max(np.abs(contributions - budgets)))
    if not np.isfinite(contributions).all() or max_budget_error > 1e-5:
        raise RuntimeError(
            "risk-parity solution failed the equal-risk-contribution check: "
            f"maximum budget error {max_budget_error:.3e}"
        )

    result = pd.Series(weights, index=clean.columns, dtype=float)
    _validate_weights(result)
    return result


def _validate_estimation_window(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("estimation returns must be a non-empty DataFrame")
    if returns.shape[0] < 2:
        raise ValueError("at least two estimation observations are required")
    numeric = returns.apply(pd.to_numeric, errors="raise").astype(float)
    if numeric.isna().any(axis=None) or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("estimation returns must be complete and finite")
    return numeric


def _validate_weights(weights: pd.Series, tolerance: float = 1e-10) -> None:
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("portfolio weights must be finite")
    if np.any(values < -tolerance):
        raise RuntimeError("portfolio weights must be long-only")
    if not np.isclose(values.sum(), 1.0, atol=tolerance, rtol=0.0):
        raise RuntimeError("portfolio weights must sum to one")


def _complete_return_history(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty wide DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")
    if returns.index.has_duplicates:
        raise ValueError("return dates must be unique")
    if returns.columns.has_duplicates:
        raise ValueError("return tickers must be unique")

    ordered = returns.sort_index().apply(pd.to_numeric, errors="raise").astype(float)
    if ordered.index.max() > SAMPLE_END:
        raise ValueError("returns exceed the approved sample end of 2023-12-31")
    if (ordered < -1.0).any(axis=None):
        raise ValueError("simple asset returns cannot be below -100%")
    complete_rows = ~ordered.isna().any(axis=1)
    if not complete_rows.any():
        raise ValueError("returns contain no complete cross-asset observation")
    first_complete_position = int(np.flatnonzero(complete_rows.to_numpy())[0])
    complete_history = ordered.iloc[first_complete_position:]
    if complete_history.isna().any(axis=None):
        raise ValueError(
            "missing returns after the first complete observation require investigation"
        )
    if not np.isfinite(complete_history.to_numpy()).all():
        raise ValueError("returns must be finite")
    return complete_history


def _monthly_rebalance_dates(
    return_dates: pd.DatetimeIndex,
    estimation_window: int,
) -> pd.DatetimeIndex:
    month = return_dates.to_period("M")
    first_in_month = np.r_[True, month[1:] != month[:-1]]
    positions = np.flatnonzero(first_in_month)
    eligible = positions[positions >= estimation_window]
    return return_dates.take(eligible)


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, float]:
    """Compute geometric annual return, risk, Sharpe, drawdown, and growth."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = pd.to_numeric(daily_returns, errors="raise").astype(float)
    if values.empty or values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("daily returns must be non-empty, complete, and finite")
    if (values <= -1.0).any():
        raise ValueError("fund returns must be greater than -100%")

    growth = (1.0 + values).cumprod()
    final_growth = float(growth.iloc[-1])
    annualised_return = final_growth ** (periods_per_year / len(values)) - 1.0
    annualised_volatility = (
        float(values.std(ddof=1) * np.sqrt(periods_per_year))
        if len(values) > 1
        else float("nan")
    )
    daily_risk_free = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    daily_volatility = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    sharpe = (
        float((values.mean() - daily_risk_free) / daily_volatility * np.sqrt(periods_per_year))
        if np.isfinite(daily_volatility) and daily_volatility > 0.0
        else float("nan")
    )
    running_peak = growth.cummax().clip(lower=1.0)
    maximum_drawdown = float((growth / running_peak - 1.0).min())
    return {
        "annualised_return": float(annualised_return),
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": maximum_drawdown,
        "final_growth_of_1": final_growth,
    }


def oos_backtest(
    returns: pd.DataFrame,
    method: str = EQUAL_WEIGHT,
    *,
    fund_name: str | None = None,
    asset_family: str = "Combined",
    asset_classes: dict[str, str] | None = None,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE,
    transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
) -> BacktestResult:
    """Run a monthly walk-forward OOS backtest with pre-return weights.

    The first eligible monthly rebalance follows ``estimation_window`` complete
    prior observations.  A target formed before date *t* is applied to date *t*'s
    return.  Holdings then drift self-financingly until the next monthly target.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    if not isinstance(estimation_window, int) or estimation_window < 2:
        raise ValueError("estimation_window must be an integer of at least two")
    if transaction_cost_rate != 0.0:
        raise ValueError("R2 implements the stated zero-transaction-cost baseline only")

    history = _complete_return_history(returns)
    if len(history) <= estimation_window:
        raise ValueError("return history is too short for an OOS period")
    rebalance_dates = _monthly_rebalance_dates(history.index, estimation_window)
    if rebalance_dates.empty:
        raise ValueError("no eligible first-trading-day monthly rebalance exists")

    missing_classes = set(history.columns).difference(asset_classes or {})
    if asset_classes is not None and missing_classes:
        raise ValueError(f"asset class metadata missing for: {sorted(missing_classes)}")
    holding_classes = asset_classes or {ticker: "unspecified" for ticker in history.columns}
    estimator: Callable[[pd.DataFrame], pd.Series] = (
        equal_weight_weights if method == EQUAL_WEIGHT else risk_parity_weights
    )
    resolved_fund_name = fund_name or f"{asset_family} {method}"

    positions = pd.Series(np.arange(len(history)), index=history.index)
    rebalance_set = set(rebalance_dates)
    live_dates = history.loc[rebalance_dates[0] :].index
    current_weights: pd.Series | None = None
    active_rebalance_date: pd.Timestamp | None = None
    daily_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for date in live_dates:
        if date in rebalance_set:
            position = int(positions.loc[date])
            window = history.iloc[position - estimation_window : position]
            if not (window.index.max() < date):
                raise AssertionError("estimation window must end before rebalance date")
            current_weights = estimator(window).reindex(history.columns)
            _validate_weights(current_weights)
            active_rebalance_date = date
            for ticker, target_weight in current_weights.items():
                weight_rows.append(
                    {
                        "rebalance_date": date,
                        "fund_name": resolved_fund_name,
                        "asset_family": asset_family,
                        "portfolio_method": method,
                        "ticker": ticker,
                        "holding_asset_class": holding_classes[ticker],
                        "target_weight": float(target_weight),
                        "estimation_start_date": window.index.min(),
                        "estimation_end_date": window.index.max(),
                        "estimation_observations": len(window),
                    }
                )

        if current_weights is None or active_rebalance_date is None:
            raise AssertionError("live returns cannot precede the first target weights")
        asset_return = history.loc[date]
        fund_return = float(current_weights @ asset_return)
        if fund_return <= -1.0:
            raise RuntimeError("portfolio wealth became non-positive")
        daily_rows.append(
            {
                "date": date,
                "fund_name": resolved_fund_name,
                "asset_family": asset_family,
                "portfolio_method": method,
                "daily_return": fund_return,
                "active_target_rebalance_date": active_rebalance_date,
            }
        )
        current_weights = current_weights * (1.0 + asset_return) / (1.0 + fund_return)
        current_weights = current_weights / current_weights.sum()
        _validate_weights(current_weights, tolerance=1e-9)

    fund_returns = pd.DataFrame(daily_rows)
    growth = (1.0 + fund_returns["daily_return"]).cumprod()
    fund_returns["growth_of_1"] = growth
    fund_returns["drawdown"] = growth / growth.cummax().clip(lower=1.0) - 1.0
    fund_returns = fund_returns.loc[:, FUND_RETURN_COLUMNS]

    fund_weights = pd.DataFrame(weight_rows, columns=FUND_WEIGHT_COLUMNS)
    metric_values = performance_metrics(
        fund_returns["daily_return"],
        periods_per_year=periods_per_year,
        risk_free_rate_annual=risk_free_rate_annual,
    )
    start = pd.Timestamp(fund_returns["date"].min())
    end = pd.Timestamp(fund_returns["date"].max())
    metrics: dict[str, object] = {
        "fund_name": resolved_fund_name,
        "asset_family": asset_family,
        "portfolio_method": method,
        "evaluation_period": f"{start.date()} to {end.date()}",
        "evaluation_start_date": start,
        "evaluation_end_date": end,
        "observations": len(fund_returns),
        **metric_values,
        "current_target_holdings_source": (
            "results/data/fund_weights.csv; filter fund_name and latest rebalance_date"
        ),
        "latest_rebalance_date": pd.Timestamp(fund_weights["rebalance_date"].max()),
        "annualisation_days_per_year": periods_per_year,
        "risk_free_rate_annual": risk_free_rate_annual,
        "transaction_cost_rate": transaction_cost_rate,
        "estimation_window_type": "Rolling complete equity-calendar observations",
        "estimation_window_observations": estimation_window,
        "rebalance_rule": "First observed equity trading day of each calendar month",
        "constraints": "Long-only; fully invested; no leverage",
    }
    return BacktestResult(fund_returns, fund_weights, metrics)


def build_r2_artifacts(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    *,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
) -> R2Artifacts:
    """Build the two required combined-fund R2 artifacts in memory."""
    combined_returns, asset_classes = build_combined_return_matrix(
        equity_prices,
        crypto_prices,
    )
    results = [
        oos_backtest(
            combined_returns,
            method=method,
            fund_name=f"Combined {method}",
            asset_family="Combined equity + cryptocurrency",
            asset_classes=asset_classes,
            estimation_window=estimation_window,
        )
        for method in SUPPORTED_METHODS
    ]
    fund_returns = pd.concat([result.fund_returns for result in results], ignore_index=True)
    fund_weights = pd.concat([result.fund_weights for result in results], ignore_index=True)
    performance = pd.DataFrame(
        [result.metrics for result in results],
        columns=PERFORMANCE_METRIC_COLUMNS,
    )

    grouped_sums = fund_weights.groupby(
        ["fund_name", "rebalance_date"], sort=False
    )["target_weight"].sum()
    if not np.allclose(grouped_sums.to_numpy(), 1.0, atol=1e-10, rtol=0.0):
        raise AssertionError("at least one saved target portfolio does not sum to one")
    if fund_returns["date"].max() > SAMPLE_END:
        raise AssertionError("R2 fund output exceeds 2023-12-31")

    latest = fund_weights["rebalance_date"].max()
    latest_weights = fund_weights.loc[fund_weights["rebalance_date"] == latest]
    pivot = latest_weights.pivot(index="ticker", columns="portfolio_method", values="target_weight")
    if set(SUPPORTED_METHODS).issubset(pivot.columns):
        max_difference = float((pivot[EQUAL_WEIGHT] - pivot[RISK_PARITY]).abs().max())
        if max_difference <= 1e-6:
            raise RuntimeError("Equal Weight and Risk Parity are not meaningfully different")

    return R2Artifacts(fund_returns, fund_weights, performance)


def write_r2_artifacts(
    artifacts: R2Artifacts,
    repository_root: str | Path,
) -> tuple[Path, Path, Path]:
    """Write exactly the three required R2 CSV files."""
    root = Path(repository_root)
    returns_path = root / "results" / "data" / "fund_returns.csv"
    weights_path = root / "results" / "data" / "fund_weights.csv"
    metrics_path = root / "results" / "tables" / "performance_metrics.csv"
    for path in (returns_path, weights_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.fund_returns.to_csv(returns_path, index=False, date_format="%Y-%m-%d")
    artifacts.fund_weights.to_csv(weights_path, index=False, date_format="%Y-%m-%d")
    artifacts.performance_metrics.to_csv(metrics_path, index=False, date_format="%Y-%m-%d")
    return returns_path, weights_path, metrics_path
