"""Transparent multi-family portfolio construction and walk-forward backtesting.

The fund shelf spans equity-only, cryptocurrency-only, and combined universes,
each implemented with Equal Weight, long-only equal-risk-contribution Risk
Parity, and long-only Minimum Variance. Targets use only prior observations,
reset on the first native-calendar observation of each month, and drift between
rebalances. No sentiment, forecasts, forward-filled data, or post-2023 evidence
enters these fund backtests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import features


SAMPLE_END = pd.Timestamp("2023-12-31")
DEFAULT_ESTIMATION_WINDOW = 252
DEFAULT_CRYPTO_ESTIMATION_WINDOW = 365
DEFAULT_PERIODS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_TRANSACTION_COST_RATE = 0.0
DEFAULT_COVARIANCE_DIAGONAL_SHRINKAGE = 0.05

EQUAL_WEIGHT = "Equal Weight"
RISK_PARITY = "Risk Parity"
MINIMUM_VARIANCE = "Minimum Variance"
SUPPORTED_METHODS = (EQUAL_WEIGHT, RISK_PARITY, MINIMUM_VARIANCE)
DEFAULT_TRANSACTION_COST_BPS = (0.0, 10.0)

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
    "total_one_way_turnover",
    "mean_one_way_turnover",
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

FUND_TURNOVER_COLUMNS = [
    "rebalance_date",
    "fund_name",
    "asset_family",
    "portfolio_method",
    "initial_establishment",
    "one_way_turnover",
]

FUND_COST_CHECK_COLUMNS = [
    "fund_name",
    "asset_family",
    "portfolio_method",
    "cost_rate_bps_one_way",
    "gross_annualised_return",
    "net_annualised_return",
    "gross_sharpe_ratio",
    "net_sharpe_ratio",
    "gross_final_growth_of_1",
    "net_final_growth_of_1",
    "total_one_way_turnover",
]


@dataclass(frozen=True)
class BacktestResult:
    """One fund's daily OOS path, monthly target weights, and metrics."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    fund_turnover: pd.DataFrame
    metrics: dict[str, object]


@dataclass(frozen=True)
class R2Artifacts:
    """Required fund outputs plus turnover and cost evidence."""

    fund_returns: pd.DataFrame
    fund_weights: pd.DataFrame
    performance_metrics: pd.DataFrame
    fund_turnover: pd.DataFrame
    fund_cost_check: pd.DataFrame


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


def build_native_return_matrix(
    prices: pd.DataFrame,
    *,
    asset_class: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build a ticker return matrix on one asset family's native calendar."""
    if asset_class not in {"equity", "cryptocurrency"}:
        raise ValueError("asset_class must be equity or cryptocurrency")
    returned = features.daily_returns(prices, asset_class=asset_class)
    if returned.empty or returned["date"].max() > SAMPLE_END:
        raise ValueError("native return panel is empty or exceeds 2023-12-31")
    matrix = (
        returned.pivot(index="date", columns="ticker", values="simple_return")
        .sort_index()
        .reindex(columns=sorted(returned["ticker"].unique(), key=str))
    )
    classes = {ticker: asset_class for ticker in matrix.columns}
    return matrix, classes


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


def minimum_variance_weights(estimation_returns: pd.DataFrame) -> pd.Series:
    """Estimate a long-only fully invested minimum-variance portfolio."""
    clean = _validate_estimation_window(estimation_returns)
    covariance = clean.cov(ddof=1).to_numpy(dtype=float)
    diagonal = np.diag(covariance)
    if not np.isfinite(covariance).all() or np.any(diagonal <= 0.0):
        raise ValueError("minimum variance requires finite positive variances")
    regularised = (
        (1.0 - DEFAULT_COVARIANCE_DIAGONAL_SHRINKAGE) * covariance
        + DEFAULT_COVARIANCE_DIAGONAL_SHRINKAGE * np.diag(diagonal)
    )
    scale = float(diagonal.mean())
    scaled = regularised / scale
    n_assets = clean.shape[1]
    objective = lambda weights: float(weights @ scaled @ weights)
    gradient = lambda weights: 2.0 * scaled @ weights
    result = minimize(
        objective,
        np.full(n_assets, 1.0 / n_assets),
        jac=gradient,
        bounds=[(0.0, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0,
                     "jac": lambda weights: np.ones_like(weights)},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"minimum-variance optimisation failed: {result.message}")
    weights = pd.Series(result.x, index=clean.columns, dtype=float)
    weights[weights.abs() < 1e-12] = 0.0
    weights = weights / weights.sum()
    _validate_weights(weights, tolerance=1e-8)
    equal = np.full(n_assets, 1.0 / n_assets)
    if objective(weights.to_numpy()) > objective(equal) + 1e-9:
        raise RuntimeError("minimum-variance solution is worse than equal weight")
    return weights


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
    estimators: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        EQUAL_WEIGHT: equal_weight_weights,
        RISK_PARITY: risk_parity_weights,
        MINIMUM_VARIANCE: minimum_variance_weights,
    }
    estimator = estimators[method]
    resolved_fund_name = fund_name or f"{asset_family} {method}"

    positions = pd.Series(np.arange(len(history)), index=history.index)
    rebalance_set = set(rebalance_dates)
    live_dates = history.loc[rebalance_dates[0] :].index
    current_weights: pd.Series | None = None
    active_rebalance_date: pd.Timestamp | None = None
    daily_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    turnover_rows: list[dict[str, object]] = []

    for date in live_dates:
        if date in rebalance_set:
            pre_trade_weights = (
                current_weights.copy() if current_weights is not None else None
            )
            position = int(positions.loc[date])
            window = history.iloc[position - estimation_window : position]
            if not (window.index.max() < date):
                raise AssertionError("estimation window must end before rebalance date")
            current_weights = estimator(window).reindex(history.columns)
            _validate_weights(current_weights)
            active_rebalance_date = date
            turnover = (
                0.0
                if pre_trade_weights is None
                else 0.5 * float((current_weights - pre_trade_weights).abs().sum())
            )
            turnover_rows.append(
                {
                    "rebalance_date": date,
                    "fund_name": resolved_fund_name,
                    "asset_family": asset_family,
                    "portfolio_method": method,
                    "initial_establishment": pre_trade_weights is None,
                    "one_way_turnover": turnover,
                }
            )
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
    fund_turnover = pd.DataFrame(turnover_rows, columns=FUND_TURNOVER_COLUMNS)
    measured_turnover = fund_turnover.loc[~fund_turnover["initial_establishment"]]
    total_turnover = float(measured_turnover["one_way_turnover"].sum())
    mean_turnover = float(measured_turnover["one_way_turnover"].mean())
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
        "total_one_way_turnover": total_turnover,
        "mean_one_way_turnover": mean_turnover,
        "current_target_holdings_source": (
            "results/data/fund_weights.csv; filter fund_name and latest rebalance_date"
        ),
        "latest_rebalance_date": pd.Timestamp(fund_weights["rebalance_date"].max()),
        "annualisation_days_per_year": periods_per_year,
        "risk_free_rate_annual": risk_free_rate_annual,
        "transaction_cost_rate": transaction_cost_rate,
        "estimation_window_type": f"Rolling complete {asset_family.lower()} observations",
        "estimation_window_observations": estimation_window,
        "rebalance_rule": "First observed native-calendar date of each calendar month",
        "constraints": "Long-only; fully invested; no leverage",
    }
    return BacktestResult(fund_returns, fund_weights, fund_turnover, metrics)


def build_r2_artifacts(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    *,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    crypto_estimation_window: int = DEFAULT_CRYPTO_ESTIMATION_WINDOW,
) -> R2Artifacts:
    """Build nine funds across three families and three portfolio methods."""
    equity_returns, equity_classes = build_native_return_matrix(
        equity_prices, asset_class="equity"
    )
    crypto_returns, crypto_classes = build_native_return_matrix(
        crypto_prices, asset_class="cryptocurrency"
    )
    combined_returns, asset_classes = build_combined_return_matrix(
        equity_prices,
        crypto_prices,
    )
    families = [
        ("Equity", "Equity only", equity_returns, equity_classes,
         estimation_window, 252),
        ("Cryptocurrency", "Cryptocurrency only", crypto_returns, crypto_classes,
         crypto_estimation_window, 365),
        ("Combined", "Combined equity + cryptocurrency", combined_returns,
         asset_classes, estimation_window, 252),
    ]
    results = []
    for prefix, family, returns, classes, window, annualisation in families:
        for method in SUPPORTED_METHODS:
            results.append(
                oos_backtest(
                    returns,
                    method=method,
                    fund_name=f"{prefix} {method}",
                    asset_family=family,
                    asset_classes=classes,
                    estimation_window=window,
                    periods_per_year=annualisation,
                )
            )
    fund_returns = pd.concat([result.fund_returns for result in results], ignore_index=True)
    fund_weights = pd.concat([result.fund_weights for result in results], ignore_index=True)
    fund_turnover = pd.concat([result.fund_turnover for result in results], ignore_index=True)
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

    if len(performance) != 9 or performance["fund_name"].nunique() != 9:
        raise AssertionError("the expanded fund shelf must contain exactly nine funds")
    for family in performance["asset_family"].unique():
        family_rows = performance.loc[performance["asset_family"].eq(family)]
        if set(family_rows["portfolio_method"]) != set(SUPPORTED_METHODS):
            raise AssertionError(f"missing method in {family}")

    cost_rows: list[dict[str, object]] = []
    turnover_lookup = fund_turnover.set_index(["fund_name", "rebalance_date"])
    for _, metric_row in performance.iterrows():
        fund = metric_row["fund_name"]
        path = fund_returns.loc[fund_returns["fund_name"].eq(fund)].sort_values("date")
        for cost_bps in DEFAULT_TRANSACTION_COST_BPS:
            rate = cost_bps / 10_000.0
            net = path["daily_return"].copy()
            for index, row in path.iterrows():
                key = (fund, row["date"])
                if key in turnover_lookup.index:
                    turnover = float(turnover_lookup.loc[key, "one_way_turnover"])
                    net.loc[index] = (1.0 - turnover * rate) * (1.0 + row["daily_return"]) - 1.0
            net_metrics = performance_metrics(
                net.reset_index(drop=True),
                periods_per_year=int(metric_row["annualisation_days_per_year"]),
            )
            cost_rows.append(
                {
                    "fund_name": fund,
                    "asset_family": metric_row["asset_family"],
                    "portfolio_method": metric_row["portfolio_method"],
                    "cost_rate_bps_one_way": cost_bps,
                    "gross_annualised_return": metric_row["annualised_return"],
                    "net_annualised_return": net_metrics["annualised_return"],
                    "gross_sharpe_ratio": metric_row["sharpe_ratio"],
                    "net_sharpe_ratio": net_metrics["sharpe_ratio"],
                    "gross_final_growth_of_1": metric_row["final_growth_of_1"],
                    "net_final_growth_of_1": net_metrics["final_growth_of_1"],
                    "total_one_way_turnover": metric_row["total_one_way_turnover"],
                }
            )
    cost_check = pd.DataFrame(cost_rows, columns=FUND_COST_CHECK_COLUMNS)
    return R2Artifacts(fund_returns, fund_weights, performance, fund_turnover, cost_check)


def write_r2_artifacts(
    artifacts: R2Artifacts,
    repository_root: str | Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Write required fund files plus turnover and cost evidence."""
    root = Path(repository_root)
    returns_path = root / "results" / "data" / "fund_returns.csv"
    weights_path = root / "results" / "data" / "fund_weights.csv"
    metrics_path = root / "results" / "tables" / "performance_metrics.csv"
    turnover_path = root / "results" / "tables" / "fund_turnover.csv"
    cost_path = root / "results" / "tables" / "fund_transaction_cost_check.csv"
    for path in (returns_path, weights_path, metrics_path, turnover_path, cost_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.fund_returns.to_csv(returns_path, index=False, date_format="%Y-%m-%d")
    artifacts.fund_weights.to_csv(weights_path, index=False, date_format="%Y-%m-%d")
    artifacts.performance_metrics.to_csv(metrics_path, index=False, date_format="%Y-%m-%d")
    artifacts.fund_turnover.to_csv(turnover_path, index=False, date_format="%Y-%m-%d")
    artifacts.fund_cost_check.to_csv(cost_path, index=False, date_format="%Y-%m-%d")
    return returns_path, weights_path, metrics_path, turnover_path, cost_path
