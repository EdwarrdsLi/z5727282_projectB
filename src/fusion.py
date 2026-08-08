"""Look-ahead-safe fusion of sector sentiment with an equity strategy.

R4 applies a bounded sector-sentiment tilt to the existing Equal Weight method
on the equity universe only.  At each first-trading-day monthly rebalance, the
fusion target uses the sector index from exactly the preceding observed equity
trading day.  Same-day and future sentiment cannot enter the decision.  A
missing prior sector value receives a neutral *multiplier* of one while its
missingness remains explicit; the saved R3 index is never imputed or changed.

The base and fusion variants are evaluated on identical out-of-sample equity
dates with a 252-day annualisation convention.  Gross results, realised
rebalance turnover, an illustrative transaction-cost schedule, and a small
pre-specified multiplier sensitivity are kept separate so implementation
assumptions remain transparent.  All results are historical and descriptive,
not forecasts or trading advice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import features, portfolios, sentiment


SAMPLE_END = pd.Timestamp("2023-12-31")
DEFAULT_ESTIMATION_WINDOW = portfolios.DEFAULT_ESTIMATION_WINDOW
DEFAULT_TILT_STRENGTH = 0.5
DEFAULT_SENSITIVITY_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_TRANSACTION_COST_BPS = (0.0, 5.0, 10.0, 25.0)
BASE_VARIANT = "Base"
FUSION_VARIANT = "Sentiment fusion"
VARIANTS = (BASE_VARIANT, FUSION_VARIANT)
TURNOVER_DEFINITION = (
    "one-way turnover = 0.5 * sum(abs(target weight - pre-trade drifted weight)); "
    "initial portfolio establishment excluded"
)
TRANSACTION_COST_FORMULA = (
    "net rebalance-day return = (1 - one-way turnover * cost rate) * "
    "(1 + gross return) - 1"
)

FUSION_RETURN_COLUMNS = [
    "date",
    "strategy_name",
    "variant",
    "portfolio_method",
    "asset_family",
    "uses_sentiment",
    "daily_return",
    "growth_of_1",
    "drawdown",
    "active_target_rebalance_date",
]

FUSION_WEIGHT_COLUMNS = [
    "rebalance_date",
    "strategy_name",
    "variant",
    "portfolio_method",
    "asset_family",
    "uses_sentiment",
    "ticker",
    "sector",
    "holding_asset_class",
    "target_weight",
    "base_target_weight",
    "sentiment_observation_date",
    "lagged_sentiment_value",
    "sentiment_available",
    "tilt_multiplier",
    "estimation_start_date",
    "estimation_end_date",
    "estimation_observations",
]

FUSION_COMPARISON_COLUMNS = [
    "strategy_name",
    "variant",
    "portfolio_method",
    "asset_family",
    "uses_sentiment",
    "evaluation_period",
    "evaluation_start_date",
    "evaluation_end_date",
    "observations",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "final_growth_of_1",
    "annualised_return_difference_vs_base",
    "annualised_volatility_difference_vs_base",
    "sharpe_ratio_difference_vs_base",
    "maximum_drawdown_difference_vs_base",
    "final_growth_difference_vs_base",
    "higher_annualised_return_than_base",
    "result_basis",
    "annualisation_days_per_year",
    "risk_free_rate_annual",
    "transaction_cost_rate",
    "turnover_rebalances",
    "total_one_way_turnover",
    "average_one_way_turnover_per_rebalance",
    "annualised_one_way_turnover",
    "turnover_definition",
    "initial_establishment_included_in_turnover",
    "estimation_window_type",
    "estimation_window_observations",
    "rebalance_rule",
    "sentiment_lag_observed_trading_days",
    "sentiment_missing_policy",
    "tilt_rule",
    "constraints",
]

FUSION_TURNOVER_COLUMNS = [
    "rebalance_date",
    "strategy_name",
    "variant",
    "uses_sentiment",
    "tilt_strength",
    "initial_establishment",
    "one_way_turnover",
    "two_way_turnover",
    "turnover_definition",
]

FUSION_COST_CHECK_COLUMNS = [
    "strategy_name",
    "variant",
    "uses_sentiment",
    "tilt_strength",
    "evaluation_period",
    "cost_rate_bps_one_way",
    "cost_rate_decimal_one_way",
    "transaction_cost_formula",
    "turnover_definition",
    "initial_establishment_included",
    "charged_rebalances",
    "total_one_way_turnover",
    "gross_annualised_return",
    "net_annualised_return",
    "annualised_return_cost_impact",
    "gross_sharpe_ratio",
    "net_sharpe_ratio",
    "sharpe_ratio_cost_impact",
    "gross_final_growth_of_1",
    "net_final_growth_of_1",
    "final_growth_cost_impact",
    "net_annualised_return_difference_vs_base",
    "net_sharpe_ratio_difference_vs_base",
]

FUSION_SENSITIVITY_COLUMNS = [
    "strategy_name",
    "tilt_strength",
    "approved_design",
    "evaluation_period",
    "observations",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
    "final_growth_of_1",
    "annualised_return_difference_vs_zero_multiplier",
    "sharpe_ratio_difference_vs_zero_multiplier",
    "final_growth_difference_vs_zero_multiplier",
    "turnover_rebalances",
    "total_one_way_turnover",
    "average_one_way_turnover_per_rebalance",
    "sentiment_missing_policy",
    "tilt_rule",
    "result_basis",
]


@dataclass(frozen=True)
class R4Artifacts:
    """In-memory R4 paths, audit tables, and descriptive robustness checks."""

    fusion_returns: pd.DataFrame
    fusion_weights: pd.DataFrame
    fusion_comparison: pd.DataFrame
    fusion_turnover: pd.DataFrame
    fusion_cost_check: pd.DataFrame
    fusion_sensitivity: pd.DataFrame


def build_equity_return_matrix(
    equity_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct adjusted-close equity returns and a ticker-sector mapping."""
    required = {"date", "ticker", "sector", "adjClose"}
    missing = sorted(required.difference(equity_prices.columns))
    if missing:
        raise ValueError(f"equity prices are missing required columns: {missing}")
    if equity_prices[["ticker", "sector"]].isna().any(axis=None):
        raise ValueError("equity ticker and sector must not be missing")
    crypto_like = equity_prices["ticker"].astype(str).str.endswith("-USD")
    crypto_sector = equity_prices["sector"].astype(str).str.casefold().eq(
        "cryptocurrency"
    )
    if crypto_like.any() or crypto_sector.any():
        raise ValueError("R4 fusion accepts equity prices only, not cryptocurrency")

    returned = features.daily_returns(equity_prices, asset_class="equity")
    universe = returned.loc[:, ["ticker", "sector"]].drop_duplicates()
    sector_counts = universe.groupby("ticker", sort=False)["sector"].nunique()
    if (sector_counts != 1).any():
        raise ValueError("each equity ticker must map to exactly one sector")
    sector_by_ticker = universe.set_index("ticker")["sector"].sort_index()

    matrix = returned.pivot(index="date", columns="ticker", values="simple_return")
    matrix = matrix.reindex(columns=sorted(matrix.columns, key=str)).sort_index()
    matrix.columns.name = None
    if matrix.index.max() > SAMPLE_END:
        raise ValueError("equity returns exceed 2023-12-31")
    return matrix, sector_by_ticker.reindex(matrix.columns)


def _complete_history(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.empty:
        raise ValueError("equity returns require a non-empty DatetimeIndex")
    ordered = returns.sort_index().apply(pd.to_numeric, errors="raise").astype(float)
    if ordered.index.has_duplicates or ordered.columns.has_duplicates:
        raise ValueError("equity return dates and tickers must be unique")
    if ordered.index.max() > SAMPLE_END:
        raise ValueError("equity returns exceed 2023-12-31")
    complete = ~ordered.isna().any(axis=1)
    if not complete.any():
        raise ValueError("equity returns contain no complete observation")
    first = int(np.flatnonzero(complete.to_numpy())[0])
    history = ordered.iloc[first:]
    if history.isna().any(axis=None) or not np.isfinite(history.to_numpy()).all():
        raise ValueError("missing or non-finite equity return after history begins")
    if (history < -1.0).any(axis=None):
        raise ValueError("simple equity returns cannot be below -100%")
    return history


def _monthly_rebalance_dates(
    dates: pd.DatetimeIndex,
    estimation_window: int,
) -> pd.DatetimeIndex:
    month = dates.to_period("M")
    first_in_month = np.r_[True, month[1:] != month[:-1]]
    positions = np.flatnonzero(first_in_month)
    return dates.take(positions[positions >= estimation_window])


def sentiment_tilt_weights(
    base_weights: pd.Series,
    sector_by_ticker: pd.Series,
    lagged_sector_signal: pd.Series,
    *,
    tilt_strength: float = DEFAULT_TILT_STRENGTH,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Apply ``1 + strength * sentiment`` within the equity portfolio.

    Missing sector sentiment receives multiplier one, meaning no active tilt.
    The function returns target weights, per-ticker signals, and multipliers so
    that the missing-data and transformation decisions remain auditable.
    """
    if not 0.0 <= tilt_strength <= 1.0:
        raise ValueError("tilt_strength must lie between zero and one")
    if not base_weights.index.equals(sector_by_ticker.index):
        raise ValueError("base weights and ticker-sector mapping must align")
    values = pd.to_numeric(base_weights, errors="raise").astype(float)
    if values.isna().any() or (values < 0.0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("base equity weights must be complete, long-only, and sum to one")

    required_sectors = set(sector_by_ticker)
    if lagged_sector_signal.index.has_duplicates:
        raise ValueError("lagged sector signal must be unique by sector")
    if set(lagged_sector_signal.index) != required_sectors:
        raise ValueError("lagged sector signal must cover exactly the equity sectors")
    numeric_signal = pd.to_numeric(lagged_sector_signal, errors="raise").astype(float)
    observed = numeric_signal.dropna()
    if not np.isfinite(observed.to_numpy()).all() or not observed.between(-1.0, 1.0).all():
        raise ValueError("observed lagged sentiment must be finite and within [-1, 1]")

    ticker_signal = sector_by_ticker.map(numeric_signal)
    multiplier = 1.0 + tilt_strength * ticker_signal.fillna(0.0)
    if (multiplier <= 0.0).any():
        raise RuntimeError("sentiment tilt multiplier must remain positive")
    tilted = values * multiplier
    target = tilted / tilted.sum()
    if not np.isfinite(target.to_numpy()).all() or (target < 0.0).any():
        raise RuntimeError("sentiment fusion produced invalid equity weights")
    if not np.isclose(target.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError("sentiment fusion weights must sum to one")
    return target, ticker_signal, multiplier


def _lagged_signal_by_decision(
    sector_index: pd.DataFrame,
    equity_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    lagged = sentiment.lag_sector_sentiment_one_trading_day(sector_index)
    decision_dates = pd.DatetimeIndex(lagged["decision_date"].unique()).sort_values()
    if not decision_dates.equals(equity_dates):
        raise ValueError("sector sentiment grid must match the equity trading calendar")
    if lagged.duplicated(["decision_date", "sector"]).any():
        raise ValueError("lagged sentiment must be unique by decision date and sector")
    comparable = lagged["sentiment_observation_date"].notna()
    if not (
        lagged.loc[comparable, "sentiment_observation_date"]
        < lagged.loc[comparable, "decision_date"]
    ).all():
        raise AssertionError("every usable sentiment observation must pre-date its decision")
    return lagged


def _variant_metadata(variant: str) -> tuple[str, str, bool]:
    uses_sentiment = variant == FUSION_VARIANT
    return (
        (
            "Equity Equal Weight + Sector Sentiment"
            if uses_sentiment
            else "Equity Equal Weight"
        ),
        (
            "Equal Weight + Sector Sentiment"
            if uses_sentiment
            else portfolios.EQUAL_WEIGHT
        ),
        uses_sentiment,
    )


def _validated_strengths(
    strengths: tuple[float, ...],
    approved_strength: float,
) -> tuple[float, ...]:
    numeric = tuple(float(value) for value in strengths)
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in numeric):
        raise ValueError("sensitivity strengths must be finite and lie in [0, 1]")
    return tuple(sorted(set((*numeric, 0.0, float(approved_strength)))))


def _validated_cost_rates(cost_rates_bps: tuple[float, ...]) -> tuple[float, ...]:
    numeric = tuple(float(value) for value in cost_rates_bps)
    if any(not np.isfinite(value) or value < 0.0 for value in numeric):
        raise ValueError("transaction-cost rates must be finite and non-negative")
    return tuple(sorted(set(numeric)))


def _turnover_summary(
    turnover: pd.DataFrame,
    variant: str,
    observations: int,
) -> dict[str, float | int]:
    realised = turnover.loc[
        turnover["variant"].eq(variant) & ~turnover["initial_establishment"]
    ]
    values = realised["one_way_turnover"]
    total = float(values.sum())
    count = int(values.count())
    return {
        "turnover_rebalances": count,
        "total_one_way_turnover": total,
        "average_one_way_turnover_per_rebalance": (
            float(values.mean()) if count else 0.0
        ),
        "annualised_one_way_turnover": (
            total * 252.0 / observations if observations else np.nan
        ),
    }


def _build_transaction_cost_check(
    fusion_returns: pd.DataFrame,
    fusion_turnover: pd.DataFrame,
    comparison: pd.DataFrame,
    cost_rates_bps: tuple[float, ...],
    tilt_strength: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cost_bps in cost_rates_bps:
        cost_rate = cost_bps / 10_000.0
        for variant in VARIANTS:
            path = fusion_returns.loc[
                fusion_returns["variant"].eq(variant)
            ].sort_values("date")
            gross_returns = path.set_index("date")["daily_return"].copy()
            net_returns = gross_returns.copy()
            realised_turnover = fusion_turnover.loc[
                fusion_turnover["variant"].eq(variant)
                & ~fusion_turnover["initial_establishment"]
            ].set_index("rebalance_date")["one_way_turnover"]
            for rebalance_date, one_way_turnover in realised_turnover.items():
                gross = float(gross_returns.loc[rebalance_date])
                trading_cost = float(one_way_turnover) * cost_rate
                if not 0.0 <= trading_cost < 1.0:
                    raise ValueError("transaction-cost assumption exhausts portfolio wealth")
                net_returns.loc[rebalance_date] = (
                    (1.0 - trading_cost) * (1.0 + gross) - 1.0
                )
            gross_metrics = portfolios.performance_metrics(gross_returns)
            net_metrics = portfolios.performance_metrics(net_returns)
            strategy_name, _, uses_sentiment = _variant_metadata(variant)
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "variant": variant,
                    "uses_sentiment": uses_sentiment,
                    "tilt_strength": tilt_strength if uses_sentiment else 0.0,
                    "evaluation_period": comparison.loc[
                        comparison["variant"].eq(variant), "evaluation_period"
                    ].iloc[0],
                    "cost_rate_bps_one_way": cost_bps,
                    "cost_rate_decimal_one_way": cost_rate,
                    "transaction_cost_formula": TRANSACTION_COST_FORMULA,
                    "turnover_definition": TURNOVER_DEFINITION,
                    "initial_establishment_included": False,
                    "charged_rebalances": int(realised_turnover.count()),
                    "total_one_way_turnover": float(realised_turnover.sum()),
                    "gross_annualised_return": gross_metrics["annualised_return"],
                    "net_annualised_return": net_metrics["annualised_return"],
                    "annualised_return_cost_impact": (
                        net_metrics["annualised_return"]
                        - gross_metrics["annualised_return"]
                    ),
                    "gross_sharpe_ratio": gross_metrics["sharpe_ratio"],
                    "net_sharpe_ratio": net_metrics["sharpe_ratio"],
                    "sharpe_ratio_cost_impact": (
                        net_metrics["sharpe_ratio"] - gross_metrics["sharpe_ratio"]
                    ),
                    "gross_final_growth_of_1": gross_metrics["final_growth_of_1"],
                    "net_final_growth_of_1": net_metrics["final_growth_of_1"],
                    "final_growth_cost_impact": (
                        net_metrics["final_growth_of_1"]
                        - gross_metrics["final_growth_of_1"]
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=FUSION_COST_CHECK_COLUMNS)
    bases = result.loc[result["variant"].eq(BASE_VARIANT)].set_index(
        "cost_rate_bps_one_way"
    )
    result["net_annualised_return_difference_vs_base"] = result.apply(
        lambda row: row["net_annualised_return"]
        - bases.loc[row["cost_rate_bps_one_way"], "net_annualised_return"],
        axis=1,
    )
    result["net_sharpe_ratio_difference_vs_base"] = result.apply(
        lambda row: row["net_sharpe_ratio"]
        - bases.loc[row["cost_rate_bps_one_way"], "net_sharpe_ratio"],
        axis=1,
    )
    return result.loc[:, FUSION_COST_CHECK_COLUMNS]


def build_r4_artifacts(
    equity_prices: pd.DataFrame,
    sector_index: pd.DataFrame,
    *,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    tilt_strength: float = DEFAULT_TILT_STRENGTH,
    sensitivity_strengths: tuple[float, ...] = DEFAULT_SENSITIVITY_STRENGTHS,
    transaction_cost_bps: tuple[float, ...] = DEFAULT_TRANSACTION_COST_BPS,
    _include_sensitivity: bool = True,
) -> R4Artifacts:
    """Build separate equity base and lagged-sentiment fusion evidence."""
    if not isinstance(estimation_window, int) or estimation_window < 2:
        raise ValueError("estimation_window must be an integer of at least two")
    if not 0.0 <= tilt_strength <= 1.0:
        raise ValueError("tilt_strength must lie between zero and one")
    strengths = _validated_strengths(sensitivity_strengths, tilt_strength)
    cost_rates = _validated_cost_rates(transaction_cost_bps)
    return_matrix, sector_by_ticker = build_equity_return_matrix(equity_prices)
    history = _complete_history(return_matrix)
    if len(history) <= estimation_window:
        raise ValueError("equity return history is too short for an OOS period")
    rebalance_dates = _monthly_rebalance_dates(history.index, estimation_window)
    if rebalance_dates.empty:
        raise ValueError("no eligible monthly rebalance exists")

    lagged = _lagged_signal_by_decision(sector_index, return_matrix.index)
    base_target = pd.Series(
        1.0 / history.shape[1], index=history.columns, dtype=float
    )
    positions = pd.Series(np.arange(len(history)), index=history.index)
    rebalance_set = set(rebalance_dates)
    current: dict[str, pd.Series | None] = {variant: None for variant in VARIANTS}
    active_rebalance: pd.Timestamp | None = None
    return_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    turnover_rows: list[dict[str, object]] = []

    for date in history.loc[rebalance_dates[0] :].index:
        if date in rebalance_set:
            pre_trade = {
                variant: (
                    current[variant].copy() if current[variant] is not None else None
                )
                for variant in VARIANTS
            }
            position = int(positions.loc[date])
            window = history.iloc[position - estimation_window : position]
            if len(window) != estimation_window or not (window.index.max() < date):
                raise AssertionError("estimation data must be complete and pre-rebalance")
            decision_signal = lagged.loc[lagged["decision_date"].eq(date)].set_index(
                "sector"
            )
            if decision_signal.empty:
                raise ValueError(f"missing lagged sector signal for {date.date()}")
            observation_dates = decision_signal["sentiment_observation_date"].dropna()
            if observation_dates.empty or observation_dates.nunique() != 1:
                raise ValueError("all sectors must share one preceding observation date")
            observation_date = pd.Timestamp(observation_dates.iloc[0])
            prior_position = return_matrix.index.get_loc(date) - 1
            if prior_position < 0 or observation_date != return_matrix.index[prior_position]:
                raise AssertionError("fusion must use exactly one observed trading-day lag")

            fusion_target, ticker_signal, multiplier = sentiment_tilt_weights(
                base_target,
                sector_by_ticker,
                decision_signal["lagged_sentiment_value"],
                tilt_strength=tilt_strength,
            )
            current[BASE_VARIANT] = base_target.copy()
            current[FUSION_VARIANT] = fusion_target
            active_rebalance = date

            for variant in VARIANTS:
                strategy_name, portfolio_method, uses_sentiment = _variant_metadata(
                    variant
                )
                target = current[variant]
                if target is None:
                    raise AssertionError("rebalance target was not constructed")
                before = pre_trade[variant]
                one_way_turnover = (
                    np.nan
                    if before is None
                    else 0.5 * float((target - before).abs().sum())
                )
                turnover_rows.append(
                    {
                        "rebalance_date": date,
                        "strategy_name": strategy_name,
                        "variant": variant,
                        "uses_sentiment": uses_sentiment,
                        "tilt_strength": tilt_strength if uses_sentiment else 0.0,
                        "initial_establishment": before is None,
                        "one_way_turnover": one_way_turnover,
                        "two_way_turnover": (
                            np.nan
                            if before is None
                            else 2.0 * one_way_turnover
                        ),
                        "turnover_definition": TURNOVER_DEFINITION,
                    }
                )
                for ticker, target_weight in target.items():
                    weight_rows.append(
                        {
                            "rebalance_date": date,
                            "strategy_name": strategy_name,
                            "variant": variant,
                            "portfolio_method": portfolio_method,
                            "asset_family": "Equity",
                            "uses_sentiment": uses_sentiment,
                            "ticker": ticker,
                            "sector": sector_by_ticker.loc[ticker],
                            "holding_asset_class": "equity",
                            "target_weight": float(target_weight),
                            "base_target_weight": float(base_target.loc[ticker]),
                            "sentiment_observation_date": (
                                observation_date if uses_sentiment else pd.NaT
                            ),
                            "lagged_sentiment_value": (
                                float(ticker_signal.loc[ticker])
                                if uses_sentiment and pd.notna(ticker_signal.loc[ticker])
                                else np.nan
                            ),
                            "sentiment_available": bool(
                                uses_sentiment and pd.notna(ticker_signal.loc[ticker])
                            ),
                            "tilt_multiplier": (
                                float(multiplier.loc[ticker]) if uses_sentiment else 1.0
                            ),
                            "estimation_start_date": window.index.min(),
                            "estimation_end_date": window.index.max(),
                            "estimation_observations": len(window),
                        }
                    )

        if active_rebalance is None:
            raise AssertionError("live returns cannot precede a target rebalance")
        asset_return = history.loc[date]
        for variant in VARIANTS:
            weights = current[variant]
            if weights is None:
                raise AssertionError("live variant is missing weights")
            fund_return = float(weights @ asset_return)
            if fund_return <= -1.0:
                raise RuntimeError("fusion portfolio wealth became non-positive")
            strategy_name, portfolio_method, uses_sentiment = _variant_metadata(
                variant
            )
            return_rows.append(
                {
                    "date": date,
                    "strategy_name": strategy_name,
                    "variant": variant,
                    "portfolio_method": portfolio_method,
                    "asset_family": "Equity",
                    "uses_sentiment": uses_sentiment,
                    "daily_return": fund_return,
                    "active_target_rebalance_date": active_rebalance,
                }
            )
            drifted = weights * (1.0 + asset_return) / (1.0 + fund_return)
            current[variant] = drifted / drifted.sum()

    fusion_returns = pd.DataFrame(return_rows)
    enriched: list[pd.DataFrame] = []
    for _, group in fusion_returns.groupby("variant", sort=False):
        frame = group.copy()
        growth = (1.0 + frame["daily_return"]).cumprod()
        frame["growth_of_1"] = growth
        frame["drawdown"] = growth / growth.cummax().clip(lower=1.0) - 1.0
        enriched.append(frame)
    fusion_returns = pd.concat(enriched, ignore_index=True).sort_values(
        ["date", "variant"], kind="stable"
    )
    fusion_returns = fusion_returns.loc[:, FUSION_RETURN_COLUMNS].reset_index(drop=True)
    fusion_weights = pd.DataFrame(weight_rows, columns=FUSION_WEIGHT_COLUMNS)
    fusion_turnover = pd.DataFrame(turnover_rows, columns=FUSION_TURNOVER_COLUMNS)

    comparison_rows: list[dict[str, object]] = []
    metric_by_variant: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        path = fusion_returns.loc[fusion_returns["variant"].eq(variant)]
        metric_by_variant[variant] = portfolios.performance_metrics(path["daily_return"])
    base_metrics = metric_by_variant[BASE_VARIANT]
    start = pd.Timestamp(fusion_returns["date"].min())
    end = pd.Timestamp(fusion_returns["date"].max())
    for variant in VARIANTS:
        strategy_name, portfolio_method, uses_sentiment = _variant_metadata(variant)
        metrics = metric_by_variant[variant]
        observations = int(
            fusion_returns.loc[fusion_returns["variant"].eq(variant)].shape[0]
        )
        turnover_summary = _turnover_summary(
            fusion_turnover, variant, observations
        )
        comparison_rows.append(
            {
                "strategy_name": strategy_name,
                "variant": variant,
                "portfolio_method": portfolio_method,
                "asset_family": "Equity",
                "uses_sentiment": uses_sentiment,
                "evaluation_period": f"{start.date()} to {end.date()}",
                "evaluation_start_date": start,
                "evaluation_end_date": end,
                "observations": observations,
                **metrics,
                "annualised_return_difference_vs_base": (
                    metrics["annualised_return"] - base_metrics["annualised_return"]
                ),
                "annualised_volatility_difference_vs_base": (
                    metrics["annualised_volatility"]
                    - base_metrics["annualised_volatility"]
                ),
                "sharpe_ratio_difference_vs_base": (
                    metrics["sharpe_ratio"] - base_metrics["sharpe_ratio"]
                ),
                "maximum_drawdown_difference_vs_base": (
                    metrics["maximum_drawdown"] - base_metrics["maximum_drawdown"]
                ),
                "final_growth_difference_vs_base": (
                    metrics["final_growth_of_1"] - base_metrics["final_growth_of_1"]
                ),
                "higher_annualised_return_than_base": bool(
                    uses_sentiment
                    and metrics["annualised_return"] > base_metrics["annualised_return"]
                ),
                "result_basis": "Gross of transaction costs",
                "annualisation_days_per_year": 252,
                "risk_free_rate_annual": 0.0,
                "transaction_cost_rate": 0.0,
                **turnover_summary,
                "turnover_definition": TURNOVER_DEFINITION,
                "initial_establishment_included_in_turnover": False,
                "estimation_window_type": "Rolling complete equity-calendar observations",
                "estimation_window_observations": estimation_window,
                "rebalance_rule": "First observed equity trading day of each calendar month",
                "sentiment_lag_observed_trading_days": 1 if uses_sentiment else 0,
                "sentiment_missing_policy": (
                    "Missing prior sector score receives multiplier 1; no index imputation"
                    if uses_sentiment
                    else "Not applicable"
                ),
                "tilt_rule": (
                    f"target proportional to equal weight * (1 + {tilt_strength:g} * lagged sector sentiment)"
                    if uses_sentiment
                    else "Equal weight"
                ),
                "constraints": "Equity only; long-only; fully invested; no leverage",
            }
        )
    comparison = pd.DataFrame(comparison_rows, columns=FUSION_COMPARISON_COLUMNS)

    cost_check = _build_transaction_cost_check(
        fusion_returns,
        fusion_turnover,
        comparison,
        cost_rates,
        tilt_strength,
    )

    sensitivity_rows: list[dict[str, object]] = []
    if _include_sensitivity:
        zero_row = comparison.loc[comparison["variant"].eq(BASE_VARIANT)].iloc[0]
        for strength in strengths:
            if np.isclose(strength, 0.0):
                source = zero_row
            elif np.isclose(strength, tilt_strength):
                source = comparison.loc[
                    comparison["variant"].eq(FUSION_VARIANT)
                ].iloc[0]
            else:
                alternative = build_r4_artifacts(
                    equity_prices,
                    sector_index,
                    estimation_window=estimation_window,
                    tilt_strength=strength,
                    sensitivity_strengths=(),
                    transaction_cost_bps=(),
                    _include_sensitivity=False,
                )
                source = alternative.fusion_comparison.loc[
                    alternative.fusion_comparison["variant"].eq(FUSION_VARIANT)
                ].iloc[0]
            sensitivity_rows.append(
                {
                    "strategy_name": (
                        "Equity Equal Weight"
                        if np.isclose(strength, 0.0)
                        else "Equity Equal Weight + Sector Sentiment"
                    ),
                    "tilt_strength": strength,
                    "approved_design": bool(np.isclose(strength, tilt_strength)),
                    "evaluation_period": source["evaluation_period"],
                    "observations": int(source["observations"]),
                    "annualised_return": source["annualised_return"],
                    "annualised_volatility": source["annualised_volatility"],
                    "sharpe_ratio": source["sharpe_ratio"],
                    "maximum_drawdown": source["maximum_drawdown"],
                    "final_growth_of_1": source["final_growth_of_1"],
                    "annualised_return_difference_vs_zero_multiplier": (
                        source["annualised_return"] - zero_row["annualised_return"]
                    ),
                    "sharpe_ratio_difference_vs_zero_multiplier": (
                        source["sharpe_ratio"] - zero_row["sharpe_ratio"]
                    ),
                    "final_growth_difference_vs_zero_multiplier": (
                        source["final_growth_of_1"] - zero_row["final_growth_of_1"]
                    ),
                    "turnover_rebalances": int(source["turnover_rebalances"]),
                    "total_one_way_turnover": source["total_one_way_turnover"],
                    "average_one_way_turnover_per_rebalance": source[
                        "average_one_way_turnover_per_rebalance"
                    ],
                    "sentiment_missing_policy": (
                        "Missing prior sector score receives multiplier 1; "
                        "no index imputation"
                    ),
                    "tilt_rule": (
                        f"target proportional to equal weight * "
                        f"(1 + {strength:g} * lagged sector sentiment)"
                    ),
                    "result_basis": "Gross of transaction costs",
                }
            )
    sensitivity = pd.DataFrame(
        sensitivity_rows, columns=FUSION_SENSITIVITY_COLUMNS
    )

    target_sums = fusion_weights.groupby(
        ["variant", "rebalance_date"], sort=False
    )["target_weight"].sum()
    if not np.allclose(target_sums.to_numpy(), 1.0, atol=1e-10, rtol=0.0):
        raise AssertionError("at least one R4 target does not sum to one")
    if set(fusion_weights["holding_asset_class"]) != {"equity"}:
        raise AssertionError("R4 fusion must contain equity holdings only")
    counts = fusion_returns.groupby("variant")["date"].agg(["min", "max", "size"])
    if counts.nunique().max() != 1:
        raise AssertionError("base and fusion evaluation samples must match")
    initial_counts = fusion_turnover.groupby("variant")[
        "initial_establishment"
    ].sum()
    if not (initial_counts == 1).all():
        raise AssertionError("each R4 variant must have one excluded initial trade")
    realised_turnover = fusion_turnover.loc[
        ~fusion_turnover["initial_establishment"], "one_way_turnover"
    ]
    if (
        realised_turnover.isna().any()
        or (realised_turnover < 0.0).any()
        or not np.isfinite(realised_turnover.to_numpy()).all()
    ):
        raise AssertionError("realised R4 turnover must be finite and non-negative")
    return R4Artifacts(
        fusion_returns,
        fusion_weights,
        comparison,
        fusion_turnover,
        cost_check,
        sensitivity,
    )


def write_r4_artifacts(
    artifacts: R4Artifacts,
    repository_root: str | Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """Write the separate R4 data, descriptive checks, and comparison figure."""
    if list(artifacts.fusion_returns.columns) != FUSION_RETURN_COLUMNS:
        raise ValueError("fusion returns do not have the required schema")
    if list(artifacts.fusion_weights.columns) != FUSION_WEIGHT_COLUMNS:
        raise ValueError("fusion weights do not have the required schema")
    if list(artifacts.fusion_comparison.columns) != FUSION_COMPARISON_COLUMNS:
        raise ValueError("fusion comparison does not have the required schema")
    if list(artifacts.fusion_turnover.columns) != FUSION_TURNOVER_COLUMNS:
        raise ValueError("fusion turnover does not have the required schema")
    if list(artifacts.fusion_cost_check.columns) != FUSION_COST_CHECK_COLUMNS:
        raise ValueError("fusion cost check does not have the required schema")
    if list(artifacts.fusion_sensitivity.columns) != FUSION_SENSITIVITY_COLUMNS:
        raise ValueError("fusion sensitivity does not have the required schema")

    root = Path(repository_root)
    returns_path = root / "results" / "data" / "sentiment_fusion_returns.csv"
    weights_path = root / "results" / "data" / "sentiment_fusion_weights.csv"
    comparison_path = root / "results" / "tables" / "sentiment_fusion_comparison.csv"
    turnover_path = root / "results" / "tables" / "sentiment_fusion_turnover.csv"
    cost_path = (
        root / "results" / "tables" / "sentiment_fusion_transaction_cost_check.csv"
    )
    sensitivity_path = (
        root / "results" / "tables" / "sentiment_multiplier_sensitivity.csv"
    )
    figure_path = root / "results" / "figures" / "sentiment_fusion_comparison.png"
    for path in (
        returns_path,
        weights_path,
        comparison_path,
        turnover_path,
        cost_path,
        sensitivity_path,
        figure_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.fusion_returns.to_csv(returns_path, index=False, date_format="%Y-%m-%d")
    artifacts.fusion_weights.to_csv(weights_path, index=False, date_format="%Y-%m-%d")
    artifacts.fusion_comparison.to_csv(
        comparison_path, index=False, date_format="%Y-%m-%d"
    )
    artifacts.fusion_turnover.to_csv(
        turnover_path, index=False, date_format="%Y-%m-%d"
    )
    artifacts.fusion_cost_check.to_csv(cost_path, index=False)
    artifacts.fusion_sensitivity.to_csv(sensitivity_path, index=False)
    _write_comparison_figure(artifacts.fusion_returns, figure_path)
    return (
        returns_path,
        weights_path,
        comparison_path,
        turnover_path,
        cost_path,
        sensitivity_path,
        figure_path,
    )


def _write_comparison_figure(paths: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    start = pd.Timestamp(paths["date"].min()).date()
    end = pd.Timestamp(paths["date"].max()).date()
    colours = {BASE_VARIANT: "#334E68", FUSION_VARIANT: "#D97706"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    for variant in VARIANTS:
        frame = paths.loc[paths["variant"].eq(variant)].sort_values("date")
        axes[0].plot(
            frame["date"], frame["growth_of_1"], label=variant,
            color=colours[variant], linewidth=1.8,
        )
        axes[1].plot(
            frame["date"], frame["drawdown"] * 100.0, label=variant,
            color=colours[variant], linewidth=1.4,
        )
    axes[0].set_ylabel("Growth of $1 ($)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Observed equity trading date")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        "Equity Equal Weight: Before and After Lagged Sector-Sentiment Fusion\n"
        f"Historical out-of-sample sample: {start} to {end}"
    )
    fig.text(
        0.5,
        0.01,
        "Fusion uses the preceding observed equity day's sector index; historical and descriptive only, not trading advice.",
        ha="center",
        fontsize=8,
        color="#52606D",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(fig)
