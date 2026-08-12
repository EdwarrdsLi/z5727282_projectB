"""Focused deterministic tests for R2 portfolios and OOS backtesting.

Run from the project root with::

    python tests/test_portfolios.py
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import portfolios  # noqa: E402


def _return_fixture() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=95)
    step = np.arange(len(dates), dtype=float)
    returns = pd.DataFrame(
        {
            "AAA": 0.0005 + 0.008 * np.sin(step / 3.0),
            "BBB": 0.0002 + 0.016 * np.cos(step / 5.0),
            "BTC-USD": 0.0008 + 0.030 * np.sin(step / 7.0 + 0.4),
        },
        index=dates,
    )
    returns.iloc[0] = np.nan
    return returns


def _price_frame(
    ticker: str,
    dates: pd.DatetimeIndex,
    daily_return: np.ndarray,
    *,
    sector: str | None = None,
) -> pd.DataFrame:
    price = 100.0 * np.cumprod(1.0 + daily_return)
    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "adjClose": price,
            "volume": 1_000,
        }
    )
    if sector is not None:
        frame["sector"] = sector
    return frame


class TestWalkForwardBacktest(unittest.TestCase):
    def setUp(self) -> None:
        self.returns = _return_fixture()
        self.asset_classes = {
            "AAA": "equity",
            "BBB": "equity",
            "BTC-USD": "cryptocurrency",
        }

    def test_first_oos_date_follows_full_prior_window(self) -> None:
        result = portfolios.oos_backtest(
            self.returns,
            estimation_window=10,
            asset_classes=self.asset_classes,
        )
        complete = self.returns.iloc[1:]
        monthly_firsts = complete.groupby(complete.index.to_period("M")).head(1).index
        expected = next(date for date in monthly_firsts if complete.index.get_loc(date) >= 10)

        first_live = result.fund_returns["date"].min()
        first_weights = result.fund_weights.loc[
            result.fund_weights["rebalance_date"] == first_live
        ]
        self.assertEqual(first_live, expected)
        self.assertTrue((first_weights["estimation_end_date"] < first_live).all())
        self.assertTrue((first_weights["estimation_observations"] == 10).all())

    def test_rebalances_only_on_first_observed_date_of_each_month(self) -> None:
        result = portfolios.oos_backtest(
            self.returns,
            estimation_window=10,
            asset_classes=self.asset_classes,
        )
        actual = pd.DatetimeIndex(result.fund_weights["rebalance_date"].unique())
        live = self.returns.iloc[1:].loc[actual.min() :]
        expected = live.groupby(live.index.to_period("M")).head(1).index
        self.assertTrue(actual.equals(expected))
        self.assertTrue(result.fund_returns["date"].is_unique)

    def test_each_target_is_long_only_and_sums_to_one(self) -> None:
        for method in portfolios.SUPPORTED_METHODS:
            result = portfolios.oos_backtest(
                self.returns,
                method=method,
                estimation_window=10,
                asset_classes=self.asset_classes,
            )
            sums = result.fund_weights.groupby("rebalance_date")["target_weight"].sum()
            self.assertTrue(np.allclose(sums.to_numpy(), 1.0, atol=1e-10))
            self.assertTrue((result.fund_weights["target_weight"] >= 0.0).all())

    def test_future_returns_cannot_change_first_optimised_target(self) -> None:
        for method in (portfolios.RISK_PARITY, portfolios.MINIMUM_VARIANCE):
            baseline = portfolios.oos_backtest(
                self.returns, method=method, estimation_window=10,
                asset_classes=self.asset_classes,
            )
            first_live = baseline.fund_returns["date"].min()
            changed = self.returns.copy()
            changed.loc[first_live:, "BTC-USD"] = -4.0 * changed.loc[first_live:, "BTC-USD"]
            alternative = portfolios.oos_backtest(
                changed, method=method, estimation_window=10,
                asset_classes=self.asset_classes,
            )
            baseline_first = baseline.fund_weights.loc[
                baseline.fund_weights["rebalance_date"] == first_live,
                ["ticker", "target_weight"],
            ].set_index("ticker")
            alternative_first = alternative.fund_weights.loc[
                alternative.fund_weights["rebalance_date"] == first_live,
                ["ticker", "target_weight"],
            ].set_index("ticker")
            pd.testing.assert_frame_equal(baseline_first, alternative_first)


class TestMetricsAndSchemas(unittest.TestCase):
    def test_performance_calculations_match_hand_computation(self) -> None:
        returns = pd.Series([-0.10, 0.20])
        metrics = portfolios.performance_metrics(returns, periods_per_year=2)

        self.assertAlmostEqual(metrics["final_growth_of_1"], 1.08)
        self.assertAlmostEqual(metrics["annualised_return"], 0.08)
        self.assertAlmostEqual(metrics["annualised_volatility"], 0.30)
        self.assertAlmostEqual(metrics["sharpe_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["maximum_drawdown"], -0.10)

    def test_required_output_schemas_and_nine_fund_shelf(self) -> None:
        equity_dates = pd.bdate_range("2020-01-02", periods=75)
        crypto_dates = pd.date_range(equity_dates.min(), equity_dates.max(), freq="D")
        equity_step = np.arange(len(equity_dates), dtype=float)
        crypto_step = np.arange(len(crypto_dates), dtype=float)
        equities = pd.concat(
            [
                _price_frame(
                    "AAA",
                    equity_dates,
                    0.001 + 0.008 * np.sin(equity_step / 4.0),
                    sector="Tech",
                ),
                _price_frame(
                    "BBB",
                    equity_dates,
                    0.0005 + 0.015 * np.cos(equity_step / 6.0),
                    sector="Financials",
                ),
            ],
            ignore_index=True,
        )
        crypto = _price_frame(
            "BTC-USD",
            crypto_dates,
            0.0015 + 0.03 * np.sin(crypto_step / 8.0 + 0.2),
        )
        artifacts = portfolios.build_r2_artifacts(
            equities,
            crypto,
            estimation_window=10,
            crypto_estimation_window=10,
        )

        self.assertEqual(list(artifacts.fund_returns), portfolios.FUND_RETURN_COLUMNS)
        self.assertEqual(list(artifacts.fund_weights), portfolios.FUND_WEIGHT_COLUMNS)
        self.assertEqual(
            list(artifacts.performance_metrics),
            portfolios.PERFORMANCE_METRIC_COLUMNS,
        )
        self.assertEqual(
            set(artifacts.performance_metrics["portfolio_method"]),
            set(portfolios.SUPPORTED_METHODS),
        )
        self.assertEqual(
            set(artifacts.performance_metrics["asset_family"]),
            {"Equity only", "Cryptocurrency only", "Combined equity + cryptocurrency"},
        )
        self.assertEqual(artifacts.performance_metrics["fund_name"].nunique(), 9)
        self.assertEqual(list(artifacts.fund_turnover), portfolios.FUND_TURNOVER_COLUMNS)
        self.assertEqual(list(artifacts.fund_cost_check), portfolios.FUND_COST_CHECK_COLUMNS)
        self.assertEqual(set(artifacts.fund_cost_check["cost_rate_bps_one_way"]), {0.0, 10.0})
        self.assertTrue(
            artifacts.fund_cost_check.loc[
                artifacts.fund_cost_check["cost_rate_bps_one_way"].eq(10.0),
                "net_final_growth_of_1",
            ].le(
                artifacts.fund_cost_check.loc[
                    artifacts.fund_cost_check["cost_rate_bps_one_way"].eq(10.0),
                    "gross_final_growth_of_1",
                ]
            ).all()
        )
        self.assertTrue(
            artifacts.performance_metrics["current_target_holdings_source"]
            .str.contains("fund_weights.csv", regex=False)
            .all()
        )
        self.assertLessEqual(artifacts.fund_returns["date"].max(), portfolios.SAMPLE_END)

    def test_missing_return_after_history_starts_is_rejected_not_filled(self) -> None:
        returns = _return_fixture()
        returns.iloc[20, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "missing returns"):
            portfolios.oos_backtest(returns, estimation_window=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
