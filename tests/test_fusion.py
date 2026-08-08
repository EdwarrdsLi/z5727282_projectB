"""Focused deterministic tests for R4 sentiment fusion.

Run from the project root with::

    python tests/test_fusion.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import fusion  # noqa: E402


def _prices(*, include_crypto: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=85)
    step = np.arange(len(dates), dtype=float)
    specifications = [
        ("AAA", "Tech", 0.001 + 0.010 * np.sin(step / 4.0)),
        ("BBB", "Tech", 0.0005 + 0.012 * np.cos(step / 7.0)),
        ("CCC", "Energy", 0.0008 + 0.009 * np.sin(step / 6.0 + 0.4)),
        ("DDD", "Energy", 0.0003 + 0.011 * np.cos(step / 5.0 + 0.2)),
    ]
    if include_crypto:
        specifications.append(
            ("BTC-USD", "Cryptocurrency", 0.001 + 0.02 * np.sin(step / 3.0))
        )
    frames: list[pd.DataFrame] = []
    for ticker, sector, returns in specifications:
        adj_close = 100.0 * np.cumprod(1.0 + returns)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "adjClose": adj_close,
                    "sector": sector,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _sector_index(prices: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    rows: list[dict[str, object]] = []
    for position, date in enumerate(dates):
        for sector, sign in [("Energy", -1.0), ("Tech", 1.0)]:
            value = sign * 0.4 * np.sin(position / 5.0)
            rows.append(
                {
                    "date": date,
                    "sector": sector,
                    "sentiment_value": value,
                    "headline_count": 2,
                    "ticker_with_news_count": 2,
                    "sector_ticker_count": 2,
                    "zero_news_ticker_count": 0,
                    "sentiment_observed": True,
                }
            )
    return pd.DataFrame(rows)


class TestFusionTimingAndScope(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = _prices()
        self.index = _sector_index(self.prices)

    def test_each_fusion_decision_uses_exactly_the_prior_observed_day(self) -> None:
        artifacts = fusion.build_r4_artifacts(
            self.prices, self.index, estimation_window=10
        )
        weights = artifacts.fusion_weights.loc[
            artifacts.fusion_weights["variant"].eq(fusion.FUSION_VARIANT)
        ]
        dates = pd.DatetimeIndex(sorted(self.prices["date"].unique()))
        prior_by_date = pd.Series(dates[:-1], index=dates[1:])
        expected = weights["rebalance_date"].map(prior_by_date)

        self.assertTrue((weights["sentiment_observation_date"] < weights["rebalance_date"]).all())
        self.assertTrue(weights["sentiment_observation_date"].equals(expected))
        self.assertTrue((weights["estimation_end_date"] < weights["rebalance_date"]).all())

    def test_same_day_and_future_sentiment_cannot_change_first_target(self) -> None:
        baseline = fusion.build_r4_artifacts(
            self.prices, self.index, estimation_window=10
        )
        first_rebalance = baseline.fusion_weights["rebalance_date"].min()
        changed = self.index.copy()
        changed.loc[changed["date"].ge(first_rebalance), "sentiment_value"] *= -1.0
        alternative = fusion.build_r4_artifacts(
            self.prices, changed, estimation_window=10
        )

        def first_target(artifacts: fusion.R4Artifacts) -> pd.DataFrame:
            return (
                artifacts.fusion_weights.loc[
                    artifacts.fusion_weights["rebalance_date"].eq(first_rebalance),
                    ["variant", "ticker", "target_weight"],
                ]
                .sort_values(["variant", "ticker"])
                .reset_index(drop=True)
            )

        pd.testing.assert_frame_equal(first_target(baseline), first_target(alternative))

    def test_fusion_is_equity_only_and_rejects_crypto_prices(self) -> None:
        artifacts = fusion.build_r4_artifacts(
            self.prices, self.index, estimation_window=10
        )
        self.assertEqual(set(artifacts.fusion_weights["holding_asset_class"]), {"equity"})
        self.assertEqual(set(artifacts.fusion_weights["ticker"]), {"AAA", "BBB", "CCC", "DDD"})
        self.assertEqual(set(artifacts.fusion_returns["asset_family"]), {"Equity"})
        with self.assertRaisesRegex(ValueError, "not cryptocurrency"):
            fusion.build_equity_return_matrix(_prices(include_crypto=True))

    def test_missing_prior_sector_news_means_no_tilt_not_index_imputation(self) -> None:
        changed = self.index.copy()
        changed.loc[changed["sector"].eq("Tech"), "sentiment_value"] = np.nan
        artifacts = fusion.build_r4_artifacts(
            self.prices, changed, estimation_window=10
        )
        tech = artifacts.fusion_weights.loc[
            artifacts.fusion_weights["variant"].eq(fusion.FUSION_VARIANT)
            & artifacts.fusion_weights["sector"].eq("Tech")
        ]
        self.assertTrue(tech["lagged_sentiment_value"].isna().all())
        self.assertFalse(tech["sentiment_available"].any())
        self.assertTrue(np.allclose(tech["tilt_multiplier"], 1.0))


class TestFusionOutputsAndReproducibility(unittest.TestCase):
    def test_schemas_samples_and_build_are_reproducible(self) -> None:
        prices = _prices()
        index = _sector_index(prices)
        first = fusion.build_r4_artifacts(prices, index, estimation_window=10)
        second = fusion.build_r4_artifacts(prices, index, estimation_window=10)

        pd.testing.assert_frame_equal(first.fusion_returns, second.fusion_returns)
        pd.testing.assert_frame_equal(first.fusion_weights, second.fusion_weights)
        pd.testing.assert_frame_equal(first.fusion_comparison, second.fusion_comparison)
        pd.testing.assert_frame_equal(first.fusion_turnover, second.fusion_turnover)
        pd.testing.assert_frame_equal(first.fusion_cost_check, second.fusion_cost_check)
        pd.testing.assert_frame_equal(first.fusion_sensitivity, second.fusion_sensitivity)
        self.assertEqual(list(first.fusion_returns), fusion.FUSION_RETURN_COLUMNS)
        self.assertEqual(list(first.fusion_weights), fusion.FUSION_WEIGHT_COLUMNS)
        self.assertEqual(list(first.fusion_comparison), fusion.FUSION_COMPARISON_COLUMNS)
        self.assertEqual(list(first.fusion_turnover), fusion.FUSION_TURNOVER_COLUMNS)
        self.assertEqual(list(first.fusion_cost_check), fusion.FUSION_COST_CHECK_COLUMNS)
        self.assertEqual(list(first.fusion_sensitivity), fusion.FUSION_SENSITIVITY_COLUMNS)
        samples = first.fusion_returns.groupby("variant")["date"].agg(["min", "max", "size"])
        self.assertEqual(len(samples.drop_duplicates()), 1)
        self.assertTrue(first.fusion_comparison["evaluation_period"].str.contains(" to ").all())

    def test_turnover_uses_pre_trade_drift_and_excludes_initial_establishment(self) -> None:
        prices = _prices()
        artifacts = fusion.build_r4_artifacts(
            prices,
            _sector_index(prices),
            estimation_window=10,
            sensitivity_strengths=(0.0, 0.5),
        )
        turnover = artifacts.fusion_turnover.loc[
            artifacts.fusion_turnover["variant"].eq(fusion.BASE_VARIANT)
        ].sort_values("rebalance_date")
        self.assertTrue(bool(turnover.iloc[0]["initial_establishment"]))
        self.assertTrue(pd.isna(turnover.iloc[0]["one_way_turnover"]))

        first_date = pd.Timestamp(turnover.iloc[0]["rebalance_date"])
        second_date = pd.Timestamp(turnover.iloc[1]["rebalance_date"])
        targets = artifacts.fusion_weights.loc[
            artifacts.fusion_weights["variant"].eq(fusion.BASE_VARIANT)
        ]
        first_target = targets.loc[
            targets["rebalance_date"].eq(first_date), ["ticker", "target_weight"]
        ].set_index("ticker")["target_weight"]
        second_target = targets.loc[
            targets["rebalance_date"].eq(second_date), ["ticker", "target_weight"]
        ].set_index("ticker")["target_weight"]
        returns, _ = fusion.build_equity_return_matrix(prices)
        interval = returns.loc[first_date:].loc[lambda frame: frame.index < second_date]
        pre_trade_values = first_target * (1.0 + interval).prod()
        pre_trade_weights = pre_trade_values / pre_trade_values.sum()
        expected = 0.5 * float((second_target - pre_trade_weights).abs().sum())

        self.assertAlmostEqual(turnover.iloc[1]["one_way_turnover"], expected)
        self.assertTrue(
            (~turnover.iloc[1:]["initial_establishment"]).all()
        )

    def test_transaction_cost_check_is_explicit_and_reduces_net_growth(self) -> None:
        artifacts = fusion.build_r4_artifacts(
            _prices(),
            _sector_index(_prices()),
            estimation_window=10,
            sensitivity_strengths=(0.0, 0.5),
            transaction_cost_bps=(0.0, 10.0, 25.0),
        )
        check = artifacts.fusion_cost_check
        self.assertFalse(check["initial_establishment_included"].any())
        self.assertEqual(set(check["cost_rate_bps_one_way"]), {0.0, 10.0, 25.0})
        zero_cost = check.loc[check["cost_rate_bps_one_way"].eq(0.0)]
        self.assertTrue(
            np.allclose(
                zero_cost["net_final_growth_of_1"],
                zero_cost["gross_final_growth_of_1"],
            )
        )
        for _, group in check.groupby("variant"):
            ordered = group.sort_values("cost_rate_bps_one_way")
            self.assertTrue(
                ordered["net_final_growth_of_1"].is_monotonic_decreasing
            )
            self.assertTrue((ordered["charged_rebalances"] > 0).all())
        for variant in fusion.VARIANTS:
            row = check.loc[
                check["variant"].eq(variant)
                & check["cost_rate_bps_one_way"].eq(10.0)
            ].iloc[0]
            realised = artifacts.fusion_turnover.loc[
                artifacts.fusion_turnover["variant"].eq(variant)
                & ~artifacts.fusion_turnover["initial_establishment"],
                "one_way_turnover",
            ]
            expected_growth = row["gross_final_growth_of_1"] * float(
                (1.0 - realised * 10.0 / 10_000.0).prod()
            )
            self.assertAlmostEqual(row["net_final_growth_of_1"], expected_growth)

    def test_multiplier_sensitivity_matches_zero_base_and_approved_fusion(self) -> None:
        artifacts = fusion.build_r4_artifacts(
            _prices(), _sector_index(_prices()), estimation_window=10
        )
        sensitivity = artifacts.fusion_sensitivity.set_index("tilt_strength")
        comparison = artifacts.fusion_comparison.set_index("variant")

        self.assertEqual(
            set(sensitivity.index), set(fusion.DEFAULT_SENSITIVITY_STRENGTHS)
        )
        self.assertEqual(int(sensitivity["approved_design"].sum()), 1)
        self.assertAlmostEqual(
            sensitivity.loc[0.0, "annualised_return"],
            comparison.loc[fusion.BASE_VARIANT, "annualised_return"],
        )
        self.assertAlmostEqual(
            sensitivity.loc[0.5, "sharpe_ratio"],
            comparison.loc[fusion.FUSION_VARIANT, "sharpe_ratio"],
        )
        self.assertAlmostEqual(
            sensitivity.loc[0.0, "annualised_return_difference_vs_zero_multiplier"],
            0.0,
        )

    def test_writer_uses_separate_exact_filenames_and_creates_figure(self) -> None:
        artifacts = fusion.build_r4_artifacts(
            _prices(), _sector_index(_prices()), estimation_window=10
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = fusion.write_r4_artifacts(artifacts, temporary_directory)
            names = [path.name for path in paths]
            sizes = [path.stat().st_size for path in paths]
            saved_comparison = pd.read_csv(paths[2])

        self.assertEqual(
            names,
            [
                "sentiment_fusion_returns.csv",
                "sentiment_fusion_weights.csv",
                "sentiment_fusion_comparison.csv",
                "sentiment_fusion_turnover.csv",
                "sentiment_fusion_transaction_cost_check.csv",
                "sentiment_multiplier_sensitivity.csv",
                "sentiment_fusion_comparison.png",
            ],
        )
        self.assertTrue(all(size > 0 for size in sizes))
        self.assertEqual(list(saved_comparison), fusion.FUSION_COMPARISON_COLUMNS)
        self.assertEqual(set(saved_comparison["variant"]), set(fusion.VARIANTS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
