"""Deterministic tests for R5 report exhibits and fund fact sheets."""
from __future__ import annotations

import hashlib
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import exhibits, portfolios, sentiment  # noqa: E402


def _fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=8)
    specifications = {
        "Combined Equal Weight": ("Equal Weight", np.array([0.01, -0.02, 0.005, 0.0, 0.01, -0.005, 0.004, 0.003])),
        "Combined Risk Parity": ("Risk Parity", np.array([0.006, -0.01, 0.004, 0.002, 0.007, -0.003, 0.003, 0.002])),
    }
    return_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    tickers = [("AAA", "equity"), ("BBB", "equity"), ("BTC-USD", "cryptocurrency"), ("ETH-USD", "cryptocurrency")]
    rebalances = [dates[0], dates[5]]
    for fund_name, (method, daily) in specifications.items():
        growth = pd.Series(1.0 + daily).cumprod()
        drawdown = growth / growth.cummax().clip(lower=1.0) - 1.0
        for position, date in enumerate(dates):
            return_rows.append(
                {
                    "date": date,
                    "fund_name": fund_name,
                    "asset_family": "Combined equity + cryptocurrency",
                    "portfolio_method": method,
                    "daily_return": daily[position],
                    "growth_of_1": growth.iloc[position],
                    "drawdown": drawdown.iloc[position],
                    "active_target_rebalance_date": rebalances[0] if position < 5 else rebalances[1],
                }
            )
        latest_weights = [0.25, 0.25, 0.25, 0.25] if method == "Equal Weight" else [0.35, 0.30, 0.20, 0.15]
        for rebalance_index, rebalance in enumerate(rebalances):
            values = [0.25] * 4 if rebalance_index == 0 else latest_weights
            for (ticker, asset_class), value in zip(tickers, values, strict=True):
                weight_rows.append(
                    {
                        "rebalance_date": rebalance,
                        "fund_name": fund_name,
                        "asset_family": "Combined equity + cryptocurrency",
                        "portfolio_method": method,
                        "ticker": ticker,
                        "holding_asset_class": asset_class,
                        "target_weight": value,
                        "estimation_start_date": dates[0] - pd.Timedelta(days=10),
                        "estimation_end_date": rebalance - pd.Timedelta(days=1),
                        "estimation_observations": 5,
                    }
                )
        computed = portfolios.performance_metrics(pd.Series(daily), 252, 0.0)
        metric_rows.append(
            {
                "fund_name": fund_name,
                "asset_family": "Combined equity + cryptocurrency",
                "portfolio_method": method,
                "evaluation_period": f"{dates[0].date()} to {dates[-1].date()}",
                "evaluation_start_date": dates[0],
                "evaluation_end_date": dates[-1],
                "observations": len(dates),
                **computed,
                "current_target_holdings_source": "results/data/fund_weights.csv; filter fund_name and latest rebalance_date",
                "latest_rebalance_date": rebalances[-1],
                "annualisation_days_per_year": 252,
                "risk_free_rate_annual": 0.0,
                "transaction_cost_rate": 0.0,
                "estimation_window_type": "Rolling complete equity-calendar observations",
                "estimation_window_observations": 5,
                "rebalance_rule": "First observed equity trading day of each calendar month",
                "constraints": "Long-only; fully invested; no leverage",
            }
        )
    sector_rows: list[dict[str, object]] = []
    for position, date in enumerate(dates):
        for sector_name, sign in (("Energy", -1.0), ("Tech", 1.0)):
            missing = position == 3 and sector_name == "Energy"
            sector_rows.append(
                {
                    "date": date,
                    "sector": sector_name,
                    "sentiment_value": np.nan if missing else sign * 0.1 * position,
                    "headline_count": 0 if missing else 2,
                    "ticker_with_news_count": 0 if missing else 2,
                    "sector_ticker_count": 2,
                    "zero_news_ticker_count": 2 if missing else 0,
                    "sentiment_observed": not missing,
                }
            )
    return (
        pd.DataFrame(return_rows, columns=portfolios.FUND_RETURN_COLUMNS),
        pd.DataFrame(weight_rows, columns=portfolios.FUND_WEIGHT_COLUMNS),
        pd.DataFrame(metric_rows, columns=portfolios.PERFORMANCE_METRIC_COLUMNS),
        pd.DataFrame(sector_rows, columns=sentiment.SECTOR_SENTIMENT_COLUMNS),
    )


class TestR5Tables(unittest.TestCase):
    def test_fact_sheets_use_verified_metrics_and_all_latest_holdings(self) -> None:
        tables = exhibits.build_r5_tables(*_fixture_frames())
        self.assertEqual(list(tables.performance_comparison), exhibits.PERFORMANCE_COMPARISON_COLUMNS)
        self.assertEqual(list(tables.fact_sheet_summary), exhibits.FACT_SHEET_SUMMARY_COLUMNS)
        self.assertEqual(list(tables.fact_sheet_holdings), exhibits.FACT_SHEET_HOLDING_COLUMNS)
        self.assertEqual(len(tables.fact_sheet_summary), 2)
        self.assertTrue((tables.fact_sheet_summary["current_holdings_count"] == 4).all())
        sums = tables.fact_sheet_holdings.groupby("fund_name")["target_weight"].sum()
        self.assertTrue(np.allclose(sums, 1.0))
        metrics = tables.performance_metrics.set_index("fund_name")
        summary = tables.fact_sheet_summary.set_index("fund_name")
        self.assertTrue(np.allclose(summary["sharpe_ratio"], metrics["sharpe_ratio"]))
        self.assertEqual(set(tables.fact_sheet_holdings["latest_rebalance_date"]), {pd.Timestamp("2023-01-09")})

    def test_inconsistent_growth_metric_and_future_date_are_rejected(self) -> None:
        frames = list(_fixture_frames())
        frames[0] = frames[0].copy()
        frames[0].loc[0, "growth_of_1"] += 0.1
        with self.assertRaisesRegex(ValueError, "growth arithmetic"):
            exhibits.build_r5_tables(*frames)

        frames = list(_fixture_frames())
        frames[2] = frames[2].copy()
        frames[2].loc[0, "sharpe_ratio"] += 0.1
        with self.assertRaisesRegex(ValueError, "sharpe_ratio"):
            exhibits.build_r5_tables(*frames)

        frames = list(_fixture_frames())
        frames[3] = frames[3].copy()
        frames[3].loc[0, "date"] = pd.Timestamp("2024-01-01")
        with self.assertRaisesRegex(ValueError, "sample cap"):
            exhibits.build_r5_tables(*frames)


class TestR5Writer(unittest.TestCase):
    def test_strict_date_limits_end_on_last_observation(self) -> None:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        dates = pd.to_datetime(["2023-01-02", "2023-12-29"])
        figure, axis = plt.subplots()
        exhibits._set_strict_date_limits(axis, dates)
        left, right = axis.get_xlim()
        plt.close(figure)

        self.assertEqual(mdates.num2date(left).date(), dates.min().date())
        self.assertEqual(mdates.num2date(right).date(), dates.max().date())
        self.assertLess(mdates.num2date(right).date(), pd.Timestamp("2024-01-01").date())

    def test_date_axis_writer_only_targets_the_six_affected_figures(self) -> None:
        tables = exhibits.build_r5_tables(*_fixture_frames())
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = exhibits.write_r5_date_axis_figures(
                tables, temporary_directory
            )
            names = [path.name for path in paths]
            sizes = [path.stat().st_size for path in paths]

        self.assertEqual(
            names,
            [
                "drawdown_comparison.png",
                "fund_fact_sheet_combined_equal_weight.png",
                "fund_fact_sheet_combined_risk_parity.png",
                "growth_of_1_comparison.png",
                "portfolio_weights_over_time_risk_parity.png",
                "sector_sentiment_index.png",
            ],
        )
        self.assertTrue(all(size > 0 for size in sizes))

    def test_writer_creates_required_report_exhibits_and_pending_manifest(self) -> None:
        tables = exhibits.build_r5_tables(*_fixture_frames())
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths, manifest = exhibits.write_r5_artifacts(tables, temporary_directory)
            names = {path.name for path in paths}
            sizes = {path.name: path.stat().st_size for path in paths}
            png_headers = {
                path.name: path.read_bytes()[:8]
                for path in paths
                if path.suffix == ".png"
            }

        expected = {
            "performance_comparison.csv",
            "fund_fact_sheet_summary.csv",
            "fund_fact_sheet_holdings.csv",
            "report_exhibit_manifest.csv",
            "growth_of_1_comparison.png",
            "drawdown_comparison.png",
            "portfolio_weights_over_time_risk_parity.png",
            "risk_return_comparison.png",
            "sector_sentiment_index.png",
            "fund_fact_sheet_combined_equal_weight.png",
            "fund_fact_sheet_combined_risk_parity.png",
        }
        self.assertEqual(names, expected)
        self.assertTrue(all(size > 0 for size in sizes.values()))
        self.assertTrue(all(header == b"\x89PNG\r\n\x1a\n" for header in png_headers.values()))
        self.assertEqual(list(manifest), exhibits.MANIFEST_COLUMNS)
        self.assertTrue(manifest["student_interpretation_status"].str.startswith("PENDING").all())
        self.assertTrue(manifest["final_figure_acceptance_status"].str.startswith("PENDING").all())
        required_ids = {"performance_comparison", "growth_of_1", "drawdown", "portfolio_weights", "risk_return", "sector_sentiment"}
        self.assertTrue(required_ids.issubset(set(manifest["exhibit_id"])))
        figures = manifest.loc[manifest["artifact_type"].isin(["figure", "fact sheet"])]
        self.assertTrue(figures["title"].str.len().gt(0).all())
        self.assertTrue(figures["sample_period"].str.contains(" to ").all())
        self.assertTrue(figures["source"].str.len().gt(0).all())
        self.assertTrue(figures["technical_caption"].str.len().gt(0).all())

    def test_csv_and_png_outputs_are_byte_deterministic(self) -> None:
        tables = exhibits.build_r5_tables(*_fixture_frames())
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_paths, _ = exhibits.write_r5_artifacts(tables, first_directory)
            second_paths, _ = exhibits.write_r5_artifacts(tables, second_directory)
            first_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in first_paths}
            second_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in second_paths}
        self.assertEqual(first_hashes, second_hashes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
