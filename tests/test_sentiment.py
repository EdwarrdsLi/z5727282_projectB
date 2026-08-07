"""Focused deterministic tests for R3 headline and sector sentiment.

Run from the project root with::

    python tests/test_sentiment.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import sentiment  # noqa: E402


class FakeVader:
    """Deterministic VADER-shaped scorer that records exact input strings."""

    def __init__(self, compounds: dict[str, float]) -> None:
        self.compounds = compounds
        self.seen: list[str] = []

    def polarity_scores(self, text: str) -> dict[str, float]:
        self.seen.append(text)
        compound = self.compounds[text]
        if compound > 0:
            neg, neu, pos = 0.0, 0.25, 0.75
        elif compound < 0:
            neg, neu, pos = 0.75, 0.25, 0.0
        else:
            neg, neu, pos = 0.0, 1.0, 0.0
        return {"neg": neg, "neu": neu, "pos": pos, "compound": compound}


def _equities() -> pd.DataFrame:
    dates = pd.DatetimeIndex(["2023-01-06", "2023-01-09", "2023-01-10"])
    rows: list[dict[str, object]] = []
    for ticker, sector in [("AAA", "Tech"), ("BBB", "Tech"), ("CCC", "Energy")]:
        rows.extend(
            {"date": date, "ticker": ticker, "sector": sector}
            for date in dates
        )
    return pd.DataFrame(rows)


def _headlines() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2023-01-07T10:00:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "AAA GREAT!",
            },
            {
                "date": "2023-01-08T11:00:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "AAA gains again!",
            },
            {
                "date": "2023-01-09T09:00:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "AAA beats estimates!",
            },
            {
                "date": "2023-01-09T12:00:00Z",
                "ticker": "BBB",
                "sector": "Tech",
                "title": "BBB warns: BAD?",
            },
            {
                "date": "2023-01-10T12:00:00Z",
                "ticker": "CCC",
                "sector": "Energy",
                "title": "CCC files quarterly report.",
            },
        ]
    )


def _fake_vader() -> FakeVader:
    return FakeVader(
        {
            "AAA GREAT!": 0.9,
            "AAA gains again!": 0.9,
            "AAA beats estimates!": 0.9,
            "BBB warns: BAD?": -0.9,
            "CCC files quarterly report.": 0.0,
        }
    )


class TestHeadlineScoring(unittest.TestCase):
    def test_alignment_and_vader_preserve_exact_headline_text(self) -> None:
        aligned = sentiment.align_equity_headlines(_headlines(), _equities())
        analyzer = _fake_vader()
        scored = sentiment.score_headlines(aligned, analyzer=analyzer)

        expected_titles = set(_headlines()["title"])
        self.assertEqual(set(scored["headline_text"]), expected_titles)
        self.assertEqual(set(analyzer.seen), expected_titles)
        self.assertIn("AAA GREAT!", analyzer.seen)
        self.assertIn("BBB warns: BAD?", analyzer.seen)
        weekend = scored.loc[scored["headline_text"] == "AAA GREAT!"].iloc[0]
        self.assertEqual(weekend["source_date"], pd.Timestamp("2023-01-07"))
        self.assertEqual(weekend["trading_date"], pd.Timestamp("2023-01-09"))
        self.assertEqual(list(scored.columns), sentiment.HEADLINE_SCORE_COLUMNS)

    def test_scoring_retains_all_vader_components(self) -> None:
        aligned = sentiment.align_equity_headlines(_headlines(), _equities())
        scored = sentiment.score_headlines(aligned, analyzer=_fake_vader())
        row = scored.loc[scored["headline_text"] == "BBB warns: BAD?"].iloc[0]

        self.assertAlmostEqual(float(row["vader_negative"]), 0.75)
        self.assertAlmostEqual(float(row["vader_neutral"]), 0.25)
        self.assertAlmostEqual(float(row["vader_positive"]), 0.0)
        self.assertAlmostEqual(float(row["vader_compound"]), -0.9)

    def test_non_equity_ticker_and_post_sample_news_are_rejected(self) -> None:
        crypto_news = _headlines().iloc[[0]].copy()
        crypto_news.loc[:, "ticker"] = "BTC-USD"
        crypto_news.loc[:, "sector"] = "Cryptocurrency"
        with self.assertRaisesRegex(ValueError, "non-equity or unknown"):
            sentiment.align_equity_headlines(crypto_news, _equities())

        future_news = _headlines().iloc[[0]].copy()
        future_news.loc[:, "date"] = "2024-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "exceeds 2023-12-31"):
            sentiment.align_equity_headlines(future_news, _equities())


class TestSectorIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = sentiment.build_r3_artifacts(
            _equities(),
            _headlines(),
            analyzer=_fake_vader(),
        )
        cls.index = cls.artifacts.sector_sentiment_index

    def test_two_stage_aggregation_equal_weights_observed_tickers(self) -> None:
        monday_tech = self.index.loc[
            (self.index["date"] == pd.Timestamp("2023-01-09"))
            & (self.index["sector"] == "Tech")
        ].iloc[0]

        # AAA has three +0.9 headlines but receives one ticker weight; BBB has
        # one -0.9 headline and receives the other. A headline-weighted mean
        # would be +0.45, whereas the required ticker-equal mean is exactly 0.
        self.assertAlmostEqual(float(monday_tech["sentiment_value"]), 0.0)
        self.assertEqual(int(monday_tech["headline_count"]), 4)
        self.assertEqual(int(monday_tech["ticker_with_news_count"]), 2)
        self.assertEqual(int(monday_tech["zero_news_ticker_count"]), 0)
        self.assertTrue(bool(monday_tech["sentiment_observed"]))

    def test_zero_news_is_missing_information_not_neutral(self) -> None:
        friday_tech = self.index.loc[
            (self.index["date"] == pd.Timestamp("2023-01-06"))
            & (self.index["sector"] == "Tech")
        ].iloc[0]
        self.assertTrue(pd.isna(friday_tech["sentiment_value"]))
        self.assertEqual(int(friday_tech["headline_count"]), 0)
        self.assertEqual(int(friday_tech["ticker_with_news_count"]), 0)
        self.assertEqual(int(friday_tech["sector_ticker_count"]), 2)
        self.assertEqual(int(friday_tech["zero_news_ticker_count"]), 2)
        self.assertFalse(bool(friday_tech["sentiment_observed"]))

        tuesday_energy = self.index.loc[
            (self.index["date"] == pd.Timestamp("2023-01-10"))
            & (self.index["sector"] == "Energy")
        ].iloc[0]
        self.assertEqual(float(tuesday_energy["sentiment_value"]), 0.0)
        self.assertEqual(int(tuesday_energy["headline_count"]), 1)
        self.assertTrue(bool(tuesday_energy["sentiment_observed"]))

    def test_full_equity_calendar_sector_grid_and_schema(self) -> None:
        self.assertEqual(len(self.index), 3 * 2)
        self.assertEqual(list(self.index.columns), sentiment.SECTOR_SENTIMENT_COLUMNS)
        self.assertEqual(self.index["date"].nunique(), 3)
        self.assertEqual(self.index["sector"].nunique(), 2)
        self.assertLessEqual(self.index["date"].max(), sentiment.SAMPLE_END)

    def test_future_use_requires_one_observed_trading_day_lag(self) -> None:
        lagged = sentiment.lag_sector_sentiment_one_trading_day(self.index)
        monday_tech = lagged.loc[
            (lagged["decision_date"] == pd.Timestamp("2023-01-09"))
            & (lagged["sector"] == "Tech")
        ].iloc[0]
        tuesday_tech = lagged.loc[
            (lagged["decision_date"] == pd.Timestamp("2023-01-10"))
            & (lagged["sector"] == "Tech")
        ].iloc[0]

        self.assertEqual(
            monday_tech["sentiment_observation_date"],
            pd.Timestamp("2023-01-06"),
        )
        self.assertTrue(pd.isna(monday_tech["lagged_sentiment_value"]))
        self.assertEqual(
            tuesday_tech["sentiment_observation_date"],
            pd.Timestamp("2023-01-09"),
        )
        self.assertAlmostEqual(float(tuesday_tech["lagged_sentiment_value"]), 0.0)
        comparable = lagged["sentiment_observation_date"].notna()
        self.assertTrue(
            (
                lagged.loc[comparable, "sentiment_observation_date"]
                < lagged.loc[comparable, "decision_date"]
            ).all()
        )

    def test_required_csv_round_trip_preserves_missing_sentiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = sentiment.write_r3_artifact(self.index, temporary_directory)
            saved = pd.read_csv(path)

        self.assertEqual(path.name, "sector_sentiment_index.csv")
        self.assertEqual(list(saved.columns), sentiment.SECTOR_SENTIMENT_COLUMNS)
        self.assertEqual(len(saved), len(self.index))
        zero_news = saved["headline_count"].eq(0)
        self.assertTrue(saved.loc[zero_news, "sentiment_value"].isna().all())
        self.assertFalse(np.isinf(saved["sentiment_value"].dropna()).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
