"""Focused synthetic tests for the reused Part A data foundation.

Run from the project root with::

    python tests/test_foundation.py
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import etl, features  # noqa: E402


def _price_row(ticker: str, date: str, adj_close: float) -> dict[str, object]:
    return {
        "ticker": ticker,
        "date": date,
        "open": adj_close,
        "high": adj_close * 1.01,
        "low": adj_close * 0.99,
        "close": adj_close,
        "adjClose": adj_close,
        "volume": 1_000,
    }


def _equity_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **_price_row("AAA", "2023-12-22", 100.0),
                "close": 10.0,
                "sector": "Tech",
            },
            {
                **_price_row("AAA", "2023-12-26", 110.0),
                "close": 1_000.0,
                "sector": "Tech",
            },
            {
                **_price_row("AAA", "2023-12-26", 110.0),
                "close": 1_000.0,
                "sector": "Tech",
            },
            {
                **_price_row("AAA", "2023-12-27", 220.0),
                "close": 1_000.0,
                "sector": "Tech",
            },
            {
                **_price_row("BBB", "2023-12-22", 200.0),
                "sector": "Financials",
            },
            {
                **_price_row("BBB", "2023-12-26", 100.0),
                "sector": "Financials",
            },
        ]
    )


def _crypto_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _price_row("BTC-USD", "2023-12-22", 100.0),
            _price_row("BTC-USD", "2023-12-23", 110.0),
            _price_row("BTC-USD", "2023-12-24", 121.0),
            _price_row("BTC-USD", "2023-12-25", 133.1),
            _price_row("BTC-USD", "2023-12-26", 146.41),
            _price_row("BTC-USD", "2024-01-01", 999.0),
        ]
    )


def _news_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2023-12-22T14:30:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "Trading-day headline!",
                "url": "https://example.test/1",
                "publisher": "P1",
            },
            {
                "date": "2023-12-22T14:30:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "Trading-day headline!",
                "url": "https://example.test/duplicate",
                "publisher": "P2",
            },
            {
                "date": "2023-12-22T18:00:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "Different same-day headline",
                "url": "https://example.test/2",
                "publisher": "P1",
            },
            {
                "date": "2023-12-23T10:00:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "Weekend: Strong GAIN?",
                "url": "https://example.test/3",
                "publisher": "P1",
            },
            {
                "date": "2023-12-25T18:00:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "Holiday news stays RAW.",
                "url": "https://example.test/4",
                "publisher": "P1",
            },
            {
                "date": "2023-12-28T00:00:00Z",
                "ticker": "BBB",
                "sector": "Financials",
                "title": "After the observed calendar",
                "url": "https://example.test/5",
                "publisher": "P3",
            },
            {
                "date": "2024-01-01T00:00:00Z",
                "ticker": "AAA",
                "sector": "Tech",
                "title": "Outside approved sample",
                "url": "https://example.test/6",
                "publisher": "P1",
            },
        ]
    )


class TestFoundationCleaning(unittest.TestCase):
    def test_cutoff_price_keys_and_extreme_retention(self) -> None:
        equities = etl.clean_equities(_equity_fixture())
        crypto = etl.clean_crypto(_crypto_fixture())

        self.assertFalse(equities.duplicated(etl.PRICE_KEY).any())
        self.assertFalse(crypto.duplicated(etl.PRICE_KEY).any())
        self.assertLessEqual(equities["date"].max(), etl.SAMPLE_END)
        self.assertEqual(crypto["date"].max(), pd.Timestamp("2023-12-26"))
        retained_extreme = equities.loc[
            (equities["ticker"] == "AAA")
            & (equities["date"] == pd.Timestamp("2023-12-27")),
            "adjClose",
        ]
        self.assertEqual(retained_extreme.tolist(), [220.0])

    def test_news_uses_exact_ticker_date_title_key(self) -> None:
        news = etl.clean_news(_news_fixture())
        self.assertFalse(news.duplicated(etl.NEWS_KEY).any())
        same_day = news.loc[
            (news["ticker"] == "AAA")
            & (news["date"] == pd.Timestamp("2023-12-22"))
        ]
        self.assertEqual(len(same_day), 2)
        self.assertEqual(same_day["title"].nunique(), 2)
        self.assertNotIn("Outside approved sample", set(news["title"]))

    def test_raw_loads_use_project_b_data_access(self) -> None:
        with (
            patch.object(
                etl.data_access,
                "load_equity_prices",
                return_value=_equity_fixture(),
            ) as load_equities,
            patch.object(
                etl.data_access,
                "load_crypto_prices",
                return_value=_crypto_fixture(),
            ) as load_crypto,
            patch.object(
                etl.data_access,
                "load_news_headlines",
                return_value=_news_fixture(),
            ) as load_news,
        ):
            foundation = etl.load_part_a_foundation()

        load_equities.assert_called_once_with()
        load_crypto.assert_called_once_with()
        load_news.assert_called_once_with()
        self.assertIsInstance(foundation, etl.FoundationData)
        self.assertLessEqual(foundation.news["date"].max(), etl.SAMPLE_END)


class TestNativeCalendarReturns(unittest.TestCase):
    def test_returns_use_adjusted_close_and_stay_within_ticker(self) -> None:
        equities = etl.clean_equities(_equity_fixture())
        returned = features.daily_returns(equities)

        aaa_second = returned.loc[
            (returned["ticker"] == "AAA")
            & (returned["date"] == pd.Timestamp("2023-12-26")),
            "simple_return",
        ].iloc[0]
        bbb = returned.loc[returned["ticker"] == "BBB"].reset_index(drop=True)
        self.assertAlmostEqual(float(aaa_second), 0.10)
        self.assertTrue(pd.isna(bbb.loc[0, "simple_return"]))
        self.assertAlmostEqual(float(bbb.loc[1, "simple_return"]), -0.50)
        with self.assertRaisesRegex(ValueError, "must use adjClose"):
            features.daily_returns(equities, price_col="close")

    def test_crypto_returns_are_calculated_before_equity_alignment(self) -> None:
        equities = etl.clean_equities(_equity_fixture())
        crypto = etl.clean_crypto(_crypto_fixture())
        combined = features.build_combined_returns_panel(equities, crypto)

        expected_dates = pd.DatetimeIndex(
            ["2023-12-22", "2023-12-26", "2023-12-27"]
        )
        self.assertTrue(pd.DatetimeIndex(combined["date"]).equals(expected_dates))
        monday_return = combined.loc[
            combined["date"] == pd.Timestamp("2023-12-26"),
            "crypto_return__BTC-USD",
        ].iloc[0]
        self.assertAlmostEqual(float(monday_return), 0.10)
        self.assertNotAlmostEqual(float(monday_return), 146.41 / 100.0 - 1.0)
        # There is no 27 December crypto observation. A price or return fill
        # would manufacture a value; the left-aligned panel must keep it missing.
        missing_crypto = combined.loc[
            combined["date"] == pd.Timestamp("2023-12-27"),
            "crypto_return__BTC-USD",
        ].iloc[0]
        self.assertTrue(pd.isna(missing_crypto))


class TestHeadlinePanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cleaned_news = etl.clean_news(_news_fixture())
        calendar = pd.Series(
            [pd.Timestamp("2023-12-22"), pd.Timestamp("2023-12-26")],
            name="date",
        )
        cls.panel, cls.audit = features.assemble_headline_panel(
            cleaned_news,
            calendar,
            return_audit=True,
        )

    def test_same_day_or_next_observed_day_never_backward(self) -> None:
        friday = self.panel.loc[
            self.panel["trading_date"] == pd.Timestamp("2023-12-22")
        ].iloc[0]
        tuesday = self.panel.loc[
            self.panel["trading_date"] == pd.Timestamp("2023-12-26")
        ].iloc[0]

        self.assertEqual(int(friday["headline_count"]), 2)
        self.assertEqual(int(tuesday["headline_count"]), 2)
        self.assertLessEqual(friday["last_source_date"], friday["trading_date"])
        self.assertLessEqual(tuesday["last_source_date"], tuesday["trading_date"])
        self.assertEqual(tuesday["first_source_date"], pd.Timestamp("2023-12-23"))
        audit = self.audit.iloc[0]
        self.assertEqual(int(audit["same_day_mapping_count"]), 2)
        self.assertEqual(int(audit["next_trading_day_mapping_count"]), 2)
        self.assertEqual(int(audit["unmapped_headline_count"]), 1)
        self.assertIn("never backward", audit["trading_calendar_rule"])

    def test_headline_text_is_preserved_without_sentiment_fields(self) -> None:
        combined_text = "\n".join(self.panel["headline_text"])
        self.assertIn("Weekend: Strong GAIN?", combined_text)
        self.assertIn("Holiday news stays RAW.", combined_text)
        self.assertTrue(bool(self.audit.iloc[0]["headline_count_reconciles"]))
        self.assertTrue(bool(self.audit.iloc[0]["exact_headline_text_preserved"]))
        for column in self.panel.columns:
            self.assertNotIn("sentiment", column.lower())
            self.assertNotIn("score", column.lower())
            self.assertNotIn("signal", column.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
