"""Part A data cleaning reused as the foundation for Project B.

Raw course data enters Project B only through :mod:`src.data_access`. The
cleaning rules preserve the student's verified Part A decisions: normalise dates,
cap the sample at 2023-12-31, deduplicate price panels on ticker/date, deduplicate
news only on ticker/date/title, and retain genuine extreme observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import data_access


SAMPLE_END = pd.Timestamp("2023-12-31")
PRICE_KEY = ["ticker", "date"]
NEWS_KEY = ["ticker", "date", "title"]

EQUITY_REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjClose",
    "volume",
    "sector",
}
CRYPTO_REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjClose",
    "volume",
}
NEWS_REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "sector",
    "title",
    "url",
    "publisher",
}


@dataclass(frozen=True)
class FoundationData:
    """The three cleaned in-memory panels inherited from Part A."""

    equities: pd.DataFrame
    crypto: pd.DataFrame
    news: pd.DataFrame


def _normalise_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert dates through UTC to timezone-naive calendar dates."""
    if "date" not in frame.columns:
        raise ValueError("required column missing: date")
    result = frame.copy(deep=True)
    result["date"] = (
        pd.to_datetime(result["date"], errors="raise", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    return result


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    dataset: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def _clean_frame(
    raw: pd.DataFrame,
    *,
    dataset: str,
    key: list[str],
    required_columns: set[str],
) -> pd.DataFrame:
    """Apply the verified Part A cutoff and exact-key deduplication rules."""
    _require_columns(raw, required_columns, dataset)
    dated = _normalise_dates(raw)
    in_sample = dated.loc[dated["date"] <= SAMPLE_END].copy()
    cleaned = (
        in_sample.loc[~in_sample.duplicated(key, keep="first")]
        .sort_values(key, kind="stable")
        .reset_index(drop=True)
    )

    if cleaned.duplicated(key).any():
        raise AssertionError(f"{dataset} is not unique on {key} after cleaning")
    if not cleaned.empty and cleaned["date"].max() > SAMPLE_END:
        raise AssertionError(f"{dataset} exceeds the approved sample end")

    # Part A flagged genuine extremes for review but did not trim, winsorise, or
    # delete them. No return-based row filter is applied here.
    return cleaned


def clean_equities(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean an equity panel on the ticker/date key."""
    return _clean_frame(
        raw,
        dataset="equity_prices",
        key=PRICE_KEY,
        required_columns=EQUITY_REQUIRED_COLUMNS,
    )


def clean_crypto(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean a native-calendar cryptocurrency panel and enforce the cutoff."""
    return _clean_frame(
        raw,
        dataset="crypto_prices",
        key=PRICE_KEY,
        required_columns=CRYPTO_REQUIRED_COLUMNS,
    )


def clean_news(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove exact ticker/date/title duplicates while keeping other headlines."""
    return _clean_frame(
        raw,
        dataset="news_headlines",
        key=NEWS_KEY,
        required_columns=NEWS_REQUIRED_COLUMNS,
    )


def load_clean_equities() -> pd.DataFrame:
    """Load equities only through Project B's controlled data-access helper."""
    return clean_equities(data_access.load_equity_prices())


def load_clean_crypto() -> pd.DataFrame:
    """Load cryptocurrency only through Project B's controlled helper."""
    return clean_crypto(data_access.load_crypto_prices())


def load_clean_news() -> pd.DataFrame:
    """Load headlines only through Project B's controlled helper."""
    return clean_news(data_access.load_news_headlines())


def load_part_a_foundation() -> FoundationData:
    """Load all three clean Part A panels for later Project B stages."""
    return FoundationData(
        equities=load_clean_equities(),
        crypto=load_clean_crypto(),
        news=load_clean_news(),
    )
