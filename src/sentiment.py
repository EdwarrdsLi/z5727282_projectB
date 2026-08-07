"""Descriptive VADER headline sentiment and the daily equity-sector index.

R3 scores each supplied equity-news headline without changing its text. Scores
are first averaged within ticker/trading-day and those ticker-day means are then
equal-weighted within sector/trading-day. This prevents a firm with many
headlines from receiving a larger sector weight than another firm with news.

Ticker-days without headlines are omitted from the average. The output retains
the full observed equity-date/sector grid, reports factual coverage counts, and
uses a missing ``sentiment_value`` when a whole sector has no headlines. It does
not turn missing news into neutral sentiment. The index is descriptive and is
not consumed by the portfolio code in this phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAMPLE_END = pd.Timestamp("2023-12-31")

ALIGNED_HEADLINE_COLUMNS = [
    "source_date",
    "trading_date",
    "ticker",
    "sector",
    "headline_text",
]

HEADLINE_SCORE_COLUMNS = [
    *ALIGNED_HEADLINE_COLUMNS,
    "vader_negative",
    "vader_neutral",
    "vader_positive",
    "vader_compound",
]

SECTOR_SENTIMENT_COLUMNS = [
    "date",
    "sector",
    "sentiment_value",
    "headline_count",
    "ticker_with_news_count",
    "sector_ticker_count",
    "zero_news_ticker_count",
    "sentiment_observed",
]

LAGGED_SENTIMENT_COLUMNS = [
    "decision_date",
    "sector",
    "sentiment_observation_date",
    "lagged_sentiment_value",
]


@dataclass(frozen=True)
class R3Artifacts:
    """In-memory headline scores and the required saved sector index."""

    headline_scores: pd.DataFrame
    sector_sentiment_index: pd.DataFrame


def _normalise_dates(values: pd.Series) -> pd.Series:
    """Convert timestamps through UTC to timezone-naive calendar dates."""
    return (
        pd.to_datetime(values, errors="raise", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )


def _equity_universe(equity_prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "sector"}
    missing = sorted(required.difference(equity_prices.columns))
    if missing:
        raise ValueError(f"equity universe is missing required columns: {missing}")
    if equity_prices[["ticker", "sector"]].isna().any(axis=None):
        raise ValueError("equity ticker and sector must not be missing")

    universe = (
        equity_prices.loc[:, ["ticker", "sector"]]
        .drop_duplicates()
        .sort_values(["sector", "ticker"], kind="stable")
        .reset_index(drop=True)
    )
    sector_count = universe.groupby("ticker", sort=False)["sector"].nunique()
    if (sector_count != 1).any():
        raise ValueError("each equity ticker must map to exactly one sector")
    return universe


def _equity_calendar(equity_prices: pd.DataFrame) -> pd.DatetimeIndex:
    if "date" not in equity_prices.columns:
        raise ValueError("equity prices require a date column")
    dates = _normalise_dates(equity_prices["date"])
    if dates.isna().any():
        raise ValueError("equity trading dates must not be missing")
    if (dates > SAMPLE_END).any():
        raise ValueError("equity trading calendar exceeds 2023-12-31")
    calendar = pd.DatetimeIndex(dates.unique()).sort_values()
    if calendar.empty:
        raise ValueError("equity trading calendar must not be empty")
    return calendar


def align_equity_headlines(
    headlines: pd.DataFrame,
    equity_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Map supplied equity headlines to the same or next equity trading day.

    The original title string becomes ``headline_text`` without lowercasing,
    punctuation removal, tokenisation, or any other transformation. Headlines
    after the last observed trading date cannot be used and are excluded.
    """
    required = {"date", "ticker", "sector", "title"}
    missing = sorted(required.difference(headlines.columns))
    if missing:
        raise ValueError(f"headlines is missing required columns: {missing}")
    if headlines[["ticker", "sector", "title"]].isna().any(axis=None):
        raise ValueError("headline ticker, sector, and title must not be missing")
    if not headlines["title"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("every headline title must be a string")

    universe = _equity_universe(equity_prices)
    sector_by_ticker = universe.set_index("ticker")["sector"]
    unknown_tickers = sorted(set(headlines["ticker"]).difference(sector_by_ticker.index))
    if unknown_tickers:
        raise ValueError(
            "sentiment input contains non-equity or unknown ticker(s): "
            f"{unknown_tickers}"
        )
    expected_sector = headlines["ticker"].map(sector_by_ticker)
    sector_mismatch = ~headlines["sector"].eq(expected_sector)
    if sector_mismatch.any():
        raise ValueError("headline sector does not match the equity universe")

    calendar = _equity_calendar(equity_prices)
    aligned = headlines.loc[:, ["date", "ticker", "sector", "title"]].copy()
    aligned["source_date"] = _normalise_dates(aligned["date"])
    if aligned["source_date"].isna().any():
        raise ValueError("headline dates must not be missing")
    if (aligned["source_date"] > SAMPLE_END).any():
        raise ValueError("headline input exceeds 2023-12-31")
    aligned["_input_order"] = np.arange(len(aligned))

    positions = calendar.searchsorted(aligned["source_date"], side="left")
    can_map = positions < len(calendar)
    mapped = aligned.loc[can_map].copy()
    if not mapped.empty:
        mapped["trading_date"] = calendar.take(positions[can_map]).to_numpy()
    else:
        mapped["trading_date"] = pd.Series(dtype="datetime64[ns]")
    mapped = mapped.rename(columns={"title": "headline_text"})
    mapped = mapped.sort_values(
        ["trading_date", "ticker", "source_date", "headline_text", "_input_order"],
        kind="stable",
    ).reset_index(drop=True)
    result = mapped.loc[:, ALIGNED_HEADLINE_COLUMNS]

    if not result.empty:
        if (result["source_date"] > result["trading_date"]).any():
            raise AssertionError("a headline was mapped backward")
        if result["trading_date"].max() > SAMPLE_END:
            raise AssertionError("aligned headline output exceeds 2023-12-31")
    return result


def _make_vader_analyzer() -> Any:
    """Create NLTK's VADER analyser with an actionable lexicon error."""
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise LookupError(
            "NLTK's vader_lexicon is required for the R3 build. Install it once "
            "with: python -m nltk.downloader vader_lexicon"
        ) from exc


def score_headlines(
    panel: pd.DataFrame,
    *,
    analyzer: Any | None = None,
) -> pd.DataFrame:
    """Apply VADER separately to every original, aligned headline.

    ``vader_compound`` is the headline-level sentiment used by the index. The
    negative, neutral, and positive proportions are retained in memory for
    auditability. An analyser can be injected for deterministic unit tests; the
    production default is NLTK's ``SentimentIntensityAnalyzer``.
    """
    missing = sorted(set(ALIGNED_HEADLINE_COLUMNS).difference(panel.columns))
    if missing:
        raise ValueError(f"headline panel is missing required columns: {missing}")
    if panel[ALIGNED_HEADLINE_COLUMNS].isna().any(axis=None):
        raise ValueError("aligned headline fields must not be missing")
    if not panel["headline_text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("every headline_text value must be a string")

    scored = panel.loc[:, ALIGNED_HEADLINE_COLUMNS].copy(deep=True)
    if scored.empty:
        return pd.DataFrame(columns=HEADLINE_SCORE_COLUMNS)
    resolved_analyzer = analyzer if analyzer is not None else _make_vader_analyzer()
    score_rows = [
        resolved_analyzer.polarity_scores(headline)
        for headline in scored["headline_text"]
    ]
    raw_scores = pd.DataFrame(score_rows, index=scored.index)
    required_score_fields = {"neg", "neu", "pos", "compound"}
    missing_scores = sorted(required_score_fields.difference(raw_scores.columns))
    if missing_scores:
        raise ValueError(f"VADER output is missing fields: {missing_scores}")

    renamed = raw_scores.loc[:, ["neg", "neu", "pos", "compound"]].rename(
        columns={
            "neg": "vader_negative",
            "neu": "vader_neutral",
            "pos": "vader_positive",
            "compound": "vader_compound",
        }
    )
    renamed = renamed.apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(renamed.to_numpy()).all():
        raise ValueError("VADER returned a non-finite score")
    if (renamed[["vader_negative", "vader_neutral", "vader_positive"]] < 0).any(
        axis=None
    ):
        raise ValueError("VADER component proportions must be non-negative")
    if not renamed["vader_compound"].between(-1.0, 1.0).all():
        raise ValueError("VADER compound scores must lie between -1 and 1")

    result = pd.concat([scored, renamed], axis=1)
    return result.loc[:, HEADLINE_SCORE_COLUMNS]


def sector_sentiment_index(
    scores: pd.DataFrame,
    equity_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full daily equity-sector sentiment grid.

    Headline compound scores are averaged within ticker-day first. Observed
    ticker-day means are then equally weighted within each sector-day. Missing
    ticker-days are excluded, not scored zero; a sector-day with no headlines
    keeps ``sentiment_value`` missing and has zero factual coverage counts.
    """
    missing = sorted(set(HEADLINE_SCORE_COLUMNS).difference(scores.columns))
    if missing:
        raise ValueError(f"headline scores are missing required columns: {missing}")
    universe = _equity_universe(equity_prices)
    calendar = _equity_calendar(equity_prices)
    sectors = sorted(universe["sector"].unique(), key=str)

    if not scores.empty:
        score_dates = _normalise_dates(scores["trading_date"])
        if not pd.DatetimeIndex(score_dates.unique()).isin(calendar).all():
            raise ValueError("headline score date is outside the equity calendar")
        universe_pairs = pd.MultiIndex.from_frame(universe[["ticker", "sector"]])
        score_pairs = pd.MultiIndex.from_frame(scores[["ticker", "sector"]])
        if not score_pairs.isin(universe_pairs).all():
            raise ValueError("headline scores contain a non-equity ticker-sector pair")
        compound = pd.to_numeric(scores["vader_compound"], errors="raise")
        if compound.isna().any() or not np.isfinite(compound.to_numpy()).all():
            raise ValueError("headline compound scores must be complete and finite")

        ticker_day = (
            scores.assign(
                trading_date=score_dates,
                vader_compound=compound.astype(float),
            )
            .groupby(["trading_date", "ticker", "sector"], sort=True)
            .agg(
                ticker_day_sentiment=("vader_compound", "mean"),
                headline_count=("vader_compound", "size"),
            )
            .reset_index()
        )
        observed = (
            ticker_day.groupby(["trading_date", "sector"], sort=True)
            .agg(
                sentiment_value=("ticker_day_sentiment", "mean"),
                headline_count=("headline_count", "sum"),
                ticker_with_news_count=("ticker", "nunique"),
            )
            .reset_index()
            .rename(columns={"trading_date": "date"})
        )
    else:
        observed = pd.DataFrame(
            columns=[
                "date",
                "sector",
                "sentiment_value",
                "headline_count",
                "ticker_with_news_count",
            ]
        )

    grid = pd.MultiIndex.from_product(
        [calendar, sectors], names=["date", "sector"]
    ).to_frame(index=False)
    result = grid.merge(observed, on=["date", "sector"], how="left", validate="1:1")
    sector_sizes = universe.groupby("sector")["ticker"].nunique()
    result["headline_count"] = result["headline_count"].fillna(0).astype("int64")
    result["ticker_with_news_count"] = (
        result["ticker_with_news_count"].fillna(0).astype("int64")
    )
    result["sector_ticker_count"] = (
        result["sector"].map(sector_sizes).astype("int64")
    )
    result["zero_news_ticker_count"] = (
        result["sector_ticker_count"] - result["ticker_with_news_count"]
    )
    result["sentiment_observed"] = result["ticker_with_news_count"].gt(0)

    missing_sector_news = ~result["sentiment_observed"]
    if result.loc[missing_sector_news, "sentiment_value"].notna().any():
        raise AssertionError("zero-news sector-day was assigned a sentiment value")
    if result.loc[~missing_sector_news, "sentiment_value"].isna().any():
        raise AssertionError("observed sector-day is missing its sentiment value")
    if result["date"].max() > SAMPLE_END:
        raise AssertionError("sector sentiment output exceeds 2023-12-31")

    result = result.sort_values(["date", "sector"], kind="stable").reset_index(drop=True)
    return result.loc[:, SECTOR_SENTIMENT_COLUMNS]


def lag_sector_sentiment_one_trading_day(index: pd.DataFrame) -> pd.DataFrame:
    """Express the minimum timing rule for a future trading phase.

    This helper is deliberately not called by the R3 pipeline or portfolio code.
    It maps each sector's value from the preceding observed equity trading date
    to the current decision date. Missing prior news stays missing; there is no
    neutral fill or carry-forward. A later fusion phase must use this lag (or a
    longer one), never same-day sentiment.
    """
    required = {"date", "sector", "sentiment_value"}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"sector index is missing required columns: {missing}")
    if index[["date", "sector"]].isna().any(axis=None):
        raise ValueError("sector index date and sector must not be missing")

    lagged = index.loc[:, ["date", "sector", "sentiment_value"]].copy()
    lagged["date"] = _normalise_dates(lagged["date"])
    if lagged.duplicated(["date", "sector"]).any():
        raise ValueError("sector index must be unique on date and sector")
    lagged = lagged.sort_values(["sector", "date"], kind="stable")
    lagged["sentiment_observation_date"] = lagged.groupby(
        "sector", sort=False
    )["date"].shift(1)
    lagged["lagged_sentiment_value"] = lagged.groupby(
        "sector", sort=False
    )["sentiment_value"].shift(1)
    lagged = lagged.rename(columns={"date": "decision_date"})

    usable = lagged["sentiment_observation_date"].notna()
    if not (
        lagged.loc[usable, "sentiment_observation_date"]
        < lagged.loc[usable, "decision_date"]
    ).all():
        raise AssertionError("lagged sentiment must pre-date its decision date")
    return lagged.loc[:, LAGGED_SENTIMENT_COLUMNS].reset_index(drop=True)


def build_r3_artifacts(
    equity_prices: pd.DataFrame,
    headlines: pd.DataFrame,
    *,
    analyzer: Any | None = None,
) -> R3Artifacts:
    """Build R3 scores and the descriptive index without portfolio fusion."""
    aligned = align_equity_headlines(headlines, equity_prices)
    scores = score_headlines(aligned, analyzer=analyzer)
    index = sector_sentiment_index(scores, equity_prices)
    return R3Artifacts(scores, index)


def write_r3_artifact(
    index: pd.DataFrame,
    repository_root: str | Path,
) -> Path:
    """Write the exact required R3 CSV artifact."""
    if list(index.columns) != SECTOR_SENTIMENT_COLUMNS:
        raise ValueError("sector sentiment index does not have the required schema")
    path = Path(repository_root) / "results" / "data" / "sector_sentiment_index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path
