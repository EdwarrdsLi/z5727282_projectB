"""Part A return, calendar-alignment, and headline-panel features.

Inputs are the cleaned in-memory panels produced by :mod:`src.etl`. Equity and
cryptocurrency returns are calculated separately on their native calendars
before cross-asset alignment. Headline work is limited to exact text assembly;
sentiment scoring and signal lags remain future Project B work.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


EQUITY_ASSET_CLASS = "equity"
CRYPTO_ASSET_CLASS = "cryptocurrency"

DESCRIPTIVE_STAT_COLUMNS = [
    "asset_class",
    "valid_return_observations",
    "arithmetic_mean_daily_return",
    "daily_standard_deviation",
    "annualised_volatility",
    "minimum_daily_return",
    "maximum_daily_return",
    "skewness",
    "return_frequency",
    "calendar_convention",
    "annualisation_days_per_year",
    "annualisation_factor",
    "statistics_basis",
]

HEADLINE_PANEL_COLUMNS = [
    "trading_date",
    "ticker",
    "sector",
    "headline_text",
    "headline_count",
    "first_source_date",
    "last_source_date",
]

HEADLINE_AUDIT_COLUMNS = [
    "input_headline_count",
    "input_source_date_min",
    "input_source_date_max",
    "same_day_mapping_count",
    "next_trading_day_mapping_count",
    "unmapped_headline_count",
    "unmapped_source_date_min",
    "unmapped_source_date_max",
    "output_headline_count",
    "output_group_count",
    "output_trading_date_min",
    "output_trading_date_max",
    "headline_count_reconciles",
    "exact_headline_text_preserved",
    "date_normalisation",
    "trading_calendar_rule",
]


def _normalise_date_series(values: pd.Series) -> pd.Series:
    """Convert timestamps through UTC to timezone-naive calendar dates."""
    return (
        pd.to_datetime(values, errors="raise", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
    *,
    asset_class: str | None = None,
) -> pd.DataFrame:
    """Calculate native-calendar simple returns separately within each ticker.

    The formula is ``adjClose[t] / adjClose[t-1] - 1`` after stable sorting by
    ticker and date. The first return for each ticker remains missing because no
    earlier observation exists. Additional missing returns raise an error rather
    than being silently dropped or filled.
    """
    if price_col != "adjClose":
        raise ValueError("daily_returns must use adjClose")

    required = {"date", "ticker", "adjClose"}
    missing_columns = sorted(required.difference(prices.columns))
    if missing_columns:
        raise ValueError(f"prices is missing required columns: {missing_columns}")

    returned = prices.copy(deep=True)
    returned["date"] = _normalise_date_series(returned["date"])
    if returned[["ticker", "date"]].isna().any(axis=None):
        raise ValueError("ticker and date must not be missing")
    if returned.duplicated(["ticker", "date"]).any():
        raise ValueError("prices must be unique on ticker + date")

    returned["adjClose"] = pd.to_numeric(returned["adjClose"], errors="raise")
    returned = returned.sort_values(
        ["ticker", "date"], kind="stable"
    ).reset_index(drop=True)
    grouped_prices = returned.groupby("ticker", sort=False)["adjClose"]
    returned["simple_return"] = grouped_prices.pct_change(fill_method=None)

    first_return = returned.groupby("ticker", sort=False).cumcount().eq(0)
    additional_missing = returned["simple_return"].isna() & ~first_return
    if additional_missing.any():
        raise ValueError(
            "additional missing returns require investigation: "
            f"{int(additional_missing.sum())} row(s)"
        )

    inferred_asset_class = (
        EQUITY_ASSET_CLASS if "sector" in returned.columns else CRYPTO_ASSET_CLASS
    )
    returned["asset_class"] = asset_class or inferred_asset_class

    columns = ["date", "ticker", "simple_return", "asset_class"]
    if "sector" in returned.columns:
        columns.append("sector")
    return returned.loc[:, columns]


def _wide_returns(return_panel: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Pivot a unique long return panel to deterministic date/ticker columns."""
    wide = return_panel.pivot(index="date", columns="ticker", values="simple_return")
    ordered_tickers = sorted(wide.columns, key=str)
    wide = wide.reindex(columns=ordered_tickers)
    wide.columns = [f"{prefix}{ticker}" for ticker in ordered_tickers]
    return wide


def build_combined_returns_panel(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build a return panel on exactly the observed equity calendar.

    Equity and cryptocurrency returns are calculated independently first.
    Already-calculated cryptocurrency returns are then left-joined by date to
    the observed equity trading calendar. No prices or returns are forward-filled.
    """
    equity_returns = daily_returns(
        equity_prices,
        asset_class=EQUITY_ASSET_CLASS,
    )
    crypto_returns = daily_returns(
        crypto_prices,
        asset_class=CRYPTO_ASSET_CLASS,
    )

    equity_wide = _wide_returns(equity_returns, "equity_return__")
    crypto_wide = _wide_returns(crypto_returns, "crypto_return__")
    combined = equity_wide.join(crypto_wide, how="left").reset_index()
    combined = combined.sort_values("date", kind="stable").reset_index(drop=True)

    expected_dates = pd.DatetimeIndex(sorted(equity_returns["date"].unique()))
    actual_dates = pd.DatetimeIndex(combined["date"])
    if not actual_dates.equals(expected_dates):
        raise AssertionError("combined panel does not exactly match the equity calendar")
    return combined


def descriptive_return_statistics(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise native-calendar returns with asset-specific conventions."""
    conventions = [
        (
            EQUITY_ASSET_CLASS,
            equity_returns,
            252,
            "Observed equity trading days",
            "Daily on each equity's observed trading dates",
        ),
        (
            CRYPTO_ASSET_CLASS,
            crypto_returns,
            365,
            "Native seven-day cryptocurrency calendar",
            "Daily on each cryptocurrency's seven-day calendar",
        ),
    ]
    rows: list[dict[str, object]] = []
    for asset_class, frame, annualisation_days, calendar, frequency in conventions:
        if "simple_return" not in frame.columns:
            raise ValueError(f"{asset_class} return panel lacks simple_return")
        returns = pd.to_numeric(frame["simple_return"], errors="raise")
        valid = returns.dropna()
        if np.isinf(valid.to_numpy()).any():
            raise ValueError(f"{asset_class} returns contain infinite values")

        daily_volatility = valid.std(ddof=1)
        annualisation_factor = float(np.sqrt(annualisation_days))
        rows.append(
            {
                "asset_class": asset_class,
                "valid_return_observations": int(valid.count()),
                "arithmetic_mean_daily_return": valid.mean(),
                "daily_standard_deviation": daily_volatility,
                "annualised_volatility": daily_volatility * annualisation_factor,
                "minimum_daily_return": valid.min(),
                "maximum_daily_return": valid.max(),
                "skewness": valid.skew(),
                "return_frequency": frequency,
                "calendar_convention": calendar,
                "annualisation_days_per_year": annualisation_days,
                "annualisation_factor": annualisation_factor,
                "statistics_basis": (
                    "Native-calendar daily simple returns from adjClose"
                ),
            }
        )
    return pd.DataFrame(rows, columns=DESCRIPTIVE_STAT_COLUMNS)


def _normalise_trading_calendar(
    equity_trading_calendar: pd.DataFrame
    | pd.Series
    | pd.Index
    | Iterable[object],
) -> pd.DatetimeIndex:
    if isinstance(equity_trading_calendar, pd.DataFrame):
        if "date" not in equity_trading_calendar.columns:
            raise ValueError("equity trading calendar DataFrame requires a date column")
        values = equity_trading_calendar["date"]
    elif isinstance(equity_trading_calendar, (pd.Series, pd.Index)):
        values = pd.Series(equity_trading_calendar)
    else:
        values = pd.Series(list(equity_trading_calendar))

    if values.empty:
        raise ValueError("equity trading calendar must not be empty")
    calendar = pd.DatetimeIndex(
        _normalise_date_series(values).dropna().unique()
    ).sort_values()
    if calendar.empty:
        raise ValueError("equity trading calendar has no valid dates")
    return calendar


def _align_and_assemble_headlines(
    headlines: pd.DataFrame,
    equity_trading_calendar: pd.DataFrame
    | pd.Series
    | pd.Index
    | Iterable[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", "ticker", "sector", "title"}
    missing_columns = sorted(required.difference(headlines.columns))
    if missing_columns:
        raise ValueError(f"headlines is missing required columns: {missing_columns}")
    if headlines["title"].isna().any():
        raise ValueError("headline title must not be missing")

    calendar = _normalise_trading_calendar(equity_trading_calendar)
    aligned = headlines.copy(deep=True)
    aligned["source_date"] = _normalise_date_series(aligned["date"])
    aligned["_input_order"] = np.arange(len(aligned))

    positions = calendar.searchsorted(aligned["source_date"], side="left")
    can_map = positions < len(calendar)
    mapped_dates = pd.Series(pd.NaT, index=aligned.index, dtype="datetime64[ns]")
    if can_map.any():
        mapped_dates.loc[can_map] = calendar.take(positions[can_map]).to_numpy()
    aligned["trading_date"] = mapped_dates

    same_day = can_map & aligned["trading_date"].eq(
        aligned["source_date"]
    ).to_numpy()
    aligned["mapping_status"] = "unmapped_after_last_trading_date"
    aligned.loc[can_map, "mapping_status"] = "next_observed_trading_day"
    aligned.loc[same_day, "mapping_status"] = "same_observed_trading_day"

    mapped = aligned.loc[can_map].copy()
    mapped = mapped.sort_values(
        ["trading_date", "ticker", "sector", "source_date", "title", "_input_order"],
        kind="stable",
    )

    if mapped.empty:
        panel = pd.DataFrame(columns=HEADLINE_PANEL_COLUMNS)
        grouped_titles: list[str] = []
    else:
        grouped = (
            mapped.groupby(
                ["trading_date", "ticker", "sector"],
                sort=True,
                dropna=False,
            )
            .agg(
                _headline_list=("title", list),
                headline_count=("title", "size"),
                first_source_date=("source_date", "min"),
                last_source_date=("source_date", "max"),
            )
            .reset_index()
        )
        grouped_titles = [
            title
            for title_list in grouped["_headline_list"]
            for title in title_list
        ]
        grouped["headline_text"] = grouped["_headline_list"].map("\n".join)
        panel = grouped.loc[:, HEADLINE_PANEL_COLUMNS]

    input_count = len(aligned)
    output_headline_count = (
        int(panel["headline_count"].sum()) if not panel.empty else 0
    )
    exact_text_preserved = grouped_titles == mapped["title"].tolist()
    unmapped = aligned.loc[~can_map]
    audit = pd.DataFrame(
        [
            {
                "input_headline_count": input_count,
                "input_source_date_min": aligned["source_date"].min(),
                "input_source_date_max": aligned["source_date"].max(),
                "same_day_mapping_count": int(same_day.sum()),
                "next_trading_day_mapping_count": int((can_map & ~same_day).sum()),
                "unmapped_headline_count": int((~can_map).sum()),
                "unmapped_source_date_min": unmapped["source_date"].min(),
                "unmapped_source_date_max": unmapped["source_date"].max(),
                "output_headline_count": output_headline_count,
                "output_group_count": len(panel),
                "output_trading_date_min": panel["trading_date"].min(),
                "output_trading_date_max": panel["trading_date"].max(),
                "headline_count_reconciles": (
                    output_headline_count + int((~can_map).sum()) == input_count
                ),
                "exact_headline_text_preserved": exact_text_preserved,
                "date_normalisation": "UTC to timezone-naive calendar date",
                "trading_calendar_rule": (
                    "Same observed equity trading day, otherwise next observed "
                    "equity trading day; never backward"
                ),
            }
        ],
        columns=HEADLINE_AUDIT_COLUMNS,
    )
    return panel, audit


def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_trading_calendar: pd.DataFrame
    | pd.Series
    | pd.Index
    | Iterable[object],
    *,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Map cleaned headlines forward to observed equity dates and assemble text.

    Same-day headlines remain on that date. Every other mappable headline moves
    to the next observed equity trading date. Headlines after the last trading
    date are excluded from the panel and counted in the optional audit.
    """
    panel, audit = _align_and_assemble_headlines(
        headlines,
        equity_trading_calendar,
    )
    if return_audit:
        return panel, audit
    return panel
