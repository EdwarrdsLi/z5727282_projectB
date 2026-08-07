"""Reproduce the currently implemented Part B artifacts from the project root.

R3 includes the R2 portfolio outputs and the descriptive sector sentiment index::

    python scripts/run_part_b.py

Later phases will extend this orchestrator with fusion, figures, and the app.
"""
from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import etl, portfolios, sentiment  # noqa: E402


def main() -> None:
    equities = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()
    news = etl.load_clean_news()
    portfolio_artifacts = portfolios.build_r2_artifacts(equities, crypto)
    portfolio_paths = portfolios.write_r2_artifacts(portfolio_artifacts, ROOT)
    sentiment_artifacts = sentiment.build_r3_artifacts(equities, news)
    sentiment_path = sentiment.write_r3_artifact(
        sentiment_artifacts.sector_sentiment_index,
        ROOT,
    )

    print(
        "R3 clean inputs:",
        f"equities={equities.shape}",
        f"crypto={crypto.shape}",
        f"news={news.shape}",
    )
    for path, frame in zip(
        portfolio_paths,
        (
            portfolio_artifacts.fund_returns,
            portfolio_artifacts.fund_weights,
            portfolio_artifacts.performance_metrics,
        ),
        strict=True,
    ):
        print(f"wrote {path.relative_to(ROOT)}: {frame.shape}")
    sector_index = sentiment_artifacts.sector_sentiment_index
    print(
        f"scored aligned equity headlines: {len(sentiment_artifacts.headline_scores):,}"
    )
    print(f"wrote {sentiment_path.relative_to(ROOT)}: {sector_index.shape}")
    print(
        "sector sentiment coverage:",
        f"dates={sector_index['date'].min().date()} to "
        f"{sector_index['date'].max().date()}",
        f"sectors={sector_index['sector'].nunique()}",
        f"observed_sector_days={int(sector_index['sentiment_observed'].sum()):,}",
        f"zero_news_sector_days={int((~sector_index['sentiment_observed']).sum()):,}",
    )
    print(
        portfolio_artifacts.performance_metrics[
            [
                "fund_name",
                "evaluation_period",
                "annualised_return",
                "annualised_volatility",
                "sharpe_ratio",
                "maximum_drawdown",
                "final_growth_of_1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
