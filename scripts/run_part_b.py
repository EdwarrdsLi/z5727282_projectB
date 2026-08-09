"""Reproduce the currently implemented Part B artifacts from the project root.

R5 includes the unchanged R2 portfolios, R3 descriptive sentiment index, R4
equity sentiment-fusion evidence, and downstream report exhibits and fact sheets::

    python scripts/run_part_b.py

The R5 renderer reads only the precomputed R2/R3 CSVs written earlier in this
command.  It does not rerun or alter portfolio, sentiment, or fusion rules.
"""
from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import etl, exhibits, fusion, portfolios, sentiment  # noqa: E402


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
    fusion_artifacts = fusion.build_r4_artifacts(
        equities,
        sentiment_artifacts.sector_sentiment_index,
    )
    fusion_paths = fusion.write_r4_artifacts(fusion_artifacts, ROOT)
    r5_tables, r5_paths, r5_manifest = (
        exhibits.build_and_write_r5_from_precomputed_results(ROOT)
    )

    print(
        "R4 clean inputs:",
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
    for path, frame in zip(
        fusion_paths[:6],
        (
            fusion_artifacts.fusion_returns,
            fusion_artifacts.fusion_weights,
            fusion_artifacts.fusion_comparison,
            fusion_artifacts.fusion_turnover,
            fusion_artifacts.fusion_cost_check,
            fusion_artifacts.fusion_sensitivity,
        ),
        strict=True,
    ):
        print(f"wrote {path.relative_to(ROOT)}: {frame.shape}")
    print(f"wrote {fusion_paths[6].relative_to(ROOT)}")
    print(
        fusion_artifacts.fusion_comparison[
            [
                "strategy_name",
                "evaluation_period",
                "annualised_return",
                "annualised_volatility",
                "sharpe_ratio",
                "maximum_drawdown",
                "final_growth_of_1",
                "total_one_way_turnover",
                "higher_annualised_return_than_base",
            ]
        ].to_string(index=False)
    )
    print(
        "transaction-cost schedule (one-way bps):",
        sorted(fusion_artifacts.fusion_cost_check["cost_rate_bps_one_way"].unique()),
    )
    print(
        fusion_artifacts.fusion_sensitivity[
            [
                "tilt_strength",
                "approved_design",
                "annualised_return",
                "sharpe_ratio",
                "total_one_way_turnover",
            ]
        ].to_string(index=False)
    )
    for path in r5_paths:
        print(f"wrote {path.relative_to(ROOT)}")
    print(
        "R5 report artifacts:",
        f"exhibits={len(r5_manifest)}",
        f"fact_sheets={len(r5_tables.fact_sheet_summary)}",
        "student_interpretation=PENDING",
        "final_figure_acceptance=PENDING",
    )


if __name__ == "__main__":
    main()
