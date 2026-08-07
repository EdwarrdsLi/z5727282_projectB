"""Reproduce the currently implemented Part B artifacts from the project root.

R2 scope is limited to the portfolio and walk-forward backtest outputs::

    python scripts/run_part_b.py

Later phases will extend this orchestrator with sentiment, fusion, and figures.
"""
from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import etl, portfolios  # noqa: E402


def main() -> None:
    equities = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()
    artifacts = portfolios.build_r2_artifacts(equities, crypto)
    paths = portfolios.write_r2_artifacts(artifacts, ROOT)

    print(
        "R2 clean inputs:",
        f"equities={equities.shape}",
        f"crypto={crypto.shape}",
    )
    for path, frame in zip(
        paths,
        (
            artifacts.fund_returns,
            artifacts.fund_weights,
            artifacts.performance_metrics,
        ),
        strict=True,
    ):
        print(f"wrote {path.relative_to(ROOT)}: {frame.shape}")
    print(
        artifacts.performance_metrics[
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
