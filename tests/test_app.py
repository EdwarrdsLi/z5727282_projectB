"""Focused tests for the results-only Streamlit investor app."""
from __future__ import annotations

import ast
import pathlib
import sys
import unittest

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import app_data  # noqa: E402


class TestAppDataLoading(unittest.TestCase):
    def test_loads_required_precomputed_results_with_approved_dates(self) -> None:
        loaded = app_data.load_app_data(ROOT)

        self.assertEqual(set(loaded), set(app_data.APP_CSV_PATHS))
        self.assertEqual(
            set(loaded["fund_returns"]["fund_name"]),
            {"Combined Equal Weight", "Combined Risk Parity"},
        )
        self.assertLessEqual(
            loaded["fund_returns"]["date"].max(), app_data.MAX_APPROVED_DATE
        )
        self.assertLessEqual(
            loaded["sector_sentiment"]["date"].max(), app_data.MAX_APPROVED_DATE
        )
        self.assertTrue(all(path.is_file() for path in app_data.figure_paths(ROOT).values()))

    def test_date_cap_rejects_post_sample_artifact(self) -> None:
        future = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]})
        with self.assertRaisesRegex(ValueError, "2023-12-31 sample cap"):
            app_data._validate_date_cap(future, "fund_returns")

    def test_historical_allocation_uses_saved_growth_paths(self) -> None:
        fund_returns = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-02", "2023-01-03"] * 2),
                "fund_name": ["Fund A", "Fund A", "Fund B", "Fund B"],
                "growth_of_1": [1.10, 1.20, 0.90, 1.00],
            }
        )
        paths, summary = app_data.historical_allocation_scenarios(
            fund_returns, 1_000.0, {"Fund A": 0.25, "Fund B": 0.75}
        )

        self.assertAlmostEqual(paths["Custom allocation"].iloc[-1], 1_050.0)
        custom = summary.loc[summary["Scenario"].eq("Custom allocation")].iloc[0]
        self.assertAlmostEqual(custom["Historical ending value"], 1_050.0)
        self.assertAlmostEqual(custom["Historical return"], 0.05)


class TestAppSectionsAndDeploymentBoundary(unittest.TestCase):
    def test_streamlit_app_renders_required_investor_sections(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0, [item.value for item in app.exception])
        subheaders = {item.value for item in app.subheader}
        required = {
            "Compare completed fund methods",
            "Fund fact sheets and current target holdings",
            "Historical allocation lab",
            "Growth, drawdown, risk-return and portfolio weights",
            "Sector sentiment index and approved fusion comparison",
            "Method notes and limits",
        }
        self.assertTrue(required.issubset(subheaders))

    def test_app_runtime_has_no_raw_data_or_model_build_imports(self) -> None:
        forbidden_modules = {
            "nltk", "src.data_access", "src.etl", "src.features", "src.portfolios",
            "src.sentiment", "src.fusion", "src.exhibits",
        }
        for path in (ROOT / "streamlit_app.py", ROOT / "src" / "app_data.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            called: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        called.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        called.add(node.func.attr)
            self.assertTrue(forbidden_modules.isdisjoint(imported), (path, imported))
            self.assertTrue(
                {"load_equity_prices", "load_crypto_prices", "load_news_headlines",
                 "run_part_b", "SentimentIntensityAnalyzer"}.isdisjoint(called),
                (path, called),
            )

        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("nltk", "\n".join(
            line for line in requirements.splitlines() if not line.lstrip().startswith("#")
        ))
        self.assertTrue(
            all(path.startswith("results/") for path in app_data.APP_CSV_PATHS.values())
        )
        self.assertTrue(
            all(path.startswith("results/") for path in app_data.APP_FIGURE_PATHS.values())
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
