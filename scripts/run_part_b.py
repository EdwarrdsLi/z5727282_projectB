"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import data_access  # noqa: E402


def main():
    eq = data_access.load_equity_prices()
    cr = data_access.load_crypto_prices()
    print("equities:", eq.shape, "crypto:", cr.shape)
    # TODO: returns -> out-of-sample funds + fact sheets (Station 3)
    # TODO: sentiment index + fusion extension (Station 3)
    # TODO: save figures/tables under results/ for the app and the report


if __name__ == "__main__":
    main()
