"""Download and validate the BRFSS 2020 heart-disease extract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heartrisk.data import dataset_summary, download_data, load_raw, numeric_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if verified.")
    args = parser.parse_args()

    path = download_data(force=args.force)
    df = load_raw(path)
    summary = dataset_summary(df)
    print("\nDataset summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("\nNumeric ranges")
    print(numeric_summary(df).to_string(index=False))


if __name__ == "__main__":
    main()
