"""Train and evaluate the early and full risk models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heartrisk import config  # noqa: E402
from heartrisk.train import train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a 30k-row subsample and a single HGB configuration for a smoke run.",
    )
    args = parser.parse_args()

    metrics = train_and_evaluate(
        models_dir=config.MODELS_DIR,
        figures_dir=config.FIGURES_DIR,
        reports_dir=config.REPORTS_DIR,
        quick=args.quick,
    )

    print("\nSummary")
    for feature_set, payload in metrics["models"].items():
        test = payload["test"]
        print(
            f"  {feature_set:5s} | PR-AUC {test['pr_auc']:.3f} | "
            f"ROC-AUC {test['roc_auc']:.3f} | sensitivity {test['sensitivity']:.3f} | "
            f"specificity {test['specificity']:.3f} | threshold {test['threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
