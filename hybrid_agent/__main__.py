from __future__ import annotations

import argparse
import json
import sys

from .orchestrator import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid agentic system: find literature, datasets, models; run cloud inference.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="Generate and reconstruct MNIST digits with a variational autoencoder",
        help="Natural-language research/ML task",
    )
    parser.add_argument(
        "--out",
        default="runs/latest",
        help="Output directory for plan, candidates, and inference artifacts",
    )
    args = parser.parse_args(argv)

    report = run_pipeline(args.task, out_dir=args.out)
    print(json.dumps(report.to_dict(), indent=2))
    if report.errors and report.inference is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
