#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from paxbench.evaluation import run_evaluation_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Paxbench predictions or train the demo baseline.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["esm", "calm", "mrnabert", "evo2"], default="esm")
    parser.add_argument("--split-column", default="split_species")
    parser.add_argument("--target-column", default="log10_abundance")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--learner", choices=["ridge", "mlp"], default="ridge")
    args = parser.parse_args()
    metrics = run_evaluation_cli(
        output=args.output,
        data=args.data,
        predictions=args.predictions,
        model=args.model,
        split_column=args.split_column,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        limit_rows=args.limit_rows,
        learner=args.learner,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
