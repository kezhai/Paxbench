#!/usr/bin/env python
from __future__ import annotations

import argparse

from paxbench.finetuning import run_finetuning_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified Paxbench finetuning.")
    parser.add_argument("--model", choices=["esm", "calm", "mrnabert", "evo2"], default="esm")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    out = run_finetuning_cli(
        data=args.data,
        output=args.output,
        model=args.model,
        epochs=args.epochs,
        limit_rows=args.limit_rows,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Saved finetuning outputs to {out}")


if __name__ == "__main__":
    main()
