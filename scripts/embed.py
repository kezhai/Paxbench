#!/usr/bin/env python
from __future__ import annotations

import argparse

from paxbench.embeddings import run_embedding_cli
from paxbench.models import MODEL_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Paxbench embeddings.")
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), default="esm")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    path = run_embedding_cli(
        data=args.data,
        output=args.output,
        model=args.model,
        limit_rows=args.limit_rows,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Saved embeddings to {path}")


if __name__ == "__main__":
    main()
