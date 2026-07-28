#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paxbench.embeddings import run_embedding_cli
from paxbench.evaluation import fit_embedding_ml
from paxbench.finetuning import run_finetuning_cli
from paxbench.zeroshot import run_zeroshot_cli


MODE_CHOICES = ("embedding", "zeroshot", "lora")
MODEL_CHOICES = ("esm", "calm", "mrnabert", "evo2")


def read_test_metrics(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, sep="\t")
    if "test" in set(frame["split"].astype(str)):
        row = frame[frame["split"].astype(str) == "test"].iloc[0]
    else:
        row = frame.iloc[-1]
    return {key: row[key] for key in frame.columns if key != "split"}


def run_mode(
    *,
    mode: str,
    model: str,
    data: str,
    output_root: Path,
    split_column: str,
    target_column: str,
    limit_rows: int | None,
    epochs: int,
    learner: str,
) -> dict[str, object]:
    output = output_root / mode / split_column / (
        f"{model}_{learner}" if mode == "embedding" else model
    )
    if mode == "embedding":
        embedding_path = output_root / "embeddings" / f"{model}.npz"
        run_embedding_cli(
            data=data,
            output=embedding_path,
            model=model,
            limit_rows=limit_rows,
        )
        fit_embedding_ml(
            data=data,
            embeddings=embedding_path,
            output=output,
            model=model,
            split_column=split_column,
            target_column=target_column,
            learner=learner,
        )
    elif mode == "zeroshot":
        run_zeroshot_cli(
            data=data,
            output=output,
            model=model,
            split_column=split_column,
            target_column=target_column,
            limit_rows=limit_rows,
            batch_size=2,
        )
    elif mode == "lora":
        run_finetuning_cli(
            data=data,
            output=output,
            model=model,
            target_column=target_column,
            split_column=split_column,
            epochs=epochs,
            limit_rows=limit_rows,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    row = {"mode": mode, "model": model, **read_test_metrics(output / "metrics.tsv")}
    if mode == "embedding":
        row["learner"] = learner
    return row


def failure_row(*, mode: str, model: str, learner: str, error: Exception) -> dict[str, object]:
    row: dict[str, object] = {
        "mode": mode,
        "model": model,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }
    if mode == "embedding":
        row["learner"] = learner
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paxbench models and benchmark modes.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODE_CHOICES, required=True)
    parser.add_argument("--split-column", default="split_species")
    parser.add_argument("--target-column", default="log10_abundance")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learners", nargs="+", choices=["ridge", "mlp"], default=["ridge"])
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in args.modes:
        for model in args.models:
            learners = args.learners if mode == "embedding" else ["ridge"]
            for learner in learners:
                try:
                    row = run_mode(
                        mode=mode,
                        model=model,
                        data=args.data,
                        output_root=output_root,
                        split_column=args.split_column,
                        target_column=args.target_column,
                        limit_rows=args.limit_rows,
                        epochs=args.epochs,
                        learner=learner,
                    )
                    row["status"] = "ok"
                    rows.append(row)
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append(failure_row(mode=mode, model=model, learner=learner, error=exc))
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "scores.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
