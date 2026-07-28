#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paxbench.metrics import regression_metrics


DEFAULT_GROUPS = ("esm", "best_two_level", "best_three_level")
BIN_EDGES = (-np.inf, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, np.inf)
BIN_LABELS = ("<20%", "20-30%", "30-40%", "40-50%", "50-70%", "70-90%", ">=90%")


def index_prediction_files(results_dir: str | Path) -> dict[tuple[str, str, str, int], Path]:
    index: dict[tuple[str, str, str, int], Path] = {}
    for metrics_path in sorted(Path(results_dir).glob("**/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        config = payload.get("config", {})
        required = {"split_column", "feature_set", "learner", "seed"}
        if not required.issubset(config):
            continue
        predictions = metrics_path.with_name("predictions.tsv")
        if not predictions.exists():
            raise FileNotFoundError(predictions)
        key = (
            str(config["split_column"]),
            str(config["feature_set"]),
            str(config["learner"]),
            int(config["seed"]),
        )
        if key in index:
            raise ValueError(f"Duplicate prediction run for {key}")
        index[key] = predictions
    if not index:
        raise FileNotFoundError(f"No prediction runs found under {results_dir}")
    return index


def summarize_identity_performance(
    *,
    identity_path: str | Path,
    selection_path: str | Path,
    results_dir: str | Path,
    selection_groups: tuple[str, ...] = DEFAULT_GROUPS,
    id_column: str = "paxbench_id",
    target_column: str = "log10_abundance",
    require_seeds: int = 3,
    min_bin_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identity = pd.read_csv(identity_path, sep="\t")
    required_identity = {"query", "identity", "split_column"}
    missing = sorted(required_identity - set(identity.columns))
    if missing:
        raise KeyError(f"Identity table is missing columns: {missing}")
    identity["query"] = identity["query"].astype(str)
    if identity.duplicated(["split_column", "query"]).any():
        raise ValueError("Identity table contains duplicate split/query rows")

    selection = pd.read_csv(selection_path, sep="\t")
    prediction_index = index_prediction_files(results_dir)
    rows = []
    for split_column in sorted(identity["split_column"].astype(str).unique()):
        split_identity = identity[identity["split_column"].eq(split_column)][
            ["query", "identity"]
        ]
        for group in selection_groups:
            chosen = selection[
                selection["split_column"].eq(split_column)
                & selection["selection_group"].eq(group)
            ]
            if len(chosen) != 1:
                raise ValueError(f"Expected one selection row for {split_column}/{group}")
            chosen = chosen.iloc[0]
            feature_set = str(chosen["selected_feature_set"])
            learner = str(chosen["selected_learner"])
            for seed in range(42, 42 + require_seeds):
                key = (split_column, feature_set, learner, seed)
                if key not in prediction_index:
                    raise FileNotFoundError(f"Missing predictions for {key}")
                predictions = pd.read_csv(prediction_index[key], sep="\t")
                predictions = predictions[predictions["eval_split"].eq("test")].copy()
                required_prediction = {id_column, target_column, "prediction"}
                absent = sorted(required_prediction - set(predictions.columns))
                if absent:
                    raise KeyError(
                        f"{prediction_index[key]} is missing columns: {absent}"
                    )
                predictions[id_column] = predictions[id_column].astype(str)
                if predictions[id_column].duplicated().any():
                    raise ValueError(f"Duplicate test IDs in {prediction_index[key]}")
                merged = predictions.merge(
                    split_identity,
                    left_on=id_column,
                    right_on="query",
                    how="left",
                    validate="one_to_one",
                )
                merged["identity_bin"] = pd.cut(
                    pd.to_numeric(merged["identity"], errors="coerce"),
                    bins=BIN_EDGES,
                    labels=BIN_LABELS,
                    right=False,
                ).astype("string").fillna("No match")
                for identity_bin, part in merged.groupby(
                    "identity_bin", observed=True, sort=False
                ):
                    metrics = regression_metrics(
                        pd.to_numeric(part[target_column], errors="coerce").to_numpy(),
                        pd.to_numeric(part["prediction"], errors="coerce").to_numpy(),
                    )
                    if metrics["n"] < min_bin_n:
                        metrics["spearman"] = np.nan
                        metrics["pearson"] = np.nan
                    rows.append(
                        {
                            "split_column": split_column,
                            "selection_group": group,
                            "feature_set": feature_set,
                            "learner": learner,
                            "seed": seed,
                            "identity_bin": identity_bin,
                            **metrics,
                        }
                    )

    long = pd.DataFrame(rows)
    keys = [
        "split_column",
        "selection_group",
        "feature_set",
        "learner",
        "identity_bin",
    ]
    summary = (
        long.groupby(keys, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n=("n", "min"),
        )
    )
    incomplete = summary[summary["n_seeds"].ne(require_seeds)]
    if not incomplete.empty:
        raise ValueError("Identity-bin summaries do not contain all required seeds")

    gains = []
    protein = long[long["selection_group"].eq("esm")]
    for group in selection_groups:
        if group == "esm":
            continue
        fusion = long[long["selection_group"].eq(group)]
        paired = fusion.merge(
            protein,
            on=["split_column", "seed", "identity_bin"],
            suffixes=("_fusion", "_protein"),
            validate="one_to_one",
        )
        paired["spearman_gain"] = paired["spearman_fusion"] - paired["spearman_protein"]
        paired["mae_change"] = paired["mae_fusion"] - paired["mae_protein"]
        paired["fusion_group"] = group
        gains.append(
            paired[
                [
                    "split_column",
                    "fusion_group",
                    "seed",
                    "identity_bin",
                    "spearman_gain",
                    "mae_change",
                    "n_fusion",
                ]
            ].rename(columns={"n_fusion": "n"})
        )
    gain_long = pd.concat(gains, ignore_index=True)
    gain_summary = (
        gain_long.groupby(
            ["split_column", "fusion_group", "identity_bin"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            spearman_gain_mean=("spearman_gain", "mean"),
            spearman_gain_std=("spearman_gain", "std"),
            mae_change_mean=("mae_change", "mean"),
            mae_change_std=("mae_change", "std"),
            n=("n", "min"),
        )
    )
    return long, summary, gain_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratify validation-selected test performance by nearest-training identity."
    )
    parser.add_argument("--identity-path", required=True)
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--id-column", default="paxbench_id")
    parser.add_argument("--target-column", default="log10_abundance")
    parser.add_argument("--require-seeds", type=int, default=3)
    parser.add_argument("--min-bin-n", type=int, default=20)
    args = parser.parse_args()

    groups = tuple(value.strip() for value in args.selection_groups.split(",") if value.strip())
    long, summary, gains = summarize_identity_performance(
        identity_path=args.identity_path,
        selection_path=args.selection_path,
        results_dir=args.results_dir,
        selection_groups=groups,
        id_column=args.id_column,
        target_column=args.target_column,
        require_seeds=args.require_seeds,
        min_bin_n=args.min_bin_n,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    long.to_csv(output / "identity_stratified_performance.tsv", sep="\t", index=False)
    summary.to_csv(output / "identity_stratified_summary.tsv", sep="\t", index=False)
    gains.to_csv(output / "identity_stratified_fusion_gain.tsv", sep="\t", index=False)
    print(gains.to_string(index=False))


if __name__ == "__main__":
    main()
