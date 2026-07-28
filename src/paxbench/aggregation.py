from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .fusion import molecular_levels, parse_feature_set


def _one_sample_pvalue(values: pd.Series) -> float:
    data = values.to_numpy(dtype=float)
    if len(data) < 2:
        return np.nan
    if np.allclose(data, data[0]):
        return 1.0 if np.isclose(data[0], 0.0) else 0.0
    return float(stats.ttest_1samp(data, 0.0).pvalue)


def collect_run_metrics(
    results_dir: str | Path,
    expected_runs: set[tuple[str, str, str, int]] | None = None,
) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob("**/metrics.json")):
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        if "feature_set" not in config:
            continue
        for eval_split, values in payload.get("metrics", {}).items():
            rows.append(
                {
                    "result_file": str(path),
                    "feature_set": config["feature_set"],
                    "learner": config["learner"],
                    "split_column": config["split_column"],
                    "seed": int(config["seed"]),
                    "modalities": "+".join(config.get("modalities", molecular_levels(config["feature_set"]))),
                    "eval_split": eval_split,
                    **values,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No fusion metrics.json files found under {results_dir}")
    frame = pd.DataFrame(rows)
    run_columns = ["feature_set", "learner", "split_column", "seed"]
    duplicate_columns = run_columns + ["eval_split"]
    duplicates = frame[frame.duplicated(duplicate_columns, keep=False)]
    if not duplicates.empty:
        examples = duplicates[duplicate_columns + ["result_file"]].head(10)
        raise ValueError(f"Duplicate fusion results: {examples.to_dict('records')}")
    if expected_runs is not None:
        observed = set(frame[run_columns].itertuples(index=False, name=None))
        missing = sorted(expected_runs - observed)
        unexpected = sorted(observed - expected_runs)
        if missing or unexpected:
            raise ValueError(
                "Fusion run mismatch; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
    return frame


def aggregate_three_seeds(long: pd.DataFrame, require_seeds: int = 3) -> pd.DataFrame:
    keys = ["split_column", "feature_set", "learner", "modalities", "eval_split"]
    grouped = (
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
    incomplete = grouped[grouped["n_seeds"] != require_seeds]
    if not incomplete.empty:
        examples = incomplete[keys + ["n_seeds"]].head(10).to_dict("records")
        raise ValueError(f"Expected {require_seeds} seeds for every run group; examples: {examples}")
    return grouped


def _validation_candidates(
    summary: pd.DataFrame,
    split_column: str,
    *,
    feature_sets: set[str] | None = None,
    level_count: int | None = None,
) -> pd.DataFrame:
    candidates = summary[
        summary["split_column"].eq(split_column)
        & summary["eval_split"].eq("validation")
    ].copy()
    if feature_sets is not None:
        candidates = candidates[candidates["feature_set"].isin(feature_sets)]
    if level_count is not None:
        candidates = candidates[
            candidates["feature_set"].map(lambda value: len(molecular_levels(str(value))) == level_count)
        ]
    return candidates


def validation_selected_rows(
    summary: pd.DataFrame,
    *,
    split_column: str,
    selection_group: str,
    feature_sets: set[str] | None = None,
    level_count: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    candidates = _validation_candidates(
        summary,
        split_column,
        feature_sets=feature_sets,
        level_count=level_count,
    )
    if candidates.empty:
        raise ValueError(f"No validation candidates for {split_column}/{selection_group}")
    selected = candidates.sort_values(
        ["spearman_mean", "rmse_mean", "feature_set", "learner"],
        ascending=[False, True, True, True],
    ).iloc[0]
    test = summary[
        summary["split_column"].eq(split_column)
        & summary["eval_split"].eq("test")
        & summary["feature_set"].eq(selected["feature_set"])
        & summary["learner"].eq(selected["learner"])
    ]
    if len(test) != 1:
        raise ValueError(f"Expected one locked test row for {split_column}/{selection_group}")
    return selected, test.iloc[0]


def build_main_table(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = set(summary["feature_set"].astype(str))
    single_models = sorted(
        feature_set
        for feature_set in observed
        if len(parse_feature_set(feature_set)) == 1
    )
    level_counts = {
        len(molecular_levels(feature_set))
        for feature_set in observed
    }
    table_rows = []
    selection_rows = []
    for split_column in sorted(summary["split_column"].unique()):
        groups = [(model, {model}, None) for model in single_models]
        groups.extend(
            (f"best_{level}_level", None, level)
            for level in (2, 3)
            if level in level_counts
        )
        row: dict[str, object] = {"split_column": split_column}
        for label, feature_sets, level_count in groups:
            validation, test = validation_selected_rows(
                summary,
                split_column=split_column,
                selection_group=label,
                feature_sets=feature_sets,
                level_count=level_count,
            )
            row[f"{label}_spearman"] = test["spearman_mean"]
            row[f"{label}_sd"] = test["spearman_std"]
            row[f"{label}_n"] = int(test["n"])
            row[f"{label}_configuration"] = f"{test['feature_set']}|{test['learner']}"
            selection_rows.append(
                {
                    "split_column": split_column,
                    "selection_group": label,
                    "selected_feature_set": validation["feature_set"],
                    "selected_learner": validation["learner"],
                    "validation_spearman": validation["spearman_mean"],
                    "test_spearman": test["spearman_mean"],
                    "test_spearman_std": test["spearman_std"],
                    "n_test": int(test["n"]),
                }
            )
        table_rows.append(row)
    return pd.DataFrame(table_rows), pd.DataFrame(selection_rows)


def build_incremental_and_ablation_tables(
    long: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    incremental_rows = []
    ablation_rows = []
    observed = set(summary["feature_set"].astype(str))
    available_levels = sorted(
        {len(molecular_levels(feature_set)) for feature_set in observed}
    )
    for split_column in sorted(summary["split_column"].unique()):
        previous_test = None
        for level_count in available_levels:
            validation, test = validation_selected_rows(
                summary,
                split_column=split_column,
                selection_group=f"best_{level_count}_level",
                level_count=level_count,
            )
            incremental_rows.append(
                {
                    "split_column": split_column,
                    "level_count": level_count,
                    "selected_feature_set": validation["feature_set"],
                    "selected_learner": validation["learner"],
                    "validation_spearman": validation["spearman_mean"],
                    "test_spearman": test["spearman_mean"],
                    "test_spearman_std": test["spearman_std"],
                    "incremental_gain": (
                        np.nan
                        if previous_test is None
                        else test["spearman_mean"] - previous_test["spearman_mean"]
                    ),
                    "n_test": int(test["n"]),
                }
            )
            previous_test = test

        if 3 not in available_levels:
            continue
        full_validation, full_test = validation_selected_rows(
            summary,
            split_column=split_column,
            selection_group="best_three_level",
            level_count=3,
        )
        full = str(full_validation["feature_set"])
        full_scores = long[
            long["split_column"].eq(split_column)
            & long["eval_split"].eq("test")
            & long["feature_set"].eq(full)
            & long["learner"].eq(full_validation["learner"])
        ][["seed", "spearman"]]
        for removed in ("protein", "cds_codon", "mrna"):
            retained = {"protein", "cds_codon", "mrna"} - {removed}
            reduced_sets = {
                feature_set
                for feature_set in observed
                if set(molecular_levels(feature_set)) == retained
            }
            if not reduced_sets:
                continue
            reduced_validation, reduced_test = validation_selected_rows(
                summary,
                split_column=split_column,
                selection_group=f"without_{removed}",
                feature_sets=reduced_sets,
            )
            reduced_scores = long[
                long["split_column"].eq(split_column)
                & long["eval_split"].eq("test")
                & long["feature_set"].eq(reduced_validation["feature_set"])
                & long["learner"].eq(reduced_validation["learner"])
            ][["seed", "spearman"]]
            paired = full_scores.merge(
                reduced_scores,
                on="seed",
                suffixes=("_full", "_reduced"),
                validate="one_to_one",
            )
            deltas = paired["spearman_full"] - paired["spearman_reduced"]
            ablation_rows.append(
                {
                    "split_column": split_column,
                    "removed_modality": removed,
                    "full_feature_set": full,
                    "reduced_feature_set": reduced_validation["feature_set"],
                    "full_test_spearman": full_test["spearman_mean"],
                    "reduced_test_spearman": reduced_test["spearman_mean"],
                    "delta_spearman": full_test["spearman_mean"] - reduced_test["spearman_mean"],
                    "paired_seed_delta_mean": deltas.mean(),
                    "paired_t_pvalue": _one_sample_pvalue(deltas),
                    "n_seeds": len(deltas),
                    "n_test": int(full_test["n"]),
                }
            )
    return pd.DataFrame(incremental_rows), pd.DataFrame(ablation_rows)


def build_pairwise_consistency_table(
    long: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    observed_pairs = sorted(
        feature_set
        for feature_set in summary["feature_set"].astype(str).unique()
        if len(parse_feature_set(feature_set)) == 2
        and len(molecular_levels(feature_set)) == 2
    )
    for split_column in sorted(summary["split_column"].unique()):
        for pair in observed_pairs:
            members = set(pair.split("+"))
            pair_validation, pair_test = validation_selected_rows(
                summary,
                split_column=split_column,
                selection_group=f"pair:{pair}",
                feature_sets={pair},
            )
            single_validation, single_test = validation_selected_rows(
                summary,
                split_column=split_column,
                selection_group=f"pair_components:{pair}",
                feature_sets=members,
            )
            pair_scores = long[
                long["split_column"].eq(split_column)
                & long["eval_split"].eq("test")
                & long["feature_set"].eq(pair)
                & long["learner"].eq(pair_validation["learner"])
            ][["seed", "spearman"]]
            single_scores = long[
                long["split_column"].eq(split_column)
                & long["eval_split"].eq("test")
                & long["feature_set"].eq(single_validation["feature_set"])
                & long["learner"].eq(single_validation["learner"])
            ][["seed", "spearman"]]
            paired = pair_scores.merge(
                single_scores,
                on="seed",
                suffixes=("_pair", "_single"),
                validate="one_to_one",
            )
            deltas = paired["spearman_pair"] - paired["spearman_single"]
            rows.append(
                {
                    "split_column": split_column,
                    "pair": pair,
                    "selected_pair_learner": pair_validation["learner"],
                    "validation_pair_spearman": pair_validation["spearman_mean"],
                    "test_pair_spearman": pair_test["spearman_mean"],
                    "test_pair_sd": pair_test["spearman_std"],
                    "validation_selected_component": single_validation["feature_set"],
                    "selected_component_learner": single_validation["learner"],
                    "validation_component_spearman": single_validation["spearman_mean"],
                    "test_component_spearman": single_test["spearman_mean"],
                    "test_component_sd": single_test["spearman_std"],
                    "test_pair_gain": pair_test["spearman_mean"] - single_test["spearman_mean"],
                    "paired_seed_gain": deltas.mean(),
                    "paired_t_pvalue": _one_sample_pvalue(deltas),
                    "n_seeds": len(deltas),
                    "n_test": int(pair_test["n"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["holm_pvalue_within_split"] = np.nan
    for _, indices in result.groupby("split_column").groups.items():
        ordered = result.loc[indices, "paired_t_pvalue"].sort_values().index
        adjusted = []
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            value = min(1.0, (count - rank) * result.at[index, "paired_t_pvalue"])
            running = max(running, value)
            adjusted.append((index, running))
        for index, value in adjusted:
            result.at[index, "holm_pvalue_within_split"] = value
    return result


def regenerate_tables(
    *,
    results_dir: str | Path,
    output_dir: str | Path,
    require_seeds: int = 3,
    expected_runs: set[tuple[str, str, str, int]] | None = None,
    include_diagnostics: bool = True,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    long = collect_run_metrics(results_dir, expected_runs=expected_runs)
    summary = aggregate_three_seeds(long, require_seeds=require_seeds)
    main, selection = build_main_table(summary)
    outputs = {
        "long": out / "runs.tsv",
        "summary": out / "repeated_results.tsv",
        "selection": out / "validation_selection.tsv",
        "main": out / "results.tsv",
    }
    long.to_csv(outputs["long"], sep="\t", index=False)
    summary.to_csv(outputs["summary"], sep="\t", index=False)
    selection.to_csv(outputs["selection"], sep="\t", index=False)
    main.to_csv(outputs["main"], sep="\t", index=False)
    if include_diagnostics:
        incremental, ablation = build_incremental_and_ablation_tables(long, summary)
        pairwise = build_pairwise_consistency_table(long, summary)
        outputs.update(
            {
                "incremental": out / "incremental_fusion_gains.tsv",
                "ablation": out / "leave_one_modality_out.tsv",
                "pairwise": out / "pairwise_fusion_consistency.tsv",
            }
        )
        incremental.to_csv(outputs["incremental"], sep="\t", index=False)
        ablation.to_csv(outputs["ablation"], sep="\t", index=False)
        pairwise.to_csv(outputs["pairwise"], sep="\t", index=False)
    return outputs
