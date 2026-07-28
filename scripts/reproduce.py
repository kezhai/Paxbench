#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from paxbench.aggregation import regenerate_tables
from paxbench.embeddings import run_embedding_cli
from paxbench.fusion import (
    MODEL_MODALITY,
    build_complete_case_manifest,
    feature_sets_for_models,
    file_sha256,
    load_embedding,
    molecular_levels,
    parse_feature_set,
    run_fusion_experiment,
)


REGISTERED_MODELS = tuple(MODEL_MODALITY)
DEFAULT_MODELS = ("esm", "calm", "mrnabert")
DEMO_SPLITS = ("split_species",)
DEFAULT_LEARNERS = ("ridge", "mlp")
DEFAULT_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class FusionRun:
    feature_set: str
    learner: str
    split_column: str
    seed: int

    @property
    def name(self) -> str:
        features = self.feature_set.replace("+", "__")
        return f"{features}__{self.learner}__{self.split_column}__seed{self.seed}"


def build_run_plan(
    *,
    feature_sets: tuple[str, ...],
    learners: tuple[str, ...],
    split_columns: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[FusionRun]:
    return [
        FusionRun(feature_set, learner, split_column, seed)
        for feature_set, learner, split_column, seed in product(
            feature_sets, learners, split_columns, seeds
        )
    ]


def completed_run_matches(path: Path, expected: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        config = json.loads(path.read_text())["config"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all(config.get(key) == value for key, value in expected.items())


def validate_feature_sets(
    feature_sets: tuple[str, ...],
    models: tuple[str, ...],
) -> None:
    available = set(models)
    invalid = [
        feature_set
        for feature_set in feature_sets
        if not set(parse_feature_set(feature_set)).issubset(available)
    ]
    if invalid:
        raise ValueError(
            f"Feature sets require models not listed in --models: {invalid}"
        )


def extract_embeddings(args: argparse.Namespace, cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {model: cache_dir / f"{model}.npz" for model in args.models}
    for model, path in paths.items():
        if path.exists() and not args.force:
            print(f"embedding: reuse {path}")
            continue
        print(f"embedding: extract {model} -> {path}")
        run_embedding_cli(
            data=args.data,
            output=path,
            model=model,
            batch_size=args.embedding_batch_size,
            max_length=args.max_length,
            max_sequence_length=args.max_sequence_length,
            limit_rows=args.limit_rows,
        )
    return paths


def write_fusion_sets(
    *,
    mode: str,
    feature_sets: tuple[str, ...],
    embedding_paths: dict[str, Path],
    sample_files: dict[str, Path],
    output: Path,
) -> Path:
    dimensions = {
        model: load_embedding(path)[1].shape[1]
        for model, path in embedding_paths.items()
    }
    rows = []
    for feature_set in feature_sets:
        models = parse_feature_set(feature_set)
        samples = pd.read_csv(sample_files[feature_set], sep="\t")
        rows.append(
            {
                "sample_scope": mode,
                "feature_set": feature_set,
                "models": "+".join(models),
                "sequence_levels": "+".join(molecular_levels(feature_set)),
                "model_dimensions": ";".join(
                    f"{model}:{dimensions[model]}" for model in models
                ),
                "concatenated_dimension": sum(dimensions[model] for model in models),
                "n_observations": len(samples),
                "sample_policy": (
                    "shared selected-model intersection"
                    if mode == "matched"
                    else "available for this feature set"
                ),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    return output


def select_samples(
    *,
    mode: str,
    data: str,
    feature_sets: tuple[str, ...],
    embedding_paths: dict[str, Path],
    sample_dir: Path,
) -> dict[str, Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    if mode == "matched":
        path = sample_dir / "matched.tsv"
        build_complete_case_manifest(
            data=data,
            embedding_paths=embedding_paths,
            output=path,
        )
        return {feature_set: path for feature_set in feature_sets}

    sample_files = {}
    for feature_set in feature_sets:
        path = sample_dir / f"{feature_set.replace('+', '__')}.tsv"
        selected = {
            model: embedding_paths[model]
            for model in parse_feature_set(feature_set)
        }
        build_complete_case_manifest(
            data=data,
            embedding_paths=selected,
            output=path,
        )
        sample_files[feature_set] = path
    return sample_files


def expected_run_config(
    run: FusionRun,
    *,
    data: str,
    sample_file: Path,
    embeddings: dict[str, Path],
) -> dict[str, object]:
    selected_models = parse_feature_set(run.feature_set)
    return {
        "feature_set": run.feature_set,
        "learner": run.learner,
        "split_column": run.split_column,
        "seed": run.seed,
        "data_sha256": file_sha256(data),
        "cohort_sha256": file_sha256(sample_file),
        "embedding_sha256": {
            model: file_sha256(embeddings[model])
            for model in selected_models
        },
        "requested_parameters": {
            "regressor": "canonical_frozen_regressor",
        },
    }


def run_reproduction(args: argparse.Namespace) -> None:
    models = tuple(args.models)
    feature_sets = tuple(args.feature_sets or feature_sets_for_models(models))
    validate_feature_sets(feature_sets, models)
    plan = build_run_plan(
        feature_sets=feature_sets,
        learners=tuple(args.learners),
        split_columns=tuple(args.split_columns),
        seeds=tuple(args.seeds),
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir)
    cache_dir = cache_root / "representations"
    sample_dir = cache_root / f"{args.mode}_samples"
    runs_dir = cache_root / f"{args.mode}_runs"

    print(f"Mode: {args.mode}")
    print(f"Models: {', '.join(models)}")
    print(f"Planned fusion runs: {len(plan)}")
    print(f"Selection: mean validation Spearman over {len(args.seeds)} repeats")
    if args.dry_run:
        return

    embedding_paths = extract_embeddings(args, cache_dir)
    sample_files = select_samples(
        mode=args.mode,
        data=args.data,
        feature_sets=feature_sets,
        embedding_paths=embedding_paths,
        sample_dir=sample_dir,
    )

    fusion_sets = write_fusion_sets(
        mode=args.mode,
        feature_sets=feature_sets,
        embedding_paths=embedding_paths,
        sample_files=sample_files,
        output=output_root / "fusion_sets.tsv",
    )
    print(f"fusion sets: {fusion_sets}")

    for index, run in enumerate(plan, start=1):
        metrics = runs_dir / run.name / "metrics.json"
        expected = expected_run_config(
            run,
            data=args.data,
            sample_file=sample_files[run.feature_set],
            embeddings=embedding_paths,
        )
        if completed_run_matches(metrics, expected) and not args.force:
            print(f"[{index}/{len(plan)}] skip {run.name}")
            continue
        print(f"[{index}/{len(plan)}] run  {run.name}")
        run_fusion_experiment(
            data=args.data,
            embeddings_dir=cache_dir,
            cohort_manifest=sample_files[run.feature_set],
            output_dir=runs_dir,
            feature_set=run.feature_set,
            learner=run.learner,
            split_column=run.split_column,
            seed=run.seed,
            target_column=args.target_column,
        )

    generated = regenerate_tables(
        results_dir=runs_dir,
        output_dir=output_root,
        require_seeds=len(args.seeds),
        expected_runs={
            (run.feature_set, run.learner, run.split_column, run.seed)
            for run in plan
        },
        include_diagnostics=False,
    )
    for name, path in generated.items():
        print(f"{name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract model representations, run multimodal comparisons, "
            "aggregate repeats, select on validation, and regenerate main analyses."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("paper", "matched"),
        default="paper",
        help=(
            "paper uses all observations available to each feature set; matched "
            "uses the shared selected-model intersection"
        ),
    )
    parser.add_argument("--data", default="data/demo/paxbench_demo.tsv")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=REGISTERED_MODELS,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--output-dir", default="data/results/demo")
    parser.add_argument("--cache-dir", default=".cache/paxbench")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        help="defaults to every standard combination supported by --models",
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        choices=DEFAULT_LEARNERS,
        default=list(DEFAULT_LEARNERS),
    )
    parser.add_argument(
        "--split-columns",
        nargs="+",
        default=list(DEMO_SPLITS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--target-column", default="log10_abundance")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_reproduction(parse_args())
