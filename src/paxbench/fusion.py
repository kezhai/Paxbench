from __future__ import annotations

import hashlib
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN, read_table
from .evaluation import make_regressor
from .metrics import regression_metrics
from .utils import ensure_dir, write_json


MODEL_MODALITY = {
    "esm": "protein",
    "esmc_300m": "protein",
    "esmc_600m": "protein",
    "calm": "cds_codon",
    "evo2": "cds_codon",
    "dnabert2_cds": "cds_codon",
    "mrnabert": "mrna",
    "dnabert2_mrna": "mrna",
}

DEFAULT_FEATURE_SETS = (
    "esm",
    "calm",
    "evo2",
    "mrnabert",
    "esm+calm",
    "esm+evo2",
    "esm+mrnabert",
    "calm+mrnabert",
    "evo2+mrnabert",
    "esm+calm+mrnabert",
    "esm+evo2+mrnabert",
)

VALID_PARTITIONS = ("train", "validation", "test")


def feature_sets_for_models(models: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return every single-model and cross-level combination."""
    selected = tuple(dict.fromkeys(model.lower() for model in models))
    unknown = sorted(set(selected) - set(MODEL_MODALITY))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    level_order = ("protein", "cds_codon", "mrna")
    by_level = {
        level: tuple(model for model in selected if MODEL_MODALITY[model] == level)
        for level in level_order
    }
    feature_sets = list(selected)
    present_levels = tuple(level for level in level_order if by_level[level])
    for level_count in (2, 3):
        for levels in combinations(present_levels, level_count):
            feature_sets.extend(
                "+".join(models_at_levels)
                for models_at_levels in product(*(by_level[level] for level in levels))
            )
    return tuple(feature_sets)


def parse_feature_set(value: str) -> tuple[str, ...]:
    models = tuple(part.strip().lower() for part in value.split("+") if part.strip())
    unknown = sorted(set(models) - set(MODEL_MODALITY))
    if unknown:
        raise ValueError(f"Unknown models in feature set: {unknown}")
    if not models:
        raise ValueError("Feature set is empty")
    if len(models) != len(set(models)):
        raise ValueError(f"Feature set contains duplicates: {value}")
    return models


def molecular_levels(feature_set: str) -> tuple[str, ...]:
    order = ("protein", "cds_codon", "mrna")
    present = {MODEL_MODALITY[model] for model in parse_feature_set(feature_set)}
    return tuple(level for level in order if level in present)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embedding(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(path, allow_pickle=True)
    if "ids" not in arrays or "embeddings" not in arrays:
        raise ValueError(f"{path} must contain ids and embeddings")
    ids = arrays["ids"].astype(str)
    matrix = np.asarray(arrays["embeddings"], dtype=np.float32)
    if matrix.ndim != 2 or len(ids) != len(matrix):
        raise ValueError(f"Invalid embedding shape in {path}: ids={len(ids)}, matrix={matrix.shape}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs found in {path}")
    return ids, matrix


def embedding_paths_from_dir(embeddings_dir: str | Path) -> dict[str, Path]:
    root = Path(embeddings_dir)
    return {model: root / f"{model}.npz" for model in MODEL_MODALITY}


def build_complete_case_manifest(
    *,
    data: str | Path,
    embedding_paths: dict[str, str | Path],
    output: str | Path,
    id_column: str = DEFAULT_ID_COLUMN,
) -> pd.DataFrame:
    frame = read_table(data)
    if id_column not in frame.columns:
        raise KeyError(f"Missing ID column: {id_column}")
    if frame[id_column].astype(str).duplicated().any():
        raise ValueError(f"{id_column} must be unique")

    common = set(frame[id_column].astype(str))
    availability: dict[str, set[str]] = {}
    for model, path in embedding_paths.items():
        ids, _ = load_embedding(path)
        availability[model] = set(ids)
        common &= availability[model]

    sequence_columns = {
        "esm": "protein_sequence",
        "calm": "cds_sequence",
        "evo2": "cds_sequence",
        "mrnabert": "mrna_sequence",
    }
    for model, column in sequence_columns.items():
        if column in frame.columns:
            valid = set(
                frame.loc[frame[column].fillna("").astype(str).str.len().gt(0), id_column].astype(str)
            )
            common &= valid

    manifest = frame.loc[
        frame[id_column].astype(str).isin(common),
        [id_column, "species"],
    ].copy()
    manifest[id_column] = manifest[id_column].astype(str)
    manifest = manifest.sort_values(id_column).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, sep="\t", index=False)

    coverage_rows = []
    for species, part in frame.groupby("species", dropna=False):
        row = {
            "species": species,
            "all_rows": len(part),
            "complete_case_rows": int(part[id_column].astype(str).isin(common).sum()),
        }
        for model, ids in availability.items():
            row[f"{model}_rows"] = int(part[id_column].astype(str).isin(ids).sum())
        coverage_rows.append(row)
    pd.DataFrame(coverage_rows).to_csv(
        output.with_name(output.stem + "_coverage.tsv"), sep="\t", index=False
    )
    return manifest


def _align_embedding(
    ids: np.ndarray,
    matrix: np.ndarray,
    ordered_ids: np.ndarray,
) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(ids)}
    missing = [value for value in ordered_ids if value not in lookup]
    if missing:
        raise ValueError(f"Embedding is missing {len(missing)} cohort IDs; examples: {missing[:3]}")
    indices = np.fromiter((lookup[value] for value in ordered_ids), dtype=np.int64)
    return matrix[indices]


def load_fusion_data(
    *,
    data: str | Path,
    embedding_paths: dict[str, str | Path],
    feature_set: str,
    split_column: str,
    cohort_manifest: str | Path,
    id_column: str = DEFAULT_ID_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    models = parse_feature_set(feature_set)
    frame = read_table(data)
    required = {id_column, target_column, split_column, "species"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")
    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique")

    cohort = read_table(cohort_manifest)
    if id_column not in cohort.columns:
        raise KeyError(f"Cohort manifest is missing {id_column}")
    cohort_ids = cohort[id_column].astype(str)
    if cohort_ids.duplicated().any():
        raise ValueError("Cohort manifest contains duplicate IDs")

    indexed = frame.set_index(id_column, drop=False)
    absent = cohort_ids[~cohort_ids.isin(indexed.index)]
    if not absent.empty:
        raise ValueError(f"Dataset is missing {len(absent)} cohort IDs")
    aligned = indexed.loc[cohort_ids].reset_index(drop=True)
    aligned[target_column] = pd.to_numeric(aligned[target_column], errors="coerce")
    aligned = aligned.dropna(subset=[target_column, split_column]).reset_index(drop=True)
    ordered_ids = aligned[id_column].astype(str).to_numpy()

    blocks = []
    dimensions = {}
    for model in models:
        path = embedding_paths[model]
        ids, matrix = load_embedding(path)
        block = _align_embedding(ids, matrix, ordered_ids)
        blocks.append(block)
        dimensions[model] = int(block.shape[1])
    return aligned, np.hstack(blocks).astype(np.float32, copy=False), dimensions


def run_fusion_experiment(
    *,
    data: str | Path,
    embeddings_dir: str | Path,
    cohort_manifest: str | Path,
    output_dir: str | Path,
    feature_set: str,
    learner: str,
    split_column: str,
    seed: int,
    id_column: str = DEFAULT_ID_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> Path:
    paths = embedding_paths_from_dir(embeddings_dir)
    selected_paths = {model: paths[model] for model in parse_feature_set(feature_set)}
    for path in selected_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    frame, raw_x, dimensions = load_fusion_data(
        data=data,
        embedding_paths=selected_paths,
        feature_set=feature_set,
        split_column=split_column,
        cohort_manifest=cohort_manifest,
        id_column=id_column,
        target_column=target_column,
    )
    split_values = frame[split_column].astype(str).to_numpy()
    masks = {part: split_values == part for part in VALID_PARTITIONS}
    y = {
        part: frame.loc[masks[part], target_column].to_numpy(dtype=np.float32)
        for part in VALID_PARTITIONS
    }
    empty = [part for part, mask in masks.items() if not mask.any()]
    if empty:
        raise ValueError(f"Empty partitions in selected cohort: {empty}")

    regressor = make_regressor(learner, seed=seed)
    regressor.fit(raw_x[masks["train"]], y["train"])
    predictions = {
        part: regressor.predict(raw_x[mask])
        for part, mask in masks.items()
    }

    metrics = {part: regression_metrics(y[part], predictions[part]) for part in VALID_PARTITIONS}
    run_name = f"{feature_set.replace('+', '__')}__{learner}__{split_column}__seed{seed}"
    out = ensure_dir(Path(output_dir) / run_name)
    prediction_rows = []
    for part in VALID_PARTITIONS:
        selected = frame.loc[masks[part], [id_column, "species", split_column, target_column]].copy()
        selected["prediction"] = predictions[part]
        selected["eval_split"] = part
        prediction_rows.append(selected)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        out / "predictions.tsv", sep="\t", index=False
    )

    config = {
        "feature_set": feature_set,
        "models": list(parse_feature_set(feature_set)),
        "modalities": list(molecular_levels(feature_set)),
        "learner": learner,
        "split_column": split_column,
        "seed": int(seed),
        "target_column": target_column,
        "id_column": id_column,
        "embedding_dimensions": dimensions,
        "preprocessing": {
            "raw_dimension": int(raw_x.shape[1]),
            "transformed_dimension": int(raw_x.shape[1]),
            "pipeline": "canonical_frozen_regressor",
            "dimensionality_reduction": "none",
        },
        "rows": {part: int(masks[part].sum()) for part in VALID_PARTITIONS},
        "data_sha256": file_sha256(data),
        "cohort_sha256": file_sha256(cohort_manifest),
        "embedding_sha256": {model: file_sha256(path) for model, path in selected_paths.items()},
        "requested_parameters": {"regressor": "canonical_frozen_regressor"},
    }
    write_json(
        {
            "config": config,
            "metrics": metrics,
            "learner_details": {
                "estimator": type(regressor.steps[-1][1]).__name__,
            },
        },
        out / "metrics.json",
    )
    return out
