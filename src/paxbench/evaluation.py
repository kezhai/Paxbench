from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    DEFAULT_ID_COLUMN,
    DEFAULT_SPLIT_COLUMN,
    DEFAULT_TARGET_COLUMN,
    read_table,
    split_frame,
    validate_table,
)
from .embeddings import extract_sequence_feature_embeddings
from .metrics import regression_metrics
from .utils import ensure_dir, write_json


def make_regressor(learner: str, *, seed: int = 42):
    if learner == "ridge":
        estimator = Ridge(alpha=1.0)
    elif learner == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=(64,),
            max_iter=500,
            random_state=seed,
        )
    else:
        raise ValueError("learner must be 'ridge' or 'mlp'")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def prediction_frame(
    part: pd.DataFrame,
    *,
    predictions: np.ndarray,
    target_column: str,
    split_column: str,
    id_column: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            id_column: part[id_column].astype(str).to_numpy(),
            "species": part["species"].astype(str).to_numpy(),
            split_column: part[split_column].astype(str).to_numpy(),
            target_column: pd.to_numeric(part[target_column], errors="coerce").to_numpy(dtype=float),
            "prediction": predictions.astype(float),
        }
    )


def fit_frozen_ml(
    frame: pd.DataFrame,
    *,
    model: str = "esm",
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    learner: str = "ridge",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    validate_table(frame, id_column=id_column, target_column=target_column, split_column=split_column)
    train, validation, test = split_frame(frame, split_column=split_column)
    _, x_train = extract_sequence_feature_embeddings(train, model=model, id_column=id_column)
    _, x_validation = extract_sequence_feature_embeddings(validation, model=model, id_column=id_column)
    _, x_test = extract_sequence_feature_embeddings(test, model=model, id_column=id_column)
    y_train = pd.to_numeric(train[target_column], errors="coerce").to_numpy(dtype=float)

    regressor = make_regressor(learner, seed=seed)
    regressor.fit(x_train, y_train)

    predictions = []
    metrics = {}
    for name, part, matrix in [
        ("train", train, x_train),
        ("validation", validation, x_validation),
        ("test", test, x_test),
    ]:
        pred = regressor.predict(matrix)
        actual = pd.to_numeric(part[target_column], errors="coerce").to_numpy(dtype=float)
        metrics[name] = regression_metrics(actual, pred)
        predictions.append(
            prediction_frame(
                part,
                predictions=pred,
                target_column=target_column,
                split_column=split_column,
                id_column=id_column,
            )
        )
    return pd.concat(predictions, ignore_index=True), metrics


def fit_embedding_ml(
    *,
    data: str | Path,
    embeddings: str | Path,
    model: str,
    output: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    learner: str = "ridge",
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    frame = read_table(data)
    if id_column not in frame.columns and "paxbench_id" in frame.columns:
        id_column = "paxbench_id"
    arrays = np.load(embeddings, allow_pickle=True)
    ids = arrays["ids"].astype(str)
    matrix = np.asarray(arrays["embeddings"], dtype=np.float32)
    features = pd.DataFrame({"_feature_row": np.arange(len(ids)), id_column: ids})
    merged = frame.merge(features, on=id_column, how="inner", validate="one_to_one")
    merged["_feature_row"] = merged["_feature_row"].astype(int)
    x_all = matrix[merged["_feature_row"].to_numpy()]
    y_all = pd.to_numeric(merged[target_column], errors="coerce").to_numpy(dtype=float)
    split_values = merged[split_column].astype(str).to_numpy()

    train_mask = split_values == "train"
    regressor = make_regressor(learner, seed=seed)
    regressor.fit(x_all[train_mask], y_all[train_mask])

    rows = []
    metrics = {}
    for split in ["train", "validation", "test"]:
        mask = split_values == split
        part = merged.loc[mask].copy()
        pred = regressor.predict(x_all[mask])
        metrics[split] = regression_metrics(part[target_column].to_numpy(dtype=float), pred)
        rows.append(
            prediction_frame(
                part,
                predictions=pred,
                target_column=target_column,
                split_column=split_column,
                id_column=id_column,
            )
        )
    out = ensure_dir(output)
    pd.concat(rows, ignore_index=True).to_csv(out / "predictions.tsv", sep="\t", index=False)
    write_json({"model": model, "learner": learner, "metrics": metrics}, out / "metrics.json")
    pd.DataFrame([{"split": split, **values} for split, values in metrics.items()]).to_csv(
        out / "metrics.tsv", sep="\t", index=False
    )
    return metrics


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    prediction_column: str = "prediction",
) -> dict[str, dict[str, float]]:
    needed = [target_column, split_column, prediction_column]
    missing = [column for column in needed if column not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction table missing columns: {missing}")
    metrics = {}
    for split, part in predictions.groupby(predictions[split_column].astype(str)):
        metrics[str(split)] = regression_metrics(
            pd.to_numeric(part[target_column], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(part[prediction_column], errors="coerce").to_numpy(dtype=float),
        )
    return metrics


def run_evaluation_cli(
    *,
    output: str | Path,
    data: str | Path | None = None,
    predictions: str | Path | None = None,
    model: str = "esm",
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    prediction_column: str = "prediction",
    limit_rows: int | None = None,
    learner: str = "ridge",
) -> dict[str, dict[str, float]]:
    out = ensure_dir(output)
    if predictions:
        pred = read_table(predictions)
        metrics = evaluate_predictions(
            pred, target_column=target_column, split_column=split_column, prediction_column=prediction_column
        )
        pred.to_csv(out / "predictions.tsv", sep="\t", index=False)
    elif data:
        frame = read_table(data)
        if limit_rows is not None:
            frame = frame.head(limit_rows).copy()
        pred, metrics = fit_frozen_ml(
            frame,
            model=model,
            target_column=target_column,
            split_column=split_column,
            id_column=id_column,
            learner=learner,
        )
        pred.to_csv(out / "predictions.tsv", sep="\t", index=False)
    else:
        raise ValueError("Provide either --data or --predictions")
    write_json({"model": model, "learner": learner, "metrics": metrics}, out / "metrics.json")
    pd.DataFrame(
        [{"split": split, **values} for split, values in metrics.items()]
    ).to_csv(out / "metrics.tsv", sep="\t", index=False)
    return metrics
