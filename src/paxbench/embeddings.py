from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_ID_COLUMN, clean_sequence, read_table
from .models import get_model_spec

AA = "ACDEFGHIKLMNPQRSTVWY"
DNA = "ACGT"


def sequence_feature_matrix(sequences: list[str], *, alphabet: str = AA) -> np.ndarray:
    rows = []
    letters = list(alphabet)
    for raw in sequences:
        seq = clean_sequence(raw, set(letters))
        length = max(len(seq), 1)
        counts = np.array([seq.count(ch) / length for ch in letters], dtype=np.float32)
        gc = 0.0
        if set(letters) == set(DNA):
            gc = (seq.count("G") + seq.count("C")) / length
        rows.append(np.concatenate([[np.log1p(length), gc], counts]).astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def extract_sequence_feature_embeddings(
    frame: pd.DataFrame,
    *,
    model: str,
    id_column: str = DEFAULT_ID_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    spec = get_model_spec(model)
    column = spec.sequence_column if spec.sequence_column in frame.columns else "protein_sequence"
    alphabet = DNA if spec.modality in {"cds", "mrna"} else AA
    ids = frame[id_column].astype(str).to_numpy() if id_column in frame.columns else np.arange(len(frame)).astype(str)
    embeddings = sequence_feature_matrix(frame[column].fillna("").astype(str).tolist(), alphabet=alphabet)
    return ids, embeddings


def save_npz(ids: np.ndarray, embeddings: np.ndarray, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=ids.astype(str), embeddings=embeddings.astype(np.float32))


def run_embedding_cli(
    *,
    data: str | Path,
    output: str | Path,
    model: str,
    id_column: str = DEFAULT_ID_COLUMN,
    limit_rows: int | None = None,
    model_name: str | None = None,
    batch_size: int = 2,
    max_length: int = 512,
    max_sequence_length: int | None = None,
) -> Path:
    from .model_backends import extract_model_embeddings

    return extract_model_embeddings(
        data=data,
        output=output,
        model=model,
        model_name=model_name,
        id_column=id_column,
        batch_size=batch_size,
        max_length=max_length,
        max_sequence_length=max_sequence_length,
        limit_rows=limit_rows,
    )
