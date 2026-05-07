from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_ID_COLUMN = "paxbench_id"
DEFAULT_TARGET_COLUMN = "log10_abundance"
DEFAULT_SPLIT_COLUMN = "split_species"


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t", low_memory=False)
    elif suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    elif suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    elif suffixes := path.suffixes:
        if suffixes[-2:] == [".tsv", ".gz"]:
            frame = pd.read_csv(path, sep="\t", low_memory=False)
        else:
            raise ValueError(f"Unsupported table format: {path}")
    else:
        raise ValueError(f"Unsupported table format: {path}")
    if "paxbench_id" not in frame.columns and "obs_id" in frame.columns:
        frame = frame.copy()
        frame["paxbench_id"] = frame["obs_id"].astype(str)
    if "obs_id" not in frame.columns and "paxbench_id" in frame.columns:
        frame = frame.copy()
        frame["obs_id"] = frame["paxbench_id"].astype(str)
    return frame


def validate_table(
    frame: pd.DataFrame,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
) -> None:
    required = [id_column, target_column, split_column, "species", "protein_sequence"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.notna().sum() == 0:
        raise ValueError(f"No numeric values found in target column {target_column!r}")
    split_values = set(frame[split_column].astype(str))
    needed = {"train", "validation", "test"}
    if not needed.issubset(split_values):
        raise ValueError(f"{split_column!r} must contain train, validation, and test")


def clean_sequence(value: object, alphabet: set[str] | None = None) -> str:
    if value is None or pd.isna(value):
        return ""
    seq = str(value).upper().replace(" ", "").replace("\n", "").strip()
    if alphabet is not None:
        seq = "".join(ch for ch in seq if ch in alphabet)
    return seq


def split_frame(
    frame: pd.DataFrame,
    *,
    split_column: str = DEFAULT_SPLIT_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame[split_column].astype(str) == "train"].copy()
    validation = frame[frame[split_column].astype(str) == "validation"].copy()
    test = frame[frame[split_column].astype(str) == "test"].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"{split_column!r} must contain non-empty train, validation, and test rows")
    return train, validation, test
