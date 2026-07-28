#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from paxbench.data import read_table


BEST_HIT_COLUMNS = (
    "target",
    "local_identity",
    "global_identity_shorter",
    "global_identity_longer",
    "global_identity_mean",
    "query_coverage",
    "target_coverage",
    "minimum_coverage",
    "alignment_length",
    "evalue",
    "bits",
)


def clean_protein_sequence(value: object) -> str:
    return "".join(str(value).upper().replace("*", "").split())


def normalize_fraction(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.le(1.0), numeric / 100.0)


def write_fasta(
    frame: pd.DataFrame,
    *,
    path: Path,
    id_column: str,
    sequence_column: str,
    prefix: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("w") as handle:
        for position, row in enumerate(frame.itertuples(index=False)):
            internal_id = f"{prefix}{position}"
            observation_id = str(getattr(row, id_column))
            sequence = str(getattr(row, sequence_column))
            mapping[internal_id] = observation_id
            handle.write(f">{internal_id}\n{sequence}\n")
    return mapping


def read_hits(
    path: Path,
    *,
    query_map: dict[str, str],
    target_map: dict[str, str],
) -> pd.DataFrame:
    columns = [
        "query_internal",
        "target_internal",
        "pident",
        "alignment_length",
        "query_length",
        "target_length",
        "query_coverage",
        "target_coverage",
        "evalue",
        "bits",
    ]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    hits = pd.read_csv(path, sep="\t", names=columns)
    for column in [
        "pident",
        "alignment_length",
        "query_length",
        "target_length",
        "query_coverage",
        "target_coverage",
        "evalue",
        "bits",
    ]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits = hits.dropna(
        subset=["pident", "alignment_length", "query_length", "target_length"]
    ).copy()
    hits["local_identity"] = normalize_fraction(hits["pident"])
    hits["query_coverage"] = normalize_fraction(hits["query_coverage"])
    hits["target_coverage"] = normalize_fraction(hits["target_coverage"])
    hits["minimum_coverage"] = hits[
        ["query_coverage", "target_coverage"]
    ].min(axis=1)
    hits["n_identical"] = hits["local_identity"] * hits["alignment_length"]
    hits["global_identity_shorter"] = hits["n_identical"] / hits[
        ["query_length", "target_length"]
    ].min(axis=1)
    hits["global_identity_longer"] = hits["n_identical"] / hits[
        ["query_length", "target_length"]
    ].max(axis=1)
    hits["global_identity_mean"] = (
        2.0 * hits["n_identical"] / (hits["query_length"] + hits["target_length"])
    )
    hits["query"] = hits["query_internal"].astype(str).map(query_map)
    hits["target"] = hits["target_internal"].astype(str).map(target_map)
    return hits.dropna(subset=["query", "target"]).copy()


def best_hit_per_query(hits: pd.DataFrame) -> pd.DataFrame:
    if hits.empty:
        return hits
    return (
        hits.sort_values(
            [
                "query",
                "global_identity_shorter",
                "minimum_coverage",
                "bits",
            ],
            ascending=[True, False, False, False],
        )
        .groupby("query", as_index=False)
        .head(1)
    )


def merge_best_hits(
    queries: pd.DataFrame,
    hits: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    if hits.empty:
        return queries
    columns = ["query", *BEST_HIT_COLUMNS]
    renamed = hits[columns].rename(
        columns={column: f"{prefix}_{column}" for column in BEST_HIT_COLUMNS}
    )
    return queries.merge(renamed, on="query", how="left", validate="one_to_one")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def audit_canonical_split(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    split_column: str = "split_cluster",
    id_column: str = "paxbench_id",
    sequence_column: str = "protein_sequence",
    coverage_threshold: float = 0.8,
    search_min_identity: float = 0.2,
    sensitivity: float = 7.5,
    max_hits: int = 100,
    threads: int = 16,
    mmseqs_binary: str = "mmseqs",
) -> dict[str, Path]:
    executable = shutil.which(mmseqs_binary)
    if executable is None:
        raise FileNotFoundError(f"Could not find MMseqs2 executable: {mmseqs_binary}")

    frame = read_table(data_path).copy()
    required = {id_column, sequence_column, split_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Dataset is missing columns: {missing}")
    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique")
    frame[sequence_column] = frame[sequence_column].map(clean_protein_sequence)
    frame = frame[frame[sequence_column].str.len().gt(0)].copy()

    canonical = frame[frame[split_column].isin(["train", "validation", "test"])].copy()
    train = canonical[canonical[split_column].eq("train")].copy()
    test = canonical[canonical[split_column].eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError(f"{split_column} must contain non-empty train and test partitions")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    work = output / "mmseqs_work"
    work.mkdir(parents=True, exist_ok=True)
    train_fasta = work / "canonical_train.fasta"
    test_fasta = work / "canonical_test.fasta"
    query_map = write_fasta(
        test,
        path=test_fasta,
        id_column=id_column,
        sequence_column=sequence_column,
        prefix="q",
    )
    target_map = write_fasta(
        train,
        path=train_fasta,
        id_column=id_column,
        sequence_column=sequence_column,
        prefix="t",
    )

    raw_hits_path = work / "canonical_test_to_train.tsv"
    tmp = work / "tmp"
    command = [
        executable,
        "easy-search",
        str(test_fasta),
        str(train_fasta),
        str(raw_hits_path),
        str(tmp),
        "--min-seq-id",
        str(search_min_identity),
        "-s",
        str(sensitivity),
        "--threads",
        str(threads),
        "--max-seqs",
        str(max_hits),
        "--format-output",
        "query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue,bits",
    ]
    subprocess.run(command, check=True)
    hits = read_hits(raw_hits_path, query_map=query_map, target_map=target_map)
    best_any = best_hit_per_query(hits)
    coverage_hits = hits[hits["minimum_coverage"].ge(coverage_threshold)].copy()
    best_coverage = best_hit_per_query(coverage_hits)

    test_sequences = test.set_index(id_column)[sequence_column]
    train_sequence_set = set(train[sequence_column])
    long = pd.DataFrame({"query": test[id_column].astype(str)})
    long["split_column"] = split_column
    long["exact_train_sequence_match"] = long["query"].map(
        lambda value: test_sequences[value] in train_sequence_set
    )
    long = merge_best_hits(long, best_any, prefix="best_any")
    long = merge_best_hits(long, best_coverage, prefix="coverage_filtered")

    long["identity"] = long.get(
        "coverage_filtered_global_identity_shorter",
        pd.Series(np.nan, index=long.index),
    )
    n_test = len(long)
    identity = pd.to_numeric(long["identity"], errors="coerce")
    summary: dict[str, object] = {
        "split_column": split_column,
        "identity_definition": "global_identity_shorter",
        "coverage_definition": "min(query_coverage,target_coverage)",
        "coverage_threshold": coverage_threshold,
        "n_train": len(train),
        "n_test": n_test,
        "n_with_any_hit": int(long.get("best_any_target", pd.Series()).notna().sum()),
        "n_with_coverage_hit": int(identity.notna().sum()),
        "fraction_with_coverage_hit": float(identity.notna().mean()),
        "n_exact_sequence_overlap": int(long["exact_train_sequence_match"].sum()),
        "fraction_exact_sequence_overlap": float(
            long["exact_train_sequence_match"].mean()
        ),
        "identity_median_coverage_filtered": float(identity.median()),
        "identity_q90_coverage_filtered": float(identity.quantile(0.9)),
        "identity_max_coverage_filtered": float(identity.max()),
    }
    for threshold in (0.4, 0.5, 0.7, 0.9):
        label = str(threshold).replace(".", "_")
        summary[f"n_test_identity_ge_{label}"] = int(identity.ge(threshold).sum())
        summary[f"fraction_test_identity_ge_{label}"] = float(
            identity.ge(threshold).sum() / n_test
        )

    manifest = canonical[[id_column, split_column]].copy()
    manifest["protein_sequence_sha256"] = canonical[sequence_column].map(sequence_sha256)
    manifest_path = output / "canonical_split_manifest.tsv"
    long_path = output / "canonical_test_to_train_identity.tsv"
    summary_path = output / "canonical_identity_summary.tsv"
    metadata_path = output / "canonical_identity_run_metadata.json"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    long.to_csv(long_path, sep="\t", index=False)
    pd.DataFrame([summary]).to_csv(summary_path, sep="\t", index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "data_path": str(Path(data_path).resolve()),
                "split_column": split_column,
                "mmseqs_binary": executable,
                "mmseqs_version": subprocess.run(
                    [executable, "version"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "command": command,
                **summary,
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "manifest": manifest_path,
        "long": long_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit train-test identity using a saved canonical split column."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-column", default="split_cluster")
    parser.add_argument("--id-column", default="paxbench_id")
    parser.add_argument("--sequence-column", default="protein_sequence")
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--search-min-identity", type=float, default=0.2)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--max-hits", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--mmseqs-binary", default="mmseqs")
    args = parser.parse_args()
    outputs = audit_canonical_split(
        data_path=args.data,
        output_dir=args.output_dir,
        split_column=args.split_column,
        id_column=args.id_column,
        sequence_column=args.sequence_column,
        coverage_threshold=args.coverage_threshold,
        search_min_identity=args.search_min_identity,
        sensitivity=args.sensitivity,
        max_hits=args.max_hits,
        threads=args.threads,
        mmseqs_binary=args.mmseqs_binary,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
