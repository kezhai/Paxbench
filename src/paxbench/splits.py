from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_ID_COLUMN, read_table


FUNGI = {"Saccharomyces cerevisiae", "Schizosaccharomyces pombe"}
NON_FUNGI = {
    "Arabidopsis thaliana",
    "Drosophila melanogaster",
    "Escherichia coli",
    "Homo sapiens",
    "Mus musculus",
}


def _random_labels(size: int, ratios: tuple[float, float, float], seed: int) -> np.ndarray:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to one")
    order = list(range(size))
    random.Random(seed).shuffle(order)
    n_train = int(round(size * ratios[0]))
    n_validation = int(round(size * ratios[1]))
    labels = np.full(size, "test", dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train : n_train + n_validation]] = "validation"
    return labels


def _fungi_species_labels(size: int, validation_fraction: float, seed: int) -> np.ndarray:
    labels = np.full(size, "train", dtype=object)
    validation_size = int(round(size * validation_fraction))
    validation_indices = random.Random(seed).sample(
        range(size), k=min(validation_size, size)
    )
    labels[validation_indices] = "validation"
    return labels


def add_standard_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    out = frame.copy()
    species = out["species"].astype(str)
    out["split_random"] = _random_labels(len(out), (0.8, 0.1, 0.1), seed)

    abundance = pd.to_numeric(out["log10_abundance"], errors="coerce").to_numpy()
    order = sorted(
        range(len(out)),
        key=lambda index: (
            np.inf if not np.isfinite(abundance[index]) else abundance[index],
            index,
        ),
        reverse=True,
    )
    labels = pd.Series("test", index=out.index, dtype="string")
    n_train = int(round(len(out) * 0.4))
    n_validation = int(round(len(out) * 0.1))
    labels.iloc[order[:n_train]] = "train"
    labels.iloc[order[n_train : n_train + n_validation]] = "validation"
    out["split_highlow"] = labels

    out["split_species"] = np.select(
        [species.isin(NON_FUNGI), species.eq("Schizosaccharomyces pombe")],
        ["train", "validation"],
        default="test",
    )
    out["split_species2"] = np.select(
        [species.isin(NON_FUNGI), species.eq("Saccharomyces cerevisiae")],
        ["train", "validation"],
        default="test",
    )
    out["split_species3"] = np.select(
        [
            species.isin(
                {
                    "Saccharomyces cerevisiae",
                    "Schizosaccharomyces pombe",
                    "Homo sapiens",
                    "Mus musculus",
                    "Escherichia coli",
                }
            ),
            species.eq("Drosophila melanogaster"),
        ],
        ["train", "validation"],
        default="test",
    )

    fungi_mask = species.isin(FUNGI)
    fungi_random = pd.Series(pd.NA, index=out.index, dtype="string")
    fungi_random.loc[fungi_mask] = _random_labels(int(fungi_mask.sum()), (0.8, 0.1, 0.1), seed)
    out["split_fungirandom"] = fungi_random

    fungi_species = pd.Series(pd.NA, index=out.index, dtype="string")
    budding = species.eq("Saccharomyces cerevisiae")
    budding_labels = _fungi_species_labels(int(budding.sum()), 0.15, seed)
    fungi_species.loc[budding] = budding_labels
    fungi_species.loc[species.eq("Schizosaccharomyces pombe")] = "test"
    out["split_fungi2fungi"] = fungi_species
    return out


def split_agreement(
    reference: pd.DataFrame,
    reconstructed: pd.DataFrame,
    split_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for split_column in split_columns:
        if split_column not in reference.columns or split_column not in reconstructed.columns:
            continue
        expected = reference[split_column].astype("string")
        observed = reconstructed[split_column].astype("string")
        comparable = expected.notna() & observed.notna()
        n_comparable = int(comparable.sum())
        n_agree = int(expected[comparable].eq(observed[comparable]).sum())
        rows.append(
            {
                "split_column": split_column,
                "n_comparable": n_comparable,
                "n_agree": n_agree,
                "agreement_fraction": n_agree / n_comparable if n_comparable else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_mmseqs_clustering(
    *,
    frame: pd.DataFrame,
    output_dir: str | Path,
    id_column: str = DEFAULT_ID_COLUMN,
    identity: float = 0.4,
    coverage: float = 0.8,
    threads: int = 16,
    mmseqs_binary: str = "mmseqs",
) -> Path:
    executable = shutil.which(mmseqs_binary)
    if executable is None:
        raise FileNotFoundError(f"Could not find MMseqs2 executable: {mmseqs_binary}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fasta = out / "proteins.fasta"
    with fasta.open("w") as handle:
        for _, row in frame.iterrows():
            sequence = str(row["protein_sequence"]).strip()
            if sequence:
                handle.write(f">{row[id_column]}\n{sequence}\n")
    prefix = out / "paxbench_cluster"
    tmp = out / "mmseqs_tmp"
    subprocess.run(
        [
            executable,
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(tmp),
            "--min-seq-id",
            str(identity),
            "-c",
            str(coverage),
            "--cov-mode",
            "1",
            "--threads",
            str(threads),
        ],
        check=True,
    )
    cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(cluster_tsv)
    return cluster_tsv


def assign_cluster_split(
    frame: pd.DataFrame,
    *,
    cluster_tsv: str | Path,
    id_column: str = DEFAULT_ID_COLUMN,
    seed: int = 42,
) -> pd.DataFrame:
    pairs = pd.read_csv(cluster_tsv, sep="\t", names=["cluster_id", id_column], dtype=str)
    if pairs[id_column].duplicated().any():
        raise ValueError("MMseqs2 cluster membership contains duplicate members")
    out = frame.merge(pairs, on=id_column, how="left", validate="one_to_one")
    out["cluster_id"] = out["cluster_id"].fillna(out[id_column]).astype(str)
    sizes = out.groupby("cluster_id").size().rename("size").reset_index()
    sizes = sizes.sample(frac=1.0, random_state=seed).sort_values("size", ascending=False, kind="mergesort")
    targets = {"train": 0.8 * len(out), "validation": 0.1 * len(out), "test": 0.1 * len(out)}
    counts = {key: 0 for key in targets}
    assignment = {}
    for _, row in sizes.iterrows():
        partition = max(targets, key=lambda key: targets[key] - counts[key])
        assignment[row["cluster_id"]] = partition
        counts[partition] += int(row["size"])
    out["split_cluster"] = out["cluster_id"].map(assignment)
    return out


def split_audit(frame: pd.DataFrame, split_columns: list[str]) -> pd.DataFrame:
    rows = []
    sequence_hash = pd.util.hash_pandas_object(
        frame["protein_sequence"].fillna("").astype(str), index=False
    )
    for split_column in split_columns:
        if split_column not in frame.columns:
            continue
        for partition, part in frame.groupby(split_column, dropna=False):
            rows.append(
                {
                    "split_column": split_column,
                    "partition": partition,
                    "rows": len(part),
                    "species": part["species"].nunique(),
                    "unique_protein_sequences": sequence_hash.loc[part.index].nunique(),
                }
            )
        if "cluster_id" in frame.columns and split_column == "split_cluster":
            overlap = (
                frame.dropna(subset=[split_column])
                .groupby("cluster_id")[split_column]
                .nunique()
                .gt(1)
                .sum()
            )
            if overlap:
                raise ValueError(f"{overlap} clusters cross split_cluster partitions")
    return pd.DataFrame(rows)


def audit_train_test_identity(
    *,
    frame: pd.DataFrame,
    split_columns: list[str],
    output_dir: str | Path,
    threads: int = 16,
    mmseqs_binary: str = "mmseqs",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    executable = shutil.which(mmseqs_binary)
    if executable is None:
        raise FileNotFoundError(f"Could not find MMseqs2 executable: {mmseqs_binary}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    summary_rows = []
    for split_column in split_columns:
        if split_column not in frame.columns:
            continue
        train = frame[frame[split_column].astype(str).eq("train")]
        test = frame[frame[split_column].astype(str).eq("test")]
        if train.empty or test.empty:
            continue
        split_dir = output_dir / split_column
        split_dir.mkdir(parents=True, exist_ok=True)
        train_fasta = split_dir / "train.fasta"
        test_fasta = split_dir / "test.fasta"
        for part, path in [(train, train_fasta), (test, test_fasta)]:
            with path.open("w") as handle:
                for _, row in part.iterrows():
                    handle.write(f">{row[DEFAULT_ID_COLUMN]}\n{row['protein_sequence']}\n")
        result = split_dir / "test_to_train.tsv"
        tmp = split_dir / "tmp"
        subprocess.run(
            [
                executable,
                "easy-search",
                str(test_fasta),
                str(train_fasta),
                str(result),
                str(tmp),
                "--format-output",
                "query,target,fident,alnlen,qcov,tcov,evalue,bits",
                "--max-seqs",
                "10",
                "-s",
                "7.5",
                "--threads",
                str(threads),
            ],
            check=True,
        )
        columns = ["query", "target", "identity", "alignment_length", "query_coverage", "target_coverage", "evalue", "bits"]
        matches = pd.read_csv(result, sep="\t", names=columns)
        if matches.empty:
            best = pd.DataFrame({"query": test[DEFAULT_ID_COLUMN].astype(str), "identity": np.nan})
        else:
            matches["identity"] = pd.to_numeric(matches["identity"], errors="coerce")
            if matches["identity"].max() > 1.0:
                matches["identity"] /= 100.0
            best = matches.sort_values(
                ["query", "identity", "bits"], ascending=[True, False, False]
            ).groupby("query", as_index=False).head(1)
        best["split_column"] = split_column
        long_rows.append(best)
        values = best["identity"].dropna()
        summary_rows.append(
            {
                "split_column": split_column,
                "n_test": len(test),
                "n_with_match": len(values),
                "identity_median": values.median(),
                "identity_q90": values.quantile(0.9),
                "identity_max": values.max(),
                "fraction_ge_0_4": (values >= 0.4).mean(),
                "fraction_ge_0_5": (values >= 0.5).mean(),
                "fraction_ge_0_7": (values >= 0.7).mean(),
                "fraction_ge_0_9": (values >= 0.9).mean(),
            }
        )
    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    return long, pd.DataFrame(summary_rows)


def build_splits(
    *,
    data: str | Path,
    output: str | Path,
    audit_output: str | Path,
    agreement_output: str | Path | None = None,
    cluster_output_dir: str | Path | None = None,
    cluster_tsv: str | Path | None = None,
    seed: int = 42,
    threads: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_table(data)
    frame[DEFAULT_ID_COLUMN] = frame[DEFAULT_ID_COLUMN].astype(str)
    reference = frame.copy()
    frame = add_standard_splits(frame, seed=seed)
    reconstructed_columns = [
        "split_random",
        "split_species",
        "split_species2",
        "split_species3",
        "split_fungirandom",
        "split_fungi2fungi",
        "split_highlow",
    ]
    if cluster_tsv is None and cluster_output_dir is not None:
        cluster_tsv = run_mmseqs_clustering(
            frame=frame,
            output_dir=cluster_output_dir,
            threads=threads,
        )
    if cluster_tsv is not None:
        frame = assign_cluster_split(frame, cluster_tsv=cluster_tsv, seed=seed)
        reconstructed_columns.append("split_cluster")
    audit = split_audit(
        frame,
        [
            "split_random",
            "split_cluster",
            "split_species",
            "split_species2",
            "split_species3",
            "split_fungirandom",
            "split_fungi2fungi",
            "split_highlow",
        ],
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix in {".pkl", ".pickle"}:
        frame.to_pickle(output)
    elif output.suffix == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, sep="\t", index=False)
    audit_output = Path(audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_output, sep="\t", index=False)
    if agreement_output is not None:
        agreement = split_agreement(
            reference,
            frame,
            reconstructed_columns,
        )
        agreement_output = Path(agreement_output)
        agreement_output.parent.mkdir(parents=True, exist_ok=True)
        agreement.to_csv(agreement_output, sep="\t", index=False)
    return frame, audit
