from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    sequence_column: str
    default_model_name: str
    modality: str
    target_modules: tuple[str, ...]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "esm": ModelSpec("esm", "protein_sequence", "facebook/esm2_t33_650M_UR50D", "protein", ("query", "value")),
    "esmc_300m": ModelSpec("esmc_300m", "protein_sequence", "biohub/ESMC-300M", "protein", ("query", "value")),
    "esmc_600m": ModelSpec("esmc_600m", "protein_sequence", "biohub/ESMC-600M", "protein", ("query", "value")),
    "calm": ModelSpec("calm", "cds_sequence", "multimolecule/calm", "cds", ("query", "value")),
    "dnabert2_cds": ModelSpec("dnabert2_cds", "cds_sequence", "zhihan1996/DNABERT-2-117M", "cds", ("query", "value")),
    "mrnabert": ModelSpec("mrnabert", "mrnabert_sequence", "YYLY66/mRNABERT", "mrna", ("Wqkv",)),
    "dnabert2_mrna": ModelSpec("dnabert2_mrna", "mrna_sequence", "zhihan1996/DNABERT-2-117M", "mrna", ("query", "value")),
    "evo2": ModelSpec("evo2", "cds_sequence", "evo2_7b_base", "cds", ("Wqkv",)),
}


def get_model_spec(model: str) -> ModelSpec:
    key = model.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]
