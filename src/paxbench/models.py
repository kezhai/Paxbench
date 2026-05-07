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
    "calm": ModelSpec("calm", "cds_sequence", "multimolecule/calm", "cds", ("query", "value")),
    "mrnabert": ModelSpec("mrnabert", "mrnabert_sequence", "YYLY66/mRNABERT", "mrna", ("Wqkv",)),
    "evo2": ModelSpec("evo2", "cds_sequence", "evo2_7b_base", "cds", ("Wqkv",)),
}


def get_model_spec(model: str) -> ModelSpec:
    key = model.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]
