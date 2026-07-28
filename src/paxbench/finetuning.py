from __future__ import annotations

from pathlib import Path

def run_finetuning_cli(
    *,
    data: str | Path,
    output: str | Path,
    model: str,
    target_column: str = "log10_abundance",
    split_column: str = "split_species",
    epochs: int = 1,
    limit_rows: int | None = None,
    model_name: str | None = None,
    batch_size: int = 1,
    max_length: int = 512,
    seed: int = 42,
) -> Path:
    from .model_backends import run_model_lora

    return run_model_lora(
        data=data,
        output=output,
        model=model,
        model_name=model_name,
        target_column=target_column,
        split_column=split_column,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
        limit_rows=limit_rows,
    )
