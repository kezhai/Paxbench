from __future__ import annotations

from pathlib import Path

def run_finetuning_cli(
    *,
    data: str | Path,
    output: str | Path,
    model: str,
    epochs: int = 1,
    limit_rows: int | None = None,
    model_name: str | None = None,
    batch_size: int = 1,
    max_length: int = 512,
) -> Path:
    from .model_backends import run_model_lora

    return run_model_lora(
        data=data,
        output=output,
        model=model,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        limit_rows=limit_rows,
    )
