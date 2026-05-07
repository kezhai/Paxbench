from __future__ import annotations

from pathlib import Path

from .data import DEFAULT_ID_COLUMN, DEFAULT_SPLIT_COLUMN, DEFAULT_TARGET_COLUMN


def run_zeroshot_cli(
    *,
    data: str | Path,
    output: str | Path,
    model: str,
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    limit_rows: int | None = None,
    model_name: str | None = None,
    batch_size: int = 2,
    max_length: int = 512,
) -> Path:
    from .model_backends import run_model_zeroshot

    return run_model_zeroshot(
        data=data,
        output=output,
        model=model,
        model_name=model_name,
        id_column=id_column,
        target_column=target_column,
        split_column=split_column,
        batch_size=batch_size,
        max_length=max_length,
        limit_rows=limit_rows,
    )
