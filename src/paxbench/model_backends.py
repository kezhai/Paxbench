from __future__ import annotations

import builtins
import contextlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_ID_COLUMN, DEFAULT_SPLIT_COLUMN, DEFAULT_TARGET_COLUMN, clean_sequence, read_table, split_frame
from .evaluation import evaluate_predictions
from .metrics import regression_metrics
from .models import get_model_spec
from .utils import ensure_dir, write_json

RNA_ALPHABET = set("ACGTU")


def require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Model mode requires torch.") from exc
    return torch


def require_transformers():
    try:
        from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Model mode requires transformers.") from exc
    return AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer


def load_checkpoint_state_dict(model_name: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors_file
        from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("mRNABERT manual checkpoint loading requires huggingface_hub and safetensors.") from exc

    try:
        weights_path = hf_hub_download(model_name, SAFE_WEIGHTS_NAME)
        return load_safetensors_file(weights_path, device="cpu")
    except Exception as safetensors_exc:
        try:
            weights_path = hf_hub_download(model_name, WEIGHTS_NAME)
        except Exception as torch_exc:
            raise RuntimeError(f"Could not download weights for {model_name}.") from torch_exc
        print(f"[model] safetensors load failed ({safetensors_exc}); falling back to torch weights", flush=True)
        torch = require_torch()
        return torch.load(weights_path, map_location="cpu")


def load_mrnabert_model(
    model_name: str,
    model_class,
    *,
    num_labels: int | None = None,
    problem_type: str | None = None,
):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.is_decoder = False
    config.add_cross_attention = False
    config.use_cache = False
    if num_labels is not None:
        config.num_labels = num_labels
    if problem_type is not None:
        config.problem_type = problem_type
    config.attention_probs_dropout_prob = max(float(getattr(config, "attention_probs_dropout_prob", 0.0)), 0.1)
    config.hidden_dropout_prob = max(float(getattr(config, "hidden_dropout_prob", 0.0)), 0.1)
    with disable_mrnabert_flash_attention():
        model = model_class.from_config(config, trust_remote_code=True)
        missing_keys, unexpected_keys = model.load_state_dict(load_checkpoint_state_dict(model_name), strict=False)
    print(
        f"[model] mRNABERT manual checkpoint load missing={len(missing_keys)} unexpected={len(unexpected_keys)}",
        flush=True,
    )
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"mRNABERT still has meta parameters: {meta_parameters[:10]}")
    return model


@contextlib.contextmanager
def disable_mrnabert_flash_attention():
    original_import = builtins.__import__
    sentinel = object()
    original_triton = sys.modules.get("triton", sentinel)
    original_triton_language = sys.modules.get("triton.language", sentinel)
    fake_triton = types.ModuleType("triton")
    fake_triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
    fake_triton_language = types.ModuleType("triton.language")
    fake_triton_language.__spec__ = importlib.machinery.ModuleSpec("triton.language", loader=None)
    fake_triton.language = fake_triton_language
    sys.modules.setdefault("triton", fake_triton)
    sys.modules.setdefault("triton.language", fake_triton_language)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "flash_attn_triton" in name:
            raise ImportError("mRNABERT flash attention is disabled for Paxbench")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import
        if original_triton is sentinel:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = original_triton
        if original_triton_language is sentinel:
            sys.modules.pop("triton.language", None)
        else:
            sys.modules["triton.language"] = original_triton_language


def device_for_inference():
    torch = require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_sequence_frame(
    data: str | Path,
    *,
    model: str,
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    spec = get_model_spec(model)
    frame = read_table(data).copy()
    if model == "mrnabert":
        if "cdna_sequence" not in frame.columns or "cds_sequence" not in frame.columns:
            raise ValueError("mrnabert requires columns 'cdna_sequence' and 'cds_sequence'")
        frame["mrnabert_sequence"] = [
            make_mrnabert_sequence(cdna, cds)
            for cdna, cds in zip(frame["cdna_sequence"], frame["cds_sequence"])
        ]
    if spec.sequence_column not in frame.columns:
        raise ValueError(f"{model} requires column {spec.sequence_column!r}")
    frame[spec.sequence_column] = frame[spec.sequence_column].map(clean_sequence)
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[target_column, split_column]).copy()
    frame = frame[frame[spec.sequence_column] != ""].copy()
    if model == "calm":
        frame = frame[frame[spec.sequence_column].str.len() % 3 == 0].copy()
    if limit_rows is not None:
        frame = frame.head(limit_rows).copy()
    if frame.empty:
        raise ValueError("No rows remain after full-model input filtering.")
    return frame.reset_index(drop=True)


def clean_nucleotide_sequence(value: object) -> str:
    return clean_sequence(value, RNA_ALPHABET).replace("U", "T")


def make_mrnabert_sequence(cdna: object, cds: object) -> str:
    cdna_seq = clean_nucleotide_sequence(cdna)
    cds_seq = clean_nucleotide_sequence(cds)
    if not cdna_seq or not cds_seq:
        return ""
    start = cdna_seq.find(cds_seq)
    if start < 0:
        return ""
    end = start + len(cds_seq)
    five_prime = list(cdna_seq[:start])
    codons = [cds_seq[i : i + 3] for i in range(0, len(cds_seq) - len(cds_seq) % 3, 3)]
    three_prime = list(cdna_seq[end:])
    return " ".join(token for token in [*five_prime, *codons, *three_prime] if token)


def load_tokenizer_and_encoder(model: str, model_name: str | None = None):
    spec = get_model_spec(model)
    resolved = model_name or spec.default_model_name
    AutoModel, _, _, AutoTokenizer = require_transformers()
    if model == "calm":
        from multimolecule import CaLmModel, RnaTokenizer

        return RnaTokenizer.from_pretrained(resolved), CaLmModel.from_pretrained(resolved)
    if model == "evo2":
        raise NotImplementedError("Evo2 uses its native backend; call extract_evo2_embeddings.")
    if model == "mrnabert":
        with disable_mrnabert_flash_attention():
            tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        encoder = load_mrnabert_model(resolved, AutoModel)
        return tokenizer, encoder
    return AutoTokenizer.from_pretrained(resolved, trust_remote_code=True), AutoModel.from_pretrained(
        resolved, trust_remote_code=True
    )


def mean_pool(last_hidden_state, attention_mask):
    torch = require_torch()
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (summed / counts).detach().cpu().numpy().astype(np.float32)


def last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise TypeError(f"Could not find hidden states in model output type {type(outputs)!r}")


def extract_model_embeddings(
    *,
    data: str | Path,
    output: str | Path,
    model: str,
    model_name: str | None = None,
    id_column: str = DEFAULT_ID_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    batch_size: int = 2,
    max_length: int = 512,
    limit_rows: int | None = None,
) -> Path:
    if model == "evo2":
        return extract_evo2_embeddings(
            data=data,
            output=output,
            model_name=model_name,
            id_column=id_column,
            target_column=target_column,
            split_column=split_column,
            max_length=max_length,
            limit_rows=limit_rows,
        )
    torch = require_torch()
    spec = get_model_spec(model)
    frame = model_sequence_frame(
        data, model=model, target_column=target_column, split_column=split_column, limit_rows=limit_rows
    )
    tokenizer, encoder = load_tokenizer_and_encoder(model, model_name)
    device = device_for_inference()
    encoder.to(device)
    encoder.eval()
    embeddings: list[np.ndarray] = []
    sequences = frame[spec.sequence_column].astype(str).tolist()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output_obj = encoder(**tokens, output_hidden_states=True, return_dict=True)
            embeddings.append(mean_pool(last_hidden_state(output_obj), tokens["attention_mask"]))
    matrix = np.vstack(embeddings)
    ids = frame[id_column].astype(str).to_numpy() if id_column in frame.columns else frame["paxbench_id"].astype(str).to_numpy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=ids,
        embeddings=matrix,
        targets=frame[target_column].to_numpy(dtype=float),
        splits=frame[split_column].astype(str).to_numpy(),
    )
    return output


def load_masked_lm(model: str, model_name: str | None = None):
    spec = get_model_spec(model)
    resolved = model_name or spec.default_model_name
    _, AutoModelForMaskedLM, _, AutoTokenizer = require_transformers()
    if model == "calm":
        from multimolecule import CaLmForMaskedLM, RnaTokenizer

        return RnaTokenizer.from_pretrained(resolved), CaLmForMaskedLM.from_pretrained(resolved)
    if model == "evo2":
        raise NotImplementedError("Evo2 uses its native backend; call run_evo2_zeroshot.")
    if model == "mrnabert":
        with disable_mrnabert_flash_attention():
            tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        masked_lm = load_mrnabert_model(resolved, AutoModelForMaskedLM)
        return tokenizer, masked_lm
    return AutoTokenizer.from_pretrained(resolved, trust_remote_code=True), AutoModelForMaskedLM.from_pretrained(
        resolved, trust_remote_code=True
    )


def build_skip_token_ids(tokenizer) -> set[int]:
    skip_token_ids = set()
    for attr in ("pad_token_id", "cls_token_id", "bos_token_id", "sep_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            skip_token_ids.add(int(token_id))
    return skip_token_ids


def gather_token_log_probs(log_probs, input_ids, attention_mask, *, skip_token_ids: set[int]) -> list[float]:
    torch = require_torch()
    scores = []
    for row_index in range(input_ids.size(0)):
        valid_positions = torch.nonzero(attention_mask[row_index].bool(), as_tuple=False).flatten()
        token_ids = input_ids[row_index, valid_positions]
        keep_mask = torch.ones_like(token_ids, dtype=torch.bool)
        for token_id in skip_token_ids:
            keep_mask &= token_ids != token_id
        kept_positions = valid_positions[keep_mask]
        kept_token_ids = input_ids[row_index, kept_positions]
        if kept_positions.numel() == 0:
            scores.append(float("nan"))
            continue
        token_log_probs = log_probs[row_index, kept_positions]
        observed = token_log_probs.gather(dim=-1, index=kept_token_ids.unsqueeze(-1)).squeeze(-1)
        scores.append(float(observed.mean().item()))
    return scores


def score_wildtype_marginals(
    *,
    sequences: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device,
) -> np.ndarray:
    torch = require_torch()
    scores: list[float] = []
    skip_token_ids = build_skip_token_ids(tokenizer)
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        encoded = tokenizer(batch_sequences, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            log_probs = torch.log_softmax(logits, dim=-1)
        scores.extend(
            gather_token_log_probs(
                log_probs=log_probs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                skip_token_ids=skip_token_ids,
            )
        )
    return np.asarray(scores, dtype=np.float32)


def run_model_zeroshot(
    *,
    data: str | Path,
    output: str | Path,
    model: str,
    model_name: str | None = None,
    id_column: str = DEFAULT_ID_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    batch_size: int = 2,
    max_length: int = 512,
    limit_rows: int | None = None,
) -> Path:
    if model == "evo2":
        return run_evo2_zeroshot(
            data=data,
            output=output,
            model_name=model_name,
            id_column=id_column,
            target_column=target_column,
            split_column=split_column,
            max_length=max_length,
            limit_rows=limit_rows,
        )
    spec = get_model_spec(model)
    frame = model_sequence_frame(
        data, model=model, target_column=target_column, split_column=split_column, limit_rows=limit_rows
    )
    tokenizer, masked_lm = load_masked_lm(model, model_name)
    device = device_for_inference()
    masked_lm.to(device)
    masked_lm.eval()
    scores = score_wildtype_marginals(
        sequences=frame[spec.sequence_column].astype(str).tolist(),
        tokenizer=tokenizer,
        model=masked_lm,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    out = ensure_dir(output)
    ids = frame[id_column].astype(str).to_numpy() if id_column in frame.columns else frame["paxbench_id"].astype(str).to_numpy()
    predictions = pd.DataFrame(
        {
            id_column: ids,
            "species": frame["species"].astype(str).to_numpy(),
            split_column: frame[split_column].astype(str).to_numpy(),
            target_column: frame[target_column].to_numpy(dtype=float),
            "prediction": np.asarray(scores, dtype=float),
        }
    )
    predictions.to_csv(out / "predictions.tsv", sep="\t", index=False)
    metrics = evaluate_predictions(predictions, target_column=target_column, split_column=split_column)
    write_json({"model": model, "mode": "zeroshot", "metrics": metrics}, out / "metrics.json")
    pd.DataFrame([{"split": split, **values} for split, values in metrics.items()]).to_csv(
        out / "metrics.tsv", sep="\t", index=False
    )
    return out


def run_model_lora(
    *,
    data: str | Path,
    output: str | Path,
    model: str,
    model_name: str | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    epochs: int = 1,
    batch_size: int = 1,
    max_length: int = 512,
    limit_rows: int | None = None,
) -> Path:
    torch = require_torch()
    spec = get_model_spec(model)
    if model == "evo2":
        raise NotImplementedError("Evo2 LoRA requires the Evo2/Savanna CUDA backend.")
    _, _, AutoModelForSequenceClassification, AutoTokenizer = require_transformers()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    frame = model_sequence_frame(
        data, model=model, target_column=target_column, split_column=split_column, limit_rows=limit_rows
    )
    train, validation, test = split_frame(frame, split_column=split_column)
    resolved = model_name or spec.default_model_name
    if model == "calm":
        from multimolecule import CaLmForSequencePrediction, RnaTokenizer

        tokenizer = RnaTokenizer.from_pretrained(resolved)
        base = CaLmForSequencePrediction.from_pretrained(resolved, num_labels=1)
    else:
        if model == "mrnabert":
            with disable_mrnabert_flash_attention():
                tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
            base = load_mrnabert_model(
                resolved,
                AutoModelForSequenceClassification,
                num_labels=1,
                problem_type="regression",
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
            base = AutoModelForSequenceClassification.from_pretrained(
                resolved, num_labels=1, problem_type="regression", trust_remote_code=True
            )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=list(spec.target_modules),
    )
    model_obj = get_peft_model(base, lora)
    device = device_for_inference()
    model_obj.to(device)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, table: pd.DataFrame):
            self.table = table.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.table)

        def __getitem__(self, index: int):
            row = self.table.iloc[index]
            item = tokenizer(str(row[spec.sequence_column]), truncation=True, max_length=max_length)
            item["labels"] = float(row[target_column])
            return item

    out = ensure_dir(output)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model_obj,
        args=args,
        train_dataset=Dataset(train),
        eval_dataset=Dataset(validation),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    raw_predictions = trainer.predict(Dataset(test)).predictions
    if isinstance(raw_predictions, (tuple, list)):
        raw_predictions = raw_predictions[0]
    pred = np.asarray(raw_predictions).reshape(len(test), -1)[:, 0]
    ids = test[DEFAULT_ID_COLUMN].astype(str).to_numpy() if DEFAULT_ID_COLUMN in test.columns else test["paxbench_id"].astype(str).to_numpy()
    predictions = pd.DataFrame(
        {
            DEFAULT_ID_COLUMN: ids,
            "species": test["species"].astype(str).to_numpy(),
            split_column: test[split_column].astype(str).to_numpy(),
            target_column: test[target_column].to_numpy(dtype=float),
            "prediction": pred.astype(float),
        }
    )
    predictions.to_csv(out / "predictions.tsv", sep="\t", index=False)
    metrics = {"test": regression_metrics(predictions[target_column].to_numpy(), predictions["prediction"].to_numpy())}
    write_json({"model": model, "mode": "lora", "metrics": metrics}, out / "metrics.json")
    pd.DataFrame([{"split": split, **values} for split, values in metrics.items()]).to_csv(
        out / "metrics.tsv", sep="\t", index=False
    )
    model_obj.save_pretrained(out / "adapter")
    return out


def load_evo2(model_name: str):
    try:
        from evo2 import Evo2
    except ImportError as exc:
        raise ImportError(
            "Could not import Evo2. Install the Evo2/Savanna environment before running Evo2 modes."
        ) from exc
    return Evo2(model_name)


def prepare_evo2_frame(
    data: str | Path,
    *,
    id_column: str,
    target_column: str,
    split_column: str,
    max_length: int,
    limit_rows: int | None,
) -> pd.DataFrame:
    frame = read_table(data).copy()
    if "cds_sequence" not in frame.columns:
        raise ValueError("Evo2 requires column 'cds_sequence'")
    frame["cds_sequence"] = frame["cds_sequence"].map(lambda value: clean_nucleotide_sequence(value).replace("N", ""))
    frame["sequence_length"] = frame["cds_sequence"].map(len)
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[target_column, split_column]).copy()
    frame = frame[(frame["cds_sequence"] != "") & (frame["sequence_length"] <= max_length)].copy()
    if limit_rows is not None:
        frame = frame.head(limit_rows).copy()
    if frame.empty:
        raise ValueError("No rows remain after Evo2 input filtering.")
    if id_column not in frame.columns and "paxbench_id" in frame.columns:
        id_column = "paxbench_id"
    frame["_id_string"] = frame[id_column].astype(str)
    return frame.reset_index(drop=True)


def extract_evo2_embeddings(
    *,
    data: str | Path,
    output: str | Path,
    model_name: str | None,
    id_column: str,
    target_column: str,
    split_column: str,
    max_length: int,
    limit_rows: int | None,
    layer_name: str = "blocks.28.mlp.l3",
) -> Path:
    torch = require_torch()
    frame = prepare_evo2_frame(
        data,
        id_column=id_column,
        target_column=target_column,
        split_column=split_column,
        max_length=max_length,
        limit_rows=limit_rows,
    )
    evo2_model = load_evo2(model_name or get_model_spec("evo2").default_model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for sequence in frame["cds_sequence"].astype(str).tolist():
            token_ids = evo2_model.tokenizer.tokenize(sequence)
            input_ids = torch.tensor(token_ids, dtype=torch.int).unsqueeze(0).to(device)
            _, hidden_by_layer = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
            if layer_name not in hidden_by_layer:
                raise KeyError(f"Layer {layer_name!r} not found in Evo2 outputs.")
            hidden = hidden_by_layer[layer_name]
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(0)
            pooled = hidden.float().mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
            embeddings.append(pooled)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=frame["_id_string"].to_numpy(dtype=str),
        embeddings=np.vstack(embeddings).astype(np.float32),
        targets=frame[target_column].to_numpy(dtype=float),
        splits=frame[split_column].astype(str).to_numpy(),
    )
    return output


def run_evo2_zeroshot(
    *,
    data: str | Path,
    output: str | Path,
    model_name: str | None,
    id_column: str,
    target_column: str,
    split_column: str,
    max_length: int,
    limit_rows: int | None,
) -> Path:
    torch = require_torch()
    frame = prepare_evo2_frame(
        data,
        id_column=id_column,
        target_column=target_column,
        split_column=split_column,
        max_length=max_length,
        limit_rows=limit_rows,
    )
    evo2_model = load_evo2(model_name or get_model_spec("evo2").default_model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scores = []
    with torch.inference_mode():
        for sequence in frame["cds_sequence"].astype(str).tolist():
            token_ids = evo2_model.tokenizer.tokenize(sequence)
            if len(token_ids) < 2:
                scores.append(float("nan"))
                continue
            input_ids = torch.tensor(token_ids, dtype=torch.int).unsqueeze(0).to(device)
            outputs, _ = evo2_model(input_ids)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            labels = input_ids[:, 1:].long()
            observed = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            scores.append(float(observed.mean().item()))
    out = ensure_dir(output)
    predictions = pd.DataFrame(
        {
            DEFAULT_ID_COLUMN: frame["_id_string"].to_numpy(dtype=str),
            "species": frame["species"].astype(str).to_numpy(),
            split_column: frame[split_column].astype(str).to_numpy(),
            target_column: frame[target_column].to_numpy(dtype=float),
            "prediction": np.asarray(scores, dtype=float),
        }
    )
    predictions.to_csv(out / "predictions.tsv", sep="\t", index=False)
    metrics = evaluate_predictions(predictions, target_column=target_column, split_column=split_column)
    write_json({"model": "evo2", "mode": "zeroshot", "metrics": metrics}, out / "metrics.json")
    pd.DataFrame([{"split": split, **values} for split, values in metrics.items()]).to_csv(
        out / "metrics.tsv", sep="\t", index=False
    )
    return out
