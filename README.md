# Paxbench

Paxbench is a compact benchmark package for protein abundance prediction. The repository keeps the public evaluation code small: one environment and one benchmark runner.

## Install

```bash
conda env create -f environment.yml
conda activate paxbench
pip install -e .
```

This environment is for ESM, CaLM, mRNABERT, embedding + ML, zero-shot, LoRA, and evaluation.

Evo2 must be installed and run in a separate Linux CUDA environment. Follow the official Evo2 full install instructions from [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2#installation):

```bash
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
pip install -e .
```

Keep the environments separate:

- Use `paxbench` for `esm`, `calm`, and `mrnabert`.
- Use the Evo2 environment only for `--models evo2`.
- Run the same Paxbench CLI from the relevant activated environment; do not mix Evo2 dependencies into the base `paxbench` environment.

## Benchmark Schema

The modeling table uses `paxbench_id` as the public row identifier. The default target is `log10_abundance`.

Core fields:

- `paxbench_id`
- `species`
- `protein_sequence`
- `cds_sequence`
- `mrna_sequence`
- `cdna_sequence`
- `log10_abundance`
- split columns: `split_species`, `split_homology`, `split_highlow`, `split_random`, `split_fungi2fungi`, `split_fungirandom`, `split_cluster`, `split_species2`, `split_species3`

Identifier mapping and source metadata are stored separately under `data/metadata/`, keyed by `paxbench_id`.

## Run Models

Use one command to run any number of models and any combination of the three benchmark modes:

```bash
python scripts/run_benchmark.py \
  --data data/demo/paxbench_demo.tsv \
  --output results/demo \
  --models esm calm mrnabert evo2 \
  --modes embedding zeroshot lora \
  --learners ridge mlp \
  --epochs 1
```

Modes:

- `embedding`: model embedding extraction followed by Ridge or MLP.
- `zeroshot`: masked-marginal zero-shot scoring.
- `lora`: LoRA finetuning for abundance regression.

The command writes per-run outputs plus a single score table:

```text
results/demo/scores.tsv
```

Score tables report RMSE, MAE, Spearman, Pearson, and row count. R2 is intentionally not reported.

## Demo Command

The complete demo shell script runs all three benchmark modes across all four model names:

```bash
bash scripts/run_demo_all.sh
```

Equivalent explicit form:

```bash
bash scripts/run_demo_all.sh data/demo/paxbench_demo.tsv results/demo
```

## Model Modules

Paxbench exposes three model modules through the same runner:

- Model embedding extraction + Ridge/MLP: `--modes embedding --learners ridge mlp`
- LoRA finetuning: `--modes lora`
- Model zero-shot: `--modes zeroshot`

LoRA can run on CPU for small demo checks, but CUDA is recommended. Evo2 requires its separate Evo2/Vortex/Savanna CUDA stack.



## License

Code, documentation, and packaged benchmark artifacts are released under the Creative Commons Attribution 4.0 International License. See [LICENSE](LICENSE).
