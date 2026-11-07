# Paxbench

Paxbench is a compact benchmark package for protein abundance prediction. The repository keeps the public evaluation code small: one environment and one benchmark runner.

The full [PaxBench dataset and documentation](https://huggingface.co/datasets/cke2333/Paxbench) are hosted on Hugging Face. The GitHub repository includes a small demo dataset.

## Install

```bash
conda env create -f environment.yml
conda activate paxbench
pip install -e .
```

This environment supports ESM2, ESMC, CaLM, DNABERT2, mRNABERT, embedding + ML, zero-shot, LoRA, and evaluation.

Evo2 must be installed and run in a separate Linux CUDA environment. Follow the official Evo2 full install instructions from [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2#installation):

```bash
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
pip install -e .
```

Keep the environments separate:

- Use `paxbench` for ESM2, ESMC, CaLM, DNABERT2, and mRNABERT.
- Use the Evo2 environment only for Evo2 commands.
- Run the same Paxbench CLI from the relevant activated environment.

## Benchmark Schema

The modeling table uses `paxbench_id` as the public row identifier. The default target is `log10_abundance`.

Core fields:

- `paxbench_id`
- `species`
- `protein_sequence`
- `cds_sequence`
- `mrna_sequence`
- `log10_abundance`
- released random, cluster, species, fungi, and abundance split columns

Identifier mapping and source metadata are available in the dataset release's `metadata/` directory, keyed by `paxbench_id`.

## Run Models

The benchmark runner is the entry point for individual models:

```bash
python scripts/run_benchmark.py \
  --data data/demo/paxbench_demo.tsv \
  --output results/demo \
  --models esm calm mrnabert \
  --modes embedding zeroshot lora \
  --learners ridge mlp \
  --epochs 1
```

Modes:

- `embedding`: model embedding extraction followed by Ridge or MLP.
- `zeroshot`: sequence likelihood scoring without abundance supervision.
- `lora`: LoRA fine-tuning for abundance regression.

Frozen-embedding regressors fit preprocessing and prediction components using the training partition.

Run Evo2 separately in its Linux CUDA environment. This repository provides Evo2 embedding extraction and zero-shot scoring, but not Evo2 LoRA. Model weights are downloaded on first use unless already cached.

## Reproduce

`scripts/reproduce.py` reproduces the path from model representations to the reported fusion tables. The bundled demo uses fixed train, validation, and test assignments.

```bash
python scripts/reproduce.py
```

The command extracts or reuses ESM2, CaLM, and mRNABERT representations, constructs every supported single-model and cross-level fusion set, runs Ridge and MLP with three seeds, selects the best configuration by mean validation Spearman, and evaluates the locked configuration on test. The 36-row demo verifies the code path; its metrics are not benchmark estimates.

Five files are written to `data/results/demo/`:

- `fusion_sets.tsv`: model combinations, molecular levels, dimensions, and coverage.
- `runs.tsv`: every combination, learner, seed, partition, and metric.
- `repeated_results.tsv`: three-seed means and standard deviations.
- `validation_selection.tsv`: configurations selected using validation only.
- `results.tsv`: locked test results used to regenerate the main comparison.

Download the released table to run the full benchmark. Use a separate representation cache from the demo:

```bash
curl -fL https://huggingface.co/datasets/cke2333/Paxbench/resolve/main/paxbench_tracka.csv.gz \
  -o data/paxbench_tracka.csv.gz

python scripts/reproduce.py \
  --data data/paxbench_tracka.csv.gz \
  --cache-dir .cache/paxbench/full \
  --output-dir data/results/paper \
  --models esm calm evo2 mrnabert \
  --split-columns split_random split_cluster split_species \
                  split_species2 split_species3 split_fungirandom \
                  split_fungi2fungi split_highlow
```

Model representations are aligned by `paxbench_id` and concatenated for fusion. Extract the Evo2 representation once in the separate Evo2 environment before running the command above:

```bash
python scripts/embed.py \
  --model evo2 \
  --data data/paxbench_tracka.csv.gz \
  --max-length 1024 \
  --output .cache/paxbench/full/representations/evo2.npz
```

Run the Evo2 extraction before the full benchmark command. Reuse an embedding cache only for the same input table and extraction settings.

The same frozen-representation pipeline supports ESM2, ESMC 300M, ESMC 600M,
CaLM, Evo2, DNABERT2 on CDS or mRNA, and mRNABERT. Passing these names through
`--models` makes `scripts/reproduce.py` construct all valid cross-level
combinations before applying the same three-repeat aggregation and validation
selection.

Zero-shot and LoRA are available through `scripts/run_benchmark.py`.

The released table includes the benchmark splits. To audit sequence identity across a split, install MMseqs2 and see `python scripts/split_audit.py --help`.

## License

Code, documentation, and packaged benchmark artifacts are released under the Creative Commons Attribution 4.0 International License. See [LICENSE](LICENSE).
