# Represented Is Not Computed

Self-contained code and artifact release for the paper **“Represented Is Not Computed: A Causal Test of Symbolic Intermediates in a Transformer.”**

This repository is organized around the following pieces:

```text
.
├── config.py, data.py, model.py, tokenizer.py   # task + model definition
├── training/                                    # model training
├── evaluation/                                  # checkpoint evaluation
├── analysis/paper_pipeline/                     # paper analyses, saved results, methods ledger
├── logs/                                        # training logs for reported runs
└── checkpoints/                                 # expected checkpoint location (not bundled here)
```

## What is included

- all source code needed to train the models used in the paper
- evaluation code that reconstructs each checkpoint's own held-out split from checkpoint metadata
- one clean analysis script per paper analysis
- the final machine-readable result tables used in the manuscript
- final/supporting analysis figures, including the figure assets used in the paper
- the exact Methods and Results ledgers used to keep the paper auditable
- training logs for the reported model family

## What is intentionally not included

The trained `.pt` checkpoints are omitted from this GitHub-sized release because the 10-layer checkpoints are large binary artifacts. They are hosted separately on OSF at <https://osf.io/vzj72/>. The analyses are fully reproducible once the expected checkpoint files are placed in `checkpoints/`; see [`checkpoints/README.md`](checkpoints/README.md). The repository can also retrain the same model variants from source.

Large transient artifacts such as cached activation dumps and development-only smoke/timing outputs are not part of the release.

## Setup

A minimal Python environment needs:

```bash
python3 -m pip install -r requirements.txt
```

The release was smoke-tested with Python `3.10.19`, PyTorch `2.7.1`, NumPy `2.2.6`, Matplotlib `3.10.8`, and tqdm `4.67.3`.

The code uses PyTorch and will choose CUDA, then MPS, then CPU when available.

## What works without downloaded checkpoints

- inspect the Methods ledger, Results ledger, saved result tables, and saved figures
- regenerate paper Figures 2 and 3 from the committed result tables
- run the lightweight packaging check:

```bash
python3 scripts/check_release.py
```

Evaluation and fresh analysis reruns require the expected `.pt` files in `checkpoints/`; download them from <https://osf.io/vzj72/> and see [`checkpoints/README.md`](checkpoints/README.md).

## Typical use

### 1. Train a model

Set the desired architecture/seed/training-order values in `config.py`, then run:

```bash
python3 training/train.py
```

Training writes named checkpoints into `checkpoints/` and logs into `logs/`. The checkpoint payload stores `split_info`; downstream evaluation and analyses use that saved split metadata rather than rebuilding splits from the current config.

### 2. Evaluate one checkpoint

```bash
python3 evaluation/test.py \
  --checkpoint checkpoints/model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
```

### 3. Reproduce the paper analyses

Use the compact runbook in [`analysis/paper_pipeline/README.md`](analysis/paper_pipeline/README.md). Each analysis has one script under `analysis/paper_pipeline/analyses/` and writes tables under `analysis/paper_pipeline/data/`.

The two paper figures generated directly from saved result tables can be rebuilt with:

```bash
python3 analysis/paper_pipeline/plot_paper_figures.py
```

The committed tables under `analysis/paper_pipeline/data/` are the authoritative saved outputs used by the manuscript; rerunning the scripts reproduces those outputs from checkpoints.

### 4. Read the audit trail

- [`analysis/paper_pipeline/Methods.md`](analysis/paper_pipeline/Methods.md): exact protocols
- [`analysis/paper_pipeline/RESULTS.md`](analysis/paper_pipeline/RESULTS.md): authoritative paper-facing numbers
- [`analysis/paper_pipeline/Scratch.md`](analysis/paper_pipeline/Scratch.md): informative exploratory checks that did not enter the final paper path

## Reproducibility principle

The checkpoint is the unit of truth. A checkpoint filename identifies the model variant; the checkpoint payload stores the exact held-out `N` and `B` sets used for that run. Every evaluation and analysis script reconstructs the checkpoint-specific split from that metadata, so analyses remain valid even when the current `config.py` has changed.

## License

This repository is released under the [MIT License](LICENSE).
