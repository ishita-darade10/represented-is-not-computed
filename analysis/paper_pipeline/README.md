# Paper Analysis Pipeline

This folder contains the reproducible analysis path from trained checkpoints to the paper-facing result tables.

```text
paper_pipeline/
├── analyses/      # one script per paper analysis
├── helpers/       # checkpoint loading, split reconstruction, runtime helpers
├── data/          # saved final result tables
├── figures/       # saved supporting figures
├── Methods.md     # exact protocol ledger
├── RESULTS.md     # authoritative numerical ledger
├── Scratch.md     # exploratory checks outside the final paper path
└── plot_paper_figures.py
```

## Operating rule

Every script takes named checkpoints as input and reconstructs that checkpoint's own validation/test split from the `split_info` stored inside it. Do not substitute current `config.py` split values for saved checkpoint metadata.

## Final analyses used in the paper

Run commands are shown from the repository root. Use the `0` setting shown for `--max-examples`, `--max-pairs`, or `--max-sources` to run on the full checkpoint-specific test set.

### 01. Held-out autoregressive test performance

```bash
python3 analysis/paper_pipeline/analyses/01_test_set_performance.py \
  --run-label main_10layer_ptrue \
  --checkpoints \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s0_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s42_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
```

The 5-layer companion check uses the same script with:

```text
companion_5layer_ptrue              model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt
```

### 02. Closed-form linear probing

```bash
python3 analysis/paper_pipeline/analyses/02_linear_probing_closed_form.py \
  --run-label main_10layer_ptrue \
  --checkpoints \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s0_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s42_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
```

Companion 5-layer run:

```bash
python3 analysis/paper_pipeline/analyses/02_linear_probing_closed_form.py \
  --run-label companion_5layer_ptrue \
  --checkpoints checkpoints/model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt
```

### 03. Cumulative `D_ones -> O` attention ablation

```bash
python3 analysis/paper_pipeline/analyses/03_dones_to_output_attention_ablation.py \
  --run-label main_10layer_ptrue_fulltest \
  --max-examples 0 \
  --checkpoints \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s0_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s42_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
```

Companion 5-layer run:

```bash
python3 analysis/paper_pipeline/analyses/03_dones_to_output_attention_ablation.py \
  --run-label companion_5layer_ptrue_fulltest \
  --max-examples 0 \
  --checkpoints checkpoints/model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt
```

### 04. Route-split `D_ones -> O` K/V patching

```bash
python3 analysis/paper_pipeline/analyses/04_dones_output_attention_route_patching.py \
  --run-label main_10layer_ptrue_fulltestpairs_outputrows_only \
  --max-pairs 0 \
  --checkpoints \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s0_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s42_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
```

Companion 5-layer run:

```bash
python3 analysis/paper_pipeline/analyses/04_dones_output_attention_route_patching.py \
  --run-label companion_5layer_ptrue_fulltestpairs_outputrows_only \
  --max-pairs 0 \
  --checkpoints checkpoints/model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt
```

### 05. Information carried by the full `D_ones -> O` route

```bash
python3 analysis/paper_pipeline/analyses/05_dones_information_content_patching.py \
  --run-label main_10layer_ptrue_test_matchedsources_clean_correct \
  --max-sources 0 \
  --checkpoints \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s0_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s42_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
```

Companion 5-layer run:

```bash
python3 analysis/paper_pipeline/analyses/05_dones_information_content_patching.py \
  --run-label companion_5layer_ptrue_test_matchedsources_clean_correct \
  --max-sources 0 \
  --checkpoints checkpoints/model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt
```

### 06. Greedy sparse-circuit discovery

```bash
python3 analysis/paper_pipeline/analyses/06_sparse_circuit_discovery.py \
  --run-label main_10layer_ptrue_fullval \
  --checkpoints \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s0_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s42_best.pt \
  checkpoints/model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
```

## Paper figures

Figures 2 and 3 are regenerated from the saved final tables with:

```bash
python3 analysis/paper_pipeline/plot_paper_figures.py
```

Figures 1 and 4 are maintained as final released figure assets in `figures/paper_figures/`.

## Where to look first

- exact methods: `Methods.md`
- final reported numbers: `RESULTS.md`
- exploratory side checks: `Scratch.md`
- saved tables for each analysis: `data/<analysis_slug>/<run_label>/`
- supporting generated figures: `figures/<analysis_slug>/`
