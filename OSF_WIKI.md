# Represented Is Not Computed: Model Checkpoints

This OSF project hosts the trained PyTorch checkpoint files used in the paper:

**Represented Is Not Computed: A Causal Test of Symbolic Intermediates in a Transformer**

The accompanying code repository contains the training code, evaluation code, paper-analysis pipeline, saved result tables, figures, and full documentation. This OSF project exists only to host the larger model checkpoint files that are impractical to include directly in the GitHub release.

## Contents

The following `best` checkpoints are provided:

```text
model_t_l10_mby_NB_intersection_pTrue_s0_best.pt
model_t_l10_mby_NB_intersection_pTrue_s42_best.pt
model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt
```

## How these files are used

The three 10-layer `pTrue` checkpoints are the main models used for the paper's multi-seed analyses.

The remaining checkpoint is a companion model used for a qualitative depth-robustness check:

- 5-layer `pTrue`

Checkpoint filenames encode:

- model depth (`l5`, `l10`)
- split regime (`by_NB_intersection`)
- whether all six training field-order permutations were used (`pTrue` for the released paper checkpoints)
- random seed (`s0`, `s42`, `s1337`)
- checkpoint selection (`best`)

Each checkpoint also stores the exact held-out split metadata used during training. The public analysis pipeline reconstructs evaluation splits from this saved metadata rather than from the current config file.

SHA-256 checksums for these exact released files are provided in the code repository at `checkpoints/CHECKSUMS.sha256`.

## Reproduction

To reproduce the paper analyses:

1. Download these checkpoint files.
2. Place them in the `checkpoints/` directory of the accompanying code repository.
3. Follow the repository README and `analysis/paper_pipeline/README.md`.

**Checkpoint archive:**  
<https://osf.io/vzj72/>

**Code repository:**  
_Add GitHub URL here once public._

## Notes

- These files are provided to support reproducibility of the reported paper results.
- The code repository is the authoritative source for methods, final results, figure generation, and navigation of the full pipeline.
