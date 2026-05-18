# Checkpoints

Checkpoint binaries are intentionally not bundled in this GitHub-sized release. Download the released checkpoints from OSF at <https://osf.io/vzj72/> and place them in this directory before running evaluation or paper analyses.

After downloading, verify the files against the released SHA-256 manifest:

```bash
cd checkpoints
# macOS
shasum -a 256 -c CHECKSUMS.sha256

# Linux
sha256sum -c CHECKSUMS.sha256
```

The paper analyses expect these **best** checkpoints:

```text
model_t_l10_mby_NB_intersection_pTrue_s0_best.pt
model_t_l10_mby_NB_intersection_pTrue_s42_best.pt
model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt
```

Only the named `best` checkpoints are required to reproduce the paper-facing analyses. The code can also generate corresponding `last` checkpoints during training, but those are not needed for the reported results.

Why the names matter:

- `l{n}` encodes the number of Transformer layers
- `m...` encodes the split regime
- `pTrue` records that all six training field-order permutations were used for the released paper checkpoints
- `s...` encodes the seed
- `best` / `last` encodes checkpoint selection

Each checkpoint also stores the exact `split_info` object used during training. Evaluation and analysis reconstruct the held-out split from that object rather than from the current contents of `config.py`.
