# Scratch Ledger

This file preserves useful exploratory checks and discarded branches that informed the final analysis but are **not** part of the canonical paper-facing pipeline.

- `Methods.md` records the final protocols used in the paper.
- `RESULTS.md` records the final reported findings and numbers.
- `Scratch.md` records informative side experiments, development observations, and choices that should remain visible without being mistaken for final evidence.

Unless stated otherwise, the experiments below were exploratory, often on an earlier single checkpoint or an earlier analysis framing, and were superseded by the cleaner multi-seed analyses in the final pipeline. They are preserved because they helped shape the final questions, not because they should be cited as additional paper results.

## Earlier route-localization work pointed away from a single serial handoff

Before the final paper pipeline was rebuilt around checkpoint-specific splits and the `D_ones` notation, we used the legacy name `pre_O` for the stream immediately before the output prompt token. Several early route tests already suggested that this stream was important but not sufficient:

- removing the full `pre_O -> output` route reduced exact accuracy from `100%` to `35.8%`
- keeping only `pre_O` among input-side sources gave a similar `35.0%` exact accuracy
- blocking direct number-digit routes was also damaging, with the strongest `N`-digit route-window ablation reducing exact accuracy to `24.8%`
- blocking tag-token inputs into `pre_O` at layer 0 reduced exact accuracy to `31.8%`, suggesting that the early `D`-adjacent state behaved partly as a routing/control state rather than as an isolated symbolic calculator

Why this mattered:

- These checks made a purely serial story less plausible before the final circuit search existed.
- They suggested the eventual paper should ask not only **what is represented** in the `D`-adjacent stream, but also **which other routes remain necessary** for behavior.
- The final sparse-circuit analysis is the cleaner version of that idea, because it is multi-seed, checkpoint-aware, and reports one stable sufficient circuit family rather than a collection of one-off ablations.

## Earlier probe trajectories foreshadowed the final dissociation

Single-checkpoint exploratory probes already showed a suggestive separation:

| Quantity | Strongest legacy `pre_O` locus | Strongest output-stream locus |
| --- | ---: | ---: |
| `B^D` | layer 2, `R² = 0.939` | layer 3, `R² = 0.669` |
| `floor(N / B^D)` | layer 7, `R² = 0.925` | layer 2, `R² = 0.818` |
| final digit | layer 6, `R² = 0.226` | layer 9, `R² = 0.849` |

Why this mattered:

- This was the first clear hint that quotient-like quantities could be easy to decode from the `D`-adjacent stream while the answer itself consolidated later on the output side.
- The final paper does not rely on these single-checkpoint values; the three-seed linear-probing analysis replaces them with the final reported result.

## Probe-control checks around correlation, residualization, and initialization

After the paper-facing probe result was in place, we ran a few additional checks to understand why quotient-like quantities could be highly decodable from `D_ones` even very early. These checks are exploratory qualifiers for interpretation, not separate causal evidence.

### Correlation with `B^D` did not explain early quotient-like decoding

One worry was that early `N / B^D` decodability might be a side effect of correlation with `B^D`, since `B^D` itself is nearly perfectly decodable from `D_ones` at layer 0. A direct linear regression from `B^D` to `N / B^D` over the same held-out examples explained little variance:

- linear `B^D -> N / B^D`: mean `R² = 0.029`, 95% CI `[0.025, 0.037]`
- `D_ones` layer 0 probe for original `N / B^D`: mean `R² = 0.948`
- `D_ones` layer 0 probe for `N / B^D` residualized against `B^D`: mean `R² = 0.946`
- `D_ones` layer 2 probe for original `N / B^D`: mean `R² = 0.960`
- `D_ones` layer 2 probe for `N / B^D` residualized against `B^D`: mean `R² = 0.956`

Takeaway: the early quotient-like signal is not just a linear echo of the `B^D` signal.

### Incremental residualization preserved the same broad probe structure

We then residualized each target against progressively richer scalar baselines:

- `B^D` residualized against raw `B,D`
- `N / B^D` residualized against raw `N,B,D` plus `B^D`
- `floor(N / B^D)` residualized against raw `N,B,D`, `B^D`, and `N / B^D`
- answer residualized against raw `N,B,D`, `B^D`, `N / B^D`, and `floor(N / B^D)`

The scalar baselines were strong for some targets and weak for others:

- scalar baseline for residualized `B^D`: mean `R² = 0.222`, 95% CI `[0.134, 0.317]`
- scalar baseline for residualized `N / B^D`: mean `R² = 0.503`, 95% CI `[0.392, 0.570]`
- scalar baseline for residualized `floor(N / B^D)`: essentially `R² = 1.000` after including `N / B^D`, making this residual target uninformative
- scalar baseline for residualized answer: mean `R² = 0.366`, 95% CI `[0.358, 0.374]`

The residual-stream probes still carried substantial incremental signal:

- residualized `B^D` remained nearly perfectly decodable from `D_ones` at layer 0 (`R² ≈ 0.998`)
- residualized `N / B^D` remained strongly decodable from `D_ones`, peaking around layer 6 (`R² ≈ 0.935`) and already high at layer 0 (`R² ≈ 0.923`)
- residualized answer information was strongest late in the output streams (`O[1]` late `R² ≈ 0.891`; `O[0]` late `R² ≈ 0.846`)

Takeaway: the high probe scores are not fully reducible to trivial scalar correlations among the target variables. The residual stream makes additional target-aligned structure linearly accessible, especially in `D_ones` and the output streams.

### Initialization probes showed that high decodability is not automatically learned computation

We also reconstructed untrained models using the same architectures, seeds, and held-out splits, without loading checkpoint weights. These initialized models had near-zero autoregressive task competence:

| Seed | Exact-answer accuracy | First digit | Second digit |
| ---: | ---: | ---: | ---: |
| 0 | `0.15%` | `2.43%` | `7.34%` |
| 42 | `0.05%` | `0.83%` | `7.76%` |
| 1337 | `0.00%` | `0.11%` | `6.30%` |

Despite this, some closed-form quantities were already highly linearly decodable at initialization. The final paper therefore reports initialization baselines and the positive fraction of the initialization-to-ceiling `R²` gap closed by training:

| Quantity | Stream/layer | Gap-to-ceiling closed by training | Init `R²` | Trained `R²` |
| --- | --- | ---: | ---: | ---: |
| `B^D` | `D_ones`, layer 0 | `98.00%` `[96.00%, 99.20%]` | `0.913` | `0.998` |
| `N / B^D` | `D_ones`, layer 6 | `32.80%` `[15.30%, 55.00%]` | `0.941` | `0.960` |
| `floor(N / B^D)` | `D_ones`, layer 6 | `32.80%` `[15.30%, 55.00%]` | `0.942` | `0.960` |
| answer | `O[1]`, layer 9 | `89.10%` `[87.20%, 90.50%]` | `0.447` | `0.940` |

Takeaway: training clearly sculpts task-relevant accessibility, especially for early `B^D` in `D_ones` and late answer information in the output streams. But the initialized baselines also show why decodability alone should not be read as evidence of learned causal use.

## Output-side exploratory tests supported late integration

Several exploratory interventions focused directly on the output streams rather than the `D`-adjacent stream. These were useful for thinking, but they were single-checkpoint / development analyses and were superseded by the cleaner sparse-circuit result.

### Output-stream zeroing

Cumulative and single-layer zeroing suggested that the final output-side states were behaviorally fragile:

- prompt output stream `O`: exact accuracy fell from `100%` to `0%` when layer 9 alone was zeroed
- first generated output stream `O+1`: exact accuracy fell to `5.4%` when layer 9 alone was zeroed

### Output-stream contrastive patching for `N`, `B`, and `D`

We also patched output-side residual states using donor examples where exactly one factor changed while the other two were held fixed. The exploratory script used the legacy names `target` for the unpatched prompt and `source` for the donor residual being inserted; the values below are donor-answer match rates after patching one output-side layer.

| Patched stream | Changed factor | Best layer | Best donor-exact rate |
| --- | --- | ---: | ---: |
| `O` | `N` | 8 | `19.2%` |
| `O` | `B` | 5 | `45.2%` |
| `O` | `D` | 7 | `17.6%` |
| `O+1` | `N` | 9 | `71.6%` |
| `O+1` | `B` | 9 | `70.8%` |
| `O+1` | `D` | 9 | `70.4%` |

Digit-level behavior was sharper than exact-answer behavior. For example, patching `O` late could almost completely set the first digit while leaving the second digit less donor-like; patching `O+1` at layer 9 made the second digit donor-like for all three factor changes.

Why this mattered:

- These tests helped sharpen the late-integration interpretation: output-side computation is not merely a passive readout of one upstream symbolic variable.
- The output streams can carry answer-relevant changes induced by `N`, `B`, and `D`, consistent with the paper's statement that output-side integration remains to be decomposed.
- They were left out of the final paper because they were exploratory, single-checkpoint, and less clean than the sparse-circuit result, which communicates the same broader point with a reproducible multi-seed relation-level summary.

## Head-level decoding and ablation shenanigans

We also explored whether individual attention heads explained the early `D`-adjacent route. This was done before the final paper pipeline settled on route-level K/V patching, so the results are best read as development notes rather than evidence for the manuscript.

Head-level probes found several heads whose outputs or routed values made `D`, `B`, or `B^D` decodable. Examples from the archived `pre_O` head sweep include:

- several layer-0 heads had near-perfect `D` decodability (`R²` close to `1.0`)
- layer-0 head 10 had the strongest single-head `B^D` signal in that sweep (`R² = 0.639`) and a residualized `B^D` signal after controlling for `D` (`R² = 0.418`)
- per-`D` specialization plots showed that many heads behaved differently across digit positions, especially for low `D` values with more examples

But causal ablations did not turn this into a clean head-level story:

| Exploratory ablation | Exact accuracy |
| --- | ---: |
| baseline, 100-sample check | `100%` |
| ablate layer-0 head 10 only | `100%` |
| ablate layer-0 head 11 only | `100%` |
| ablate top 6 ranked heads per layer | `100%` |
| ablate top 10 ranked heads per layer | `67%` |
| ablate bottom 10 ranked heads per layer | `35%` |
| ablate all heads per layer in the tested route | `36%` |

Why this mattered:

- Individual heads could make attractive representational stories, but single-head causal tests were not decisive.
- Large grouped ablations were damaging, but they were coarse and hard to interpret because they removed many heads at once.
- This pushed the final paper toward route-level interventions: asking what information the output streams read from `D_ones`, while keeping the source `D_ones` computation itself intact.
- The paper therefore does not claim a head-level mechanism; it explicitly stops at the route level.

## Bounded-domain competence did not imply extrapolation

We also ran two deliberately out-of-grammar checks on an earlier checkpoint:

| Extrapolation condition | Digit accuracy | Exact-answer accuracy |
| --- | ---: | ---: |
| four-digit `N > 999` | 28.5% | 0.0% |
| two-digit `D > 9` | 42.0% | 0.0% |

Why this mattered:

- These checks reinforced the final paper's deliberately bounded wording: the trained model generalizes over held-out number--base intersections **within** the task domain, but this should not be mistaken for unbounded algorithmic generalization.
- They were not elevated into the paper because the paper's target question is causal mechanism after robust in-domain task competence, not extrapolation benchmarking.

## Sparse-circuit discovery: subset-size development sweeps

Before using the full validation set for the final sparse-circuit discovery run, we tested capped discovery subsets to understand runtime and stability of the greedy prefix-selection procedure.

Main 10-layer kept-only held-out exact-answer accuracy by discovery cap:

| Discovery subset | Seed 0 | Seed 42 | Seed 1337 |
| --- | ---: | ---: | ---: |
| 128 examples | 81.46% | 92.28% | 96.23% |
| 256 examples | 90.48% | 96.40% | 98.17% |
| 512 examples | 83.34% | 96.49% | 75.39% |
| 2048 examples | 90.83% | 91.68% | 97.68% |

Takeaways:

- Capped discovery subsets were useful for runtime development but materially affected the greedy elbow-selected circuit.
- The qualitative route backbone remained stable, but exact retained depths and kept-only accuracy were sensitive to the local prefix-selection heuristic and the discovery sample.
- Because the 512-example run showed that larger random subsets were not automatically more reliable, the final paper analysis used the full validation set rather than a capped random subset.

## Fixed-order training produced a different `D` route

We also trained a 10-layer seed-1337 model without field-order permutation during training (`pFalse`). It solved the held-out task almost perfectly (`99.838%` exact-answer accuracy), but its internal route organization differed from the permutation-trained family used in the paper.

Exploratory findings:

- closed-form `B^D` remained strongly decodable from `D_ones` at layer 0 (`R² = 0.996`), but the quotient-like quantities peaked more strongly in `O[0]` than in `D_ones`
- masking only the layer-0 `D_ones -> O` attention edge reduced exact-answer accuracy from `99.838%` to `32.418%`
- route-split K/V patching reversed the main-paper pattern:
  - `L1`-only donor substitution transferred essentially nothing (`0.041%` donor-exact)
  - `L0 + L2+` donor substitution transferred almost perfectly (`99.816%` donor-exact)
- full-route patching remained strongly `D`-selective but less perfectly factorized than in the `pTrue` models:
  - donor change in `N`: `0.968%` donor-exact
  - donor change in `B`: `2.130%` donor-exact
  - donor change in `D`: `100.000%` donor-exact

The sparse-circuit search was not reliable for this model under the paper's original greedy selection rule. It retained only `10 / 65` candidate relations and `20 / 246` candidate layer-edges, but the resulting kept-only circuit achieved only `40.926%` exact-answer accuracy. This likely reflects a search-rule failure rather than a genuinely sufficient ultra-sparse mechanism: several individually weak routes may be jointly necessary in the fixed-order model, so pruning by single-edge ablation underestimates the supporting circuit.

Why this stayed out of the paper:

- the paper's central causal story is about the three permutation-trained 10-layer seeds
- the fixed-order model is a scientifically interesting contrast condition, not a robustness check for the reported mechanism
- interpreting it properly would require a circuit search that tests joint sufficiency or add-back rescue, which is beyond the scope of the current manuscript

## A 3-layer model compressed the same broad mechanism

We also analyzed a 3-layer seed-1337 permutation-trained model. It solved the held-out task at `99.623%` exact-answer accuracy and showed the same broad causal story as the deeper permutation-trained models, but compressed into fewer layers.

Exploratory findings:

- the closed-form quantities remained strongly decodable:
  - `B^D`: `D_ones@L0`, `R² = 0.999`
  - `N / B^D`: `D_ones@L0`, `R² = 0.983`
  - `floor(N / B^D)`: `D_ones@L0`, `R² = 0.983`
  - final answer: `O[1]@L2`, `R² = 0.912`
- cumulative `D_ones -> O` ablation again showed a sharp early dependence:
  - clean exact accuracy: `99.623%`
  - mask `L0` only: `99.246%`
  - mask `L0-L1`: `49.004%`
- route-split K/V patching again favored `L1`:
  - `L1`-only donor exact: `83.177%`
  - `L0 + L2+` donor exact: `6.827%`
- full-route information patching again transferred behavior only when `D` changed:
  - donor change in `N`: `0.000%` donor-exact
  - donor change in `B`: `0.000%` donor-exact
  - donor change in `D`: `99.741%` donor-exact
- its sparse circuit retained `30 / 87` candidate relations and `52 / 204` candidate layer-edges while preserving `92.299%` exact-answer accuracy

The 3-layer circuit also retained two small tag-level cross-field links, `N_tag -> B_tag -> B_tens` and `N_tag -> B_tag -> B_ones`, suggesting that the shallow model reuses some structural scaffold across fields when depth is scarce.

Why this stayed out of the paper:

- the 5-layer companion already provides a cleaner depth-robustness check without requiring extra prose
- the 3-layer result is best read as an informative compression experiment, not as evidence needed for the manuscript's central claim

## Release note

The clean public release keeps the final reproducible paper path compact. Development-only code branches, timing runs, and intermediate artifacts that were superseded by the final analyses are intentionally not promoted into the canonical pipeline unless they answer a durable scientific question.

Some exploratory patching ideas were also tried and then superseded while the causal intervention was being specified more precisely. Those implementation branches are intentionally not documented here: once the final question became “what key/value information from `D_ones` is visible to the output streams, while the source stream itself remains untouched?”, the final route-patching analyses in `Methods.md` and `RESULTS.md` became the only versions worth preserving as evidence.

## Untrained LSTM probe baseline: high decodability is not only a Transformer quirk

To test whether the high linear readability of closed-form quantities was specific to Transformer residual streams, we ran a scratch baseline with a different random architecture.

Protocol:

- 10-layer stacked LSTM, hidden size 384, random/untrained weights.
- Same tokenizer and canonical `N...B...D...O` examples as the paper pipeline.
- The paper checkpoints were used only to recover seed identities and checkpoint-specific validation/test splits; trained checkpoint weights were not loaded into the LSTM.
- Linear probes used the same pooled held-out validation+test examples and 5-fold CV protocol as the paper's closed-form probe analysis.
- For `O[1]`, the LSTM generated its own first answer digit and then conditioned on that generated digit before the second-pass state was collected, matching the autoregressive protocol.

Key across-seed results:

| Target | Best relevant LSTM state | Mean CV `R²` | 95% CI |
| --- | --- | ---: | ---: |
| `B^D` | `D_ones`, layer 0 | 0.986 | [0.975, 0.998] |
| `N / B^D` | `D_ones`, layer 1 | 0.806 | [0.797, 0.813] |
| `floor(N / B^D)` | `D_ones`, layer 1 | 0.806 | [0.797, 0.813] |
| `floor(N / B^D) mod B` | `O[1]`, layer 0 | 0.529 | [0.504, 0.566] |

Takeaway:

- Very high `B^D` decodability and substantial quotient-like decodability can appear even in an untrained non-Transformer architecture.
- This makes it less plausible that high closed-form decodability is a Transformer-specific quirk or, by itself, evidence of learned causal computation.
- The result is exploratory and stays out of the paper-facing pipeline; it is useful mainly as a qualifier for interpreting probe success.

Workspace files:

- script: `analysis/scratch/lstm_probe_baseline/run_untrained_lstm_linear_probes.py`
- results: `analysis/scratch_results/lstm_probe_baseline/untrained_10layer_hidden384_main_splits/`
