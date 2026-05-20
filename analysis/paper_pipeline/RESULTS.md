# Results Ledger

This file records the paper-facing findings as they are finalized. It is intentionally concise: `Methods.md` preserves exactly what was done, while this file preserves what we learned and the numbers we will later need when writing the manuscript.

## Result 01 — Transformers learn the base-conversion task on held-out `(N, B)` combinations

### Main three-seed 10-layer model result

The main 10-layer models trained with shuffled `N/B/D` field order during training (`pTrue`) achieved near-perfect autoregressive performance on their own held-out canonical test splits.

| Seed | Held-out test examples | Exact 2-digit answer accuracy | `O[0]` token accuracy | `O[1]` token accuracy |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2017 | 99.504% | 99.950% | 99.504% |
| 42 | 2164 | 100.000% | 100.000% | 100.000% |
| 1337 | 1857 | 100.000% | 100.000% | 100.000% |

Across the three independently trained networks:

- exact 2-digit answer accuracy: mean `99.835%`, 95% bootstrap CI `[99.504%, 100.000%]`
- `O[0]` token accuracy: mean `99.983%`, 95% bootstrap CI `[99.950%, 100.000%]`
- `O[1]` token accuracy: mean `99.835%`, 95% bootstrap CI `[99.504%, 100.000%]`

Interpretation:

- The trained Transformer generalizes essentially perfectly to held-out `(N, B)` intersections defined by checkpoint-specific `N` and `B` sets.
- This supports the opening empirical claim that the task is solved systematically enough to justify mechanistic analysis, rather than by memorizing the exact held-out `(N, B)` training pairs.

### Companion checks

Using the same held-out autoregressive evaluation protocol on the available seed-1337 5-layer companion checkpoint:

| Model condition | Held-out test examples | Exact 2-digit answer accuracy | `O[0]` token accuracy | `O[1]` token accuracy |
| --- | ---: | ---: | ---: | ---: |
| 5-layer, shuffled training order (`pTrue`) | 1857 | 99.892% | 99.946% | 99.892% |

Interpretation:

- The task is already solved to very high held-out accuracy by the 5-layer companion model.
- Because only one trained checkpoint is currently available for this companion condition, it should be described as a qualitative robustness check rather than as a seed-averaged comparison.

## Result 02 — Closed-form quantities are linearly represented

Using autoregressively collected residual streams and 5-fold cross-validated linear probes on pooled held-out validation-plus-test examples, quantities aligned with the closed-form solution are strongly decodable from the network.

Main 10-layer, three-seed summary at the strongest across-seed locus for each quantity:

| Quantity | Reported stream/layer | Seed 0 CV `R²` | Seed 42 CV `R²` | Seed 1337 CV `R²` | Mean CV `R²` | 95% bootstrap CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `B^D` | `D_ones`, layer `0` | 0.999 | 0.996 | 1.000 | 0.998 | [0.996, 1.000] |
| `N / B^D` | `D_ones`, layer `2` | 0.968 | 0.955 | 0.957 | 0.960 | [0.955, 0.968] |
| `floor(N / B^D)` | `D_ones`, layer `2` | 0.968 | 0.955 | 0.957 | 0.960 | [0.955, 0.968] |
| `floor(N / B^D) mod B` | `O[1]`, layer `9` | 0.947 | 0.931 | 0.943 | 0.940 | [0.931, 0.947] |

Interpretation and nuance:

- The first three closed-form quantities are most strongly decoded from the `D_ones` stream, with `B^D` peaking earlier than the quotient-like quantities in the across-seed mean curves.
- The final answer quantity is strongest in the output stream, especially late `O[1]`.
- A scalar-input control confirms that the high activation-probe scores are not explained merely by linear dependence of the targets on raw `N`, `B`, and `D`.
- The layer of the single-seed maximum is not identical for every seed for the quotient-like quantities; the paper-facing claim is about the across-seed layer-wise pattern, not exact equality of every individual maximum.
- The separate 5-layer checkpoint shows the same qualitative pattern.
- Control streams do not show the same closed-form decodability pattern: across the main 10-layer seeds, the best observed mean CV `R²` for `B_ones` is approximately `0.034` for the final-answer quantity and negative for the other three quantities; `N_ones` remains negative for all four quantities in this sweep.

Scalar-input control over the checkpoint-specific held-out test sets:

| Target | Seed 0 CV `R²` | Seed 42 CV `R²` | Seed 1337 CV `R²` | Mean CV `R²` | 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B^D` | 0.254 | 0.233 | 0.328 | 0.272 | [0.233, 0.328] |
| `N / B^D` | 0.532 | 0.544 | 0.547 | 0.541 | [0.532, 0.547] |
| `floor(N / B^D)` | 0.532 | 0.544 | 0.547 | 0.541 | [0.532, 0.547] |
| `floor(N / B^D) mod B` | 0.323 | 0.321 | 0.368 | 0.337 | [0.321, 0.368] |

This is the source for the paper's brief scalar-input-control sentence in the probing section.

## Result 03 — Output streams depend primarily on early `D_ones` information

We cumulatively masked only the attention edges from the output streams `O[0]` and `O[1]` to the `D_ones` source stream, leaving all other edges intact, and measured exact two-digit answer accuracy on the full checkpoint-specific held-out test sets.

Main 10-layer, three-seed summary:

### Forward sweep: mask shallow-to-deep

| Masked layers | Mean exact accuracy | 95% bootstrap CI |
| --- | ---: | ---: |
| clean | 99.83% | [99.50%, 100.00%] |
| `0` | 99.83% | [99.50%, 100.00%] |
| `0-1` | 73.08% | [69.36%, 78.08%] |
| `0-2` | 55.85% | [52.36%, 58.50%] |
| `0-3` | 45.71% | [41.08%, 50.19%] |
| all layers | 32.06% | [31.72%, 32.62%] |

### Reverse sweep: mask deep-to-shallow

| Masked layers | Mean exact accuracy | 95% bootstrap CI |
| --- | ---: | ---: |
| clean | 99.83% | [99.50%, 100.00%] |
| `9` | 99.83% | [99.50%, 100.00%] |
| `4-9` | 99.75% | [99.45%, 100.00%] |
| `3-9` | 97.94% | [96.02%, 99.40%] |
| `2-9` | 91.04% | [82.50%, 95.79%] |
| `1-9` | 35.06% | [32.36%, 38.47%] |
| all layers | 32.06% | [31.72%, 32.62%] |

Per-seed exact accuracies for the full forward sweep:

| Masked layers | Seed 0 | Seed 42 | Seed 1337 |
| --- | ---: | ---: | ---: |
| clean | 99.50% | 100.00% | 100.00% |
| `0` | 99.50% | 100.00% | 100.00% |
| `0-1` | 71.79% | 69.36% | 78.08% |
| `0-2` | 58.50% | 52.36% | 56.70% |
| `0-3` | 45.86% | 41.08% | 50.19% |
| `0-4` | 44.03% | 36.32% | 45.29% |
| `0-5` | 39.41% | 35.21% | 40.98% |
| `0-6` | 37.13% | 33.23% | 38.07% |
| `0-7` | 35.20% | 32.12% | 35.43% |
| `0-8` | 33.32% | 32.12% | 33.17% |
| all layers | 32.62% | 31.84% | 31.72% |

Per-seed exact accuracies for the full reverse sweep:

| Masked layers | Seed 0 | Seed 42 | Seed 1337 |
| --- | ---: | ---: | ---: |
| clean | 99.50% | 100.00% | 100.00% |
| `9` | 99.50% | 100.00% | 100.00% |
| `8-9` | 99.50% | 100.00% | 100.00% |
| `7-9` | 99.50% | 100.00% | 100.00% |
| `6-9` | 99.45% | 100.00% | 100.00% |
| `5-9` | 99.45% | 100.00% | 99.95% |
| `4-9` | 99.45% | 100.00% | 99.78% |
| `3-9` | 98.41% | 99.40% | 96.02% |
| `2-9` | 95.79% | 94.82% | 82.50% |
| `1-9` | 38.47% | 34.33% | 32.36% |
| all layers | 32.62% | 31.84% | 31.72% |

Interpretation and nuance:

- Ablating layer `0` alone has essentially no effect, but adding layer `1` causes the first large behavioral drop.
- Deep routes alone are weakly causal: masking through `4-9` leaves performance near baseline, and the reverse sweep does not collapse until layer `1` is included.
- The result does not say that later `D_ones` states are uninformative; Analysis 02 shows that quotient-like quantities are strongly decodable there. It says that the output streams rely most heavily on the early route for behavior.
- The separate 5-layer checkpoint shows the same qualitative structure.

## Result 04 — Layer-1 `D_ones -> O` attention is the stronger normal-operation route on average

Using ordered source→donor test pairs matched on the same `(N, B)` and differing only in `D`, we patched donor `D_ones` `K/V` into the output-stream query rows while leaving the source `D_ones` stream itself unchanged.

Main 10-layer, three-seed exact-answer result:

Figure: `figures/05_dones_information_content_patching/main_10layer_ptrue_test_matchedsources_clean_correct_source_vs_donor_exact_accuracy.png`

| condition | Seed 0 source / donor | Seed 42 source / donor | Seed 1337 source / donor | Mean source / donor | 95% CI source | 95% CI donor |
|---|---:|---:|---:|---:|---:|---:|
| clean | 99.55% / 0.00% | 100.00% / 0.00% | 100.00% / 0.00% | 99.85% / 0.00% | [99.55%, 100.00%] | [0.00%, 0.00%] |
| `L1` | 13.58% / 81.13% | 16.59% / 79.03% | 45.18% / 42.72% | 25.11% / 67.63% | [13.58%, 45.18%] | [42.72%, 81.13%] |
| `L0 + L2+` | 81.13% / 13.58% | 79.03% / 16.59% | 42.72% / 45.18% | 67.63% / 25.11% | [42.72%, 81.13%] | [13.58%, 45.18%] |

Per-seed digit-level results:

| condition | seed | source exact | donor exact | source `O[0]` | source `O[1]` | donor `O[0]` | donor `O[1]` |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0 | 99.55% | 0.00% | 99.97% | 99.55% | 79.11% | 2.80% |
| `L1` | 0 | 13.58% | 81.13% | 86.44% | 15.57% | 91.96% | 83.00% |
| `L0 + L2+` | 0 | 81.13% | 13.58% | 91.96% | 83.00% | 86.44% | 15.57% |
| clean | 42 | 100.00% | 0.00% | 100.00% | 100.00% | 88.19% | 1.30% |
| `L1` | 42 | 16.59% | 79.03% | 92.45% | 17.65% | 95.20% | 80.01% |
| `L0 + L2+` | 42 | 79.03% | 16.59% | 95.20% | 80.01% | 92.45% | 17.65% |
| clean | 1337 | 100.00% | 0.00% | 100.00% | 100.00% | 70.24% | 3.39% |
| `L1` | 1337 | 45.18% | 42.72% | 86.02% | 47.81% | 81.38% | 45.99% |
| `L0 + L2+` | 1337 | 42.72% | 45.18% | 81.38% | 45.99% | 86.02% | 47.81% |

Main 10-layer digit-level means across seeds:

| condition | source `O[0]` | source `O[1]` | donor `O[0]` | donor `O[1]` |
|---|---:|---:|---:|---:|
| clean | 99.99% | 99.85% | 79.18% | 2.50% |
| `L1` | 88.30% | 27.01% | 89.51% | 69.67% |
| `L0 + L2+` | 89.51% | 69.67% | 88.30% | 27.01% |

5-layer companion (`pTrue`, seed 1337 only):

| condition | source exact | donor exact |
|---|---:|---:|
| clean | 99.90% | 0.00% |
| `L1` | 21.42% | 63.43% |
| `L0 + L2+` | 63.43% | 21.42% |

Interpretation and nuance:

- During normal operation, substituting only the layer-1 `D_ones -> O` attentional readout strongly steers the answer toward the donor `D` value on average across seeds.
- The complementary `L0 + L2+` routes produce the opposite balance: on average they preserve substantially more source behavior than donor behavior.
- The two patched conditions are nearly mirror images under ordered source→donor pairing, which is a useful sanity check on the route split.
- The 5-layer checkpoint reproduces the same qualitative dissociation.


## Result 05 — The full `D_ones -> O` route is selectively `D`-dependent

Using the same held-out test sources across all three conditions, we paired each retained source with one donor that differed only in `N`, one that differed only in `B`, and one that differed only in `D`. Sources were retained only if the unpatched model answered them exactly correctly, so the analysis isolates transfer away from a solved source computation. We then patched the donor `D_ones` `K/V` readout into the output-stream query rows at every layer while leaving the source `D_ones` stream itself unchanged.

Matched held-out test sources before and after the clean-correctness filter:

| seed | before filter | retained |
| ---: | ---: | ---: |
| 0 | 1696 | 1686 |
| 42 | 1821 | 1821 |
| 1337 | 1552 | 1552 |

Main 10-layer, three-seed exact-answer result:

| condition | Seed 0 source / donor | Seed 42 source / donor | Seed 1337 source / donor | Mean source / donor | 95% CI source | 95% CI donor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 100.00% / 0.00% | 100.00% / 0.00% | 100.00% / 0.00% | 100.00% / 0.00% | [100.00%, 100.00%] | [0.00%, 0.00%] |
| `vary_N` | 100.00% / 0.00% | 100.00% / 0.00% | 100.00% / 0.00% | 100.00% / 0.00% | [100.00%, 100.00%] | [0.00%, 0.00%] |
| `vary_B` | 100.00% / 0.00% | 100.00% / 0.00% | 100.00% / 0.00% | 100.00% / 0.00% | [100.00%, 100.00%] | [0.00%, 0.00%] |
| `vary_D` | 0.00% / 99.53% | 0.00% / 100.00% | 0.00% / 100.00% | 0.00% / 99.84% | [0.00%, 0.00%] | [99.53%, 100.00%] |

Main 10-layer digit-level means across seeds:

| condition | source `O[0]` | source `O[1]` | donor `O[0]` | donor `O[1]` |
| --- | ---: | ---: | ---: | ---: |
| clean | 100.00% | 100.00% | 78.85% | 2.16% |
| `vary_N` | 100.00% | 100.00% | 78.85% | 2.16% |
| `vary_B` | 100.00% | 100.00% | 70.96% | 3.38% |
| `vary_D` | 72.95% | 3.05% | 100.00% | 99.84% |

5-layer companion (`pTrue`, seed 1337 only):

| matched before filter | retained |
| ---: | ---: |
| 1552 | 1550 |

| condition | source exact | donor exact |
| --- | ---: | ---: |
| clean | 100.00% | 0.00% |
| `vary_N` | 100.00% | 0.00% |
| `vary_B` | 100.00% | 0.00% |
| `vary_D` | 0.00% | 99.87% |

Interpretation and nuance:

- Replacing the **entire** `D_ones -> O` route flips behavior almost perfectly when only `D` changes.
- The same intervention leaves the model completely source-like when only `N` or only `B` changes.
- Because the same clean-correct source examples are used in all three donor conditions, the contrast is not an artifact of comparing different source populations or of pre-existing source errors.
- Thus, the behaviorally effective information carried from `D_ones` to the output streams is highly selective for `D`-dependent information, even though `D_ones` residuals also linearly encode quantities involving `N` and `B` in Analysis 02.
- The 5-layer checkpoint shows the same qualitative selectivity.

## Result 06 — A sparse, mostly factorized circuit preserves most held-out performance

Using the greedy right-to-left circuit search, we identified checkpoint-specific sparse attention circuits from each checkpoint's full validation split and evaluated each frozen kept-only circuit autoregressively on that checkpoint's held-out test split.

Main 10-layer, three-seed result:

| Seed | Clean test exact accuracy | Kept-only test exact accuracy | Retained relations | Retained layer-edges |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 99.50% | 90.83% | 22 / 79 (27.85%) | 106 / 500 (21.20%) |
| 42 | 100.00% | 91.68% | 24 / 84 (28.57%) | 97 / 483 (20.08%) |
| 1337 | 100.00% | 97.68% | 24 / 84 (28.57%) | 103 / 529 (19.47%) |

Across the three 10-layer checkpoints:

- clean held-out exact accuracy: mean `99.83%`, 95% bootstrap CI `[99.50%, 100.00%]`
- kept-only held-out exact accuracy: mean `93.40%`, 95% bootstrap CI `[90.83%, 97.68%]`
- retained relation fraction: mean `28.33%`, 95% bootstrap CI `[27.85%, 28.57%]`
- retained layer-edge fraction: mean `20.25%`, 95% bootstrap CI `[19.47%, 21.20%]`

The three independently discovered relation sets are highly overlapping:

| Comparison | Shared relations | Union relations | Relation-level overlap |
| --- | ---: | ---: | ---: |
| Three-seed intersection | 17 | 18 | 94.44% intersection-over-union |
| Seed `0` vs seed `42` | 17 | 18 | 94.44% Jaccard overlap |
| Seed `0` vs seed `1337` | 17 | 18 | 94.44% Jaccard overlap |
| Seed `42` vs seed `1337` | 18 | 18 | 100.00% Jaccard overlap |

Coverage of each checkpoint-specific circuit by the all-seed intersection:

| Seed | Retained relations in that seed | Relations also in all-seed intersection | Coverage |
| ---: | ---: | ---: | ---: |
| 0 | 17 | 17 | 100.00% |
| 42 | 18 | 17 | 94.44% |
| 1337 | 18 | 17 | 94.44% |

The only retained relation that is not universal is `B_tens -> O[0]`, which is present in seeds `42` and `1337` but absent in seed `0`.


Threshold-sweep robustness check:

- Grid: first-drop threshold `{0.01, 0.02, 0.05}` crossed with later-drop fraction `{0.10, 0.20, 0.30}`, for `27` seed-by-threshold checks.
- Reference: the all-seed shared circuit from the default `0.02 / 0.20` cell, containing `17` relations.
- Mean relation-set overlap with the reference circuit: `92.77%`.
- Range of relation-set overlap with the reference circuit: `83.33%` to `100.00%`.
- Checks containing all `17` shared relations: `21 / 27`.
- Exact reference matches: `9 / 27`.
- Upstream routes first combining `N`, `B`, and `D`: no stable route was found. The only exception was `N_tag -> B_tag` for seed `1337` at the loosest first-drop threshold (`3 / 27` checks).

Relations retained in all three circuits:

```text
N_tag -> N_hundreds
N_tag -> N_tens
N_hundreds -> O[0], O[1]
N_tens     -> O[0], O[1]
N_ones     -> O[0], O[1]

B_tag  -> B_tens
B_tag  -> B_ones
B_tens -> B_ones
B_ones -> O[0], O[1]

D_tag  -> D_ones
D_ones -> O[0], O[1]

O[0]   -> O[1]
```

Interpretation and nuance:

- The retained circuits are sparse yet behaviorally substantial: roughly one-fifth of candidate layer-edges preserve over `93%` mean held-out exact accuracy.
- Relation-level stability is very high across independent training seeds: the all-seed intersection contains `17 / 18` union routes, and the only non-universal route is `B_tens -> O[0]`.
- The shared backbone is mostly factorized before the outputs. `N` streams show local number-structure routes, `B` streams show local base-structure routes, and `D_tag` supports `D_ones`; these routes then project largely into `O[0]` and `O[1]`.
- The outputs are not independent: `O[0] -> O[1]` is retained in every checkpoint-specific circuit.
- The required depth differs by route family. The local upstream scaffold routes are shallow, whereas many routes from the digit-bearing streams into the outputs remain important much deeper into the stack, especially for the `N` pathways.
- The search is greedy and does not establish formal minimality or uniqueness. The paper-facing claim should be that the model admits a sparse, performance-preserving, mostly factorized circuit consistent across independently trained checkpoints.
