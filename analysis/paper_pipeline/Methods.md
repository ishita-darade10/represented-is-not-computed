# Methods Ledger

This file records the methodological decisions for the clean paper-results pipeline. It is the authoritative protocol ledger for drafting the paper and for the linked code release; `README.md` remains the navigational document for running the code.

## Global experimental setting

### Task

Each example presents:

- a decimal integer `N`
- a base `B`
- a digit position `D`

The target is the coefficient of `B^D` in the base-`B` representation of `N`:

```text
floor(N / B^D) mod B
```

### Dataset domain

- `N ∈ {0, ..., 999}`
- `B ∈ {2, ..., 30}`
- for each `(N, B)`, valid `D` values run from `0` through the highest required base digit position, plus one additional out-of-range position with target `00`
- for `N = 0`, `D = 0` is treated as the in-range digit position
- because `D` is represented by a single character token, queries are capped at `D <= 9`

### Input formatting

- canonical analysis order: `N...B...D...O`
- training may use all six permutations of the `N`, `B`, and `D` fields, depending on the checkpoint variant
- validation and test examples remain canonical in the current data pipeline

### Model family

- decoder-only Transformer
- default hidden size and attention/MLP dimensions are inherited from `config.py`
- checkpoint filenames determine analysis-time overrides that can vary across runs:
  - number of layers
  - split mode
  - train-time field-order permutation flag
  - seed
  - selected checkpoint (`best` or `last`)

## Split reconstruction policy

All paper analyses must reconstruct evaluation splits from the `split_info` object embedded in each checkpoint, rather than from the current mutable values in `config.py`.

This is necessary because:

1. different seeds imply different held-out value sets used to define the evaluation intersections
2. the realized sample counts depend on which intersection-defining values were selected, because valid `D` counts vary with `(N, B)`
3. rerunning the split logic from today's config can silently mismatch the network that was actually trained

### `by_NB_intersection` splits

The current main checkpoints use the `by_NB_intersection` regime.

Configured holdout policy in `config.py`:

- validation `N` fraction: `0.10`
- test `N` fraction: `0.10`
- validation `B` fraction: `0.20`
- test `B` fraction: `0.20`

Realized value sets used to define the held-out intersections under the current domain:

- `100 / 1000` held-out `N` values for validation
- `100 / 1000` held-out `N` values for test
- `5 / 29` held-out `B` values for validation
- `5 / 29` held-out `B` values for test

The base fractions realize as `5 / 29 ≈ 17.24%` rather than exactly `20%` because the current splitter uses integer truncation when selecting held-out base sets.

Assignment rule:

- validation examples satisfy `(N ∈ val_Ns) AND (B ∈ val_Bs)`
- test examples satisfy `(N ∈ test_Ns) AND (B ∈ test_Bs)`
- all remaining examples are assigned to train

The exact intersection-defining value sets and sample counts are checkpoint-specific and must be read from the saved `split_info`. Values in `test_Ns` and `test_Bs` can each still appear in training outside their held-out intersection; only examples satisfying both conditions are assigned to test.

## Checkpoint contract

Named checkpoints currently follow:

```text
model_t_l{layers}_m{split_mode}_p{train_all_permutations}_s{seed}_{best|last}.pt
```

Example:

```text
model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt
```

means:

- 10-layer Transformer
- `by_NB_intersection` split mode
- train-time six-permutation expansion enabled
- seed `1337`
- best validation checkpoint

## Analysis ledger

Each finalized analysis section below records:

1. scientific question
2. checkpoint(s) used
3. split(s) used
4. intervention / probe / metric definition
5. parameter values and selection logic
6. outputs written to disk
7. caveats needed for paper interpretation

## Analysis 01 — Held-out autoregressive test performance

### Scientific question

Can the trained 10-layer Transformer solve the task on held-out `(N, B)` combinations when it must emit the answer autoregressively?

### Checkpoints

Main three-seed condition:

- `model_t_l10_mby_NB_intersection_pTrue_s0_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s42_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt`

All three checkpoints use:

- 10 Transformer layers
- `by_NB_intersection` split construction
- train-time expansion over all six `N/B/D` field-order permutations
- best-checkpoint selection by validation digit accuracy

Companion checkpoint evaluated with the same protocol:

- `model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt`

This is reported as a companion single-run check rather than as an across-seed estimate, because only one trained checkpoint is currently available for that condition.

### Evaluation split and input order

- Each checkpoint is evaluated on its own checkpoint-stored canonical test split reconstructed from embedded `split_info`.
- Test inputs use the canonical `N,B,D` field order, even though training used shuffled field orders.
- Because `split_info` is checkpoint-specific, seeds `0`, `42`, and `1337` can have different held-out `N` and `B` identities and different realized test-set sizes.

### Generation protocol

- The model receives the prompt through the `O` marker only.
- It then greedily generates `Config.target_len = 3` tokens autoregressively:
  - first answer digit `O[0]`
  - second answer digit `O[1]`
  - end token `E`
- Only `O[0]` and `O[1]` are used for the reported answer accuracies below.

### Metrics

- `exact_answer_accuracy`: fraction of examples for which both generated answer digits, `O[0]` and `O[1]`, exactly match the two target digits
- `o0_token_accuracy`: fraction of examples for which `O[0]` matches its target digit
- `o1_token_accuracy`: fraction of examples for which `O[1]` matches its target digit

The end token is generated during evaluation but excluded from these reported answer metrics because the paper claim concerns correctness of the two-digit answer itself.

### Aggregation across seeds

- Per-seed accuracies are reported directly from each model's full held-out test set.
- The paper-level three-seed summary reports:
  - the arithmetic mean across the three independently trained networks
  - a two-sided 95% percentile-bootstrap confidence interval over the seed-level mean accuracy
  - `100,000` bootstrap resamples with bootstrap RNG seed `20260517`

Rationale:

- Wilson intervals are appropriate for binomial uncertainty over examples within a fixed model/test set.
- The cross-run claim here is about variation across trained networks, so the model seed is the replicate unit.
- Bootstrap resampling keeps the interval on the natural `[0, 1]` support of accuracy without post hoc clipping.
- With only three seeds, the interval is necessarily coarse and should be read as descriptive rather than as a finely estimated population interval.

### Outputs

The analysis script writes:

- one row per checkpoint under `data/01_test_set_performance/<run_label>/per_seed_results.csv`
- one row per metric under `data/01_test_set_performance/<run_label>/across_seed_summary.csv`
- auditable metadata under `data/01_test_set_performance/<run_label>/metadata.json`

Run labels are used to keep statistically distinct conditions separate on disk:

- `main_10layer_ptrue`
- `companion_5layer_ptrue`

## Analysis 02 — Linear probing of closed-form quantities

### Scientific question

Are quantities aligned with the natural closed-form solution linearly decodable from the model's residual streams, and how does decodability evolve across depth and token stream?

### Checkpoints

Main three-seed condition:

- `model_t_l10_mby_NB_intersection_pTrue_s0_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s42_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt`

Separate companion condition:

- `model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt`

The 5-layer model is analyzed with the same protocol but is not pooled with the three 10-layer seeds.

### Probe targets

Four scalar quantities from the closed-form solution are probed:

1. `B^D`
2. `N / B^D`
3. `floor(N / B^D)`
4. `floor(N / B^D) mod B`

The final quantity equals the requested answer value.

### Evaluation pool and split reconstruction

- Each checkpoint uses its own checkpoint-stored `split_info`.
- Canonical validation and test examples are reconstructed from that stored split metadata and pooled together before cross-validation.
- The pooled held-out set is used to expose the probes to a wider range of held-out `N` and `B` values than either validation or test alone.
- No train examples are used in this analysis.

### Autoregressive activation protocol

All activations are collected autoregressively, not with teacher forcing.

For each canonical held-out prompt:

1. run the prompt through the `O` marker
2. greedily generate the first answer digit
3. append that generated digit and run the model again to obtain the stream that predicts the second answer digit

Streams analyzed:

- `N_tag`
- `N_hundreds`
- `N_tens`
- `N_ones`
- `B_tag`
- `B_tens`
- `B_ones`
- `D_tag`
- `D_ones`
- `O[0]`
- `O[1]`

The analysis includes non-hypothesis control streams such as `B_ones` and `N_ones` in the full sweep. In the current exploratory figure they are rendered with low opacity, while `D_ones`, `O[0]`, and `O[1]` are visually emphasized.

Definitions:

- `O[0]` is the residual stream at the `O` marker position, from which the first answer digit is predicted.
- `O[1]` is the residual stream at the first generated answer-digit position, from which the second answer digit is predicted.
- `O[2]` is excluded because it is the stream used to predict the end token `E`, not an answer digit.

Representations analyzed:

- embedding-plus-position residual stream before any Transformer block, labeled `input`
- post-block residual outputs for every Transformer layer

### Probe model and scoring

- For each `(target, representation, stream)` combination, fit an ordinary least squares linear regression probe with:
  - feature standardization fit on the training folds only
  - an explicit intercept term
- The implementation uses a QR-based least-squares solver (`torch.linalg.lstsq(..., driver="gelsy")`) for numerical robustness while preserving the unregularized OLS estimand.
- No regularization is used in the primary analysis.
- Rationale for unregularized OLS:
  - the pooled held-out dataset contains several thousand examples per checkpoint
  - the feature dimension is 384
  - OLS is therefore well-posed and keeps the primary estimand simple
- Probes are scored with shuffled 5-fold cross-validated `R²` within the pooled validation-plus-test set.
- Fold assignment uses analysis RNG seed `20260517`.

### Scalar-input control

The scalar-input control asks whether the probe targets are already easy linear functions of the raw input scalars alone. This is a control on the interpretation of probe magnitude, not an activation analysis.

For each of the three main 10-layer seeds:

1. reconstruct that seed's checkpoint-specific canonical test split from `split_info`
2. build a 3-dimensional feature matrix containing raw scalar `N`, `B`, and `D`
3. use the same four target definitions as the activation probes
4. fit ordinary least squares regressions with an explicit intercept and train-fold standardization
5. score with shuffled 5-fold cross-validated `R²`

The fold RNG seed is the checkpoint seed for this control (`0`, `42`, or `1337`). Across-seed means and 95% percentile-bootstrap confidence intervals use the same seed-level bootstrap convention as the rest of the paper (`100,000` resamples, bootstrap RNG seed `20260517`).

### Aggregation and visualization

Main 10-layer figure:

- one 2×2 figure with one panel per closed-form quantity
- x-axis: `input`, then post-block layers `0...9`
- y-axis: mean cross-validated `R²`
- one line per stream
- seed-level probe scores are averaged across the three 10-layer seeds
- 95% percentile-bootstrap confidence intervals are computed over seed-level mean scores with:
  - `100,000` bootstrap resamples
  - bootstrap RNG seed `20260517`
- `D_ones`, `O[0]`, and `O[1]` are visually emphasized; all other streams are shown with lower opacity

Separate 5-layer figure:

- same 2×2 structure
- layers `input`, `0...4`
- no across-seed interval because only one 5-layer checkpoint is currently available

### Outputs

The analysis writes:

- seed-level probe rows under `data/02_linear_probing/<run_label>/per_seed_probe_rows.csv`
- across-seed summaries under `data/02_linear_probing/<run_label>/across_seed_summary.csv`
- scalar-input control rows under `data/02_linear_probing/<run_label>/scalar_input_control_per_seed.csv`
- scalar-input control summaries under `data/02_linear_probing/<run_label>/scalar_input_control_summary.csv`
- metadata under `data/02_linear_probing/<run_label>/metadata.json`
- scalar-input control metadata under `data/02_linear_probing/<run_label>/scalar_input_control_metadata.json`
- figures under `figures/02_linear_probing/`

## Analysis 03 — Cumulative attention ablation from `D_ones` into the output streams

### Scientific question

Which depths of the `D_ones` stream are causally read by the output streams during answer generation?

### Checkpoints

Main three-seed condition:

- `model_t_l10_mby_NB_intersection_pTrue_s0_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s42_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt`

Separate companion condition:

- `model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt`

### Evaluation split and generation protocol

- Each checkpoint uses its own checkpoint-stored canonical test split reconstructed from embedded `split_info`.
- Evaluation is greedy and autoregressive.
- The model generates both answer digits and the end token, but the reported metric is exact two-digit answer accuracy over `O[0]` and `O[1]`.

### Intervention

At selected layers, attention from both output queries jointly to the `D_ones` source stream is blocked:

- query positions:
  - `O[0]`: the `O` marker position during first-digit generation
  - `O[1]`: the first generated answer-digit position during second-digit generation
- source position:
  - `D_ones`
- only these query-to-source attention edges are masked
- masking is implemented by setting the selected pre-softmax attention logits to `-1e9`, yielding zero post-softmax attention weight after renormalization over the remaining source positions
- all other attention edges and all residual streams remain unchanged

### Cumulative sweeps

Two cumulative masking orders are evaluated:

- forward sweep:
  - mask layer `0`
  - then layers `0-1`
  - then `0-2`
  - continuing through all layers
- reverse sweep:
  - mask the deepest layer only
  - then the deepest two layers
  - continuing upward until all layers are masked

For each sweep, a clean unmasked baseline is also evaluated.

### Run-size policy

- Development/timing pass: first `200` checkpoint-specific test examples
- Final paper pass: all checkpoint-specific held-out test examples

### Aggregation and visualization

- Main three-seed 10-layer condition:
  - mean exact-answer accuracy across seeds
  - 95% percentile-bootstrap confidence intervals over seed-level means
  - `100,000` bootstrap resamples with RNG seed `20260517`
- Separate 5-layer condition:
  - plotted separately, without across-seed intervals because only one trained checkpoint is currently available
- Forward and reverse sweeps are plotted side by side.

### Outputs

- seed-level sweep rows: `data/03_dones_to_output_attention_ablation/<run_label>/per_seed_rows.csv`
- across-seed summaries: `data/03_dones_to_output_attention_ablation/<run_label>/across_seed_summary.csv`
- metadata: `data/03_dones_to_output_attention_ablation/<run_label>/metadata.json`
- figures: `figures/03_dones_to_output_attention_ablation/`

## Analysis 04 — `D_ones -> O` attentional K/V patching: layer 1 versus `L0 + L2+`

### Scientific question

When `D` changes while `(N, B)` are fixed, how much of the resulting behavioral change reaches the output streams through the layer-1 `D_ones -> O` attention route, versus through the complementary `D_ones -> O` routes at layer `0` and layers `2+`?

### Pair construction

- Examples are drawn from each checkpoint's own checkpoint-stored canonical test split.
- Source and donor examples are matched on identical `(N, B)` and differ in `D`.
- Only ordered source→donor pairs with different two-digit answers are retained.
- Ordered source→donor pairs available in the current test splits:
  - 10-layer seed 0: `5812`
  - 10-layer seed 42: `6624`
  - 10-layer seed 1337: `4892`
  - 5-layer seed 1337: `4892`

### Intervention logic

This is an output-attention route substitution, not a forward-propagating residual patch.

For each matched source-donor pair, both source and donor are run normally to obtain the layerwise attention-side states needed for patching. The patched model is then evaluated autoregressively on the source prompt.

Layer indexing follows the actual attention computation: at layer `L`, the `D_ones` key/value vectors and the output-stream query vectors are computed from the residual stream entering layer `L` (that is, after block `L-1`, or from the embedding-plus-position residual for layer `0`).

Two causal conditions are defined:

1. `l1_only`
   - the source run remains the base computation
   - for output-stream queries (`O[0]` during first-digit generation and `O[1]` during second-digit generation), donor-derived `K` and `V` for the `D_ones` source position are substituted at Transformer layer `1` only
   - all other edges and all residual-stream trajectories remain source-like

2. `l0_plus_l2plus`
   - the source run again remains the base computation
   - donor-derived `K` and `V` for the `D_ones` source position are substituted for output-stream queries at layer `0` and at all layers `>= 2`
   - layer `1` is left source-like
   - the source `D_ones` computation itself is not overwritten or rolled out from a donor residual state at any layer

Operationally, at every patched layer:

- the output queries remain source-like
- only the `D_ones` key/value slot seen by the output-stream query rows is donor-derived
- the source `D_ones -> D_ones` self-read remains source-like
- all non-output query rows and all non-`D_ones` attention edges remain source-like
- no donor residual state is propagated through the source network

Implementation detail needed to preserve this estimand:

- each Transformer block computes all query rows in parallel
- therefore, the patched attention result is constructed by first computing the fully source-like attention output for every query row, then replacing only the output-query rows with outputs recomputed using donor `D_ones` `K/V`
- globally replacing the `D_ones` value vector would also alter the `D_ones -> D_ones` self-read and would no longer be the intended intervention

This design isolates the attentional readout route from `D_ones` into the output streams while keeping the source `D_ones` computation intact.

### Metrics

Autoregressive generation is evaluated on source prompts after patching.

- `source_exact_rate`: fraction of patched outputs matching the original source answer
- `donor_exact_rate`: fraction of patched outputs matching the donor answer
- digit-level source/donor rates for `O[0]` and `O[1]`

### Run-size policy

- Development/timing pass: sample `200` valid ordered pairs per checkpoint
- Final paper pass: all valid ordered source→donor pairs from each checkpoint's own held-out canonical test split

### Aggregation

- Main three-seed 10-layer condition:
  - mean source/donor exact-match rates across seeds
  - 95% percentile-bootstrap confidence intervals over seed-level means
  - `100,000` bootstrap resamples with RNG seed `20260517`
- Separate 5-layer condition:
  - reported separately as a single-checkpoint companion result without across-seed intervals

### Outputs

- seed-level condition rows: `data/04_dones_output_attention_route_patching/<run_label>/per_seed_rows.csv`
- across-seed summaries: `data/04_dones_output_attention_route_patching/<run_label>/across_seed_summary.csv`
- metadata: `data/04_dones_output_attention_route_patching/<run_label>/metadata.json`


## Analysis 05 — Information communicated by the full `D_ones -> O` attentional route

### Scientific question

Which task variables are communicated from the `D_ones` stream to the output streams during normal operation? In particular, if the same source example is paired with donors that change only `N`, only `B`, or only `D`, which donor substitution changes the output behavior?

### Checkpoints

Main three-seed condition:

- `model_t_l10_mby_NB_intersection_pTrue_s0_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s42_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt`

Separate companion condition:

- `model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt`

### Evaluation pool

- Each checkpoint uses its own checkpoint-stored canonical test split reconstructed from embedded `split_info`.
- Validation and training examples are not used in the primary analysis because the held-out test split already contains enough sources that admit all three controlled donor types.

### Matched-source donor construction

The analysis first identifies source examples for which all three donor types exist within the same held-out test split:

1. `vary_N`: same `(B, D)`, different `N`
2. `vary_B`: same `(N, D)`, different `B`
3. `vary_D`: same `(N, B)`, different `D`

For a source to be retained:

- one valid donor must exist for each of the three conditions above
- each donor must differ from the source in exactly the named field and no others
- each donor must have a different two-digit answer from the source
- the unpatched model must answer the source exactly correctly autoregressively

The clean-correctness requirement is applied uniformly to every checkpoint so that the experiment measures behavioral transfer from a solved source computation rather than mixing transfer effects with pre-existing model errors.

For every retained source, one donor per condition is sampled deterministically with analysis RNG seed `20260517`. The same retained source set is therefore evaluated under all three donor conditions.

Number of matched held-out test sources before and after the clean-correctness filter:

| model | seed | matched before filter | retained after filter |
| --- | ---: | ---: | ---: |
| 10-layer `pTrue` | 0 | 1696 | 1686 |
| 10-layer `pTrue` | 42 | 1821 | 1821 |
| 10-layer `pTrue` | 1337 | 1552 | 1552 |
| 5-layer `pTrue` | 1337 | 1552 | 1550 |

### Intervention logic

The intervention is the full-route version of Analysis 04.

For each source-donor pair:

- both source and donor are run normally to compute attention-side states
- for every Transformer layer, donor-derived `K` and `V` for the `D_ones` source position are substituted only into the output-stream query rows:
  - `O[0]` during first-digit generation
  - `O[1]` during second-digit generation
- all output queries remain source-like
- the source `D_ones -> D_ones` self-read remains source-like
- all non-output query rows and all non-`D_ones` edges remain source-like
- no donor residual state is propagated through the source network

Thus, the experiment asks what behaviorally effective information the **full `D_ones -> O` route** carries, rather than what is merely represented within the `D_ones` residual stream.

### Generation and metrics

- Evaluation is greedy and autoregressive on source prompts.
- Reported metrics:
  - `source_exact_rate`
  - `donor_exact_rate`
  - source/donor token-level match rates for `O[0]` and `O[1]`
- The clean baseline is evaluated on the same retained source set used in all three donor conditions and is therefore exactly `100%` by construction for source accuracy.

### Aggregation

- Main three-seed 10-layer condition:
  - mean rates across seeds
  - 95% percentile-bootstrap confidence intervals over seed-level means
  - `100,000` bootstrap resamples with RNG seed `20260517`
- Separate 5-layer condition:
  - reported separately as a single-checkpoint companion result without across-seed intervals

### Outputs

- seed-level rows: `data/05_dones_information_content_patching/<run_label>/per_seed_rows.csv`
- across-seed summaries: `data/05_dones_information_content_patching/<run_label>/across_seed_summary.csv`
- metadata: `data/05_dones_information_content_patching/<run_label>/metadata.json`
- grouped-bar summary figure: `figures/05_dones_information_content_patching/<run_label>_source_vs_donor_exact_accuracy.png`

## Analysis 06 — Greedy sparse-circuit discovery

### Scientific question

Which stream-to-stream attention routes are necessary for final task performance, how deep must each route remain available, and does the resulting retained circuit suggest mostly factorized routing of `N`, `B`, and `D` information before late output-side integration?

### Checkpoints

Main three-seed condition:

- `model_t_l10_mby_NB_intersection_pTrue_s0_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s42_best.pt`
- `model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt`

Separate companion condition:

- `model_t_l5_mby_NB_intersection_pTrue_s1337_best.pt`

Each checkpoint is analyzed independently to produce its own circuit. Circuits are compared qualitatively across seeds after discovery rather than pooled during the search itself.

### Split usage

- Discovery split: each checkpoint's own canonical validation split reconstructed from checkpoint-stored `split_info`
- Final sufficiency evaluation split: each checkpoint's own canonical held-out test split reconstructed from checkpoint-stored `split_info`
- The finalized paper analysis uses all available validation examples for discovery and all available held-out test examples for final sufficiency evaluation.
- The script also supports deterministic random validation subsets for development-time smoke tests and runtime checks; these capped runs are not used for the finalized paper result.

### Graph and search roots

The analyzed stream set is:

- prompt-side streams: `N_tag`, `N_hundreds`, `N_tens`, `N_ones`, `B_tag`, `B_tens`, `B_ones`, `D_tag`, `D_ones`
- output-side streams: `O[0]`, `O[1]`

The search begins from both output streams. `O[0]` and `O[1]` are treated as important through all available layers at the start of the search. The route `O[0] -> O[1]` is included inside the same discovery pass, so the analysis estimates which `O[0]` layers are required for `O[1]` rather than treating that connection as a separate post hoc check.

### Single-edge intervention

For a tested source stream `s`, destination stream `d`, and cumulative layer prefix `L0..Lk`, the analysis blocks only the attention relation `s -> d` at the selected layers by setting the corresponding pre-softmax attention logits to `-1e9` before the softmax. Every other attention logit remains unchanged.

All metrics are measured autoregressively. The model first generates `O[0]`, then conditions on its own generated first digit to generate `O[1]`.

### Recursive right-to-left discovery

For a given destination task, the analysis tests all causally admissible earlier source streams. After each cumulative prefix ablation, it measures the drop in validation-set exact two-digit answer accuracy relative to the clean validation baseline.

A relation is tagged as important only if its cumulative ablation curve exhibits a meaningful validation-set drop. The retained prefix is chosen by the same greedy elbow rule used in the exploratory analysis:

- minimum exact-accuracy drop threshold: `0.02`
- later-layer increment threshold: later layers are retained only if they add more than `20%` of the first meaningful jump
- plateau patience: stop after `2` consecutive non-meaningful increments

### Threshold-sweep robustness check

The paper-facing robustness check repeats the discovery over a `3 x 3` grid of greedy-rule thresholds:

- first-drop threshold: `0.01`, `0.02`, `0.05`
- later-drop fraction: `0.10`, `0.20`, `0.30`
- plateau patience fixed at `2`

For each grid cell, discovery uses the full validation split for each of the three main 10-layer seeds, and kept-only sufficiency evaluation uses that seed's held-out test split. The default paper circuit is the `0.02 / 0.20` cell. The threshold-sweep outputs include relation overlap with the Fig. 4 shared circuit and whether each sweep circuit contains all `17` shared relations.

If a source stream is retained as important for a downstream destination, the analysis recursively asks which earlier streams support that source stream, but only through the deepest layer prefix that was required downstream. This constrains the upstream search to the depth at which that stream actually mattered for final performance.

### Route provenance: “important through”

Importance is cumulative with depth, but the analysis also records *through which downstream stream* each upstream dependency mattered. Each recursive task therefore carries a `through` field. For example, a row can encode that an upstream source was important for `D_ones` *through* `O[0]`, rather than merely stating that `D_ones` was important in the aggregate. This preserves the circuit's connectivity and makes the final graph interpretable as a routed support structure for end performance.

### Final kept-only circuit evaluation

After discovery is complete for one checkpoint, all retained source-to-destination layer prefixes are unioned into a single checkpoint-specific circuit. The model is then rerun autoregressively on the held-out test split while every non-retained admissible incoming attention relation into every constrained destination is masked at the relevant layers. This produces a direct sufficiency test of the discovered circuit.

The procedure is greedy and performance-oriented; it identifies a sparse performance-preserving circuit but does not prove formal minimality.

### Logged outputs and summary metrics

For each checkpoint, the script records:

- every cumulative prefix sweep result
- every tested relation, retained or rejected
- retained layer prefix for each relation
- the downstream stream through which each route was discovered
- validation clean accuracy
- held-out test clean accuracy
- held-out test kept-only accuracy
- per-destination required depth
- candidate and retained relation counts
- candidate and retained layer-edge counts
- retained fractions for relations and layer-edges

Across the three main 10-layer seeds, the script additionally reports means and 95% percentile-bootstrap confidence intervals over checkpoint-level values for:

- clean held-out test exact accuracy
- kept-only held-out test exact accuracy
- retained-relation fraction
- retained-layer-edge fraction

Bootstrap settings:

- `100,000` resamples
- bootstrap RNG seed `20260517`

### Interpretation guardrails

This analysis supports claims about sparse performance-preserving flow, not about formal circuit minimality or uniqueness. Because the search is greedy, different retained circuits can in principle support similar performance. The paper-facing interpretation should therefore emphasize qualitative structure shared across independently discovered checkpoint-specific circuits: which families route locally, where they project to the outputs, and how required depth differs across pathways.
