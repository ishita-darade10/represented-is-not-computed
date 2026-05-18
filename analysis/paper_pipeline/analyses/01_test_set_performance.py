from __future__ import annotations

"""
Analysis 01: autoregressive held-out test performance across trained seeds.

This script evaluates named checkpoints on the exact canonical test split stored
inside each checkpoint. It reports answer correctness for the two answer digits
only (`O[0]`, `O[1]`), while still generating the end token autoregressively to
match the task setup used at inference time.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from helpers.checkpoints import LoadedCheckpoint, load_checkpoint_bundle
from helpers.paths import analysis_data_dir
from helpers.runtime import get_device
from helpers.splits import reconstruct_canonical_splits, summarize_split_info


ANALYSIS_SLUG = "01_test_set_performance"
BOOTSTRAP_SEED = 20260517
N_BOOTSTRAP_RESAMPLES = 100_000


@torch.no_grad()
def evaluate_two_digit_answer_accuracy(
    bundle: LoadedCheckpoint,
    test_samples: Sequence[tuple[str, str]],
    device: torch.device,
) -> Dict[str, Any]:
    """Run greedy autoregressive inference and score only `O[0]` and `O[1]`."""
    model = bundle.model
    tokenizer = bundle.tokenizer
    model.eval()

    total = 0
    exact_answer_correct = 0
    o0_correct = 0
    o1_correct = 0

    for prompt, target in tqdm(
        test_samples,
        desc=f"seed {bundle.spec.seed}",
        leave=False,
    ):
        x = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.long).unsqueeze(0)
        generated = model.generate(x, max_new_tokens=3)
        predicted_answer = tokenizer.decode(generated[0, -3:].tolist())[:2]
        target_answer = target[:2]

        total += 1
        exact_answer_correct += int(predicted_answer == target_answer)
        o0_correct += int(predicted_answer[0] == target_answer[0])
        o1_correct += int(predicted_answer[1] == target_answer[1])

    return {
        "n_test_examples": total,
        "exact_answer_correct": exact_answer_correct,
        "o0_correct": o0_correct,
        "o1_correct": o1_correct,
        "exact_answer_accuracy": exact_answer_correct / max(1, total),
        "o0_token_accuracy": o0_correct / max(1, total),
        "o1_token_accuracy": o1_correct / max(1, total),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a deterministic CSV with the requested column order."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_across_seeds(per_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute mean and percentile-bootstrap 95% CI across seed-level accuracies."""
    metrics = (
        "exact_answer_accuracy",
        "o0_token_accuracy",
        "o1_token_accuracy",
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in per_seed_rows], dtype=np.float64)
        n = len(values)
        avg = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else 0.0
        bootstrap_means = rng.choice(values, size=(N_BOOTSTRAP_RESAMPLES, n), replace=True).mean(axis=1)
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "n_seeds": n,
                "mean": avg,
                "std_across_seeds": sd,
                "bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "ci95_low_bootstrap_percentile": float(ci_low),
                "ci95_high_bootstrap_percentile": float(ci_high),
            }
        )
    return rows


def run_analysis(checkpoints: Iterable[str | Path], run_label: str) -> Dict[str, Any]:
    """Evaluate all checkpoints and persist auditable outputs."""
    checkpoint_paths = [Path(path) for path in checkpoints]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required.")

    device = get_device()
    out_dir = analysis_data_dir(ANALYSIS_SLUG) / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {
        "analysis_slug": ANALYSIS_SLUG,
        "run_label": run_label,
        "device": str(device),
        "metric_definitions": {
            "exact_answer_accuracy": "Both generated answer digits O[0] and O[1] match the target answer digits.",
            "o0_token_accuracy": "Generated first answer digit O[0] matches target digit 0.",
            "o1_token_accuracy": "Generated second answer digit O[1] matches target digit 1.",
        },
        "confidence_interval_policy": {
            "across_seed_interval": (
                "Two-sided 95% percentile bootstrap interval over seed-level mean accuracies "
                f"using {N_BOOTSTRAP_RESAMPLES} bootstrap resamples and bootstrap seed {BOOTSTRAP_SEED}."
            ),
            "rationale": (
                "Seed is the replicate unit for cross-run uncertainty; bootstrap resampling keeps "
                "the interval on the natural [0, 1] support of accuracy. Wilson intervals address "
                "within-test-set binomial uncertainty instead."
            ),
        },
        "checkpoints": [],
    }

    for checkpoint_path in checkpoint_paths:
        bundle = load_checkpoint_bundle(checkpoint_path, device=device)
        _, _, test_samples = reconstruct_canonical_splits(bundle.split_info)
        split_summary = summarize_split_info(bundle.split_info)
        metrics = evaluate_two_digit_answer_accuracy(bundle, test_samples, device)

        row = {
            "checkpoint": str(bundle.path),
            "seed": bundle.spec.seed,
            "num_layers": bundle.spec.num_layers,
            "split_mode": bundle.spec.split_mode,
            "train_all_permutations": bundle.spec.train_all_permutations,
            "selection": bundle.spec.selection,
            **metrics,
        }
        per_seed_rows.append(row)
        metadata["checkpoints"].append(
            {
                "path": str(bundle.path),
                "spec": asdict(bundle.spec),
                "split_summary": split_summary,
            }
        )

    per_seed_rows.sort(key=lambda row: int(row["seed"]))
    across_seed_rows = summarize_across_seeds(per_seed_rows)

    _write_csv(
        out_dir / "per_seed_results.csv",
        per_seed_rows,
        fieldnames=(
            "checkpoint",
            "seed",
            "num_layers",
            "split_mode",
            "train_all_permutations",
            "selection",
            "n_test_examples",
            "exact_answer_correct",
            "o0_correct",
            "o1_correct",
            "exact_answer_accuracy",
            "o0_token_accuracy",
            "o1_token_accuracy",
        ),
    )
    _write_csv(
        out_dir / "across_seed_summary.csv",
        across_seed_rows,
        fieldnames=(
            "metric",
            "n_seeds",
            "mean",
            "std_across_seeds",
            "bootstrap_resamples",
            "bootstrap_seed",
            "ci95_low_bootstrap_percentile",
            "ci95_high_bootstrap_percentile",
        ),
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "per_seed_rows": per_seed_rows,
        "across_seed_rows": across_seed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate autoregressive held-out test accuracy across checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Named checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--run-label",
        required=True,
        help=(
            "Output subdirectory label for this homogeneous condition, "
            "e.g. main_10layer_ptrue or companion_3layer_ptrue."
        ),
    )
    args = parser.parse_args()
    results = run_analysis(args.checkpoints, run_label=args.run_label)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
