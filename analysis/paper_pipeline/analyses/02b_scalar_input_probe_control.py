from __future__ import annotations

"""
Analysis 02b: scalar-input control for the closed-form probe targets.

The linear-probing result asks whether closed-form quantities are accessible
from residual-stream activity. This control asks the narrower baseline question:
how much of the same targets can be recovered by an ordinary linear regression
from the raw scalar inputs N, B, and D alone, over the same pooled held-out
validation and test examples used for the activation probes?

The script is intentionally checkpoint-centric: every held-out split is rebuilt
from the `split_info` stored inside each checkpoint.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from helpers.checkpoints import load_checkpoint_payload, parse_checkpoint_name
from helpers.paths import analysis_data_dir
from helpers.splits import CanonicalRecord, records_for_splits, summarize_split_info


ANALYSIS_SLUG = "02_linear_probing"
CV_FOLDS = 5
BOOTSTRAP_SEED = 20260517
N_BOOTSTRAP_RESAMPLES = 100_000
TARGETS = (
    ("BpowD", "B^D"),
    ("NdivBpowD", "N / B^D"),
    ("floorNdivBpowD", "floor(N / B^D)"),
    ("floorNdivBpowD_modB", "floor(N / B^D) mod B"),
)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 1e-18:
        return 1.0 if ss_res <= 1e-18 else 0.0
    return 1.0 - ss_res / ss_tot


def _make_folds(num_rows: int, cv_folds: int, fold_seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(num_rows)
    fold_sizes = np.full(cv_folds, num_rows // cv_folds, dtype=np.int64)
    fold_sizes[: num_rows % cv_folds] += 1
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + int(fold_size)
        test_idx = perm[start:stop]
        train_mask = np.ones(num_rows, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_idx))
        start = stop
    return folds


def _fit_predict_ols(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Ordinary least squares with train-fold standardization and intercept.

    Standardization is not needed for the scalar inputs mathematically, but it
    matches the activation-probe protocol and keeps the control definition
    parallel to Analysis 02.
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    x_train = (X_train - mu) / sd
    x_test = (X_test - mu) / sd
    x_train_aug = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
    x_test_aug = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=1)
    weights = np.linalg.lstsq(x_train_aug, y_train, rcond=None)[0]
    return x_test_aug @ weights


def _cv_r2(X: np.ndarray, y: np.ndarray, folds: Sequence[Tuple[np.ndarray, np.ndarray]]) -> float:
    pred = np.empty_like(y, dtype=np.float64)
    for train_idx, test_idx in folds:
        pred[test_idx] = _fit_predict_ols(X[train_idx], y[train_idx], X[test_idx])
    return _r2_score(y, pred)


def _build_scalar_inputs_and_targets(records: Sequence[CanonicalRecord]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n = np.asarray([record.N for record in records], dtype=np.float64)
    b = np.asarray([record.B for record in records], dtype=np.float64)
    d = np.asarray([record.D for record in records], dtype=np.float64)
    bpow = np.asarray([record.B ** record.D for record in records], dtype=np.float64)
    floor_q = np.floor(n / bpow)
    X = np.stack([n, b, d], axis=1)
    targets = {
        "BpowD": bpow,
        "NdivBpowD": n / bpow,
        "floorNdivBpowD": floor_q,
        "floorNdivBpowD_modB": floor_q % b,
    }
    return X, targets


def analyze_one_checkpoint(checkpoint_path: str | Path, cv_folds: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(checkpoint_path)
    spec = parse_checkpoint_name(path)
    payload = load_checkpoint_payload(path, map_location="cpu")
    split_info = payload["split_info"]
    records = records_for_splits(split_info, ("val", "test"))
    X, targets = _build_scalar_inputs_and_targets(records)
    fold_seed = int(spec.seed)
    folds = _make_folds(len(records), cv_folds=cv_folds, fold_seed=fold_seed)

    rows: List[Dict[str, Any]] = []
    target_labels = dict(TARGETS)
    for target_name, y in targets.items():
        rows.append(
            {
                "checkpoint": str(path),
                "seed": spec.seed,
                "num_layers": spec.num_layers,
                "target": target_name,
                "target_label": target_labels[target_name],
                "features": "raw_N_B_D",
                "cv_r2": float(_cv_r2(X, y, folds)),
                "n_examples": len(records),
                "cv_folds": cv_folds,
                "fold_seed": fold_seed,
            }
        )

    metadata = {
        "checkpoint": str(path),
        "checkpoint_spec": asdict(spec),
        "split_summary": summarize_split_info(split_info),
        "n_heldout_val_test_examples": len(records),
        "fold_seed": fold_seed,
    }
    return rows, metadata


def summarize_across_seeds(per_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = {}
    for row in per_seed_rows:
        grouped.setdefault(str(row["target"]), []).append(float(row["cv_r2"]))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows: List[Dict[str, Any]] = []
    target_labels = dict(TARGETS)
    for target_name, values_list in sorted(grouped.items()):
        values = np.asarray(values_list, dtype=np.float64)
        bootstrap_means = rng.choice(values, size=(N_BOOTSTRAP_RESAMPLES, len(values)), replace=True).mean(axis=1)
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        summary_rows.append(
            {
                "target": target_name,
                "target_label": target_labels[target_name],
                "features": "raw_N_B_D",
                "n_seeds": len(values),
                "mean_cv_r2": float(values.mean()),
                "ci95_low_bootstrap_percentile": float(ci_low),
                "ci95_high_bootstrap_percentile": float(ci_high),
            }
        )
    return summary_rows


def run_analysis(checkpoints: Iterable[str | Path], run_label: str, cv_folds: int) -> Dict[str, Any]:
    checkpoint_paths = [Path(path) for path in checkpoints]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required.")

    out_dir = analysis_data_dir(ANALYSIS_SLUG) / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, Any]] = []
    checkpoint_metadata: List[Dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        rows, metadata = analyze_one_checkpoint(checkpoint_path, cv_folds=cv_folds)
        per_seed_rows.extend(rows)
        checkpoint_metadata.append(metadata)

    summary_rows = summarize_across_seeds(per_seed_rows)
    _write_csv(
        out_dir / "scalar_input_control_per_seed.csv",
        per_seed_rows,
        fieldnames=(
            "checkpoint",
            "seed",
            "num_layers",
            "target",
            "target_label",
            "features",
            "cv_r2",
            "n_examples",
            "cv_folds",
            "fold_seed",
        ),
    )
    _write_csv(
        out_dir / "scalar_input_control_summary.csv",
        summary_rows,
        fieldnames=(
            "target",
            "target_label",
            "features",
            "n_seeds",
            "mean_cv_r2",
            "ci95_low_bootstrap_percentile",
            "ci95_high_bootstrap_percentile",
        ),
    )
    metadata = {
        "analysis_slug": ANALYSIS_SLUG,
        "control_name": "scalar_input_control",
        "run_label": run_label,
        "cv_folds": cv_folds,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
        "features": ["N", "B", "D"],
        "targets": [{"name": name, "label": label} for name, label in TARGETS],
        "fold_assignment": "checkpoint seed, independently within each checkpoint-specific pooled val+test set",
        "checkpoints": checkpoint_metadata,
    }
    with (out_dir / "scalar_input_control_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "n_rows": len(per_seed_rows),
        "summary_path": str(out_dir / "scalar_input_control_summary.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw N,B,D linear-regression controls for closed-form probe targets.")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.checkpoints, args.run_label, args.cv_folds), indent=2))


if __name__ == "__main__":
    main()
