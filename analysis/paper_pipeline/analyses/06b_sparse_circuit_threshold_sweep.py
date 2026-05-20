from __future__ import annotations

"""
Analysis 06b: threshold sensitivity sweep for sparse-circuit discovery.

This is a reviewer-facing robustness check for Analysis 06. It repeats the
same greedy sparse-circuit discovery while varying the two threshold parameters
that define a "meaningful" cumulative ablation drop:

  - first-drop threshold
  - subsequent-drop fraction of the first meaningful jump

For each grid cell, the script reports retained sparsity, kept-only accuracy,
and relation-level overlap across seeds. The intent is not to find a new
"optimal" circuit, but to ask whether the qualitative factorized N/B/D routing
structure depends narrowly on the paper's default thresholds.
"""

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from helpers.paths import analysis_data_dir
from helpers.runtime import get_device


ANALYSIS_SLUG = "06_sparse_circuit_discovery"
BOOTSTRAP_SEED = 20260517
N_BOOTSTRAP_RESAMPLES = 100_000
DEFAULT_DROP_THRESHOLDS = (0.01, 0.02, 0.05)
DEFAULT_RELATIVE_FRACTIONS = (0.10, 0.20, 0.30)


def _load_sparse_module():
    """Load `06_sparse_circuit_discovery.py`, whose filename starts with a digit."""
    module_path = Path(__file__).with_name("06_sparse_circuit_discovery.py")
    spec = importlib.util.spec_from_file_location("sparse_circuit_discovery_impl", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sparse-circuit module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = rng.choice(arr, size=(N_BOOTSTRAP_RESAMPLES, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def _retained_relation_set(edge_rows: Sequence[Mapping[str, Any]]) -> Set[Tuple[str, str]]:
    return {
        (str(row["source"]), str(row["destination"]))
        for row in edge_rows
        if int(row["retained"]) == 1
    }


def _relation_overlap(relation_sets: Sequence[Set[Tuple[str, str]]]) -> Dict[str, Any]:
    if not relation_sets:
        return {
            "relation_intersection_count": 0,
            "relation_union_count": 0,
            "relation_iou": 0.0,
            "intersection_relations": "",
            "union_relations": "",
        }
    intersection = set.intersection(*relation_sets)
    union = set.union(*relation_sets)
    return {
        "relation_intersection_count": len(intersection),
        "relation_union_count": len(union),
        "relation_iou": len(intersection) / max(1, len(union)),
        "intersection_relations": ";".join(f"{s}->{d}" for s, d in sorted(intersection)),
        "union_relations": ";".join(f"{s}->{d}" for s, d in sorted(union)),
    }


def _core_route_flags(relation_sets: Sequence[Set[Tuple[str, str]]]) -> Dict[str, Any]:
    """
    Coarse qualitative flags for the paper's factorized-routing claim.

    These flags deliberately avoid encoding the exact paper circuit. They ask
    whether each seed retained at least one number-to-output route, at least one
    base-to-output route, both D_ones-to-output routes, and O[0]->O[1].
    """
    per_seed_ok: List[bool] = []
    for rels in relation_sets:
        has_n_to_o = any(src.startswith("N_") and dst in {"O[0]", "O[1]"} for src, dst in rels)
        has_b_to_o = any(src.startswith("B_") and dst in {"O[0]", "O[1]"} for src, dst in rels)
        has_d_to_o0 = ("D_ones", "O[0]") in rels
        has_d_to_o1 = ("D_ones", "O[1]") in rels
        has_o0_to_o1 = ("O[0]", "O[1]") in rels
        per_seed_ok.append(bool(has_n_to_o and has_b_to_o and has_d_to_o0 and has_d_to_o1 and has_o0_to_o1))
    return {
        "factorized_core_present_all_seeds": int(all(per_seed_ok)) if per_seed_ok else 0,
        "factorized_core_present_seed_fraction": float(np.mean(per_seed_ok)) if per_seed_ok else 0.0,
    }



def _relations_from_string(text: str) -> Set[Tuple[str, str]]:
    rels: Set[Tuple[str, str]] = set()
    for item in str(text).split(";"):
        if not item:
            continue
        source, destination = item.split("->", 1)
        rels.add((source, destination))
    return rels


def _write_reference_overlap_outputs(out_dir: Path, per_seed_rows: Sequence[Mapping[str, Any]]) -> None:
    """
    Compare every threshold-sweep relation set to the paper's shared circuit.

    The reference circuit is the across-seed intersection of the default grid cell
    (`drop0.02_rel0.2`), which is the circuit shown in the paper figure.
    """
    default_sets = [
        _relations_from_string(str(row["retained_relations"]))
        for row in per_seed_rows
        if row["cell_id"] == "drop0.02_rel0.2"
    ]
    if not default_sets:
        return
    reference = set.intersection(*default_sets)
    rows: List[Dict[str, Any]] = []
    for row in per_seed_rows:
        rels = _relations_from_string(str(row["retained_relations"]))
        intersection = rels & reference
        union = rels | reference
        rows.append(
            {
                "cell_id": row["cell_id"],
                "seed": row["seed"],
                "n_relations": len(rels),
                "reference_relations": len(reference),
                "intersection_with_reference": len(intersection),
                "union_with_reference": len(union),
                "iou_with_reference": len(intersection) / max(1, len(union)),
                "contains_all_reference_relations": int(reference <= rels),
                "exact_reference_match": int(rels == reference),
                "missing_reference_relations": ";".join(f"{s}->{d}" for s, d in sorted(reference - rels)),
                "extra_relations": ";".join(f"{s}->{d}" for s, d in sorted(rels - reference)),
            }
        )
    _write_csv(out_dir / "threshold_sweep_reference_overlap.csv", rows, list(rows[0].keys()))
    ious = [float(row["iou_with_reference"]) for row in rows]
    summary = [
        {
            "n_seed_threshold_checks": len(rows),
            "reference_relation_count": len(reference),
            "iou_with_reference_mean": float(np.mean(ious)),
            "iou_with_reference_min": float(np.min(ious)),
            "iou_with_reference_max": float(np.max(ious)),
            "contains_all_reference_count": int(sum(int(row["contains_all_reference_relations"]) for row in rows)),
            "exact_reference_match_count": int(sum(int(row["exact_reference_match"]) for row in rows)),
            "reference_relations": ";".join(f"{s}->{d}" for s, d in sorted(reference)),
        }
    ]
    _write_csv(out_dir / "threshold_sweep_reference_overlap_summary.csv", summary, list(summary[0].keys()))

def _metric(summary: Mapping[str, Any], variant: str, metric: str) -> float:
    row = next(row for row in summary["metrics"] if row["variant"] == variant)
    return float(row[metric])


def run_sweep(
    checkpoints: Iterable[str | Path],
    run_label: str,
    drop_thresholds: Sequence[float],
    relative_fractions: Sequence[float],
    discover_max_examples: int,
    eval_max_examples: int,
    batch_size: int,
    plateau_patience: int,
    discovery_sample_seed: int,
    skip_prefix_rows: bool,
) -> Dict[str, Any]:
    sparse = _load_sparse_module()
    device = get_device()
    checkpoint_paths = [Path(path) for path in checkpoints]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required.")

    out_dir = analysis_data_dir(ANALYSIS_SLUG) / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_summary_rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
    all_edge_rows: List[Dict[str, Any]] = []
    all_prefix_rows: List[Dict[str, Any]] = []
    metadata_cells: List[Dict[str, Any]] = []

    for drop_threshold in drop_thresholds:
        for relative_fraction in relative_fractions:
            cell_id = f"drop{drop_threshold:g}_rel{relative_fraction:g}"
            cell_start = time.time()
            print(
                f"\n[grid cell] {cell_id} "
                f"(drop_threshold={drop_threshold}, relative_fraction={relative_fraction})",
                flush=True,
            )
            relation_sets: List[Set[Tuple[str, str]]] = []
            kept_accs: List[float] = []
            clean_accs: List[float] = []
            relation_fracs: List[float] = []
            layer_edge_fracs: List[float] = []
            cell_summaries: List[Dict[str, Any]] = []

            for checkpoint_path in checkpoint_paths:
                checkpoint_start = time.time()
                print(f"  [start] {Path(checkpoint_path).name}", flush=True)
                sweep_rows, edge_rows, metric_rows, summary = sparse._discover_one_checkpoint(
                    checkpoint_path=checkpoint_path,
                    discover_max_examples=discover_max_examples,
                    eval_max_examples=eval_max_examples,
                    batch_size=batch_size,
                    drop_threshold=drop_threshold,
                    relative_increment_fraction=relative_fraction,
                    plateau_patience=plateau_patience,
                    discovery_sample_seed=discovery_sample_seed,
                    device=device,
                )
                seed = int(summary["checkpoint_spec"]["seed"])
                relation_set = _retained_relation_set(edge_rows)
                relation_sets.append(relation_set)
                clean_acc = _metric(summary, "eval_clean_test", "exact_answer_accuracy")
                kept_acc = _metric(summary, "eval_kept_only_test", "exact_answer_accuracy")
                clean_accs.append(clean_acc)
                kept_accs.append(kept_acc)
                relation_fracs.append(float(summary["sparsity"]["retained_relation_fraction"]))
                layer_edge_fracs.append(float(summary["sparsity"]["retained_layer_edge_fraction"]))
                cell_summaries.append(summary)

                per_seed_rows.append(
                    {
                        "cell_id": cell_id,
                        "drop_threshold": drop_threshold,
                        "relative_increment_fraction": relative_fraction,
                        "plateau_patience": plateau_patience,
                        "checkpoint": summary["checkpoint"],
                        "seed": seed,
                        "n_discovery_examples_used": summary["discovery_parameters"]["n_discovery_examples_used"],
                        "eval_clean_test_exact_accuracy": clean_acc,
                        "eval_kept_only_test_exact_accuracy": kept_acc,
                        **summary["sparsity"],
                        "retained_relations": ";".join(f"{s}->{d}" for s, d in sorted(relation_set)),
                    }
                )
                for row in edge_rows:
                    all_edge_rows.append(
                        {
                            "cell_id": cell_id,
                            "drop_threshold": drop_threshold,
                            "relative_increment_fraction": relative_fraction,
                            "plateau_patience": plateau_patience,
                            **row,
                        }
                    )
                if not skip_prefix_rows:
                    for row in sweep_rows:
                        all_prefix_rows.append(
                            {
                                "cell_id": cell_id,
                                "drop_threshold": drop_threshold,
                                "relative_increment_fraction": relative_fraction,
                                "plateau_patience": plateau_patience,
                                **row,
                            }
                        )
                print(
                    "  [done] "
                    f"seed={seed} "
                    f"kept_acc={kept_acc:.4f} "
                    f"rel_frac={summary['sparsity']['retained_relation_fraction']:.4f} "
                    f"layer_edge_frac={summary['sparsity']['retained_layer_edge_fraction']:.4f} "
                    f"relations={len(relation_set)} "
                    f"time_min={(time.time() - checkpoint_start) / 60.0:.2f}",
                    flush=True,
                )

            kept_mean, kept_lo, kept_hi = _bootstrap_ci(kept_accs)
            clean_mean, clean_lo, clean_hi = _bootstrap_ci(clean_accs)
            rel_frac_mean, rel_frac_lo, rel_frac_hi = _bootstrap_ci(relation_fracs)
            layer_frac_mean, layer_frac_lo, layer_frac_hi = _bootstrap_ci(layer_edge_fracs)
            overlap = _relation_overlap(relation_sets)
            flags = _core_route_flags(relation_sets)
            grid_summary_rows.append(
                {
                    "cell_id": cell_id,
                    "drop_threshold": drop_threshold,
                    "relative_increment_fraction": relative_fraction,
                    "plateau_patience": plateau_patience,
                    "n_checkpoints": len(checkpoint_paths),
                    "discover_max_examples": discover_max_examples,
                    "eval_max_examples": eval_max_examples,
                    "eval_clean_test_exact_accuracy_mean": clean_mean,
                    "eval_clean_test_exact_accuracy_ci95_low": clean_lo,
                    "eval_clean_test_exact_accuracy_ci95_high": clean_hi,
                    "eval_kept_only_test_exact_accuracy_mean": kept_mean,
                    "eval_kept_only_test_exact_accuracy_ci95_low": kept_lo,
                    "eval_kept_only_test_exact_accuracy_ci95_high": kept_hi,
                    "retained_relation_fraction_mean": rel_frac_mean,
                    "retained_relation_fraction_ci95_low": rel_frac_lo,
                    "retained_relation_fraction_ci95_high": rel_frac_hi,
                    "retained_layer_edge_fraction_mean": layer_frac_mean,
                    "retained_layer_edge_fraction_ci95_low": layer_frac_lo,
                    "retained_layer_edge_fraction_ci95_high": layer_frac_hi,
                    **overlap,
                    **flags,
                }
            )
            metadata_cells.append(
                {
                    "cell_id": cell_id,
                    "drop_threshold": drop_threshold,
                    "relative_increment_fraction": relative_fraction,
                    "checkpoint_summaries": cell_summaries,
                }
            )
            print(
                "  [cell summary] "
                f"kept_acc_mean={kept_mean:.4f} "
                f"relation_iou={overlap['relation_iou']:.4f} "
                f"core_all_seeds={flags['factorized_core_present_all_seeds']} "
                f"time_min={(time.time() - cell_start) / 60.0:.2f}",
                flush=True,
            )

    _write_csv(out_dir / "threshold_sweep_grid_summary.csv", grid_summary_rows, list(grid_summary_rows[0].keys()))
    _write_csv(out_dir / "threshold_sweep_per_seed.csv", per_seed_rows, list(per_seed_rows[0].keys()))
    _write_reference_overlap_outputs(out_dir, per_seed_rows)
    _write_csv(out_dir / "threshold_sweep_edge_rows.csv", all_edge_rows, list(all_edge_rows[0].keys()))
    if not skip_prefix_rows and all_prefix_rows:
        _write_csv(out_dir / "threshold_sweep_prefix_rows.csv", all_prefix_rows, list(all_prefix_rows[0].keys()))
    (out_dir / "threshold_sweep_metadata.json").write_text(
        json.dumps(
            {
                "analysis_slug": ANALYSIS_SLUG,
                "run_label": run_label,
                "device": str(device),
                "drop_thresholds": list(drop_thresholds),
                "relative_increment_fractions": list(relative_fractions),
                "plateau_patience": plateau_patience,
                "discover_max_examples": discover_max_examples,
                "eval_max_examples": eval_max_examples,
                "batch_size": batch_size,
                "discovery_sample_seed": discovery_sample_seed,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
                "cells": metadata_cells,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "grid_cells": len(grid_summary_rows),
        "summary_path": str(out_dir / "threshold_sweep_grid_summary.csv"),
        "per_seed_path": str(out_dir / "threshold_sweep_per_seed.csv"),
    }


def _float_list(values: Sequence[str]) -> List[float]:
    return [float(value) for value in values]


def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold sweep for greedy sparse-circuit discovery.")
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--drop-thresholds", nargs="+", default=[str(v) for v in DEFAULT_DROP_THRESHOLDS])
    ap.add_argument("--relative-increment-fractions", nargs="+", default=[str(v) for v in DEFAULT_RELATIVE_FRACTIONS])
    ap.add_argument("--discover-max-examples", type=int, default=512)
    ap.add_argument("--eval-max-examples", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--plateau-patience", type=int, default=2)
    ap.add_argument("--discovery-sample-seed", type=int, default=20260517)
    ap.add_argument("--skip-prefix-rows", action="store_true", help="Do not save large per-prefix sweep rows.")
    args = ap.parse_args()
    print(
        json.dumps(
            run_sweep(
                checkpoints=args.checkpoints,
                run_label=args.run_label,
                drop_thresholds=_float_list(args.drop_thresholds),
                relative_fractions=_float_list(args.relative_increment_fractions),
                discover_max_examples=args.discover_max_examples,
                eval_max_examples=args.eval_max_examples,
                batch_size=args.batch_size,
                plateau_patience=args.plateau_patience,
                discovery_sample_seed=args.discovery_sample_seed,
                skip_prefix_rows=args.skip_prefix_rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
