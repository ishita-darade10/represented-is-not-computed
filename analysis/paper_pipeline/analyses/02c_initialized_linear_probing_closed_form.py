from __future__ import annotations

"""
Workspace-only diagnostic: run the Fig. 2 closed-form linear probe protocol on
freshly initialized transformers with the same architecture seeds and same
checkpoint-stored val/test splits.

This tests how much closed-form decodability is already present as a random
high-dimensional projection / token-position geometry effect before training.
"""

import argparse
import csv
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = ROOT / "represented_is_not_computed_release" / "analysis" / "paper_pipeline"
PROBE_SCRIPT = PIPELINE_ROOT / "analyses" / "02_linear_probing_closed_form.py"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec_mod = importlib.util.spec_from_file_location("probe02", PROBE_SCRIPT)
probe02 = importlib.util.module_from_spec(spec_mod)
assert spec_mod.loader is not None
spec_mod.loader.exec_module(probe02)

from helpers.checkpoints import LoadedCheckpoint, load_checkpoint_payload, parse_checkpoint_name  # noqa: E402
from helpers.runtime import get_device  # noqa: E402
from helpers.splits import records_for_splits, summarize_split_info  # noqa: E402
from model import DecoderOnlyTransformer  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

BOOTSTRAP_SEED = 20260517
N_BOOTSTRAP_RESAMPLES = 100_000


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_initialized_bundle(checkpoint_path: str | Path, device: torch.device) -> LoadedCheckpoint:
    path = Path(checkpoint_path)
    spec = parse_checkpoint_name(path)
    payload = load_checkpoint_payload(path, map_location="cpu")
    tokenizer = Tokenizer()
    seed_everything(spec.seed)
    model = DecoderOnlyTransformer(vocab_size=len(tokenizer), num_layers=spec.num_layers).to(device)
    model.eval()
    return LoadedCheckpoint(
        path=path,
        spec=spec,
        payload={"init_control_from_checkpoint": str(path), "split_info": payload["split_info"]},
        split_info=payload["split_info"],
        tokenizer=tokenizer,
        model=model,
    )


def probe_initialized_checkpoint(checkpoint_path: str | Path, batch_size: int, cv_folds: int, device: torch.device):
    bundle = make_initialized_bundle(checkpoint_path, device=device)
    records = records_for_splits(bundle.split_info, ("val", "test"))
    print(f"[init probe] seed={bundle.spec.seed} layers={bundle.spec.num_layers} n={len(records)} device={device}", flush=True)
    features = probe02.extract_autoregressive_features(bundle, records, batch_size=batch_size, device=device)
    targets = probe02._build_targets(records)
    folds = probe02._make_folds(len(records), cv_folds)

    rows: List[Dict[str, Any]] = []
    for rep_idx in range(features.shape[1]):
        layer_label = "input" if rep_idx == 0 else str(rep_idx - 1)
        for stream_idx, stream_name in enumerate(probe02.STREAMS):
            scores = probe02._multioutput_cv_r2(features[:, rep_idx, stream_idx, :], targets, folds)
            for target_idx, (target_name, target_label) in enumerate(probe02.TARGETS):
                rows.append({
                    "source_checkpoint": str(bundle.path),
                    "seed": bundle.spec.seed,
                    "num_layers": bundle.spec.num_layers,
                    "model_condition": "initialized_untrained",
                    "representation_index": rep_idx,
                    "layer_label": layer_label,
                    "stream": stream_name,
                    "target": target_name,
                    "target_label": target_label,
                    "cv_r2": float(scores[target_idx]),
                    "n_examples": len(records),
                    "cv_folds": cv_folds,
                })
    metadata = {
        "source_checkpoint": str(bundle.path),
        "checkpoint_spec": asdict(bundle.spec),
        "split_summary": summarize_split_info(bundle.split_info),
        "n_pooled_val_test_examples": len(records),
        "init_seed": bundle.spec.seed,
    }
    return rows, metadata


def summarize_across_seeds(per_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, int, str], List[float]] = {}
    for row in per_seed_rows:
        grouped.setdefault((str(row["target"]), int(row["representation_index"]), str(row["stream"])), []).append(float(row["cv_r2"]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: List[Dict[str, Any]] = []
    target_labels = dict(probe02.TARGETS)
    for (target, rep_idx, stream), vals in sorted(grouped.items()):
        values = np.asarray(vals, dtype=np.float64)
        boot = rng.choice(values, size=(N_BOOTSTRAP_RESAMPLES, len(values)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({
            "target": target,
            "target_label": target_labels[target],
            "representation_index": rep_idx,
            "layer_label": "input" if rep_idx == 0 else str(rep_idx - 1),
            "stream": stream,
            "n_seeds": len(values),
            "mean_cv_r2": float(values.mean()),
            "ci95_low_bootstrap_percentile": float(lo),
            "ci95_high_bootstrap_percentile": float(hi),
        })
    return out


def plot_summary(summary_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    rows = list(summary_rows)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True, sharey=True)
    palette = {"D_ones": "#d62728", "O[0]": "#1f77b4", "O[1]": "#2ca02c"}
    other_color = "#7f7f7f"
    for ax, (target, target_label) in zip(axes.reshape(-1), probe02.TARGETS):
        for stream in probe02.STREAMS:
            sub = [r for r in rows if r["target"] == target and r["stream"] == stream]
            sub.sort(key=lambda r: int(r["representation_index"]))
            xs = np.asarray([int(r["representation_index"]) for r in sub])
            ys = np.asarray([float(r["mean_cv_r2"]) for r in sub])
            lo = np.asarray([float(r["ci95_low_bootstrap_percentile"]) for r in sub])
            hi = np.asarray([float(r["ci95_high_bootstrap_percentile"]) for r in sub])
            highlighted = stream in {"D_ones", "O[0]", "O[1]"}
            color = palette.get(stream, other_color)
            ax.plot(xs, ys, color=color, alpha=1.0 if highlighted else 0.24, linewidth=2.4 if highlighted else 1.0, label=stream if highlighted else None)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.16 if highlighted else 0.04)
        ax.set_title(target_label)
        ax.set_xlabel("Representation")
        ax.set_ylabel("5-fold CV $R^2$")
        ax.grid(alpha=0.2)
        max_rep = max(int(r["representation_index"]) for r in rows)
        ax.set_xticks(np.arange(max_rep + 1))
        ax.set_xticklabels(["input"] + [str(i) for i in range(max_rep)])
        ax.set_ylim(-0.08, 1.02)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Initialized/untrained transformers: closed-form linear probes", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def run(checkpoints: Iterable[str | Path], output_dir: Path, batch_size: int, cv_folds: int) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    per_seed_rows: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []
    for ckpt in checkpoints:
        rows, meta = probe_initialized_checkpoint(ckpt, batch_size=batch_size, cv_folds=cv_folds, device=device)
        per_seed_rows.extend(rows)
        metadata.append(meta)
    summary = summarize_across_seeds(per_seed_rows)
    write_csv(output_dir / "per_seed_init_probe_rows.csv", per_seed_rows, fieldnames=(
        "source_checkpoint", "seed", "num_layers", "model_condition", "representation_index", "layer_label",
        "stream", "target", "target_label", "cv_r2", "n_examples", "cv_folds",
    ))
    write_csv(output_dir / "across_seed_init_probe_summary.csv", summary, fieldnames=(
        "target", "target_label", "representation_index", "layer_label", "stream", "n_seeds",
        "mean_cv_r2", "ci95_low_bootstrap_percentile", "ci95_high_bootstrap_percentile",
    ))
    plot_path = output_dir / "init_closed_form_probe_grid.png"
    plot_summary(summary, plot_path)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "purpose": "workspace diagnostic: Fig. 2 closed-form probes on initialized untrained transformers",
            "splits": "same pooled val+test split_info as source checkpoints",
            "activation_protocol": "same autoregressive extractor as paper Analysis 02; O[1] follows the initialized model's own first generated token",
            "cv_seed": probe02.CV_SEED,
            "cv_folds": cv_folds,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
            "checkpoints": metadata,
        }, f, indent=2)
    return {"output_dir": str(output_dir), "plot_png": str(plot_path), "plot_pdf": str(plot_path.with_suffix(".pdf")), "n_rows": len(per_seed_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig. 2 linear probes on initialized/untrained transformers.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", default="analysis/scratch_results/init_linear_probe/main_10layer_ptrue")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoints, Path(args.output_dir), args.batch_size, args.cv_folds), indent=2))


if __name__ == "__main__":
    main()
