from __future__ import annotations

"""
Analysis 02: autoregressive linear probes for closed-form task quantities.

The implementation is checkpoint-centric throughout: every split is rebuilt
from checkpoint-stored split_info, and every output stream is obtained from
greedy autoregressive generation rather than teacher forcing.
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
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from helpers.checkpoints import LoadedCheckpoint, load_checkpoint_bundle
from helpers.paths import analysis_data_dir, analysis_figure_dir
from helpers.runtime import get_device
from helpers.splits import CanonicalRecord, records_for_splits, summarize_split_info


ANALYSIS_SLUG = "02_linear_probing"
CV_SEED = 20260517
BOOTSTRAP_SEED = 20260517
N_BOOTSTRAP_RESAMPLES = 100_000
TARGETS = (
    ("BpowD", "B^D"),
    ("NdivBpowD", "N / B^D"),
    ("floorNdivBpowD", "floor(N / B^D)"),
    ("floorNdivBpowD_modB", "floor(N / B^D) mod B"),
)
STREAMS = (
    "N_tag",
    "N_hundreds",
    "N_tens",
    "N_ones",
    "B_tag",
    "B_tens",
    "B_ones",
    "D_tag",
    "D_ones",
    "O[0]",
    "O[1]",
)
HIGHLIGHT_STREAMS = {"D_ones", "O[0]", "O[1]"}


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_targets(records: Sequence[CanonicalRecord]) -> np.ndarray:
    n = np.asarray([record.N for record in records], dtype=np.int64)
    b = np.asarray([record.B for record in records], dtype=np.int64)
    d = np.asarray([record.D for record in records], dtype=np.int64)
    bpow = np.power(b, d, dtype=np.int64)
    flo = n // bpow
    return np.stack(
        [
            bpow.astype(np.float64),
            n.astype(np.float64) / bpow.astype(np.float64),
            flo.astype(np.float64),
            (flo % b).astype(np.float64),
        ],
        axis=1,
    )


def _forward_collect(model, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return logits plus `[input, block0_out, ..., blockN_out]`.

    This mirrors `DecoderOnlyTransformer.forward` while exposing exactly the
    residual states used by the probe analysis.
    """
    _, seqlen = idx.shape
    pos = torch.arange(0, seqlen, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    x = model.drop(x)
    states = [x]
    for block in model.blocks:
        x = block(x)
        states.append(x)
    logits = model.head(model.ln_f(x))
    return logits, torch.stack(states, dim=1)


@torch.no_grad()
def extract_autoregressive_features(
    bundle: LoadedCheckpoint,
    records: Sequence[CanonicalRecord],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Collect residual states shaped `[example, representation, stream, d_model]`.

    Prefix streams and `O[0]` come from the prompt pass. `O[1]` comes from the
    second autoregressive pass after the model's own first generated digit has
    been appended.
    """
    tokenizer = bundle.tokenizer
    model = bundle.model
    num_examples = len(records)
    num_representations = len(model.blocks) + 1
    d_model = model.tok_emb.embedding_dim
    features = np.empty((num_examples, num_representations, len(STREAMS), d_model), dtype=np.float32)

    prompts = [record.prompt for record in records]
    for start in tqdm(range(0, num_examples, batch_size), desc=f"extract seed {bundle.spec.seed}", leave=False):
        stop = min(start + batch_size, num_examples)
        batch_prompts = prompts[start:stop]
        idx = torch.tensor([tokenizer.encode(prompt) for prompt in batch_prompts], device=device, dtype=torch.long)

        logits_prompt, states_prompt = _forward_collect(model, idx)
        first_digit = torch.argmax(logits_prompt[:, -1, :], dim=-1, keepdim=True)

        idx_with_first_digit = torch.cat([idx, first_digit], dim=1)
        _, states_second_pass = _forward_collect(model, idx_with_first_digit)

        # Canonical prompt layout:
        # 0 N_tag, 1 N_hundreds, 2 N_tens, 3 N_ones,
        # 4 B_tag, 5 B_tens, 6 B_ones, 7 D_tag, 8 D_ones, 9 O[0].
        prefix_and_o0 = states_prompt[:, :, :10, :]
        o1 = states_second_pass[:, :, 10:11, :]
        batch_features = torch.cat([prefix_and_o0, o1], dim=2)
        features[start:stop] = batch_features.detach().cpu().numpy().astype(np.float32)

    return features


def _make_folds(num_rows: int, cv_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(CV_SEED)
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


def _multioutput_cv_r2(X: np.ndarray, Y: np.ndarray, folds: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return mean CV R² for all target columns using one OLS fit per fold."""
    fold_scores: List[np.ndarray] = []
    X64 = X.astype(np.float64, copy=False)
    for train_idx, test_idx in folds:
        x_train = X64[train_idx]
        x_test = X64[test_idx]
        y_train = Y[train_idx]
        y_test = Y[test_idx]

        mu = x_train.mean(axis=0, keepdims=True)
        sd = x_train.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        x_train_std = (x_train - mu) / sd
        x_test_std = (x_test - mu) / sd
        x_train_aug = np.concatenate([x_train_std, np.ones((x_train_std.shape[0], 1))], axis=1)
        x_test_aug = np.concatenate([x_test_std, np.ones((x_test_std.shape[0], 1))], axis=1)
        # Use PyTorch's CPU QR-based least-squares driver for numerical
        # robustness. This is still ordinary least squares, not ridge; it
        # merely avoids occasional NumPy SVD non-convergence on very
        # ill-conditioned but finite design matrices.
        weights = torch.linalg.lstsq(
            torch.from_numpy(x_train_aug),
            torch.from_numpy(y_train),
            driver="gelsy",
        ).solution.numpy()
        pred = x_test_aug @ weights

        ss_res = ((y_test - pred) ** 2).sum(axis=0)
        ss_tot = ((y_test - y_test.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
        scores = np.where(ss_tot <= 1e-18, np.where(ss_res <= 1e-18, 1.0, 0.0), 1.0 - ss_res / ss_tot)
        fold_scores.append(scores)
    return np.stack(fold_scores, axis=0).mean(axis=0)


def probe_one_checkpoint(
    checkpoint_path: str | Path,
    batch_size: int,
    cv_folds: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bundle = load_checkpoint_bundle(checkpoint_path, device=device)
    records = records_for_splits(bundle.split_info, ("val", "test"))
    features = extract_autoregressive_features(bundle, records, batch_size=batch_size, device=device)
    targets = _build_targets(records)
    folds = _make_folds(len(records), cv_folds)

    rows: List[Dict[str, Any]] = []
    for rep_idx in range(features.shape[1]):
        layer_label = "input" if rep_idx == 0 else str(rep_idx - 1)
        for stream_idx, stream_name in enumerate(STREAMS):
            scores = _multioutput_cv_r2(features[:, rep_idx, stream_idx, :], targets, folds)
            for target_idx, (target_name, target_label) in enumerate(TARGETS):
                rows.append(
                    {
                        "checkpoint": str(bundle.path),
                        "seed": bundle.spec.seed,
                        "num_layers": bundle.spec.num_layers,
                        "representation_index": rep_idx,
                        "layer_label": layer_label,
                        "stream": stream_name,
                        "target": target_name,
                        "target_label": target_label,
                        "cv_r2": float(scores[target_idx]),
                        "n_examples": len(records),
                        "cv_folds": cv_folds,
                    }
                )

    metadata = {
        "checkpoint": str(bundle.path),
        "checkpoint_spec": asdict(bundle.spec),
        "split_summary": summarize_split_info(bundle.split_info),
        "n_pooled_val_test_examples": len(records),
    }
    return rows, metadata


def summarize_across_seeds(per_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[float]] = {}
    for row in per_seed_rows:
        key = (str(row["target"]), int(row["representation_index"]), str(row["stream"]))
        grouped.setdefault(key, []).append(float(row["cv_r2"]))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows: List[Dict[str, Any]] = []
    for (target, rep_idx, stream), values_list in sorted(grouped.items()):
        values = np.asarray(values_list, dtype=np.float64)
        bootstrap_means = rng.choice(values, size=(N_BOOTSTRAP_RESAMPLES, len(values)), replace=True).mean(axis=1)
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        target_label = dict(TARGETS)[target]
        summary_rows.append(
            {
                "target": target,
                "target_label": target_label,
                "representation_index": rep_idx,
                "layer_label": "input" if rep_idx == 0 else str(rep_idx - 1),
                "stream": stream,
                "n_seeds": len(values),
                "mean_cv_r2": float(values.mean()),
                "ci95_low_bootstrap_percentile": float(ci_low),
                "ci95_high_bootstrap_percentile": float(ci_high),
            }
        )
    return summary_rows


def plot_from_saved_data(run_label: str) -> Path:
    data_path = analysis_data_dir(ANALYSIS_SLUG) / run_label / "across_seed_summary.csv"
    rows: List[Dict[str, str]] = []
    with data_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True, sharey=True)
    palette = {
        "D_ones": "#d62728",
        "O[0]": "#1f77b4",
        "O[1]": "#2ca02c",
    }
    other_color = "#7f7f7f"

    for ax, (target, target_label) in zip(axes.reshape(-1), TARGETS):
        for stream in STREAMS:
            subset = [row for row in rows if row["target"] == target and row["stream"] == stream]
            subset.sort(key=lambda row: int(row["representation_index"]))
            xs = np.asarray([int(row["representation_index"]) for row in subset])
            ys = np.asarray([float(row["mean_cv_r2"]) for row in subset])
            lo = np.asarray([float(row["ci95_low_bootstrap_percentile"]) for row in subset])
            hi = np.asarray([float(row["ci95_high_bootstrap_percentile"]) for row in subset])
            highlighted = stream in HIGHLIGHT_STREAMS
            color = palette.get(stream, other_color)
            ax.plot(
                xs,
                ys,
                color=color,
                alpha=1.0 if highlighted else 0.28,
                linewidth=2.4 if highlighted else 1.1,
                label=stream if highlighted else None,
            )
            if int(subset[0]["n_seeds"]) > 1:
                ax.fill_between(xs, lo, hi, color=color, alpha=0.18 if highlighted else 0.05)
        ax.set_title(target_label)
        ax.set_xlabel("Representation")
        ax.set_ylabel("5-fold CV $R^2$")
        ax.grid(alpha=0.2)
        max_rep = max(int(row["representation_index"]) for row in rows)
        ax.set_xticks(np.arange(max_rep + 1))
        ax.set_xticklabels(["input"] + [str(i) for i in range(max_rep)])
        ax.set_ylim(-0.05, 1.02)
    axes[0, 0].legend(loc="lower right")

    out_dir = analysis_figure_dir(ANALYSIS_SLUG)
    out_path = out_dir / f"{run_label}_closed_form_probe_grid.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def run_analysis(
    checkpoints: Iterable[str | Path],
    run_label: str,
    batch_size: int,
    cv_folds: int,
) -> Dict[str, Any]:
    checkpoint_paths = [Path(path) for path in checkpoints]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required.")

    out_dir = analysis_data_dir(ANALYSIS_SLUG) / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    per_seed_rows: List[Dict[str, Any]] = []
    checkpoint_metadata: List[Dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        rows, metadata = probe_one_checkpoint(checkpoint_path, batch_size=batch_size, cv_folds=cv_folds, device=device)
        per_seed_rows.extend(rows)
        checkpoint_metadata.append(metadata)

    summary_rows = summarize_across_seeds(per_seed_rows)
    _write_csv(
        out_dir / "per_seed_probe_rows.csv",
        per_seed_rows,
        fieldnames=(
            "checkpoint",
            "seed",
            "num_layers",
            "representation_index",
            "layer_label",
            "stream",
            "target",
            "target_label",
            "cv_r2",
            "n_examples",
            "cv_folds",
        ),
    )
    _write_csv(
        out_dir / "across_seed_summary.csv",
        summary_rows,
        fieldnames=(
            "target",
            "target_label",
            "representation_index",
            "layer_label",
            "stream",
            "n_seeds",
            "mean_cv_r2",
            "ci95_low_bootstrap_percentile",
            "ci95_high_bootstrap_percentile",
        ),
    )
    metadata = {
        "analysis_slug": ANALYSIS_SLUG,
        "run_label": run_label,
        "device": str(device),
        "cv_seed": CV_SEED,
        "cv_folds": cv_folds,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
        "targets": [{"name": name, "label": label} for name, label in TARGETS],
        "streams": list(STREAMS),
        "activation_protocol": "autoregressive",
        "checkpoints": checkpoint_metadata,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    figure_path = plot_from_saved_data(run_label)
    return {
        "n_rows": len(per_seed_rows),
        "figure_path": str(figure_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoregressive linear probes for closed-form quantities.")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.checkpoints, args.run_label, args.batch_size, args.cv_folds), indent=2))


if __name__ == "__main__":
    main()
