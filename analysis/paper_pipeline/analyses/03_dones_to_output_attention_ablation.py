from __future__ import annotations

"""
Analysis 03: cumulative attention-edge ablation from D_ones into O[0]/O[1].
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from helpers.checkpoints import LoadedCheckpoint, load_checkpoint_bundle
from helpers.paths import analysis_data_dir, analysis_figure_dir
from helpers.runtime import get_device
from helpers.splits import reconstruct_canonical_splits, summarize_split_info


ANALYSIS_SLUG = "03_dones_to_output_attention_ablation"
BOOTSTRAP_SEED = 20260517
N_BOOTSTRAP_RESAMPLES = 100_000


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _exact_attention_scores(attn_mod, x_ln1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, d_model = x_ln1.shape
    num_heads = attn_mod.n_heads
    head_dim = attn_mod.head_dim
    qkv = attn_mod.qkv(x_ln1)
    q, k, v = qkv.split(d_model, dim=-1)
    q = q.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(head_dim))
    causal = torch.triu(torch.ones((seqlen, seqlen), device=x_ln1.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal.view(1, 1, seqlen, seqlen), -1e9)
    return scores, v


def _finish_attention(attn_mod, scores: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(scores, dim=-1)
    y_heads = torch.matmul(weights, v)
    y = y_heads.transpose(1, 2).contiguous().view(scores.shape[0], scores.shape[2], -1)
    return attn_mod.resid_drop(attn_mod.proj(y)), weights


def _forward_with_dones_block(
    bundle: LoadedCheckpoint,
    idx: torch.Tensor,
    blocked_layers: Set[int],
) -> Tuple[torch.Tensor, float]:
    """
    Block only D_ones -> O[0]/O[1] attention edges at selected layers.

    For sequence length 10 (prompt only), only the O[0] query position exists.
    For sequence length 11 (after one generated digit), both O[0] and O[1]
    query positions exist and are masked jointly.
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    _, seqlen = idx.shape
    pos = torch.arange(0, seqlen, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    x = model.drop(x)

    d_ones_pos = 8
    o0_pos = 9
    query_positions = [o0_pos]
    if seqlen >= 11:
        query_positions.append(10)

    max_forbidden = 0.0
    for layer_idx, block in enumerate(model.blocks):
        x_ln1 = block.ln1(x)
        if layer_idx in blocked_layers:
            scores, v = _exact_attention_scores(block.attn, x_ln1)
            scores = scores.clone()
            scores[:, :, query_positions, d_ones_pos] = -1e9
            attn_out, weights = _finish_attention(block.attn, scores, v)
            max_forbidden = max(max_forbidden, float(weights[:, :, query_positions, d_ones_pos].max().item()))
        else:
            attn_out = block.attn(x_ln1)
        x = x + attn_out
        x = x + block.mlp(block.ln2(x))
    logits = model.head(model.ln_f(x))
    return logits, max_forbidden


@torch.no_grad()
def evaluate_condition(
    bundle: LoadedCheckpoint,
    test_samples: Sequence[tuple[str, str]],
    blocked_layers: Set[int],
    max_examples: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    subset = list(test_samples[: max_examples if max_examples > 0 else len(test_samples)])
    exact_correct = 0
    o0_correct = 0
    o1_correct = 0
    max_forbidden = 0.0
    t0 = time.perf_counter()

    for start in range(0, len(subset), batch_size):
        batch = subset[start : start + batch_size]
        prompts = [prompt for prompt, _ in batch]
        targets = [target[:2] for _, target in batch]
        idx = torch.tensor([bundle.tokenizer.encode(prompt) for prompt in prompts], device=device, dtype=torch.long)

        logits0, forbidden0 = _forward_with_dones_block(bundle, idx, blocked_layers)
        first_digit = torch.argmax(logits0[:, -1, :], dim=-1, keepdim=True)
        idx1 = torch.cat([idx, first_digit], dim=1)
        logits1, forbidden1 = _forward_with_dones_block(bundle, idx1, blocked_layers)
        second_digit = torch.argmax(logits1[:, -1, :], dim=-1, keepdim=True)
        pred_tokens = torch.cat([first_digit, second_digit], dim=1).detach().cpu().tolist()
        preds = [bundle.tokenizer.decode(tokens) for tokens in pred_tokens]

        max_forbidden = max(max_forbidden, forbidden0, forbidden1)
        for pred, target in zip(preds, targets):
            exact_correct += int(pred == target)
            o0_correct += int(pred[0] == target[0])
            o1_correct += int(pred[1] == target[1])

    elapsed = time.perf_counter() - t0
    n = len(subset)
    return {
        "n_examples": n,
        "exact_answer_correct": exact_correct,
        "o0_correct": o0_correct,
        "o1_correct": o1_correct,
        "exact_answer_accuracy": exact_correct / max(1, n),
        "o0_token_accuracy": o0_correct / max(1, n),
        "o1_token_accuracy": o1_correct / max(1, n),
        "elapsed_seconds": elapsed,
        "examples_per_second": n / max(elapsed, 1e-12),
        "max_forbidden_attn_weight": max_forbidden,
    }


def _sweep_specs(num_layers: int) -> Dict[str, List[Set[int]]]:
    return {
        "forward": [set()] + [set(range(0, k + 1)) for k in range(num_layers)],
        "reverse": [set()] + [set(range(k, num_layers)) for k in range(num_layers - 1, -1, -1)],
    }


def run_one_checkpoint(
    checkpoint_path: str | Path,
    max_examples: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bundle = load_checkpoint_bundle(checkpoint_path, device=device)
    _, _, test_samples = reconstruct_canonical_splits(bundle.split_info)
    rows: List[Dict[str, Any]] = []
    for sweep_order, conditions in _sweep_specs(bundle.spec.num_layers).items():
        for step_index, blocked_layers in enumerate(conditions):
            metrics = evaluate_condition(
                bundle=bundle,
                test_samples=test_samples,
                blocked_layers=blocked_layers,
                max_examples=max_examples,
                batch_size=batch_size,
                device=device,
            )
            rows.append(
                {
                    "checkpoint": str(bundle.path),
                    "seed": bundle.spec.seed,
                    "num_layers": bundle.spec.num_layers,
                    "sweep_order": sweep_order,
                    "step_index": step_index,
                    "blocked_layers": ",".join(str(x) for x in sorted(blocked_layers)),
                    "num_blocked_layers": len(blocked_layers),
                    **metrics,
                }
            )
    metadata = {
        "checkpoint": str(bundle.path),
        "checkpoint_spec": asdict(bundle.spec),
        "split_summary": summarize_split_info(bundle.split_info),
    }
    return rows, metadata


def summarize_across_seeds(per_seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[float]] = {}
    label_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in per_seed_rows:
        key = (str(row["sweep_order"]), int(row["step_index"]))
        grouped.setdefault(key, []).append(float(row["exact_answer_accuracy"]))
        label_lookup[key] = {
            "blocked_layers": row["blocked_layers"],
            "num_blocked_layers": row["num_blocked_layers"],
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: List[Dict[str, Any]] = []
    for key, values_list in sorted(grouped.items()):
        values = np.asarray(values_list, dtype=np.float64)
        boot = rng.choice(values, size=(N_BOOTSTRAP_RESAMPLES, len(values)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sweep_order, step_index = key
        rows.append(
            {
                "sweep_order": sweep_order,
                "step_index": step_index,
                **label_lookup[key],
                "n_seeds": len(values),
                "mean_exact_answer_accuracy": float(values.mean()),
                "ci95_low_bootstrap_percentile": float(lo),
                "ci95_high_bootstrap_percentile": float(hi),
            }
        )
    return rows


def plot_from_saved_data(run_label: str) -> Path:
    data_path = analysis_data_dir(ANALYSIS_SLUG) / run_label / "across_seed_summary.csv"
    with data_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, sharey=True)
    for ax, order, title in zip(axes, ("forward", "reverse"), ("Forward sweep", "Reverse sweep")):
        subset = [row for row in rows if row["sweep_order"] == order]
        subset.sort(key=lambda row: int(row["step_index"]))
        xs = np.arange(len(subset))
        ys = np.asarray([float(row["mean_exact_answer_accuracy"]) for row in subset])
        lo = np.asarray([float(row["ci95_low_bootstrap_percentile"]) for row in subset])
        hi = np.asarray([float(row["ci95_high_bootstrap_percentile"]) for row in subset])
        labels = ["clean" if not row["blocked_layers"] else row["blocked_layers"] for row in subset]
        ax.plot(xs, ys, marker="o", linewidth=2.2)
        if int(subset[0]["n_seeds"]) > 1:
            ax.fill_between(xs, lo, hi, alpha=0.18)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title)
        ax.set_xlabel("Masked layers")
        ax.set_ylabel("Exact 2-digit answer accuracy")
        ax.grid(alpha=0.2)
    out_dir = analysis_figure_dir(ANALYSIS_SLUG)
    out_path = out_dir / f"{run_label}_dones_to_output_attention_ablation.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def run_analysis(
    checkpoints: Iterable[str | Path],
    run_label: str,
    max_examples: int,
    batch_size: int,
) -> Dict[str, Any]:
    device = get_device()
    out_dir = analysis_data_dir(ANALYSIS_SLUG) / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        rows, metadata = run_one_checkpoint(checkpoint, max_examples=max_examples, batch_size=batch_size, device=device)
        per_seed_rows.extend(rows)
        metadata_rows.append(metadata)
    summary_rows = summarize_across_seeds(per_seed_rows)

    _write_csv(
        out_dir / "per_seed_rows.csv",
        per_seed_rows,
        fieldnames=(
            "checkpoint",
            "seed",
            "num_layers",
            "sweep_order",
            "step_index",
            "blocked_layers",
            "num_blocked_layers",
            "n_examples",
            "exact_answer_correct",
            "o0_correct",
            "o1_correct",
            "exact_answer_accuracy",
            "o0_token_accuracy",
            "o1_token_accuracy",
            "elapsed_seconds",
            "examples_per_second",
            "max_forbidden_attn_weight",
        ),
    )
    _write_csv(
        out_dir / "across_seed_summary.csv",
        summary_rows,
        fieldnames=(
            "sweep_order",
            "step_index",
            "blocked_layers",
            "num_blocked_layers",
            "n_seeds",
            "mean_exact_answer_accuracy",
            "ci95_low_bootstrap_percentile",
            "ci95_high_bootstrap_percentile",
        ),
    )
    metadata = {
        "analysis_slug": ANALYSIS_SLUG,
        "run_label": run_label,
        "device": str(device),
        "max_examples": max_examples,
        "batch_size": batch_size,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
        "checkpoints": metadata_rows,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    fig_path = plot_from_saved_data(run_label)
    return {"n_rows": len(per_seed_rows), "figure_path": str(fig_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cumulative D_ones -> O[0]/O[1] attention ablation.")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.checkpoints, args.run_label, args.max_examples, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
