from __future__ import annotations

"""
Analysis 04: D_ones-to-output-stream attentional route patching.

Conditions:
- clean: source run with no substitution
- l1_only: donor D_ones K/V substituted only at layer 1 for output-stream queries
- l0_plus_l2plus: donor D_ones K/V substituted at layer 0 and all layers >= 2
  for output-stream queries; source D_ones computation itself remains source-like
  throughout.
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from helpers.checkpoints import LoadedCheckpoint, load_checkpoint_bundle
from helpers.paths import analysis_data_dir
from helpers.runtime import get_device
from helpers.splits import CanonicalRecord, records_for_splits, summarize_split_info


ANALYSIS_SLUG = "04_dones_output_attention_route_patching"
PAIR_SEED = 20260517
BOOTSTRAP_SEED = 20260517
N_BOOTSTRAP_RESAMPLES = 100_000
CONDITIONS = ("clean", "l1_only", "l0_plus_l2plus")
D_ONES_POSITION = 8
FIRST_OUTPUT_POSITION = 9
SECOND_OUTPUT_POSITION = 10


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_valid_pairs(records: Sequence[CanonicalRecord], max_pairs: int) -> List[Tuple[CanonicalRecord, CanonicalRecord]]:
    grouped: Dict[Tuple[int, int], List[CanonicalRecord]] = {}
    for record in records:
        grouped.setdefault((record.N, record.B), []).append(record)
    pairs: List[Tuple[CanonicalRecord, CanonicalRecord]] = []
    for group in grouped.values():
        for source in group:
            for donor in group:
                if donor.D == source.D:
                    continue
                if donor.target[:2] == source.target[:2]:
                    continue
                pairs.append((source, donor))
    rng = np.random.default_rng(PAIR_SEED)
    rng.shuffle(pairs)
    return pairs[: max_pairs if max_pairs > 0 else len(pairs)]


def _query_positions_for_length(seqlen: int) -> List[int]:
    positions = [FIRST_OUTPUT_POSITION]
    if seqlen > SECOND_OUTPUT_POSITION:
        positions.append(SECOND_OUTPUT_POSITION)
    return positions


def _exact_attention_components(attn_mod, x_ln1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seqlen, d_model = x_ln1.shape
    num_heads = attn_mod.n_heads
    head_dim = attn_mod.head_dim
    qkv = attn_mod.qkv(x_ln1)
    q, k, v = qkv.split(d_model, dim=-1)
    q = q.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    return q, k, v


def _scores_from_qk(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(q.shape[-1]))
    seqlen = q.shape[2]
    causal = torch.triu(torch.ones((seqlen, seqlen), device=q.device, dtype=torch.bool), diagonal=1)
    return scores.masked_fill(causal.view(1, 1, seqlen, seqlen), -1e9)


def _finish_attention(attn_mod, scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(scores, dim=-1)
    y_heads = torch.matmul(weights, v)
    y = y_heads.transpose(1, 2).contiguous().view(scores.shape[0], scores.shape[2], -1)
    return attn_mod.resid_drop(attn_mod.proj(y))


@torch.no_grad()
def _collect_layer_inputs(model, idx: torch.Tensor) -> List[torch.Tensor]:
    _, seqlen = idx.shape
    pos = torch.arange(0, seqlen, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    x = model.drop(x)
    layer_inputs: List[torch.Tensor] = [x.clone()]
    for layer_index, block in enumerate(model.blocks):
        x = block(x)
        if layer_index + 1 < len(model.blocks):
            layer_inputs.append(x.clone())
    return layer_inputs


def _forward_with_dones_kv_substitution(
    bundle: LoadedCheckpoint,
    source_idx: torch.Tensor,
    donor_layer_inputs: Sequence[torch.Tensor],
    patched_layers: Sequence[int],
) -> torch.Tensor:
    model = bundle.model
    _, seqlen = source_idx.shape
    pos = torch.arange(0, seqlen, device=source_idx.device)
    x = model.tok_emb(source_idx) + model.pos_emb(pos)[None, :, :]
    x = model.drop(x)
    patched_set = set(patched_layers)

    for layer_index, block in enumerate(model.blocks):
        if layer_index not in patched_set:
            x = block(x)
            continue

        x_ln1 = block.ln1(x)
        q_source, k_source, v_source = _exact_attention_components(block.attn, x_ln1)

        donor_x = donor_layer_inputs[layer_index]
        donor_ln1 = block.ln1(donor_x)
        _q_donor, k_donor, v_donor = _exact_attention_components(block.attn, donor_ln1)

        query_positions = _query_positions_for_length(seqlen)
        k_mixed = k_source.clone()
        v_mixed = v_source.clone()
        k_mixed[:, :, D_ONES_POSITION : D_ONES_POSITION + 1, :] = k_donor[:, :, D_ONES_POSITION : D_ONES_POSITION + 1, :]
        v_mixed[:, :, D_ONES_POSITION : D_ONES_POSITION + 1, :] = v_donor[:, :, D_ONES_POSITION : D_ONES_POSITION + 1, :]

        # Start from the entirely source-like attention result. Only the output
        # query rows are replaced below; all other rows, including D_ones -> D_ones,
        # must remain exactly source-like so the D_ones stream does not roll forward
        # from the donor intervention.
        scores = _scores_from_qk(q_source, k_source)
        attn_out = _finish_attention(block.attn, scores, v_source)

        query_scores = scores[:, :, query_positions, :].clone()
        donor_scores = torch.matmul(
            q_source[:, :, query_positions, :],
            k_mixed[:, :, D_ONES_POSITION : D_ONES_POSITION + 1, :].transpose(-2, -1),
        ) / math.sqrt(float(block.attn.head_dim))
        query_scores[:, :, :, D_ONES_POSITION : D_ONES_POSITION + 1] = donor_scores

        patched_query_out = _finish_attention(block.attn, query_scores, v_mixed)
        attn_out = attn_out.clone()
        attn_out[:, query_positions, :] = patched_query_out
        x = x + attn_out
        x = x + block.mlp(block.ln2(x))

    return model.head(model.ln_f(x))


@torch.no_grad()
def _evaluate_condition(
    bundle: LoadedCheckpoint,
    pairs: Sequence[Tuple[CanonicalRecord, CanonicalRecord]],
    condition: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    n_layers = bundle.spec.num_layers
    if condition == "l1_only":
        patched_layers = [1]
    elif condition == "l0_plus_l2plus":
        patched_layers = [0, *range(2, n_layers)]
    else:
        patched_layers = []

    source_exact = donor_exact = 0
    source_o0 = source_o1 = donor_o0 = donor_o1 = 0
    t0 = time.perf_counter()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        sources = [source for source, _ in batch]
        donors = [donor for _, donor in batch]
        source_idx = torch.tensor([tokenizer.encode(r.prompt) for r in sources], device=device, dtype=torch.long)
        donor_idx = torch.tensor([tokenizer.encode(d.prompt) for d in donors], device=device, dtype=torch.long)

        donor_layer_inputs_first = _collect_layer_inputs(model, donor_idx)
        if condition == "clean":
            logits0 = model(source_idx)
        else:
            logits0 = _forward_with_dones_kv_substitution(
                bundle,
                source_idx=source_idx,
                donor_layer_inputs=donor_layer_inputs_first,
                patched_layers=patched_layers,
            )

        first = torch.argmax(logits0[:, -1, :], dim=-1, keepdim=True)
        source_idx1 = torch.cat([source_idx, first], dim=1)

        donor_first_clean = torch.argmax(model(donor_idx)[:, -1, :], dim=-1, keepdim=True)
        donor_idx1 = torch.cat([donor_idx, donor_first_clean], dim=1)
        donor_layer_inputs_second = _collect_layer_inputs(model, donor_idx1)

        if condition == "clean":
            logits1 = model(source_idx1)
        else:
            logits1 = _forward_with_dones_kv_substitution(
                bundle,
                source_idx=source_idx1,
                donor_layer_inputs=donor_layer_inputs_second,
                patched_layers=patched_layers,
            )

        second = torch.argmax(logits1[:, -1, :], dim=-1, keepdim=True)
        preds = [tokenizer.decode(tokens) for tokens in torch.cat([first, second], dim=1).cpu().tolist()]

        for pred, source, donor in zip(preds, sources, donors):
            source_answer = source.target[:2]
            donor_answer = donor.target[:2]
            source_exact += int(pred == source_answer)
            donor_exact += int(pred == donor_answer)
            source_o0 += int(pred[0] == source_answer[0])
            source_o1 += int(pred[1] == source_answer[1])
            donor_o0 += int(pred[0] == donor_answer[0])
            donor_o1 += int(pred[1] == donor_answer[1])

    elapsed = time.perf_counter() - t0
    n = len(pairs)
    return {
        "pair_count": n,
        "source_exact_rate": source_exact / max(1, n),
        "donor_exact_rate": donor_exact / max(1, n),
        "source_o0_rate": source_o0 / max(1, n),
        "source_o1_rate": source_o1 / max(1, n),
        "donor_o0_rate": donor_o0 / max(1, n),
        "donor_o1_rate": donor_o1 / max(1, n),
        "elapsed_seconds": elapsed,
        "pairs_per_second": n / max(elapsed, 1e-12),
    }


def run_one_checkpoint(
    checkpoint_path: str | Path,
    max_pairs: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bundle = load_checkpoint_bundle(checkpoint_path, device=device)
    records = records_for_splits(bundle.split_info, ("test",))
    all_pairs = _build_valid_pairs(records, max_pairs=0)
    pairs = _build_valid_pairs(records, max_pairs=max_pairs)
    rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        metrics = _evaluate_condition(bundle, pairs, condition, batch_size=batch_size, device=device)
        rows.append(
            {
                "checkpoint": str(bundle.path),
                "seed": bundle.spec.seed,
                "num_layers": bundle.spec.num_layers,
                "condition": condition,
                **metrics,
            }
        )
    metadata = {
        "checkpoint": str(bundle.path),
        "checkpoint_spec": asdict(bundle.spec),
        "split_summary": summarize_split_info(bundle.split_info),
        "available_pairs_in_test_split": len(all_pairs),
        "sampled_pairs": len(pairs),
    }
    return rows, metadata


def summarize_across_seeds(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    metrics = (
        "source_exact_rate",
        "donor_exact_rate",
        "source_o0_rate",
        "source_o1_rate",
        "donor_o0_rate",
        "donor_o1_rate",
    )
    for row in rows:
        group = grouped.setdefault(str(row["condition"]), {metric: [] for metric in metrics})
        for metric in metrics:
            group[metric].append(float(row[metric]))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        for metric in metrics:
            values = np.asarray(grouped[condition][metric], dtype=np.float64)
            boot = rng.choice(values, size=(N_BOOTSTRAP_RESAMPLES, len(values)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "ci95_low_bootstrap_percentile": float(lo),
                    "ci95_high_bootstrap_percentile": float(hi),
                }
            )
    return out


def run_analysis(
    checkpoints: Iterable[str | Path],
    run_label: str,
    max_pairs: int,
    batch_size: int,
) -> Dict[str, Any]:
    device = get_device()
    out_dir = analysis_data_dir(ANALYSIS_SLUG) / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        rows, metadata = run_one_checkpoint(checkpoint, max_pairs=max_pairs, batch_size=batch_size, device=device)
        per_seed_rows.extend(rows)
        metadata_rows.append(metadata)
    summary_rows = summarize_across_seeds(per_seed_rows)
    _write_csv(out_dir / "per_seed_rows.csv", per_seed_rows, tuple(per_seed_rows[0].keys()))
    _write_csv(out_dir / "across_seed_summary.csv", summary_rows, tuple(summary_rows[0].keys()))
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "analysis_slug": ANALYSIS_SLUG,
                "run_label": run_label,
                "pair_seed": PAIR_SEED,
                "max_pairs": max_pairs,
                "batch_size": batch_size,
                "conditions": list(CONDITIONS),
                "checkpoints": metadata_rows,
            },
            f,
            indent=2,
        )
    return {"n_rows": len(per_seed_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="D_ones -> output-stream attentional source/donor substitution.")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.checkpoints, args.run_label, args.max_pairs, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
