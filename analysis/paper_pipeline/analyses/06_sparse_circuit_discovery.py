from __future__ import annotations

"""
Analysis 06: greedy sparse-circuit discovery for prompt/output information flow.

The search proceeds right-to-left from O[0] and O[1]. For each destination task,
it cumulatively masks one source -> destination attention relation over layer
prefixes L0..Lk, tags the last important prefix before the cumulative effect
plateaus, and recursively asks which earlier streams support any newly tagged
prompt-side source. Each recursive task records the downstream stream through
which that source mattered, so the final circuit preserves route provenance as
well as aggregate sparsity.
"""

import argparse
import csv
import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from helpers.checkpoints import LoadedCheckpoint, load_checkpoint_bundle
from helpers.paths import analysis_data_dir, analysis_figure_dir
from helpers.runtime import get_device
from helpers.splits import reconstruct_canonical_splits, summarize_split_info

ANALYSIS_SLUG = "06_sparse_circuit_discovery"
BOOTSTRAP_SEED = 20260517
N_BOOTSTRAP_RESAMPLES = 100_000
PROMPT_LABELS = (
    "N_tag", "N_hundreds", "N_tens", "N_ones", "B_tag", "B_tens", "B_ones", "D_tag", "D_ones"
)
PROMPT_POSITIONS = {label: i for i, label in enumerate(PROMPT_LABELS)}
O0_LABEL = "O[0]"
O1_LABEL = "O[1]"
O0_POSITION = 9
O1_POSITION = 10
ALL_STREAMS = PROMPT_LABELS + (O0_LABEL, O1_LABEL)


@dataclass(frozen=True)
class SearchTask:
    destination: str
    max_layer: int
    through: str


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _destination_position(label: str, seqlen: int) -> Optional[int]:
    if label in PROMPT_POSITIONS:
        return PROMPT_POSITIONS[label]
    if label == O0_LABEL:
        return O0_POSITION if seqlen > O0_POSITION else None
    if label == O1_LABEL:
        return O1_POSITION if seqlen > O1_POSITION else None
    raise KeyError(label)


def _candidate_sources(destination: str) -> List[str]:
    if destination == O0_LABEL:
        return list(PROMPT_LABELS)
    if destination == O1_LABEL:
        return list(PROMPT_LABELS) + [O0_LABEL]
    dest_pos = PROMPT_POSITIONS[destination]
    return list(PROMPT_LABELS[:dest_pos])


def _source_position(label: str) -> int:
    if label in PROMPT_POSITIONS:
        return PROMPT_POSITIONS[label]
    if label == O0_LABEL:
        return O0_POSITION
    raise KeyError(label)


def _exact_attention_scores(attn_mod, x_ln1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, d_model = x_ln1.shape
    qkv = attn_mod.qkv(x_ln1)
    q, k, v = qkv.split(d_model, dim=-1)
    q = q.view(bsz, seqlen, attn_mod.n_heads, attn_mod.head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, attn_mod.n_heads, attn_mod.head_dim).transpose(1, 2)
    v = v.view(bsz, seqlen, attn_mod.n_heads, attn_mod.head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(attn_mod.head_dim))
    causal = torch.triu(torch.ones((seqlen, seqlen), device=x_ln1.device, dtype=torch.bool), diagonal=1)
    return scores.masked_fill(causal.view(1, 1, seqlen, seqlen), -1e9), v


def _finish_attention(attn_mod, scores: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(scores, dim=-1)
    y_heads = torch.matmul(weights, v)
    y = y_heads.transpose(1, 2).contiguous().view(scores.shape[0], scores.shape[2], -1)
    return attn_mod.resid_drop(attn_mod.proj(y)), weights


def _forward_with_single_edge_block(
    bundle: LoadedCheckpoint,
    idx: torch.Tensor,
    destination: str,
    source: str,
    blocked_layers: Set[int],
) -> Tuple[torch.Tensor, float]:
    model = bundle.model
    _, seqlen = idx.shape
    pos = torch.arange(seqlen, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    x = model.drop(x)
    query_pos = _destination_position(destination, seqlen)
    source_pos = _source_position(source)
    max_forbidden = 0.0
    for layer_idx, block in enumerate(model.blocks):
        x_ln1 = block.ln1(x)
        if query_pos is not None and layer_idx in blocked_layers and source_pos < seqlen:
            scores, v = _exact_attention_scores(block.attn, x_ln1)
            scores = scores.clone()
            scores[:, :, query_pos, source_pos] = -1e9
            attn_out, weights = _finish_attention(block.attn, scores, v)
            max_forbidden = max(max_forbidden, float(weights[:, :, query_pos, source_pos].max().item()))
        else:
            attn_out = block.attn(x_ln1)
        x = x + attn_out
        x = x + block.mlp(block.ln2(x))
    return model.head(model.ln_f(x)), max_forbidden


def _forward_keep_only_circuit(
    bundle: LoadedCheckpoint,
    idx: torch.Tensor,
    retained_prefixes: Mapping[Tuple[str, str], int],
    constrained_destinations: Mapping[str, int],
) -> Tuple[torch.Tensor, float]:
    model = bundle.model
    _, seqlen = idx.shape
    pos = torch.arange(seqlen, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
    x = model.drop(x)
    max_forbidden = 0.0
    for layer_idx, block in enumerate(model.blocks):
        x_ln1 = block.ln1(x)
        scores = None
        v = None
        masked_pairs: List[Tuple[int, int]] = []
        for destination, max_layer in constrained_destinations.items():
            if layer_idx > max_layer:
                continue
            query_pos = _destination_position(destination, seqlen)
            if query_pos is None:
                continue
            for source in _candidate_sources(destination):
                source_pos = _source_position(source)
                if source_pos >= seqlen or source_pos >= query_pos:
                    continue
                keep_max = retained_prefixes.get((source, destination), -1)
                if layer_idx > keep_max:
                    if scores is None:
                        scores, v = _exact_attention_scores(block.attn, x_ln1)
                        scores = scores.clone()
                    scores[:, :, query_pos, source_pos] = -1e9
                    masked_pairs.append((query_pos, source_pos))
        if scores is None:
            attn_out = block.attn(x_ln1)
        else:
            assert v is not None
            attn_out, weights = _finish_attention(block.attn, scores, v)
            for query_pos, source_pos in masked_pairs:
                max_forbidden = max(max_forbidden, float(weights[:, :, query_pos, source_pos].max().item()))
        x = x + attn_out
        x = x + block.mlp(block.ln2(x))
    return model.head(model.ln_f(x)), max_forbidden


@torch.no_grad()
def _evaluate_autoregressive(
    bundle: LoadedCheckpoint,
    samples: Sequence[Tuple[str, str]],
    batch_size: int,
    max_examples: int,
    forward_fn,
) -> Dict[str, Any]:
    subset = list(samples[: max_examples if max_examples > 0 else len(samples)])
    exact = o0 = o1 = 0
    max_forbidden = 0.0
    for start in range(0, len(subset), batch_size):
        batch = subset[start : start + batch_size]
        idx = torch.tensor([bundle.tokenizer.encode(prompt) for prompt, _ in batch], device=next(bundle.model.parameters()).device)
        logits0, forbidden0 = forward_fn(idx)
        first = torch.argmax(logits0[:, -1, :], dim=-1, keepdim=True)
        idx1 = torch.cat([idx, first], dim=1)
        logits1, forbidden1 = forward_fn(idx1)
        second = torch.argmax(logits1[:, -1, :], dim=-1, keepdim=True)
        preds = [bundle.tokenizer.decode(tokens) for tokens in torch.cat([first, second], dim=1).cpu().tolist()]
        targets = [target[:2] for _, target in batch]
        max_forbidden = max(max_forbidden, forbidden0, forbidden1)
        for pred, target in zip(preds, targets):
            exact += int(pred == target)
            o0 += int(pred[0] == target[0])
            o1 += int(pred[1] == target[1])
    n = len(subset)
    return {
        "n_examples": n,
        "exact_answer_accuracy": exact / max(1, n),
        "o0_token_accuracy": o0 / max(1, n),
        "o1_token_accuracy": o1 / max(1, n),
        "max_forbidden_attn_weight": max_forbidden,
    }


def _select_discovery_samples(
    samples: Sequence[Tuple[str, str]],
    max_examples: int,
    sample_seed: int,
) -> List[Tuple[str, str]]:
    """
    Select a deterministic random validation subset for circuit discovery.

    If `max_examples <= 0`, return the full validation split unchanged. When a
    capped discovery run is requested, sampling rather than taking the first N
    examples avoids making the retained circuit depend on dataset iteration
    order.
    """
    if max_examples <= 0 or max_examples >= len(samples):
        return list(samples)
    rng = np.random.default_rng(sample_seed)
    indices = np.sort(rng.choice(len(samples), size=max_examples, replace=False))
    return [samples[int(i)] for i in indices]


def _select_last_important_prefix(
    exact_drops: Sequence[float],
    drop_threshold: float,
    relative_increment_fraction: float,
    plateau_patience: int,
) -> Optional[int]:
    best = 0.0
    first_jump: Optional[float] = None
    last: Optional[int] = None
    plateau = 0
    for layer_idx, drop in enumerate(exact_drops):
        if first_jump is None and drop < drop_threshold:
            continue
        improvement = drop - best
        if first_jump is None:
            first_jump = drop
            best = drop
            last = layer_idx
            plateau = 0
            continue
        threshold = relative_increment_fraction * first_jump
        if improvement > threshold:
            best = drop
            last = layer_idx
            plateau = 0
        else:
            plateau += 1
            if plateau >= plateau_patience:
                break
    return last


def _merge_task(tasks: MutableMapping[Tuple[str, str], int], destination: str, through: str, max_layer: int) -> bool:
    key = (destination, through)
    previous = tasks.get(key, -1)
    if max_layer > previous:
        tasks[key] = max_layer
        return True
    return False


def _discover_one_checkpoint(
    checkpoint_path: str | Path,
    discover_max_examples: int,
    eval_max_examples: int,
    batch_size: int,
    drop_threshold: float,
    relative_increment_fraction: float,
    plateau_patience: int,
    discovery_sample_seed: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    bundle = load_checkpoint_bundle(checkpoint_path, device=device)
    _, val_samples, test_samples = reconstruct_canonical_splits(bundle.split_info)
    discovery_samples = _select_discovery_samples(
        val_samples,
        max_examples=discover_max_examples,
        sample_seed=discovery_sample_seed,
    )
    discover_clean = _evaluate_autoregressive(bundle, discovery_samples, batch_size, 0, lambda idx: (bundle.model(idx), 0.0))
    eval_clean = _evaluate_autoregressive(bundle, test_samples, batch_size, eval_max_examples, lambda idx: (bundle.model(idx), 0.0))

    tasks: Dict[Tuple[str, str], int] = {}
    queue: deque[SearchTask] = deque()
    for root in (O0_LABEL, O1_LABEL):
        _merge_task(tasks, root, "ROOT", bundle.spec.num_layers - 1)
        queue.append(SearchTask(root, bundle.spec.num_layers - 1, "ROOT"))

    sweep_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    retained_prefixes: Dict[Tuple[str, str], int] = {}
    constrained_destinations: Dict[str, int] = {O0_LABEL: bundle.spec.num_layers - 1, O1_LABEL: bundle.spec.num_layers - 1}
    processed: Set[Tuple[str, str, int]] = set()

    while queue:
        task = queue.popleft()
        task_key = (task.destination, task.through, task.max_layer)
        if task_key in processed:
            continue
        processed.add(task_key)
        constrained_destinations[task.destination] = max(constrained_destinations.get(task.destination, -1), task.max_layer)
        for source in _candidate_sources(task.destination):
            exact_drops: List[float] = []
            for blocked_prefix_max_layer in range(task.max_layer + 1):
                metrics = _evaluate_autoregressive(
                    bundle,
                    discovery_samples,
                    batch_size,
                    0,
                    lambda idx, destination=task.destination, source=source, blocked_prefix_max_layer=blocked_prefix_max_layer: _forward_with_single_edge_block(
                        bundle, idx, destination, source, set(range(blocked_prefix_max_layer + 1))
                    ),
                )
                drop = discover_clean["exact_answer_accuracy"] - metrics["exact_answer_accuracy"]
                exact_drops.append(drop)
                sweep_rows.append({
                    "checkpoint": str(bundle.path), "seed": bundle.spec.seed, "num_layers": bundle.spec.num_layers,
                    "destination": task.destination, "through": task.through, "destination_max_layer": task.max_layer,
                    "source": source, "blocked_prefix_max_layer": blocked_prefix_max_layer,
                    "blocked_layers": f"L0-L{blocked_prefix_max_layer}", "exact_drop_from_clean": drop, **metrics,
                })
            keep_max = _select_last_important_prefix(exact_drops, drop_threshold, relative_increment_fraction, plateau_patience)
            retained = keep_max is not None
            edge_rows.append({
                "checkpoint": str(bundle.path), "seed": bundle.spec.seed, "num_layers": bundle.spec.num_layers,
                "source": source, "destination": task.destination, "through": task.through,
                "destination_max_layer": task.max_layer, "keep_max_layer": -1 if keep_max is None else keep_max,
                "kept_layers": "none" if keep_max is None else f"L0-L{keep_max}",
                "retained": int(retained), "max_exact_drop": max(exact_drops) if exact_drops else 0.0,
                "exact_drop_at_kept_prefix": 0.0 if keep_max is None else exact_drops[keep_max],
                "drop_threshold": drop_threshold, "relative_increment_fraction": relative_increment_fraction,
                "plateau_patience": plateau_patience,
            })
            if retained:
                retained_prefixes[(source, task.destination)] = max(retained_prefixes.get((source, task.destination), -1), int(keep_max))
                if source in PROMPT_POSITIONS:
                    if _merge_task(tasks, source, task.destination, int(keep_max)):
                        queue.append(SearchTask(source, int(keep_max), task.destination))

    kept_only = _evaluate_autoregressive(
        bundle,
        test_samples,
        batch_size,
        eval_max_examples,
        lambda idx: _forward_keep_only_circuit(bundle, idx, retained_prefixes, constrained_destinations),
    )

    candidate_relations = len(edge_rows)
    retained_relations = sum(int(row["retained"]) for row in edge_rows)
    candidate_layer_edges = sum(int(row["destination_max_layer"]) + 1 for row in edge_rows)
    retained_layer_edges = sum(int(row["keep_max_layer"]) + 1 for row in edge_rows if int(row["keep_max_layer"]) >= 0)
    required_depth_by_destination = {
        destination: max_layer for destination, max_layer in sorted(constrained_destinations.items(), key=lambda kv: ALL_STREAMS.index(kv[0]))
    }
    metrics_rows = [
        {"variant": "discover_clean_validation", **discover_clean},
        {"variant": "eval_clean_test", **eval_clean},
        {"variant": "eval_kept_only_test", **kept_only},
    ]
    summary = {
        "checkpoint": str(bundle.path),
        "checkpoint_spec": asdict(bundle.spec),
        "split_summary": summarize_split_info(bundle.split_info),
        "discovery_parameters": {
            "discover_split": "val", "eval_split": "test", "discover_max_examples": discover_max_examples,
            "discovery_sample_seed": discovery_sample_seed,
            "n_discovery_examples_used": len(discovery_samples),
            "eval_max_examples": eval_max_examples, "batch_size": batch_size, "drop_threshold": drop_threshold,
            "relative_increment_fraction": relative_increment_fraction, "plateau_patience": plateau_patience,
        },
        "required_depth_by_destination": required_depth_by_destination,
        "sparsity": {
            "candidate_relations": candidate_relations,
            "retained_relations": retained_relations,
            "retained_relation_fraction": retained_relations / max(1, candidate_relations),
            "candidate_layer_edges": candidate_layer_edges,
            "retained_layer_edges": retained_layer_edges,
            "retained_layer_edge_fraction": retained_layer_edges / max(1, candidate_layer_edges),
        },
        "metrics": metrics_rows,
    }
    return sweep_rows, edge_rows, metrics_rows, summary


def _summarize_across_checkpoints(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_specs = [
        ("eval_clean_test", "exact_answer_accuracy"),
        ("eval_kept_only_test", "exact_answer_accuracy"),
        ("sparsity", "retained_relation_fraction"),
        ("sparsity", "retained_layer_edge_fraction"),
    ]
    rows: List[Dict[str, Any]] = []
    for variant, metric in metric_specs:
        values = []
        for summary in summaries:
            if variant == "sparsity":
                values.append(float(summary["sparsity"][metric]))
            else:
                row = next(r for r in summary["metrics"] if r["variant"] == variant)
                values.append(float(row[metric]))
        arr = np.asarray(values, dtype=float)
        boot = rng.choice(arr, size=(N_BOOTSTRAP_RESAMPLES, len(arr)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"variant": variant, "metric": metric, "mean": float(arr.mean()), "ci95_low": float(lo), "ci95_high": float(hi), "n_checkpoints": len(arr)})
    return rows


def _plot_pruning_matrix(edge_rows: Sequence[Mapping[str, Any]], out_path: Path, title: str) -> None:
    dests = [stream for stream in ALL_STREAMS if any(row["destination"] == stream for row in edge_rows)]
    sources = list(ALL_STREAMS[:-1])
    mat = np.full((len(dests), len(sources)), -1.0)
    for row in edge_rows:
        if int(row["keep_max_layer"]) < 0:
            continue
        di = dests.index(str(row["destination"]))
        si = sources.index(str(row["source"]))
        mat[di, si] = max(mat[di, si], float(row["keep_max_layer"]))
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-1)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=35, ha="right")
    ax.set_yticks(range(len(dests)))
    ax.set_yticklabels(dests)
    ax.set_xlabel("Source stream")
    ax.set_ylabel("Destination stream")
    ax.set_title(title)
    for di in range(mat.shape[0]):
        for si in range(mat.shape[1]):
            if mat[di, si] >= 0:
                ax.text(si, di, f"L0-L{int(mat[di, si])}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Deepest retained layer")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_compact_circuit(edge_rows: Sequence[Mapping[str, Any]], out_path: Path, title: str) -> None:
    """Render the retained circuit with the original main-analysis diagram grammar."""
    from matplotlib.patches import FancyBboxPatch

    # Preserve the old figure's visual language exactly; only map the new
    # pipeline's labels into the old renderer's nomenclature.
    node_positions = {
        "N_tag": (0.0, 7.2),
        "N_hundreds": (0.0, 6.0),
        "N_tens": (0.0, 4.8),
        "N_ones": (0.0, 3.6),
        "B_tag": (0.0, 2.1),
        "B_tens": (0.0, 0.9),
        "B_ones": (0.0, -0.3),
        "D_tag": (0.0, -2.0),
        "pre_O": (0.0, -3.3),
        "O": (5.0, -1.0),
        "O+1": (5.0, 1.7),
    }
    node_fills = {"N": "#d9e8fb", "B": "#dff4df", "D": "#ffe9cc", "OUT": "#f7dcec"}
    node_edges = {"N": "#4a78c2", "B": "#4f9a5d", "D": "#c9852d", "OUT": "#a54f7d"}
    label_positions = {
        # Keep hand-tuned positions only for compact local scaffold edges.
        ("N_tag", "N_hundreds"): (-0.46, 6.02),
        ("N_hundreds", "N_tens"): (-0.34, 5.18),
        ("N_tag", "N_tens"): (-0.34, 5.18),
        ("B_tag", "B_tens"): (-0.42, 0.96),
        ("B_tag", "B_ones"): (-0.72, 0.10),
        ("B_tens", "B_ones"): (-0.20, -0.06),
        ("D_tag", "pre_O"): (-0.38, -2.66),
    }

    def map_label(label: str) -> str:
        return {"D_ones": "pre_O", O0_LABEL: "O", O1_LABEL: "O+1"}.get(label, label)

    kept_edges: List[Dict[str, Any]] = []
    for row in edge_rows:
        if int(row["keep_max_layer"]) < 0:
            continue
        kept_edges.append({
            "source_label": map_label(str(row["source"])),
            "destination": map_label(str(row["destination"])),
            "keep_max_layer": int(row["keep_max_layer"]),
            "max_exact_drop": float(row["max_exact_drop"]),
        })
    kept_edges.sort(key=lambda row: (node_positions[row["source_label"]][0], node_positions[row["destination"]][0]))

    def family(label: str) -> str:
        if label.startswith("N_"):
            return "N"
        if label.startswith("B_"):
            return "B"
        if label in {"D_tag", "pre_O"}:
            return "D"
        return "OUT"

    def edge_rad(source: str, dest: str) -> float:
        if source == "O" and dest == "O+1":
            return 0.68
        if dest in {"O", "O+1"}:
            delta = node_positions[source][1] - node_positions[dest][1]
            if delta > 4:
                return -0.22
            if delta > 2:
                return -0.14
            if delta > 0:
                return -0.08
            if delta > -2:
                return 0.08
            return 0.16
        if source == "D_tag" and dest == "pre_O":
            return 0.0
        if source == "B_tag" and dest == "B_ones":
            return 0.36
        if source == "B_tens" and dest == "B_ones":
            return 0.18
        if source == "B_tag" and dest == "B_tens":
            return 0.28
        if source.startswith("B_") and dest.startswith("B_"):
            return 0.22
        if source.startswith("N_") and dest.startswith("N_"):
            return 0.18
        return 0.12

    fig, ax = plt.subplots(figsize=(11.8, 6.9), constrained_layout=True)
    ax.set_facecolor("#fcfbf7")
    fig.patch.set_facecolor("#fcfbf7")
    max_drop = max(float(row["max_exact_drop"]) for row in kept_edges) if kept_edges else 1.0

    for row in kept_edges:
        source = row["source_label"]
        dest = row["destination"]
        sx, sy = node_positions[source]
        dx, dy = node_positions[dest]
        fam = family(source)
        drop = float(row["max_exact_drop"])
        keep_layer = int(row["keep_max_layer"])
        linewidth = 1.2 + 6.0 * (drop / max_drop if max_drop > 0 else 0.0)
        alpha = 0.55 + 0.4 * (drop / max_drop if max_drop > 0 else 0.0)
        start_x = sx + 0.80 if dx > sx else sx
        start_y = sy - 0.02
        end_x = dx - 0.80 if dx > sx else dx
        end_y = dy + 0.02
        if source == "O" and dest == "O+1":
            start_x = sx + 0.88
            end_x = dx + 0.88
            start_y = sy + 0.06
            end_y = dy - 0.06
        elif dx == sx and abs(dy - sy) > 1.0:
            start_x = sx - 0.82
            end_x = dx - 0.82
            start_y = sy - 0.06
            end_y = dy + 0.06
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                connectionstyle=f"arc3,rad={edge_rad(source, dest)}",
                                arrowstyle="-|>", mutation_scale=14, linewidth=linewidth,
                                color=node_edges[fam], alpha=alpha, zorder=3, shrinkA=1, shrinkB=1)
        arrow.set_sketch_params(scale=0.45, length=85.0, randomness=1.5)
        ax.add_patch(arrow)
        if (source, dest) in label_positions:
            mx, my = label_positions[(source, dest)]
        elif source == "O" and dest == "O+1":
            mx, my = (6.18, 0.38)
        else:
            # Place long-route labels near the corresponding destination-side
            # segment of their own edge rather than stacking them at mid-bundle.
            # This is an explicit visual layout map, not a data transform.
            long_route_labels = {
                ("N_hundreds", "O+1"): (3.92, 3.78),
                ("N_tens", "O+1"): (3.62, 2.92),
                ("N_ones", "O+1"): (3.38, 2.12),
                ("B_tens", "O+1"): (3.26, 0.62),
                ("B_ones", "O+1"): (3.56, -0.28),
                ("pre_O", "O+1"): (3.86, -1.72),
                ("N_hundreds", "O"): (4.10, 1.96),
                ("N_tens", "O"): (3.84, 1.20),
                ("N_ones", "O"): (3.54, 0.38),
                ("B_tens", "O"): (3.30, -0.84),
                ("B_ones", "O"): (3.64, -1.62),
                ("pre_O", "O"): (4.04, -2.88),
            }
            mx, my = long_route_labels.get((source, dest), ((sx + dx) / 2.0, (sy + dy) / 2.0))
        layer_label = f"L0-L{keep_layer}" if keep_layer > 0 else "L0"
        ax.text(mx, my, layer_label, ha="center", va="center", fontsize=9, color="#222222",
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.08", facecolor="white",
                          edgecolor="#d2d2d2", linewidth=0.8, alpha=0.95), zorder=6)

    for label, (x, y) in node_positions.items():
        fam = family(label)
        box = FancyBboxPatch((x - 0.78, y - 0.32), 1.56, 0.64,
                             boxstyle="round,pad=0.03,rounding_size=0.12",
                             linewidth=1.8, edgecolor=node_edges[fam], facecolor=node_fills[fam], zorder=4)
        box.set_sketch_params(scale=0.35, length=90.0, randomness=1.2)
        ax.add_patch(box)
        display_label = {"O": "O[0]", "O+1": "O[1]", "pre_O": "D_ones (pre_O)"}.get(label, label)
        ax.text(x, y, display_label, ha="center", va="center", fontsize=11, color="#1d1d1d",
                zorder=5, family="DejaVu Sans", fontweight="semibold")

    ax.text(2.45, 8.45, title, ha="center", va="center", fontsize=14, fontweight="bold", color="#1f1f1f")
    ax.text(-0.05, 7.95, "Prompt-side streams", ha="center", fontsize=10, color="#4b4b4b")
    ax.text(5.0, 3.1, "Output-side streams", ha="center", fontsize=10, color="#a54f7d")
    ax.text(1.20, 6.55, "Number scaffold", ha="center", fontsize=10, color="#4a78c2")
    ax.text(1.05, 1.20, "Base scaffold", ha="center", fontsize=10, color="#4f9a5d")
    ax.text(1.18, -2.55, "Query handoff", ha="center", fontsize=10, color="#c9852d")
    ax.set_xlim(-1.6, 7.15)
    ax.set_ylim(-5.2, 8.9)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260)
    fig.savefig(out_path.with_suffix(".pdf"))
    # Use live text in SVG exports so the file is friendlier to edit in Inkscape.
    old_svg_fonttype = plt.rcParams.get("svg.fonttype")
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(out_path.with_suffix(".svg"))
    fig.savefig(out_path.with_name(out_path.stem + "_editable.svg"))
    plt.rcParams["svg.fonttype"] = old_svg_fonttype
    plt.close(fig)


def plot_from_saved_data(run_label: str) -> None:
    data_dir = analysis_data_dir(ANALYSIS_SLUG) / run_label
    fig_dir = analysis_figure_dir(ANALYSIS_SLUG) / run_label
    edge_rows = list(csv.DictReader((data_dir / "edge_summary_rows.csv").open()))
    by_seed: Dict[str, List[Dict[str, Any]]] = {}
    for row in edge_rows:
        by_seed.setdefault(str(row["seed"]), []).append(row)
    for seed, rows in by_seed.items():
        _plot_pruning_matrix(rows, fig_dir / f"seed_{seed}_retained_prefix_matrix.png", f"Retained layer prefixes — seed {seed}")
        _plot_compact_circuit(rows, fig_dir / f"seed_{seed}_compact_circuit.png", f"Sparse retained circuit — seed {seed}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy right-to-left sparse-circuit discovery.")
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--discover-max-examples", type=int, default=0)
    ap.add_argument("--eval-max-examples", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--drop-threshold", type=float, default=0.02)
    ap.add_argument("--relative-increment-fraction", type=float, default=0.20)
    ap.add_argument("--plateau-patience", type=int, default=2)
    ap.add_argument("--discovery-sample-seed", type=int, default=20260517)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    device = get_device()
    all_sweeps: List[Dict[str, Any]] = []
    all_edges: List[Dict[str, Any]] = []
    all_metric_rows: List[Dict[str, Any]] = []
    all_required_depth_rows: List[Dict[str, Any]] = []
    all_sparsity_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        sweep_rows, edge_rows, metric_rows, summary = _discover_one_checkpoint(
            checkpoint, args.discover_max_examples, args.eval_max_examples, args.batch_size,
            args.drop_threshold, args.relative_increment_fraction, args.plateau_patience,
            args.discovery_sample_seed, device,
        )
        all_sweeps.extend(sweep_rows)
        all_edges.extend(edge_rows)
        for row in metric_rows:
            all_metric_rows.append({"checkpoint": summary["checkpoint"], "seed": summary["checkpoint_spec"]["seed"], **row})
        for destination, max_layer in summary["required_depth_by_destination"].items():
            all_required_depth_rows.append({
                "checkpoint": summary["checkpoint"],
                "seed": summary["checkpoint_spec"]["seed"],
                "destination": destination,
                "required_max_layer": max_layer,
                "required_layers": f"L0-L{max_layer}",
            })
        all_sparsity_rows.append({
            "checkpoint": summary["checkpoint"],
            "seed": summary["checkpoint_spec"]["seed"],
            **summary["sparsity"],
        })
        summaries.append(summary)

    out_dir = analysis_data_dir(ANALYSIS_SLUG) / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "prefix_sweep_rows.csv", all_sweeps, list(all_sweeps[0].keys()))
    _write_csv(out_dir / "edge_summary_rows.csv", all_edges, list(all_edges[0].keys()))
    _write_csv(out_dir / "per_checkpoint_metrics.csv", all_metric_rows, list(all_metric_rows[0].keys()))
    _write_csv(out_dir / "required_depth_rows.csv", all_required_depth_rows, list(all_required_depth_rows[0].keys()))
    _write_csv(out_dir / "sparsity_rows.csv", all_sparsity_rows, list(all_sparsity_rows[0].keys()))
    summary_rows = _summarize_across_checkpoints(summaries)
    _write_csv(out_dir / "across_checkpoint_summary.csv", summary_rows, list(summary_rows[0].keys()))
    (out_dir / "metadata.json").write_text(json.dumps({"run_label": args.run_label, "checkpoints": summaries}, indent=2), encoding="utf-8")
    if not args.skip_plots:
        plot_from_saved_data(args.run_label)


if __name__ == "__main__":
    main()
