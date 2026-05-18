from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import Sample, generate_all_samples


CANONICAL_ORDER = ("N", "B", "D")
VALID_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class CanonicalRecord:
    """Canonical-order sample with parsed fields attached."""

    prompt: str
    target: str
    N: int
    B: int
    D: int
    answer_value: int
    split_name: str


def parse_prompt_fields(prompt: str) -> Tuple[int, int, int]:
    """Extract `(N, B, D)` from a canonical or permuted prompt string."""
    n_idx = prompt.index("N")
    b_idx = prompt.index("B")
    d_idx = prompt.index("D")
    return (
        int(prompt[n_idx + 1 : n_idx + 4]),
        int(prompt[b_idx + 1 : b_idx + 3]),
        int(prompt[d_idx + 1]),
    )


def _as_set(values: Iterable[Any]) -> set[Any]:
    """Normalize checkpoint-stored iterables to plain Python sets."""
    return set(values)


def reconstruct_canonical_splits(
    split_info: Mapping[str, Any],
    max_n: int = 1000,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Recreate the exact canonical train/validation/test split stored in a checkpoint.

    The function intentionally uses the held-out identities from `split_info`
    instead of rerunning pseudo-random split construction from `Config.seed`.
    This keeps inference and analysis aligned with the network that was actually
    trained, even when the current repository configuration changes later.
    """
    mode = str(split_info["mode"])
    all_samples = generate_all_samples(max_n=max_n, order=CANONICAL_ORDER)

    train: List[Sample] = []
    val: List[Sample] = []
    test: List[Sample] = []

    if mode == "by_NB_intersection":
        val_ns = _as_set(split_info["val_Ns"])
        test_ns = _as_set(split_info["test_Ns"])
        val_bs = _as_set(split_info["val_Bs"])
        test_bs = _as_set(split_info["test_Bs"])

        for prompt, target in all_samples:
            n, b, _ = parse_prompt_fields(prompt)
            if n in test_ns and b in test_bs:
                test.append((prompt, target))
            elif n in val_ns and b in val_bs:
                val.append((prompt, target))
            else:
                train.append((prompt, target))
        return train, val, test

    if mode == "by_N":
        train_ns = _as_set(split_info["train_ns"])
        val_ns = _as_set(split_info["val_ns"])
        test_ns = _as_set(split_info["test_ns"])
        for prompt, target in all_samples:
            n, _, _ = parse_prompt_fields(prompt)
            if n in train_ns:
                train.append((prompt, target))
            elif n in val_ns:
                val.append((prompt, target))
            elif n in test_ns:
                test.append((prompt, target))
        return train, val, test

    if mode == "by_base":
        train_bs = _as_set(split_info["train_bs"])
        val_bs = _as_set(split_info["val_bs"])
        test_bs = _as_set(split_info["test_bs"])
        for prompt, target in all_samples:
            _, b, _ = parse_prompt_fields(prompt)
            if b in train_bs:
                train.append((prompt, target))
            elif b in val_bs:
                val.append((prompt, target))
            elif b in test_bs:
                test.append((prompt, target))
        return train, val, test

    if mode == "by_NB":
        train_pairs = {tuple(pair) for pair in split_info["train_pairs"]}
        val_pairs = {tuple(pair) for pair in split_info["val_pairs"]}
        test_pairs = {tuple(pair) for pair in split_info["test_pairs"]}
        for prompt, target in all_samples:
            n, b, _ = parse_prompt_fields(prompt)
            pair = (n, b)
            if pair in train_pairs:
                train.append((prompt, target))
            elif pair in val_pairs:
                val.append((prompt, target))
            elif pair in test_pairs:
                test.append((prompt, target))
        return train, val, test

    raise ValueError(f"Unsupported split mode in checkpoint split_info: {mode}")


def records_for_splits(
    split_info: Mapping[str, Any],
    split_names: Sequence[str],
    max_n: int = 1000,
) -> List[CanonicalRecord]:
    """Return parsed canonical records from the requested split names."""
    requested = tuple(split_names)
    invalid = sorted(set(requested) - set(VALID_SPLIT_NAMES))
    if invalid:
        raise ValueError(f"Unknown split names: {invalid}")

    train, val, test = reconstruct_canonical_splits(split_info=split_info, max_n=max_n)
    by_name = {"train": train, "val": val, "test": test}

    records: List[CanonicalRecord] = []
    for split_name in VALID_SPLIT_NAMES:
        if split_name not in requested:
            continue
        for prompt, target in by_name[split_name]:
            n, b, d = parse_prompt_fields(prompt)
            records.append(
                CanonicalRecord(
                    prompt=prompt,
                    target=target,
                    N=n,
                    B=b,
                    D=d,
                    answer_value=int(target[:2]),
                    split_name=split_name,
                )
            )
    return records


def summarize_split_info(
    split_info: Mapping[str, Any],
    max_n: int = 1000,
    num_bases: int = 29,
) -> Dict[str, Any]:
    """Build a JSON-friendly summary of checkpoint-stored split metadata."""
    mode = str(split_info["mode"])
    train, val, test = reconstruct_canonical_splits(split_info=split_info, max_n=max_n)
    summary: Dict[str, Any] = {
        "mode": mode,
        "sample_counts": {"train": len(train), "val": len(val), "test": len(test)},
    }

    if mode == "by_NB_intersection":
        val_ns = sorted(_as_set(split_info["val_Ns"]))
        test_ns = sorted(_as_set(split_info["test_Ns"]))
        val_bs = sorted(_as_set(split_info["val_Bs"]))
        test_bs = sorted(_as_set(split_info["test_Bs"]))
        summary["heldout_value_counts"] = {
            "val_Ns": len(val_ns),
            "test_Ns": len(test_ns),
            "val_Bs": len(val_bs),
            "test_Bs": len(test_bs),
        }
        summary["heldout_value_fractions"] = {
            "val_Ns": len(val_ns) / max_n,
            "test_Ns": len(test_ns) / max_n,
            "val_Bs": len(val_bs) / num_bases,
            "test_Bs": len(test_bs) / num_bases,
        }
        summary["heldout_values"] = {
            "val_Ns": val_ns,
            "test_Ns": test_ns,
            "val_Bs": val_bs,
            "test_Bs": test_bs,
        }
    return summary
