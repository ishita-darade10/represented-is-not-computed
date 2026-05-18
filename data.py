from __future__ import annotations

import random
from itertools import permutations
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from tokenizer import Tokenizer

Sample = Tuple[str, str]  # (input_seq, target_seq)

_ALL_ORDERS = list(permutations(["N", "B", "D"]))  # 6 permutations
_CANONICAL_ORDER = ("N", "B", "D")


class BaseConversionDataset(Dataset):
    """Returns full sequence tokens for inp+tgt (fixed length Config.max_seq_len)."""
    def __init__(self, tokenizer: Tokenizer, samples: List[Sample]):
        self.tokenizer = tokenizer
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        inp, tgt = self.samples[idx]
        tokens = self.tokenizer.encode(inp + tgt)
        return torch.tensor(tokens, dtype=torch.long)


def collate_batch(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


def number_to_base(n: int, base: int) -> List[str]:
    """Convert n to base; returns list of 2-char digits ('00'..'29')."""
    if n == 0:
        return ["00"]
    digits = []
    x = n
    while x > 0:
        rem = x % base
        digits.append(f"{rem:02d}")
        x //= base
    return digits[::-1]


# ---------- formatting / parsing (marker-based, order-agnostic) ----------

def _format_input(N: int, B: int, D: int, order: Tuple[str, str, str]) -> str:
    parts = {"N": f"N{N:03d}", "B": f"B{B:02d}", "D": f"D{D}"}
    return "".join(parts[k] for k in order) + "O"


def _parse_field(inp: str, tag: str, width: int) -> int:
    i = inp.find(tag)
    if i < 0:
        raise ValueError(f"Missing tag {tag} in input: {inp}")
    return int(inp[i + 1 : i + 1 + width])


def _parse_N(inp: str) -> int:
    return _parse_field(inp, "N", 3)


def _parse_B(inp: str) -> int:
    return _parse_field(inp, "B", 2)


def _parse_D(inp: str) -> int:
    i = inp.find("D")
    if i < 0:
        raise ValueError(f"Missing tag D in input: {inp}")
    return int(inp[i + 1])


# ---------- generation ----------

def generate_all_samples(max_n: int = 1000, order: Tuple[str, str, str] = _CANONICAL_ORDER) -> List[Sample]:
    samples: List[Sample] = []
    for N in range(0, max_n):
        for B in range(2, 31):
            base_digits = number_to_base(N, B)
            digit_length = len(base_digits)

            # D is 1 digit char => limit queries to 0..9
            max_D = min(digit_length, 9)

            for D in range(0, max_D + 1):
                inp = _format_input(N, B, D, order)
                pos_from_left = digit_length - 1 - D
                coef = base_digits[pos_from_left] if 0 <= pos_from_left < digit_length else "00"
                tgt = f"{coef}E"
                samples.append((inp, tgt))
    return samples


# ---------- splitting ----------

def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train_ratio+val_ratio+test_ratio must sum to 1.0, got {s}")


def split_samples(
    samples: List[Sample],
    mode: str = Config.split_mode,
    seed: int = Config.seed,
    train_ratio: float = Config.train_ratio,
    val_ratio: float = Config.val_ratio,
    test_ratio: float = Config.test_ratio,
) -> tuple[List[Sample], List[Sample], List[Sample], Dict[str, Any]]:
    """
    Modes:
      - by_N: hold out entire N values
      - by_base: hold out entire bases
      - by_NB: hold out (N,B) pairs
      - by_NB_intersection: test/val contain ONLY (N unseen) AND (B unseen) combos
    """
    rng = random.Random(seed)

    if mode == "by_NB_intersection":
        # pick heldout sets of N and B independently
        Ns = sorted({_parse_N(inp) for (inp, _tgt) in samples})
        Bs = sorted({_parse_B(inp) for (inp, _tgt) in samples})  # 2..30

        rng.shuffle(Ns)
        rng.shuffle(Bs)

        n_val = max(1, int(len(Ns) * Config.holdout_n_val))
        n_test = max(1, int(len(Ns) * Config.holdout_n_test))
        b_val = max(1, int(len(Bs) * Config.holdout_b_val))
        b_test = max(1, int(len(Bs) * Config.holdout_b_test))

        val_Ns = set(Ns[:n_val])
        test_Ns = set(Ns[n_val:n_val + n_test])

        val_Bs = set(Bs[:b_val])
        test_Bs = set(Bs[b_val:b_val + b_test])

        train, val, test = [], [], []

        # Intersection assignment (priority test > val > train)
        for inp, tgt in samples:
            n = _parse_N(inp)
            b = _parse_B(inp)

            if (n in test_Ns) and (b in test_Bs):
                test.append((inp, tgt))
            elif (n in val_Ns) and (b in val_Bs):
                val.append((inp, tgt))
            else:
                train.append((inp, tgt))

        info = {
            "mode": mode,
            "val_Ns": val_Ns, "test_Ns": test_Ns,
            "val_Bs": val_Bs, "test_Bs": test_Bs,
            "counts": {"train": len(train), "val": len(val), "test": len(test)},
            "notes": "intersection holdout: eval only when (N in heldout_Ns) AND (B in heldout_Bs). Everything else is train.",
        }
        return train, val, test, info

    # ---- classic modes ----
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    if mode == "by_base":
        by_b: Dict[int, List[Sample]] = {}
        for inp, tgt in samples:
            by_b.setdefault(_parse_B(inp), []).append((inp, tgt))

        bs = sorted(by_b.keys())
        rng.shuffle(bs)

        total = len(bs)
        t_end = int(total * train_ratio)
        v_end = t_end + int(total * val_ratio)
        te_end = v_end + int(total * test_ratio)

        train_bs = set(bs[:t_end])
        val_bs = set(bs[t_end:v_end])
        test_bs = set(bs[v_end:te_end])

        train, val, test = [], [], []
        for b, group in by_b.items():
            if b in train_bs: train.extend(group)
            elif b in val_bs: val.extend(group)
            elif b in test_bs: test.extend(group)

        info = {"mode": mode, "train_bs": train_bs, "val_bs": val_bs, "test_bs": test_bs}
        return train, val, test, info

    if mode == "by_N":
        by_n: Dict[int, List[Sample]] = {}
        for inp, tgt in samples:
            by_n.setdefault(_parse_N(inp), []).append((inp, tgt))

        ns = sorted(by_n.keys())
        rng.shuffle(ns)

        total = len(ns)
        t_end = int(total * train_ratio)
        v_end = t_end + int(total * val_ratio)
        te_end = v_end + int(total * test_ratio)

        train_ns = set(ns[:t_end])
        val_ns = set(ns[t_end:v_end])
        test_ns = set(ns[v_end:te_end])

        train, val, test = [], [], []
        for n, group in by_n.items():
            if n in train_ns: train.extend(group)
            elif n in val_ns: val.extend(group)
            elif n in test_ns: test.extend(group)

        info = {"mode": mode, "train_ns": train_ns, "val_ns": val_ns, "test_ns": test_ns}
        return train, val, test, info

    if mode == "by_NB":
        by_nb: Dict[tuple[int, int], List[Sample]] = {}
        for inp, tgt in samples:
            by_nb.setdefault((_parse_N(inp), _parse_B(inp)), []).append((inp, tgt))

        pairs = list(by_nb.keys())
        rng.shuffle(pairs)

        total = len(pairs)
        t_end = int(total * train_ratio)
        v_end = t_end + int(total * val_ratio)
        te_end = v_end + int(total * test_ratio)

        train_pairs = set(pairs[:t_end])
        val_pairs = set(pairs[t_end:v_end])
        test_pairs = set(pairs[v_end:te_end])

        train, val, test = [], [], []
        for pair, group in by_nb.items():
            if pair in train_pairs: train.extend(group)
            elif pair in val_pairs: val.extend(group)
            elif pair in test_pairs: test.extend(group)

        info = {"mode": mode, "train_pairs": train_pairs, "val_pairs": val_pairs, "test_pairs": test_pairs}
        return train, val, test, info

    raise ValueError(f"Unknown split mode: {mode}")


# ---------- loaders ----------

def _make_loader(ds: Dataset, device_type: str, shuffle: bool, drop_last: bool, batch_size: Optional[int] = None) -> DataLoader:
    bs = Config.batch_size if batch_size is None else batch_size

    if device_type == "mps":
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
    elif device_type == "cuda":
        num_workers = Config.num_workers
        pin_memory = Config.pin_memory
        persistent_workers = (num_workers > 0)
        prefetch_factor = 4 if num_workers > 0 else None
    else:
        num_workers = Config.num_workers
        pin_memory = False
        persistent_workers = (num_workers > 0)
        prefetch_factor = 4 if num_workers > 0 else None

    kw = dict(
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    if persistent_workers:
        kw["persistent_workers"] = True
    if prefetch_factor is not None:
        kw["prefetch_factor"] = prefetch_factor

    return DataLoader(ds, **kw)


def create_data_loaders(
    tokenizer: Tokenizer,
    device_type: str,
    split_mode: str = Config.split_mode,
    max_n: int = 1000,
):
    """
    Returns:
      train_loader, val_loader, test_loader, test_samples, split_info

    Train uses all-permutation expansion if enabled. Val/Test are canonical.
    """
    # Canonical dataset defines splits (semantics-only)
    all_canonical = generate_all_samples(max_n=max_n, order=_CANONICAL_ORDER)

    train_c, val_c, test_c, split_info = split_samples(
        all_canonical,
        mode=split_mode,
        seed=Config.seed,
        train_ratio=Config.train_ratio,
        val_ratio=Config.val_ratio,
        test_ratio=Config.test_ratio,
    )

    # Expand TRAIN into all 6 permutations (labels unchanged)
    if Config.train_all_permutations:
        train_s: List[Sample] = []
        for inp, tgt in train_c:
            N = _parse_N(inp)
            B = _parse_B(inp)
            D = _parse_D(inp)
            for order in _ALL_ORDERS:
                train_s.append((_format_input(N, B, D, order), tgt))
    else:
        train_s = train_c

    val_s = val_c  # canonical
    test_s = test_c  # canonical

    train_ds = BaseConversionDataset(tokenizer, train_s)
    val_ds = BaseConversionDataset(tokenizer, val_s)
    test_ds = BaseConversionDataset(tokenizer, test_s)

    train_loader = _make_loader(train_ds, device_type=device_type, shuffle=True, drop_last=True)
    val_loader = _make_loader(val_ds, device_type=device_type, shuffle=False, drop_last=False)
    test_loader = _make_loader(test_ds, device_type=device_type, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, test_s, split_info


def create_analysis_loader(
    tokenizer: Tokenizer,
    device_type: str,
    max_n: int = 1000,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
):
    """
    Canonical loader over the ENTIRE dataset (no split), for interpretability dumps.
    Returns: (analysis_loader, analysis_samples_canonical)
    """
    all_canonical = generate_all_samples(max_n=max_n, order=_CANONICAL_ORDER)
    ds = BaseConversionDataset(tokenizer, all_canonical)
    loader = _make_loader(ds, device_type=device_type, shuffle=shuffle, drop_last=False, batch_size=batch_size)
    return loader, all_canonical