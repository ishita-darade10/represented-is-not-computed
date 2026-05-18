from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import DecoderOnlyTransformer
from tokenizer import Tokenizer


CHECKPOINT_NAME_RE = re.compile(
    r"^model_t_l(?P<num_layers>\d+)_m(?P<split_mode>.+)_p"
    r"(?P<train_all_permutations>True|False)_s(?P<seed>\d+)_"
    r"(?P<selection>best|last)\.pt$"
)


@dataclass(frozen=True)
class CheckpointSpec:
    """Configuration values encoded in a named checkpoint filename."""

    num_layers: int
    split_mode: str
    train_all_permutations: bool
    seed: int
    selection: str


@dataclass
class LoadedCheckpoint:
    """Fully loaded checkpoint bundle used by downstream analyses."""

    path: Path
    spec: CheckpointSpec
    payload: Mapping[str, Any]
    split_info: Mapping[str, Any]
    tokenizer: Tokenizer
    model: DecoderOnlyTransformer


def parse_checkpoint_name(checkpoint_path: str | Path) -> CheckpointSpec:
    """
    Parse the model variant encoded in a named checkpoint filename.

    The clean paper pipeline intentionally requires named checkpoints such as
    `model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt`. Generic aliases like
    `best.pt` are ambiguous because they do not state the architecture or split
    variant needed to recreate the original run.
    """
    path = Path(checkpoint_path)
    match = CHECKPOINT_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            "Checkpoint filename does not match the required named format: "
            "model_t_l{layers}_m{split_mode}_p{True|False}_s{seed}_{best|last}.pt. "
            f"Received: {path.name}"
        )
    groups = match.groupdict()
    return CheckpointSpec(
        num_layers=int(groups["num_layers"]),
        split_mode=str(groups["split_mode"]),
        train_all_permutations=(groups["train_all_permutations"] == "True"),
        seed=int(groups["seed"]),
        selection=str(groups["selection"]),
    )


def load_checkpoint_payload(checkpoint_path: str | Path, map_location: torch.device | str = "cpu") -> Dict[str, Any]:
    """Load and validate the checkpoint dictionary expected by the project."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint payload to be a dict, got {type(payload)!r}")
    if "model_state" not in payload:
        raise KeyError(f"Checkpoint is missing required key 'model_state': {path}")
    if "split_info" not in payload:
        raise KeyError(f"Checkpoint is missing required key 'split_info': {path}")
    return payload


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> LoadedCheckpoint:
    """
    Load a named checkpoint together with the exact model architecture it needs.

    Only `num_layers` currently varies across the named model family. The other
    architectural dimensions continue to use the project defaults from
    `config.py`, which matches how these checkpoints were trained.
    """
    path = Path(checkpoint_path)
    spec = parse_checkpoint_name(path)
    runtime_device = torch.device("cpu") if device is None else device
    payload = load_checkpoint_payload(path, map_location=runtime_device)

    tokenizer = Tokenizer()
    model = DecoderOnlyTransformer(
        vocab_size=len(tokenizer),
        num_layers=spec.num_layers,
    ).to(runtime_device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    return LoadedCheckpoint(
        path=path,
        spec=spec,
        payload=payload,
        split_info=payload["split_info"],
        tokenizer=tokenizer,
        model=model,
    )

