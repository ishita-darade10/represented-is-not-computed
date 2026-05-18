from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from data import BaseConversionDataset, collate_batch
from analysis.paper_pipeline.helpers.checkpoints import load_checkpoint_bundle
from analysis.paper_pipeline.helpers.splits import reconstruct_canonical_splits, summarize_split_info


@torch.no_grad()
def teacher_forced_eval(model, tokenizer, test_loader, device):
    model.eval()
    eos_id = tokenizer.eos_token_id

    total_loss = 0.0
    total_digit_correct = 0
    total_digit_total = 0
    total_exact = 0.0
    n_batches = 0

    for full_seq in tqdm(test_loader, desc="teacher-forced"):
        full_seq = full_seq.to(device, non_blocking=(device.type == "cuda"))
        logits = model(full_seq)

        # shift
        logits_s = logits[:, :-1, :]
        targets = full_seq[:, 1:]

        out_start = Config.input_len - 1
        out_end = out_start + Config.target_len

        logits_out = logits_s[:, out_start:out_end, :]
        targets_out = targets[:, out_start:out_end]

        loss = F.cross_entropy(
            logits_out.reshape(-1, logits_out.size(-1)),
            targets_out.reshape(-1),
        )
        total_loss += loss.item()

        pred_out = torch.argmax(logits_out, dim=-1)
        digit_mask = (targets_out != eos_id)
        total_digit_correct += ((pred_out == targets_out) & digit_mask).sum().item()
        total_digit_total += digit_mask.sum().item()

        total_exact += (pred_out == targets_out).all(dim=1).float().mean().item()
        n_batches += 1

    return (
        total_loss / max(1, n_batches),
        total_digit_correct / max(1, total_digit_total),
        total_exact / max(1, n_batches),
    )


@torch.no_grad()
def autoregressive_eval(model, tokenizer, test_samples, device, n_limit: int = 5000):
    model.eval()
    total_digit_correct = 0
    total_digit_total = 0
    exact = 0

    subset = test_samples[: min(n_limit, len(test_samples))]

    for inp, tgt in tqdm(subset, desc="autoregressive"):
        x = torch.tensor(tokenizer.encode(inp), device=device).unsqueeze(0)
        y = model.generate(x, max_new_tokens=Config.target_len)
        out = y[0, -Config.target_len :].tolist()
        out_str = tokenizer.decode(out)

        # digit acc excludes 'E'
        for j in range(Config.target_len):
            if tgt[j] != "E":
                total_digit_total += 1
                if out_str[j] == tgt[j]:
                    total_digit_correct += 1

        if out_str == tgt:
            exact += 1

    digit_acc = total_digit_correct / max(1, total_digit_total)
    exact_acc = exact / max(1, len(subset))
    return digit_acc, exact_acc, len(subset)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a named checkpoint on the exact split stored in that checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Named checkpoint path, e.g. "
            "checkpoints/model_t_l10_mby_NB_intersection_pTrue_s1337_best.pt"
        ),
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=1000,
        help="Maximum N (exclusive) used to reconstruct the canonical dataset.",
    )
    parser.add_argument(
        "--gen-limit",
        type=int,
        default=5000,
        help="Maximum number of test examples for autoregressive evaluation.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Device:", device)

    bundle = load_checkpoint_bundle(args.checkpoint, device=device)
    tokenizer = bundle.tokenizer
    model = bundle.model
    train_samples, val_samples, test_samples = reconstruct_canonical_splits(
        bundle.split_info,
        max_n=args.max_n,
    )
    split_summary = summarize_split_info(bundle.split_info, max_n=args.max_n)
    test_loader = DataLoader(
        BaseConversionDataset(tokenizer, test_samples),
        batch_size=Config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    print(
        "Loaded checkpoint:",
        bundle.path,
        f"(epoch={bundle.payload.get('epoch')}, val_digit_acc={bundle.payload.get('val_digit_acc')})",
    )
    print(
        "Checkpoint spec:",
        {
            "num_layers": bundle.spec.num_layers,
            "split_mode": bundle.spec.split_mode,
            "train_all_permutations": bundle.spec.train_all_permutations,
            "seed": bundle.spec.seed,
            "selection": bundle.spec.selection,
        },
    )
    print("Split summary:", split_summary["sample_counts"])

    tf_loss, tf_digit_acc, tf_exact = teacher_forced_eval(model, tokenizer, test_loader, device)
    print(f"[teacher] loss={tf_loss:.4f} digit_acc={tf_digit_acc:.4f} exact={tf_exact:.4f}")

    ar_digit_acc, ar_exact, n = autoregressive_eval(
        model,
        tokenizer,
        test_samples,
        device,
        n_limit=args.gen_limit,
    )
    print(f"[gen]     digit_acc={ar_digit_acc:.4f} exact={ar_exact:.4f} (n={n})")


if __name__ == "__main__":
    main()
