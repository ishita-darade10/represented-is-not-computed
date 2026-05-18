from __future__ import annotations
import os
import time
import math
import random

import torch
import torch.nn as nn

from config import Config
from tokenizer import Tokenizer
from data import create_data_loaders, generate_all_samples, split_samples
from model import DecoderOnlyTransformer

import numpy as np
import json


def seed_everything(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Prefer CUDA, else MPS, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backend(device: torch.device):
    """
    Safe, realistic perf flags.
    """
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    AdamW with correct weight decay handling:
    - decay only weights of Linear/Embedding etc (ndim>=2)
    - no decay on biases and LayerNorm weights
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not name.endswith(".bias"):
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [
        {"params": decay, "weight_decay": Config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=Config.learning_rate,
        betas=Config.betas,
        eps=Config.eps,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    """
    Linear warmup then cosine decay.
    """
    warmup_steps = max(1, int(total_steps * Config.warmup_ratio))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, full_seq: torch.Tensor, eos_id: int):
    """
    - digit_acc: accuracy over 2 digits (exclude E)
    - exact_suffix_acc: exact match over the 3 output tokens (digit,digit,E)
    """
    # Next-token prediction
    pred = torch.argmax(logits[:, :-1, :], dim=-1)   # (B, T-1)
    tgt = full_seq[:, 1:]                            # (B, T-1)

    out_start = Config.input_len - 1
    out_end = out_start + Config.target_len  # includes E

    pred_out = pred[:, out_start:out_end]  # (B, 3)
    tgt_out = tgt[:, out_start:out_end]    # (B, 3)

    digit_mask = (tgt_out != eos_id)       # excludes E position
    digit_correct = ((pred_out == tgt_out) & digit_mask).sum().item()
    digit_total = digit_mask.sum().item()

    exact = (pred_out == tgt_out).all(dim=1).float().mean().item()
    return digit_correct, digit_total, exact


class Trainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, split_info, device: torch.device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.split_info = split_info

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = build_optimizer(self.model)

        total_steps = Config.epochs * len(self.train_loader)
        self.scheduler = build_scheduler(self.optimizer, total_steps)

        # AMP only on CUDA
        self.use_amp = (device.type == "cuda") and Config.amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # torch.compile only on CUDA (compile on MPS is still often flaky/slow)
        if Config.compile and device.type == "cuda":
            self.model = torch.compile(self.model)

    def _loss_on_output_only(self, logits: torch.Tensor, full_seq: torch.Tensor) -> torch.Tensor:
        """
        Next-token loss, computed only on the output window (digits + E).
        """
        logits_s = logits[:, :-1, :]      # (B, T-1, V)
        targets = full_seq[:, 1:]         # (B, T-1)

        out_start = Config.input_len - 1
        out_end = out_start + Config.target_len

        logits_out = logits_s[:, out_start:out_end, :]
        targets_out = targets[:, out_start:out_end]

        return self.criterion(
            logits_out.reshape(-1, logits_out.size(-1)),
            targets_out.reshape(-1),
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        total_loss = 0.0
        total_digit_correct = 0
        total_digit_total = 0
        total_exact = 0.0
        n_batches = 0

        for step, full_seq in enumerate(self.train_loader):
            full_seq = full_seq.to(self.device, non_blocking=(self.device.type == "cuda"))

            self.optimizer.zero_grad(set_to_none=True)

            # Autocast only on CUDA
            if self.use_amp:
                with torch.cuda.amp.autocast(True):
                    logits = self.model(full_seq)
                    loss = self._loss_on_output_only(logits, full_seq)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(full_seq)
                loss = self._loss_on_output_only(logits, full_seq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip_norm)
                self.optimizer.step()

            self.scheduler.step()

            dc, dt, ex = compute_metrics(logits, full_seq, self.tokenizer.eos_token_id)
            total_digit_correct += dc
            total_digit_total += dt
            total_exact += ex

            total_loss += loss.item()
            n_batches += 1

            if step % Config.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"[train] epoch={epoch} step={step}/{len(self.train_loader)} loss={loss.item():.4f} lr={lr:.2e}")

        avg_loss = total_loss / max(1, n_batches)
        digit_acc = total_digit_correct / max(1, total_digit_total)
        exact_acc = total_exact / max(1, n_batches)
        print(f"[train] epoch={epoch} time={time.time()-t0:.1f}s loss={avg_loss:.4f} digit_acc={digit_acc:.4f} exact={exact_acc:.4f}")
        return avg_loss, digit_acc, exact_acc

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        total_loss = 0.0
        total_digit_correct = 0
        total_digit_total = 0
        total_exact = 0.0
        n_batches = 0

        for full_seq in self.val_loader:
            full_seq = full_seq.to(self.device, non_blocking=(self.device.type == "cuda"))
            logits = self.model(full_seq)
            loss = self._loss_on_output_only(logits, full_seq)

            dc, dt, ex = compute_metrics(logits, full_seq, self.tokenizer.eos_token_id)
            total_digit_correct += dc
            total_digit_total += dt
            total_exact += ex

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        digit_acc = total_digit_correct / max(1, total_digit_total)
        exact_acc = total_exact / max(1, n_batches)
        print(f"[val]   epoch={epoch} time={time.time()-t0:.1f}s loss={avg_loss:.4f} digit_acc={digit_acc:.4f} exact={exact_acc:.4f}")
        return avg_loss, digit_acc, exact_acc

    def net_name_gen(self, epoch: int, best: bool = False) -> str:
        if not best:
            append_details = f"t_l{Config.num_layers}_m{Config.split_mode}_p{Config.train_all_permutations}_s{Config.seed}_last"
        else:
            append_details = f"t_l{Config.num_layers}_m{Config.split_mode}_p{Config.train_all_permutations}_s{Config.seed}_best"
        return f"model_{append_details}.pt"

    def update_logs(self, epoch: int, train_loss: float, train_digit: float, train_exact: float, val_loss: float, val_digit: float, val_exact: float):
        append_details = f"t_l{Config.num_layers}_m{Config.split_mode}_p{Config.train_all_permutations}_s{Config.seed}"
        log_path = os.path.join(Config.logs_dir, f"training_log_{append_details}.npy")
        fields_path = os.path.join(Config.logs_dir, f"training_log_{append_details}_fields.json")
        row = [epoch, train_loss, train_digit, train_exact, val_loss, val_digit, val_exact]
        fields = [
            "epoch",
            "train_loss",
            "train_digit",
            "train_exact",
            "val_loss",
            "val_digit",
            "val_exact",
        ]

        if os.path.exists(log_path):
            existing = np.load(log_path)
            data = np.vstack([existing, np.asarray(row, dtype=np.float64)])
        else:
            data = np.asarray([row], dtype=np.float64)

        np.save(log_path, data)
        if not os.path.exists(fields_path):
            with open(fields_path, "w", encoding="utf-8") as f:
                json.dump(fields, f, indent=2)
        return log_path

    def fit(self):
        os.makedirs(Config.ckpt_dir, exist_ok=True)
        best_path = os.path.join(Config.ckpt_dir, self.net_name_gen(0, best=True))
        last_path = os.path.join(Config.ckpt_dir, self.net_name_gen(0, best=False))

        best_val = -1.0

        for epoch in range(1, Config.epochs + 1):
            train_loss, train_digit, train_exact = self.train_epoch(epoch)
            val_loss, val_digit, val_exact = self.validate(epoch)

            updated_log_path = self.update_logs(epoch, train_loss, train_digit, train_exact, val_loss, val_digit, val_exact)
            print(f"[log] updated training log -> {updated_log_path}")

            # Save last
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_digit_acc": val_digit,
                    "split_info": self.split_info,
                },
                last_path,
            )

            # Save best (by digit acc)
            if val_digit > best_val:
                best_val = val_digit
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "val_digit_acc": val_digit,
                        "split_info": self.split_info,
                    },
                    best_path,
                )
                print(f"[ckpt] saved best -> {best_path} (val_digit_acc={best_val:.4f})")


def main():
    seed_everything(Config.seed)
    device = get_device()
    print("Device:", device)
    configure_backend(device)

    tokenizer = Tokenizer()
    train_loader, val_loader, test_loader, test_samples, split_info = create_data_loaders(
        tokenizer,
        device_type=device.type,
        split_mode=Config.split_mode,
    )
    print(f"Split mode: {split_info['mode']}")

    mode = split_info.get("mode", "unknown")

    if mode == "by_base":
        print(f"Bases seen: train={len(split_info['train_bs'])} val={len(split_info['val_bs'])} test={len(split_info['test_bs'])}")
    elif mode == "by_N":
        print(f"Ns seen: train={len(split_info['train_ns'])} val={len(split_info['val_ns'])} test={len(split_info['test_ns'])}")
    elif mode == "by_NB":
        print(f"(N,B) pairs: train={len(split_info['train_pairs'])} val={len(split_info['val_pairs'])} test={len(split_info['test_pairs'])}")
    elif mode == "by_NB_intersection":
        c = split_info.get("counts", {})
        print(f"Intersection split: train={c.get('train')} val={c.get('val')} test={c.get('test')}")
        print(f"Heldout Ns: val={len(split_info['val_Ns'])} test={len(split_info['test_Ns'])}")
        print(f"Heldout Bs: val={len(split_info['val_Bs'])} test={len(split_info['test_Bs'])}")
    else:
        print(f"Split mode: {mode} (keys={list(split_info.keys())})")
        
    print(f"Examples: train={len(train_loader.dataset):,} val={len(val_loader.dataset):,} test={len(test_loader.dataset):,}")

    if split_info.get("mode") == "by_NB_intersection":
        # rebuild canonical splits explicitly (independent of loaders)
        all_canonical = generate_all_samples(max_n=1000, order=("N", "B", "D"))
        train_c, val_c, test_c, split_info2 = split_samples(
            all_canonical,
            mode=Config.split_mode,
            seed=Config.seed,
            train_ratio=Config.train_ratio,
            val_ratio=Config.val_ratio,
            test_ratio=Config.test_ratio,
        )

        def _parse_field(inp: str, tag: str, width: int) -> int:
            i = inp.find(tag)
            return int(inp[i+1:i+1+width])

        def _parse_N(inp: str) -> int: return _parse_field(inp, "N", 3)
        def _parse_B(inp: str) -> int: return _parse_field(inp, "B", 2)

        val_Ns = split_info2["val_Ns"]; test_Ns = split_info2["test_Ns"]
        val_Bs = split_info2["val_Bs"]; test_Bs = split_info2["test_Bs"]

        def frac_intersection(samples, Ns, Bs):
            ok = 0
            for inp, _ in samples:
                n = _parse_N(inp); b = _parse_B(inp)
                ok += int((n in Ns) and (b in Bs))
            return ok / max(1, len(samples))

        print("[sanity] VAL fraction (N in val_Ns AND B in val_Bs):",
              frac_intersection(val_c, val_Ns, val_Bs))
        print("[sanity] TEST fraction (N in test_Ns AND B in test_Bs):",
              frac_intersection(test_c, test_Ns, test_Bs))
        print("[sanity] TRAIN leakage fraction for TEST intersection:",
              frac_intersection(train_c, test_Ns, test_Bs))

    model = DecoderOnlyTransformer(vocab_size=len(tokenizer))
    trainer = Trainer(model, tokenizer, train_loader, val_loader, split_info, device)
    trainer.fit()


if __name__ == "__main__":
    main()