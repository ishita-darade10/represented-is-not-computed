from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Fixed grammar lengths
    input_len: int = 10
    target_len: int = 3
    max_seq_len: int = 13

    # Split mode: "by_N" | "by_base" | "by_NB" | "by_NB_intersection"
    split_mode: str = "by_NB_intersection"

    # Model
    d_model: int = 384
    num_layers: int = 10
    num_heads: int = 12
    d_ff: int = 1536
    dropout: float = 0.01

    # Data (L40S-friendly)
    batch_size: int = 2048
    num_workers: int = 8
    pin_memory: bool = True

    # Classic split ratios (used by by_N/by_base/by_NB)
    train_ratio: float = 0.75
    val_ratio: float = 0.125
    test_ratio: float = 0.125

    # Intersection split fractions (used by by_NB_intersection)
    # These pick heldout sets of Ns and Bs; eval is the cartesian intersection (N in heldout_Ns AND B in heldout_Bs).
    holdout_n_val: float = 0.10
    holdout_n_test: float = 0.10
    holdout_b_val: float = 0.20
    holdout_b_test: float = 0.20

    # Train augmentation: expand each TRAIN sample into all 6 N/B/D permutations.
    # Val/Test remain canonical.
    train_all_permutations: bool = True

    # Repro / perf
    seed: int = 1337
    amp: bool = True
    compile: bool = False
    log_every: int = 100

    # Optim
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0

    # Schedule (train set is ~6× larger)
    epochs: int = 1000
    warmup_ratio: float = 0.05

    # Checkpoints
    ckpt_dir: str = "checkpoints"
    logs_dir: str = "logs"
    best_name: str = "best.pt"
    last_name: str = "last.pt"
