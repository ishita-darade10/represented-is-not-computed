from __future__ import annotations

"""
Render final paper-facing Figures 2 and 3 from the audited pipeline outputs.

Figure 1 and Figure 4 are hand-finished vector figures maintained directly in
`figures/paper_figures/`. This script regenerates the two
data-driven paper figures from saved CSV summaries without rerunning the
expensive analyses.
"""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PAPER_FIGURES = ROOT / "figures" / "paper_figures"


COLORS = {
    "N": "#2F6BFF",
    "B": "#169B62",
    "D": "#D97706",
    "O0": "#7C3AED",
    "O1": "#DB2777",
    "ink": "#1F2937",
    "muted": "#6B7280",
    "grid": "#D1D5DB",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "savefig.transparent": True,
        }
    )


def _panel_label(ax: plt.Axes, label: str, dx: float = -0.16, dy: float = 1.10) -> None:
    ax.text(
        dx,
        dy,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save(fig: plt.Figure, stem: str) -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / f"{stem}.svg", bbox_inches="tight")


def render_figure_2() -> None:
    rows = _read_csv(DATA / "02_linear_probing" / "main_10layer_ptrue" / "across_seed_summary.csv")

    targets = [
        ("BpowD", r"$B^D$"),
        ("NdivBpowD", r"$N / B^D$"),
        ("floorNdivBpowD", r"$\lfloor N / B^D \rfloor$"),
        ("floorNdivBpowD_modB", r"$\lfloor N / B^D \rfloor\ \mathrm{mod}\ B$"),
    ]
    stream_style = {
        "D_ones": dict(color=COLORS["D"], lw=2.5, alpha=1.0, label=r"$D_{\mathrm{ones}}$"),
        "O[0]": dict(color=COLORS["O0"], lw=2.2, alpha=1.0, label=r"$O[0]$"),
        "O[1]": dict(color=COLORS["O1"], lw=2.2, alpha=1.0, label=r"$O[1]$"),
        "N_ones": dict(color=COLORS["N"], lw=1.35, alpha=0.28, label=r"$N_{\mathrm{ones}}$"),
        "B_ones": dict(color=COLORS["B"], lw=1.35, alpha=0.28, label=r"$B_{\mathrm{ones}}$"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25), sharex=True, sharey=True)
    axes_flat = axes.reshape(-1)
    for idx, (ax, (target, title)) in enumerate(zip(axes_flat, targets)):
        for stream, style in stream_style.items():
            subset = [r for r in rows if r["target"] == target and r["stream"] == stream]
            subset.sort(key=lambda r: int(r["representation_index"]))
            xs = np.asarray([int(r["representation_index"]) for r in subset], dtype=float)
            ys = np.asarray([float(r["mean_cv_r2"]) for r in subset], dtype=float)
            lo = np.asarray([float(r["ci95_low_bootstrap_percentile"]) for r in subset], dtype=float)
            hi = np.asarray([float(r["ci95_high_bootstrap_percentile"]) for r in subset], dtype=float)
            ax.plot(xs, ys, **style)
            ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.13 if stream in {"D_ones", "O[0]", "O[1]"} else 0.025, linewidth=0)

        ax.axhline(0.0, color=COLORS["muted"], lw=0.8, ls="--", alpha=0.65)
        ax.set_title(title, pad=5)
        ax.set_ylim(-0.20, 1.03)
        ax.set_xlim(-0.12, 10.12)
        ax.set_xticks(np.arange(11))
        ax.set_xticklabels(["input"] + [f"L{i}" for i in range(10)])
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.7)
        _panel_label(ax, "ABCD"[idx], dx=-0.13, dy=1.14)

    axes[0, 0].set_ylabel(r"5-fold CV $R^2$")
    axes[1, 0].set_ylabel(r"5-fold CV $R^2$")
    axes[1, 0].set_xlabel("Representation")
    axes[1, 1].set_xlabel("Representation")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.93, bottom=0.16, wspace=0.16, hspace=0.28)
    _save(fig, "figure 2")
    plt.close(fig)


def _blocked_label(blocked_layers: str, sweep: str) -> str:
    if not blocked_layers:
        return "clean"
    pieces = [int(x) for x in blocked_layers.split(",")]
    if len(pieces) == 1:
        return f"L{pieces[0]}"
    if sweep == "forward":
        return f"L0-{pieces[-1]}"
    return f"L{pieces[0]}-9"


def render_figure_3() -> None:
    ablation = _read_csv(DATA / "03_dones_to_output_attention_ablation" / "main_10layer_ptrue_fulltest" / "across_seed_summary.csv")
    patching = _read_csv(
        DATA
        / "05_dones_information_content_patching"
        / "main_10layer_ptrue_test_matchedsources_clean_correct"
        / "across_seed_summary.csv"
    )

    fig = plt.figure(figsize=(7.0, 5.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.56, wspace=0.23)
    ax_fwd = fig.add_subplot(gs[0, 0])
    ax_rev = fig.add_subplot(gs[0, 1], sharey=ax_fwd)
    ax_patch = fig.add_subplot(gs[1, :])

    for ax, sweep, title in [
        (ax_fwd, "forward", "Forward sweep"),
        (ax_rev, "reverse", "Reverse sweep"),
    ]:
        subset = [r for r in ablation if r["sweep_order"] == sweep]
        subset.sort(key=lambda r: int(r["step_index"]))
        xs = np.arange(len(subset))
        ys = np.asarray([float(r["mean_exact_answer_accuracy"]) for r in subset])
        lo = np.asarray([float(r["ci95_low_bootstrap_percentile"]) for r in subset])
        hi = np.asarray([float(r["ci95_high_bootstrap_percentile"]) for r in subset])
        labels = [_blocked_label(r["blocked_layers"], sweep) for r in subset]
        ax.plot(xs, ys, color=COLORS["D"], lw=2.4, marker="o", ms=4.2)
        ax.fill_between(xs, lo, hi, color=COLORS["D"], alpha=0.16, linewidth=0)
        ax.set_title(title, pad=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=38, ha="right")
        ax.set_ylim(0.0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.58, linewidth=0.7)
        ax.set_xlabel("Masked $D_{ones} \\rightarrow O$ layers")
    ax_fwd.set_ylabel("Exact answer accuracy")
    plt.setp(ax_rev.get_yticklabels(), visible=False)

    lookup = {(r["condition"], r["metric"]): r for r in patching}
    conditions = ("vary_N", "vary_B", "vary_D")
    labels = (r"$N$ changed", r"$B$ changed", r"$D$ changed")
    source = np.asarray([float(lookup[(c, "source_exact_rate")]["mean"]) for c in conditions])
    donor = np.asarray([float(lookup[(c, "donor_exact_rate")]["mean"]) for c in conditions])
    source_lo = np.asarray([float(lookup[(c, "source_exact_rate")]["ci95_low_bootstrap_percentile"]) for c in conditions])
    source_hi = np.asarray([float(lookup[(c, "source_exact_rate")]["ci95_high_bootstrap_percentile"]) for c in conditions])
    donor_lo = np.asarray([float(lookup[(c, "donor_exact_rate")]["ci95_low_bootstrap_percentile"]) for c in conditions])
    donor_hi = np.asarray([float(lookup[(c, "donor_exact_rate")]["ci95_high_bootstrap_percentile"]) for c in conditions])

    x = np.arange(len(conditions))
    width = 0.31
    ax_patch.bar(
        x - width / 2,
        source + 0.01,
        width=width,
        bottom=-0.01,
        yerr=np.vstack([source - source_lo, source_hi - source]),
        capsize=3,
        color=COLORS["ink"],
        alpha=0.86,
        label="Matches source answer",
    )
    ax_patch.bar(
        x + width / 2,
        donor + 0.01,
        width=width,
        bottom=-0.01,
        yerr=np.vstack([donor - donor_lo, donor_hi - donor]),
        capsize=3,
        color=COLORS["D"],
        alpha=0.92,
        label="Matches donor answer",
    )
    ax_patch.set_xticks(x)
    ax_patch.set_xticklabels(labels)
    ax_patch.set_ylim(-0.01, 1.08)
    ax_patch.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_patch.set_ylabel("Exact answer accuracy")
    ax_patch.set_title(r"Full $D_{\mathrm{ones}} \rightarrow O$ K/V patching", pad=5)
    ax_patch.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    fig.text(0.03, 0.985, "A", fontsize=13, fontweight="bold", va="top")
    fig.text(0.5, 0.985, r"Cumulative $D_{\mathrm{ones}} \rightarrow O$ attention ablations", fontsize=10.2, ha="center", va="top")
    ax_patch.text(-0.09, 1.16, "B", transform=ax_patch.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    fig.subplots_adjust(left=0.105, right=0.995, top=0.90, bottom=0.16)
    _save(fig, "figure 3")
    plt.close(fig)


def main() -> None:
    _set_style()
    render_figure_2()
    render_figure_3()


if __name__ == "__main__":
    main()
