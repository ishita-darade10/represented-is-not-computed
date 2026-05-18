from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024
EXPECTED = [
    ROOT / "README.md",
    ROOT / "config.py",
    ROOT / "training" / "train.py",
    ROOT / "evaluation" / "test.py",
    ROOT / "checkpoints" / "CHECKSUMS.sha256",
    ROOT / "analysis" / "paper_pipeline" / "Methods.md",
    ROOT / "analysis" / "paper_pipeline" / "RESULTS.md",
    ROOT / "analysis" / "paper_pipeline" / "Scratch.md",
    ROOT / "analysis" / "paper_pipeline" / "figures" / "paper_figures" / "figure 1.pdf",
    ROOT / "analysis" / "paper_pipeline" / "figures" / "paper_figures" / "figure 2.pdf",
    ROOT / "analysis" / "paper_pipeline" / "figures" / "paper_figures" / "figure 3.pdf",
    ROOT / "analysis" / "paper_pipeline" / "figures" / "paper_figures" / "figure 4.pdf",
]


def main() -> None:
    missing = [path for path in EXPECTED if not path.exists()]
    oversized = [path for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size > MAX_FILE_BYTES]

    print(f"release root: {ROOT}")
    print(f"files: {sum(1 for p in ROOT.rglob('*') if p.is_file())}")
    print(f"total size: {sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) / (1024**2):.2f} MB")

    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(str(p.relative_to(ROOT)) for p in missing))
    if oversized:
        raise SystemExit("files above 25 MB:\n" + "\n".join(str(p.relative_to(ROOT)) for p in oversized))

    print("release check: OK")


if __name__ == "__main__":
    main()
