#!/usr/bin/env python3
"""Convert the fine-tuned HuggingFace model to MLX 4-bit format for offline inference."""

import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "mlx-yojana"
HF_MODEL = "Subh24ai/yojana-sahayak-qwen2.5-1.5b-merged"


def main():
    if OUTPUT_DIR.is_dir() and any(OUTPUT_DIR.iterdir()):
        print(f"MLX model already exists at {OUTPUT_DIR}")
        print("Delete it and re-run if you want to re-convert.")
        return

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("mlx-lm not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlx-lm"])

    print(f"Converting {HF_MODEL} -> {OUTPUT_DIR} (4-bit MLX)")
    print("   Downloads ~3 GB and takes 2-3 minutes...")

    subprocess.check_call([
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", HF_MODEL,
        "-q", "--q-bits", "4",
        "--output-dir", str(OUTPUT_DIR),
    ])

    print(f"Done. Model saved to {OUTPUT_DIR}")
    print("The pipeline will now use this model automatically (fully offline).")

    # Quick sanity check — load tokenizer only (fast, no GPU needed)
    try:
        from mlx_lm import load
        print("Verifying converted model...")
        load(str(OUTPUT_DIR))
        print("Verification passed.")
    except Exception as e:
        print(f"WARNING: Verification failed: {e}")
        print("The model files were saved but may be corrupt. Try re-running.")


if __name__ == "__main__":
    main()
