"""Train the final LDA and export the bundle used by the Streamlit app.

Usage (from the project root):
    python scripts/train_and_export.py --data data/raw/Dry_Bean_Dataset.xlsx
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fisher_lda import load_dry_bean                      # noqa: E402
from fisher_lda.app_utils import build_bundle, save_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/Dry_Bean_Dataset.xlsx")
    ap.add_argument("--out", default="app/model_bundle.joblib")
    ap.add_argument("--n-background", type=int, default=2500)
    args = ap.parse_args()

    X, y = load_dry_bean(args.data)
    keep = ~X.duplicated()                                  # same cleaning as Days 2-3
    X, y = X[keep].reset_index(drop=True), y[keep.values].to_numpy()
    bundle = build_bundle(X, y, n_background=args.n_background)
    save_bundle(bundle, args.out)
    size_kb = Path(args.out).stat().st_size / 1024
    print(f"Saved {args.out} ({size_kb:.0f} KB): {bundle['n_train']} beans, "
          f"{len(bundle['classes'])} classes, {len(bundle['feature_names'])} features")


if __name__ == "__main__":
    main()
