"""Apply the src.byzantine GaussianNoiseAttacker to a CSV dataset's feature columns.

The first column is treated as the label and left untouched; noise is added
to every other column.

Usage:
    uv run python scripts/add_byzantine_noise.py demo/dataset.csv demo/dataset_noisy.csv \
        --sigma 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.byzantine import GaussianNoiseAttacker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    label_col = df.columns[0]
    feature_cols = df.columns[1:]
    features = df[feature_cols].to_numpy(dtype=np.float32)

    attacker = GaussianNoiseAttacker(sigma=args.sigma, seed=args.seed)
    (noisy_features,) = attacker.apply_to_weights([features])

    noisy_df = pd.DataFrame(noisy_features, columns=feature_cols)
    noisy_df.insert(0, label_col, df[label_col].values)
    noisy_df.to_csv(args.output_csv, index=False)
    print(f"wrote {len(noisy_df)} noisy rows (sigma={args.sigma}) -> {args.output_csv}")


if __name__ == "__main__":
    main()
