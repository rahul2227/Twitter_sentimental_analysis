from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_datasets(train_path: str | os.PathLike, test_path: str | os.PathLike) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files.

    Expects columns: 'label' (0/1) and 'tweet' (str) in train.
    Test is expected to contain 'tweet'; 'label' is optional.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    missing_cols = {c for c in ['label', 'tweet'] if c not in train_df.columns}
    if missing_cols:
        raise ValueError(f"Train file missing required columns: {sorted(missing_cols)}")
    if 'tweet' not in test_df.columns:
        raise ValueError("Test file missing required column: 'tweet'")
    return train_df[['label', 'tweet']].copy(), test_df.copy()


def ensure_output_dirs(base_dir: str | os.PathLike) -> Path:
    """Create outputs directory structure and return base path."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / 'figures').mkdir(parents=True, exist_ok=True)
    (base / 'artifacts').mkdir(parents=True, exist_ok=True)
    (base / 'reports').mkdir(parents=True, exist_ok=True)
    return base


def save_eda_plots(train_df: pd.DataFrame, out_dir: str | os.PathLike) -> None:
    """Generate and save simple EDA plots with clear naming."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tmp = train_df.copy()
    tmp['length'] = tmp['tweet'].astype(str).map(len)

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x='label', y='length', data=tmp, hue='label', palette='PRGn', legend=False)
    ax.set_title('Average Tweet Length vs Label')
    ax.set_xlabel('Label')
    ax.set_ylabel('Average Length')
    plt.tight_layout()
    plt.savefig(out / 'barplot_label_length.png', dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x='label', data=tmp)
    ax.set_title('Label Counts')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2.0, height),
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out / 'label_counts.png', dpi=150)
    plt.close()
