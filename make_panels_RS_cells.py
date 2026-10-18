#!/usr/bin/env python3
"""
Generate two figures from original_masks:

Panel A: stacked bar plot of number of images per drug, split by phenotype (R/S).
Panel B: violin + box plot of number of cells per image, grouped by drug and colored by phenotype.

Cells per image = count of unique instance labels (>0) in each mask TIFF.

Outputs (PNG):
  /home/aisia/new_mia/panelA_images_RS.png
  /home/aisia/new_mia/panelB_cells_per_image.png
  /home/aisia/new_mia/panels_AB.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from matplotlib import pyplot as plt


ROOT = Path("/home/aisia/new_mia")
DRUGS = ["CIP", "TE", "AZI"]
SPLITS = ["train", "val", "test"]
PHENOS = ["resistant", "sensitive"]

PHENO_SHORT = {"resistant": "R", "sensitive": "S"}
PALETTE = {"R": "#D55E00", "S": "#0072B2"}  # colorblind-friendly


@dataclass(frozen=True)
class Row:
    drug: str
    split: str
    phenotype: str
    cells: int


def iter_mask_paths(drug: str, split: str, phenotype: str) -> list[Path]:
    d = ROOT / f"{drug}_joblib" / split / phenotype / "original_masks"
    if not d.is_dir():
        return []
    return sorted(d.glob("*.tiff"))


def mask_cell_count(path: Path) -> int:
    m = tifffile.imread(path)
    return int((np.unique(m) > 0).sum())


def build_dataframe() -> pd.DataFrame:
    rows: list[Row] = []
    for drug in DRUGS:
        for split in SPLITS:
            for pheno in PHENOS:
                for p in iter_mask_paths(drug, split, pheno):
                    rows.append(Row(drug=drug, split=split, phenotype=PHENO_SHORT[pheno], cells=mask_cell_count(p)))
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        raise RuntimeError("No masks found. Check ROOT and dataset folders.")
    return df


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Count images per drug/phenotype across all splits
    counts = (
        df.groupby(["drug", "phenotype"])["cells"]
        .size()
        .reset_index(name="n_images")
        .pivot(index="drug", columns="phenotype", values="n_images")
        .reindex(DRUGS)
        .fillna(0)
        .astype(int)
    )

    x = np.arange(len(DRUGS))
    r = counts.get("R", pd.Series([0] * len(DRUGS), index=DRUGS)).to_numpy()
    s = counts.get("S", pd.Series([0] * len(DRUGS), index=DRUGS)).to_numpy()

    ax.bar(x, r, color=PALETTE["R"], label="Resistant (R)")
    ax.bar(x, s, bottom=r, color=PALETTE["S"], label="Sensitive (S)")

    ax.set_xticks(x, DRUGS)
    ax.set_ylabel("Number of images")
    ax.set_title("Number of images by drug (R/S)")
    ax.grid(axis="y", alpha=0.25)

    # Annotate totals on top
    totals = r + s
    ymax = int(totals.max()) if len(totals) else 0
    ax.set_ylim(0, max(1, int(ymax * 1.18)))
    for i, t in enumerate(totals):
        ax.text(i, t + max(1, ymax) * 0.02, f"{int(t)}", ha="center", va="bottom", fontsize=11)

    # Keep legend outside to avoid overlapping bars/labels
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    sns.violinplot(
        data=df,
        x="drug",
        y="cells",
        hue="phenotype",
        order=DRUGS,
        hue_order=["R", "S"],
        palette=PALETTE,
        cut=0,
        inner=None,
        linewidth=1.0,
        ax=ax,
        dodge=True,
    )
    sns.boxplot(
        data=df,
        x="drug",
        y="cells",
        hue="phenotype",
        order=DRUGS,
        hue_order=["R", "S"],
        palette=PALETTE,
        width=0.18,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "white", "alpha": 0.85},
        whiskerprops={"linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.2},
        ax=ax,
        dodge=True,
    )

    # Remove duplicate legend from overlayed plots
    handles, labels = ax.get_legend_handles_labels()
    # seaborn duplicates legends: first half violin, second half box
    if len(handles) >= 4:
        handles, labels = handles[:2], labels[:2]
    ax.legend(handles, ["Resistant (R)", "Sensitive (S)"], frameon=False, loc="upper right")

    ax.set_xlabel("")
    ax.set_ylabel("Number of cells per image")
    ax.set_title("Cells per image distribution")
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    df = build_dataframe()

    # Panel A (standalone)
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_panel_a(ax, df)
    fig.tight_layout()
    out_a = ROOT / "panelA_images_RS.png"
    fig.savefig(out_a, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Panel B (standalone)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_panel_b(ax, df)
    fig.tight_layout()
    out_b = ROOT / "panelB_cells_per_image.png"
    fig.savefig(out_b, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")  # keep output minimal
    print(out_a)
    print(out_b)


if __name__ == "__main__":
    main()

