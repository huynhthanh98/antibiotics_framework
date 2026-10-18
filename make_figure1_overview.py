#!/usr/bin/env python3
"""Figure 1. Overview of the high-content image dataset.

(A) Number of images per antibiotic treatment, stratified by resistant (R) and sensitive (S).
(B) Distribution of cell counts per image across treatments and phenotypes (violin + embedded boxplot).
(C) Representative raw fluorescence images and corresponding instance masks for R/S under each treatment.

Representative selection: per (drug, phenotype), pick the image whose cell-count is closest to the median.
Cells per image = number of unique instance labels (>0) in original_masks.

Outputs:
  /home/aisia/new_mia/Figure1_overview.png
  /home/aisia/new_mia/Figure1_panelA.png
  /home/aisia/new_mia/Figure1_panelB.png
  /home/aisia/new_mia/Figure1_panelC.png

Scale bar:
  If pixel size metadata is unavailable, set MICRONS_PER_PIXEL manually.
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

# If None, we draw a 10µm bar with a default pixel length and warn in stdout.
MICRONS_PER_PIXEL: float | None = None
FALLBACK_SCALEBAR_PX = 100

# Output resolution (higher = sharper when zooming)
OUT_DPI = 600


@dataclass(frozen=True)
class Sample:
    drug: str
    split: str
    phenotype: str  # R/S
    orig_path: Path
    mask_path: Path
    cells: int


def iter_pairs(drug: str, split: str, pheno_long: str) -> list[tuple[Path, Path]]:
    orig_dir = ROOT / f"{drug}_joblib" / split / pheno_long / "original"
    mask_dir = ROOT / f"{drug}_joblib" / split / pheno_long / "original_masks"
    if not orig_dir.is_dir() or not mask_dir.is_dir():
        return []
    pairs = []
    for op in sorted(orig_dir.glob("*.tiff")):
        mp = mask_dir / op.name
        if mp.exists():
            pairs.append((op, mp))
    return pairs


def cell_count(mask_path: Path) -> int:
    m = tifffile.imread(mask_path)
    return int((np.unique(m) > 0).sum())


def build_df_and_samples() -> tuple[pd.DataFrame, list[Sample]]:
    rows = []
    samples = []
    for drug in DRUGS:
        for split in SPLITS:
            for pheno_long in PHENOS:
                ph = PHENO_SHORT[pheno_long]
                for op, mp in iter_pairs(drug, split, pheno_long):
                    c = cell_count(mp)
                    rows.append({"drug": drug, "split": split, "phenotype": ph, "cells": c, "orig": str(op), "mask": str(mp)})
                    samples.append(Sample(drug, split, ph, op, mp, c))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data found under *_joblib/**/original_masks")
    return df, samples


def draw_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
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
    ax.set_title("(A) Images per antibiotic treatment (R/S)")
    ax.grid(axis="y", alpha=0.25)

    totals = r + s
    ymax = int(totals.max()) if len(totals) else 0
    ax.set_ylim(0, max(1, int(ymax * 1.18)))
    for i, t in enumerate(totals):
        ax.text(i, t + max(1, ymax) * 0.02, f"{int(t)}", ha="center", va="bottom", fontsize=11)

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def draw_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
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

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 4:
        handles = handles[:2]
    ax.legend(handles, ["Resistant (R)", "Sensitive (S)"], frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax.set_xlabel("")
    ax.set_ylabel("Number of cells per image")
    ax.set_title("(B) Cell counts per image distribution")
    ax.grid(axis="y", alpha=0.25)


def normalize_uint16_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def colorize_instance_mask(mask: np.ndarray) -> np.ndarray:
    # Map each label to a pseudo-random color, background black.
    rng = np.random.default_rng(0)
    labels = np.unique(mask)
    colors = {0: np.array([0, 0, 0], dtype=np.uint8)}
    for lab in labels:
        if lab == 0:
            continue
        colors[int(lab)] = rng.integers(40, 255, size=3, dtype=np.uint8)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, col in colors.items():
        if lab == 0:
            continue
        out[mask == lab] = col
    return out


def infer_microns_per_pixel(tiff_path: Path) -> float | None:
    # Best-effort: many files won't have this; return None if unknown.
    try:
        with tifffile.TiffFile(tiff_path) as t:
            page = t.pages[0]
            tags = page.tags
            if "XResolution" in tags and "ResolutionUnit" in tags:
                xres = tags["XResolution"].value
                unit = tags["ResolutionUnit"].value
                # xres can be (num, den) or float. Interpret as pixels per unit.
                if isinstance(xres, tuple) and len(xres) == 2:
                    xres = xres[0] / xres[1]
                if unit == 3:  # CENTIMETER
                    # xres = pixels / cm => microns per pixel = 1e4 / xres
                    return 1e4 / float(xres)
                if unit == 2:  # INCH
                    return 25400.0 / float(xres)
    except Exception:
        return None
    return None


def add_scale_bar(ax: plt.Axes, img_shape: tuple[int, int], ref_path: Path) -> None:
    h, w = img_shape
    mpp = MICRONS_PER_PIXEL
    if mpp is None:
        mpp = infer_microns_per_pixel(ref_path)

    if mpp is None:
        bar_px = FALLBACK_SCALEBAR_PX
        note = "(uncalibrated)"
    else:
        bar_px = int(round(10.0 / mpp))
        note = ""

    bar_px = max(10, min(bar_px, w // 3))

    # Place bottom-left with padding
    pad = max(6, int(0.03 * w))
    x0 = pad
    y0 = h - pad
    ax.plot([x0, x0 + bar_px], [y0, y0], color="white", linewidth=4, solid_capstyle="butt")
    ax.text(x0, y0 - pad * 0.6, f"10 µm{(' ' + note) if note else ''}", color="white", fontsize=10, ha="left", va="top")


def pick_representatives(df: pd.DataFrame) -> dict[tuple[str, str], tuple[Path, Path, int]]:
    reps = {}
    for drug in DRUGS:
        for ph in ["R", "S"]:
            sub = df[(df["drug"] == drug) & (df["phenotype"] == ph)]
            if sub.empty:
                continue
            med = float(np.median(sub["cells"].to_numpy()))
            idx = (sub["cells"] - med).abs().idxmin()
            row = sub.loc[idx]
            reps[(drug, ph)] = (Path(row["orig"]), Path(row["mask"]), int(row["cells"]))
    return reps


def draw_panel_c(fig: plt.Figure, spec, df: pd.DataFrame, *, add_panel_tag: bool = True) -> None:
    reps = pick_representatives(df)

    # Layout: rows = drugs, cols = phenotype x (raw, mask)
    n_rows = len(DRUGS)
    phenos = ["R", "S"]
    n_cols = len(phenos) * 2

    gs = spec.subgridspec(n_rows, n_cols, wspace=0.03, hspace=0.12)

    for r, drug in enumerate(DRUGS):
        for c_ph, ph in enumerate(phenos):
            key = (drug, ph)
            if key not in reps:
                continue
            op, mp, cells = reps[key]
            raw = tifffile.imread(op)
            mask = tifffile.imread(mp)

            raw8 = normalize_uint16_to_uint8(raw)
            mask_rgb = colorize_instance_mask(mask)

            ax_raw = fig.add_subplot(gs[r, c_ph * 2])
            ax_msk = fig.add_subplot(gs[r, c_ph * 2 + 1])

            ax_raw.imshow(raw8, cmap="gray")
            ax_msk.imshow(mask_rgb)

            for a in (ax_raw, ax_msk):
                a.set_xticks([])
                a.set_yticks([])
                a.set_facecolor("black")

            # Labels
            if r == 0:
                ax_raw.set_title(f"{ph} raw", fontsize=12)
                ax_msk.set_title(f"{ph} mask", fontsize=12)
            if c_ph == 0:
                ax_raw.set_ylabel(drug, rotation=0, labelpad=30, va="center", fontsize=12)

            # Add scale bar only on first raw image to avoid clutter
            if r == 0 and c_ph == 0:
                add_scale_bar(ax_raw, raw8.shape[:2], op)

            # tiny cell-count annotation
            ax_raw.text(0.98, 0.02, f"n={cells}", transform=ax_raw.transAxes, ha="right", va="bottom", color="white", fontsize=10)

    if add_panel_tag:
        fig.text(0.02, 0.95, "(C)", fontsize=14, weight="bold")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    df, _ = build_df_and_samples()

    # Standalone panels A/B/C
    figA, axA = plt.subplots(figsize=(12, 6))
    draw_panel_a(axA, df)
    figA.tight_layout()
    outA = ROOT / "Figure1_panelA.png"
    figA.savefig(outA, dpi=OUT_DPI, bbox_inches="tight")
    plt.close(figA)

    figB, axB = plt.subplots(figsize=(11, 6))
    draw_panel_b(axB, df)
    figB.tight_layout()
    outB = ROOT / "Figure1_panelB.png"
    figB.savefig(outB, dpi=OUT_DPI, bbox_inches="tight")
    plt.close(figB)

    figC = plt.figure(figsize=(16, 10))
    figC.suptitle("Representative raw images and instance masks", fontsize=16, y=0.98)
    specC = figC.add_gridspec(1, 1, left=0.06, right=0.98, bottom=0.06, top=0.93)[0, 0]
    draw_panel_c(figC, specC, df, add_panel_tag=True)
    outC = ROOT / "Figure1_panelC.png"
    figC.savefig(outC, dpi=OUT_DPI, bbox_inches="tight")
    plt.close(figC)

    # Full Figure 1 layout
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], width_ratios=[1, 1.2], hspace=0.25, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    draw_panel_a(ax1, df)
    draw_panel_b(ax2, df)

    # Put explicit (A)/(B) tags
    ax1.set_title("(A) Number of images per antibiotic (R/S)")
    ax2.set_title("(B) Cell counts per image distribution")

    # Panel C uses a dedicated subgridspec to avoid any axes-overdraw artifacts.
    specC = gs[1, :]
    draw_panel_c(fig, specC, df, add_panel_tag=False)
    fig.text(0.06, 0.49, "(C) Representative raw images and instance masks (scale bar = 10 µm)", fontsize=14, weight="bold")

    out = ROOT / "Figure1_overview.png"
    fig.savefig(out, dpi=OUT_DPI, bbox_inches="tight")
    plt.close(fig)

    if MICRONS_PER_PIXEL is None and infer_microns_per_pixel(next(iter(df["orig"].map(Path)))) is None:
        print("[WARN] Pixel size metadata not found. Scale bar is uncalibrated.")
        print("       Set MICRONS_PER_PIXEL in make_figure1_overview.py for a true 10 µm bar.")

    print("Saved:")
    print(out)
    print(outA)
    print(outB)
    print(outC)


if __name__ == "__main__":
    main()
