#!/usr/bin/env python3
"""
Re-split a *_joblib_cells folder to 60/20/20 (train/val/test) at the image level.

Strategy: stratified split by 'resistant' label on unique original images.
Files in cell/, mask/, original/, original_masks/ are moved to match new split.
split_info.csv is rewritten with the new 'split' column.

Usage:
    python resplit_joblib_cells.py --drug TE --ratio 0.6 0.2 0.2 --seed 42
    python resplit_joblib_cells.py --drug TE --dry_run      # preview only
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path("/home/aisia/new_mia")
SUBDIRS = ["cell", "mask", "original", "original_masks"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--drug", required=True, choices=["AZI", "CIP", "TE"])
    p.add_argument("--suffix", default="_joblib_cells")
    p.add_argument("--ratio", nargs=3, type=float, default=[0.6, 0.2, 0.2],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="Preview changes without moving files")
    return p.parse_args()


def stratified_split(images_df: pd.DataFrame, ratio_train, ratio_val, seed):
    """
    Stratified split of image-level DataFrame by 'resistant' column.
    Returns (train_df, val_df, test_df).
    """
    ratio_test = 1.0 - ratio_train - ratio_val
    assert abs(ratio_train + ratio_val + ratio_test - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        images_df,
        test_size=(ratio_val + ratio_test),
        stratify=images_df["resistant"],
        random_state=seed,
    )
    # split temp into val and test (relative ratio)
    val_frac = ratio_val / (ratio_val + ratio_test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_frac),
        stratify=temp_df["resistant"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def image_stem(original_tiff_csv: str) -> str:
    """'Disk_1/Original_Images/r03c03f03.tiff' → 'r03c03f03.tiff'"""
    return Path(original_tiff_csv).name


def move_image(drug_root: Path, img_stem: str, label_str: str,
               old_split: str, new_split: str, dry_run: bool):
    """Move all files associated with one original image to a new split directory."""
    moves = []
    for sub in SUBDIRS:
        old_dir = drug_root / old_split / label_str / sub
        new_dir = drug_root / new_split / label_str / sub

        if sub == "original" or sub == "original_masks":
            # Single file per original image
            src = old_dir / img_stem
            if src.exists():
                moves.append((src, new_dir / img_stem))
        else:
            # cell/ and mask/: multiple files per original image
            # cell file prefix matches image_name (without extension) in the CSV,
            # which equals img_stem without extension
            stem_no_ext = Path(img_stem).stem   # e.g. "r03c03f03"
            if old_dir.exists():
                for f in old_dir.iterdir():
                    # Cell files: CM{iso}_{image_name}_{idx}.tiff
                    # image_name column = "r03c03f03d01" (field within image)
                    # We match on the original TIFF stem part (e.g. "r03c03f03")
                    # as a substring of the file stem before the "d" suffix
                    # e.g. CM3585_r03c03f03d01_00000.tiff → strip iso prefix → r03c03f03d01 → starts with r03c03f03
                    fname_stem = f.stem  # "CM3585_r03c03f03d01_00000"
                    parts = fname_stem.split("_", 1)
                    if len(parts) == 2 and parts[1].startswith(stem_no_ext):
                        moves.append((f, new_dir / f.name))

    for src, dst in moves:
        if dry_run:
            print(f"  [DRY] {src.relative_to(drug_root)}  →  {dst.relative_to(drug_root)}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    return len(moves)


def main():
    args = parse_args()
    ratio_train, ratio_val, ratio_test = args.ratio
    assert abs(ratio_train + ratio_val + ratio_test - 1.0) < 1e-6, \
        "Tỷ lệ phải cộng bằng 1.0"

    drug_root = ROOT / f"{args.drug}{args.suffix}"
    csv_path = drug_root / "split_info.csv"
    backup_path = drug_root / "split_info_backup.csv"

    print(f"\n{'='*60}")
    print(f"  Re-split: {drug_root.name}")
    print(f"  Tỷ lệ  : train={ratio_train:.0%} / val={ratio_val:.0%} / test={ratio_test:.0%}")
    print(f"  Seed    : {args.seed}")
    print(f"  Dry run : {args.dry_run}")
    print(f"{'='*60}\n")

    df = pd.read_csv(csv_path)

    # ── Image-level DataFrame ──────────────────────────────────────────────────
    img_df = (
        df.drop_duplicates(subset=["original_tiff"])
        [["original_tiff", "isolate", "resistant", "split"]]
        .copy()
        .reset_index(drop=True)
    )

    print(f"Tổng ảnh gốc: {len(img_df)} (R={( img_df['resistant']==1).sum()}, S={(img_df['resistant']==0).sum()})")
    print("\n--- Phân bố HIỆN TẠI ---")
    for sp in ["train", "val", "test"]:
        sub = img_df[img_df["split"] == sp]
        print(f"  {sp:5s}: {len(sub):3d} ảnh  (R={( sub['resistant']==1).sum():2d}, S={(sub['resistant']==0).sum():2d})")

    # ── Compute new split ──────────────────────────────────────────────────────
    train_imgs, val_imgs, test_imgs = stratified_split(
        img_df, ratio_train, ratio_val, args.seed
    )

    print("\n--- Phân bố SAU KHI chia lại ---")
    split_map = {}
    for sp, sub in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        print(f"  {sp:5s}: {len(sub):3d} ảnh  (R={( sub['resistant']==1).sum():2d}, S={(sub['resistant']==0).sum():2d})")
        for tiff in sub["original_tiff"]:
            split_map[tiff] = sp

    # ── Identify moves needed ─────────────────────────────────────────────────
    changes = img_df[img_df["original_tiff"].map(split_map) != img_df["split"]].copy()
    changes["new_split"] = changes["original_tiff"].map(split_map)
    print(f"\nẢnh cần di chuyển: {len(changes)}")
    for _, row in changes.iterrows():
        label_str = "resistant" if row["resistant"] == 1 else "sensitive"
        print(f"  {row['original_tiff'].split('/')[-1]}  {row['split']} → {row['new_split']}  [{label_str}]")

    if not args.dry_run:
        # ── Backup CSV ────────────────────────────────────────────────────────
        shutil.copy2(csv_path, backup_path)
        print(f"\n[OK] CSV backed up → {backup_path.name}")

    # ── Move files ────────────────────────────────────────────────────────────
    print("\nDi chuyển file...")
    total_moved = 0
    for _, row in changes.iterrows():
        label_str = "resistant" if row["resistant"] == 1 else "sensitive"
        n = move_image(
            drug_root,
            img_stem=image_stem(row["original_tiff"]),
            label_str=label_str,
            old_split=row["split"],
            new_split=row["new_split"],
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(f"  Moved {n:4d} files: {image_stem(row['original_tiff'])}  ({row['split']} → {row['new_split']})")
        total_moved += n

    print(f"\nTổng file {'sẽ' if args.dry_run else 'đã'} di chuyển: {total_moved}")

    # ── Update CSV ────────────────────────────────────────────────────────────
    if not args.dry_run:
        df["split"] = df["original_tiff"].map(split_map)
        df.to_csv(csv_path, index=False)
        print(f"[OK] split_info.csv đã cập nhật\n")

        # Verify
        df2 = pd.read_csv(csv_path)
        print("--- Xác nhận phân bố cells ---")
        print(df2.groupby(["split", "resistant"]).size().unstack(fill_value=0))
        img2 = df2.drop_duplicates(subset=["original_tiff"])
        print("\n--- Xác nhận phân bố ảnh gốc ---")
        for sp in ["train", "val", "test"]:
            sub = img2[img2["split"] == sp]
            print(f"  {sp:5s}: {len(sub):3d} ảnh  (R={( sub['resistant']==1).sum():2d}, S={(sub['resistant']==0).sum():2d})")
    else:
        print("\n[DRY RUN] Không có file nào bị thay đổi.")


if __name__ == "__main__":
    main()
