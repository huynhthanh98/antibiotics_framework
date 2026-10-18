#!/usr/bin/env python3
"""
Build COCO-format dataset from AZI/CIP/TE joblib datasets.

Output structure:
  coco/{split}/images/*.tiff
  coco/{split}/{split}.json              <- instance only (1 class "cell")
  coco/{split}/{split}_classified.json   <- 2 classes: resistant / sensitive

Filename encodes provenance:
  {DRUG}_{PHENOTYPE}_{stem}_x{x:04d}_y{y:04d}.tiff
"""
import json
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import tifffile

# ─── CONFIG ────────────────────────────────────────────────────────────────
ROOT        = Path("/home/aisia/new_mia")
DATASETS    = ["AZI", "CIP", "TE"]
SPLITS      = ["train", "val", "test"]
PHENOTYPES  = ["resistant", "sensitive"]

PATCH_SIZE  = 224
STRIDE      = 112
MIN_OBJECTS = 1     # min objects per patch (before area filter)
MIN_AREA    = 20    # min pixel area to keep an annotation
OUT_ROOT    = ROOT / "coco"

CATEGORIES_INSTANCE = [
    {"id": 1, "name": "cell", "supercategory": "cell"}
]
CATEGORIES_CLASSIFIED = [
    {"id": 1, "name": "resistant", "supercategory": "cell"},
    {"id": 2, "name": "sensitive",  "supercategory": "cell"},
]
PHENO_TO_CAT = {"resistant": 1, "sensitive": 2}
# ───────────────────────────────────────────────────────────────────────────


def patch_positions(H: int, W: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Return sorted unique (y, x) top-left corners covering full image."""
    ys = sorted(set(list(range(0, H - size, stride)) + [H - size]))
    xs = sorted(set(list(range(0, W - size, stride)) + [W - size]))
    return list(product(ys, xs))


def mask_to_annotations(
    patch_mask: np.ndarray,
    cat_id: int,
    img_id: int,
    ann_id: int,
    min_area: int,
) -> tuple[list[dict], int]:
    """Convert instance-label patch to COCO annotation list."""
    anns = []
    for label in np.unique(patch_mask):
        if label == 0:
            continue
        binary = (patch_mask == label).astype(np.uint8)
        area = int(binary.sum())
        if area < min_area:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segs = [c.reshape(-1).tolist() for c in contours if c.shape[0] >= 3]
        if not segs:
            continue

        ys, xs = np.where(binary)
        x0, y0 = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1

        anns.append({
            "id":           ann_id,
            "image_id":     img_id,
            "category_id":  cat_id,
            "segmentation": segs,
            "area":         area,
            "bbox":         [x0, y0, bw, bh],
            "iscrowd":      0,
        })
        ann_id += 1

    return anns, ann_id


def new_coco(description: str, categories: list) -> dict:
    return {
        "info":        {"description": description, "version": "1.0"},
        "licenses":    [],
        "categories":  categories,
        "images":      [],
        "annotations": [],
    }


def build_split(split: str) -> None:
    print(f"\n{'='*60}\n  Split: {split}\n{'='*60}")

    out_img_dir = OUT_ROOT / split / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    img_id   = 1
    aid_inst = 1
    aid_cls  = 1

    coco_inst = new_coco(f"HCS instance-only {split}", CATEGORIES_INSTANCE)
    coco_cls  = new_coco(f"HCS classified {split}",    CATEGORIES_CLASSIFIED)

    for drug, pheno in product(DATASETS, PHENOTYPES):
        orig_dir = ROOT / f"{drug}_joblib" / split / pheno / "original"
        mask_dir = ROOT / f"{drug}_joblib" / split / pheno / "original_masks"
        if not orig_dir.is_dir() or not mask_dir.is_dir():
            continue

        cat_cls = PHENO_TO_CAT[pheno]

        for orig_path in sorted(orig_dir.glob("*.tiff")):
            mask_path = mask_dir / orig_path.name
            if not mask_path.exists():
                print(f"  [WARN] no mask for {orig_path.name}")
                continue

            orig = tifffile.imread(orig_path)   # uint16, shape (H, W)
            mask = tifffile.imread(mask_path)   # uint16, instance labels
            H, W = orig.shape[:2]

            positions  = patch_positions(H, W, PATCH_SIZE, STRIDE)
            kept = 0

            for y, x in positions:
                p_orig = orig[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                p_mask = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                # fast pre-filter: count non-zero labels
                if (np.unique(p_mask) > 0).sum() < MIN_OBJECTS:
                    continue

                anns_inst, new_aid_inst = mask_to_annotations(
                    p_mask, cat_id=1,       img_id=img_id, ann_id=aid_inst, min_area=MIN_AREA)
                anns_cls, new_aid_cls = mask_to_annotations(
                    p_mask, cat_id=cat_cls, img_id=img_id, ann_id=aid_cls,  min_area=MIN_AREA)

                # Skip if all annotations filtered by area
                if not anns_inst:
                    continue

                fname = f"{drug}_{pheno}_{orig_path.stem}_x{x:04d}_y{y:04d}.tiff"
                tifffile.imwrite(out_img_dir / fname, p_orig)

                img_entry = {
                    "id":          img_id,
                    "file_name":   fname,
                    "width":       PATCH_SIZE,
                    "height":      PATCH_SIZE,
                    "drug":        drug,
                    "phenotype":   pheno,
                    "source_file": orig_path.name,
                    "patch_x":     x,
                    "patch_y":     y,
                }

                coco_inst["images"].append(img_entry)
                coco_inst["annotations"].extend(anns_inst)

                coco_cls["images"].append(img_entry)
                coco_cls["annotations"].extend(anns_cls)

                aid_inst = new_aid_inst
                aid_cls  = new_aid_cls
                img_id  += 1
                kept    += 1

            print(f"  {drug}/{pheno}/{orig_path.name}: {len(positions)} patches → {kept} kept")

    for suffix, coco in [("", coco_inst), ("_classified", coco_cls)]:
        out_json = OUT_ROOT / split / f"{split}{suffix}.json"
        with open(out_json, "w") as f:
            json.dump(coco, f)
        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        print(f"\n  Saved: {out_json}")
        print(f"         {n_img:,} images  |  {n_ann:,} annotations")


def main() -> None:
    t0 = time.time()
    print("Build COCO dataset")
    print(f"  Patch: {PATCH_SIZE}×{PATCH_SIZE}  Stride: {STRIDE}  Min objects: {MIN_OBJECTS}  Min area: {MIN_AREA}px")
    print(f"  Output: {OUT_ROOT}")

    for split in SPLITS:
        build_split(split)

    print(f"\n{'='*60}")
    print(f"  Finished in {(time.time() - t0) / 60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
