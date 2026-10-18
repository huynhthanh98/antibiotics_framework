"""
Xuất toàn bộ patch Task 2 của CIP ra thư mục ảnh.

Cấu trúc output:
    output_dir/
        train/
            resistant/  patch_<imgname>_y<y>_x<x>.png
            sensitive/  ...
        val/
            resistant/  ...
            sensitive/  ...
        test/
            resistant/  ...
            sensitive/  ...
"""

import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

from config import Config


# ─────────────────────────────────────────────────────────────────────────────

DRUG       = 'CIP'
OUTPUT_DIR = f'/home/son/new_mia/patches_{DRUG}'
PATCH_SIZE = 224
STRIDE     = 224
STD_THR    = 1000.0   # chỉ dùng khi lọc train; val/test giữ tất cả


# ─────────────────────────────────────────────────────────────────────────────

def open_as_rgb(path: str) -> Image.Image:
    """Mở ảnh 16-bit hoặc 8-bit, trả về PIL RGB 8-bit."""
    img = Image.open(path)
    if img.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        arr    = np.array(img, dtype=np.float32)
        p_low  = float(np.percentile(arr, 1))
        p_high = float(np.percentile(arr, 99))
        if p_high > p_low:
            arr = np.clip(arr, p_low, p_high)
            arr = (arr - p_low) / (p_high - p_low) * 255.0
        else:
            arr = np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8), mode='L').convert('RGB')
    else:
        img = img.convert('RGB')
    return img


def export_split(df: pd.DataFrame, drug_dir: str,
                 folder: str, out_split_dir: str,
                 filter_background: bool = False):
    """
    Cắt và lưu patch cho một split.

    Parameters
    ----------
    df                : DataFrame của split đó (từ split_info.csv)
    drug_dir          : /path/to/CIP_joblib
    folder            : tên thư mục vật lý ('train' | 'val' | 'test')
    out_split_dir     : thư mục đích để lưu patch
    filter_background : True → bỏ patch std < STD_THR (chỉ dùng cho train)
    """

    # Chuẩn hoá tên cột
    if 'resistance_label' not in df.columns:
        df = df.copy()
        df['resistance_label'] = df['resistant'].astype(int)
    if 'original_file' not in df.columns:
        df = df.copy()
        df['original_file'] = df['original_tiff']

    for label_int, label_str in [(1, 'resistant'), (0, 'sensitive')]:
        os.makedirs(os.path.join(out_split_dir, label_str), exist_ok=True)

    grouped       = df.groupby('original_file')
    total_saved   = 0
    total_skipped = 0

    for orig_file, _ in tqdm(grouped, desc=f'  {folder}', unit='img'):
        label_int = int(df[df['original_file'] == orig_file]['resistance_label'].iloc[0])
        label_str = 'resistant' if label_int == 1 else 'sensitive'

        basename = os.path.basename(orig_file)
        img_path = os.path.join(drug_dir, folder, label_str, 'original', basename)

        # Thử tìm trong cả hai nhãn nếu không thấy
        if not os.path.exists(img_path):
            for lbl in ['resistant', 'sensitive']:
                p = os.path.join(drug_dir, folder, lbl, 'original', basename)
                if os.path.exists(p):
                    img_path = p
                    break
            else:
                tqdm.write(f'    [SKIP] không tìm thấy: {basename}')
                continue

        try:
            # Đọc raw để tính std (chỉ cần khi lọc train)
            raw = np.array(Image.open(img_path), dtype=np.float32)
            if raw.ndim == 3:
                raw = raw.mean(axis=2)
            h, w = raw.shape

            # Đọc RGB để cắt patch
            rgb = open_as_rgb(img_path)

            ys = list(range(0, h - PATCH_SIZE + 1, STRIDE))
            xs = list(range(0, w - PATCH_SIZE + 1, STRIDE))

            stem = os.path.splitext(basename)[0]   # e.g. r01c03f03

            if not ys or not xs:
                # Ảnh nhỏ hơn patch → resize
                patch = rgb.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
                fname = f'{stem}_full.png'
                patch.save(os.path.join(out_split_dir, label_str, fname))
                total_saved += 1
                continue

            for y in ys:
                for x in xs:
                    if filter_background:
                        if raw[y:y + PATCH_SIZE, x:x + PATCH_SIZE].std() < STD_THR:
                            total_skipped += 1
                            continue

                    patch = rgb.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                    fname = f'{stem}_y{y:04d}_x{x:04d}.png'
                    patch.save(os.path.join(out_split_dir, label_str, fname))
                    total_saved += 1

        except Exception as e:
            tqdm.write(f'    [ERROR] {basename}: {e}')

    print(f'  → saved {total_saved:,} patches'
          + (f', skipped {total_skipped:,} background' if filter_background else ''))
    return total_saved


# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg      = Config()
    drug_dir = cfg.drug_dir(DRUG)

    print(f'Drug       : {DRUG}')
    print(f'Output dir : {OUTPUT_DIR}')
    print(f'Patch size : {PATCH_SIZE}×{PATCH_SIZE}  stride={STRIDE}')
    print(f'Std filter : ≥ {STD_THR} (train only)')
    print()

    full_df  = pd.read_csv(cfg.split_csv(DRUG))
    train_df = full_df[full_df['split'] == 'train'].reset_index(drop=True)
    # theo trainer.py: CSV.test → folder test (val), CSV.val → folder val (test)
    val_df   = full_df[full_df['split'] == 'test'].reset_index(drop=True)
    test_df  = full_df[full_df['split'] == 'val' ].reset_index(drop=True)

    summary = {}

    print('── TRAIN ──────────────────────────────────────────────────')
    n = export_split(train_df, drug_dir,
                     folder='train',
                     out_split_dir=os.path.join(OUTPUT_DIR, 'train'),
                     filter_background=True)
    summary['train'] = n

    print('\n── VAL (CSV.test → folder test/) ──────────────────────────')
    n = export_split(val_df, drug_dir,
                     folder='test',
                     out_split_dir=os.path.join(OUTPUT_DIR, 'val'),
                     filter_background=False)
    summary['val'] = n

    print('\n── TEST (CSV.val → folder val/) ───────────────────────────')
    n = export_split(test_df, drug_dir,
                     folder='val',
                     out_split_dir=os.path.join(OUTPUT_DIR, 'test'),
                     filter_background=False)
    summary['test'] = n

    print(f'\n{"="*50}')
    print(f'DONE  — total patches exported for {DRUG}:')
    for split, n in summary.items():
        print(f'  {split:<6}: {n:,}')
    print(f'  {"TOTAL":<6}: {sum(summary.values()):,}')
    print(f'\nSaved to: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
