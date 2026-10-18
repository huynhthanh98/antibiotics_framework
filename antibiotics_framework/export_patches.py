"""
Xuất toàn bộ patch Task 2 cho một hoặc nhiều drug ra thư mục ảnh.

Cấu trúc output:
    {data_root}/task2_patches/{DRUG}/
        train/
            resistant/  patch_<imgname>_y<y>_x<x>.png
            sensitive/  ...
        val/
            resistant/  ...
            sensitive/  ...
        test/
            resistant/  ...
            sensitive/  ...

Usage:
    # Export patches cho CIP
    python export_patches.py --drug CIP

    # Export patches cho tất cả drugs
    python export_patches.py --drug all

    # Với tham số tuỳ chỉnh
    python export_patches.py --drug AZI --patch_size 224 --stride 224 --std_threshold 1000

    # Train không lọc patch theo std (giữ mọi patch như val/test)
    python export_patches.py --drug all --no_train_filter
"""

import os
import argparse
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from config import Config


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


def export_split(
    df: pd.DataFrame,
    drug_dir: str,
    folder: str,
    out_split_dir: str,
    patch_size: int = 224,
    stride: int = 224,
    std_threshold: float = 1000.0,
    filter_background: bool = False,
) -> int:
    """
    Cắt và lưu patch cho một split.

    Parameters
    ----------
    df                : DataFrame của split đó (từ split_info.csv)
    drug_dir          : /path/to/DRUG_joblib
    folder            : tên thư mục vật lý ('train' | 'val' | 'test')
    out_split_dir     : thư mục đích để lưu patch
    patch_size        : kích thước patch
    stride            : bước nhảy
    std_threshold     : ngưỡng std để lọc background (chỉ dùng khi filter_background=True)
    filter_background : True → bỏ patch có std < std_threshold
    """

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
                continue

        try:
            raw = np.array(Image.open(img_path), dtype=np.float32)
            if raw.ndim == 3:
                raw = raw.mean(axis=2)
            h, w = raw.shape

            rgb = open_as_rgb(img_path)

            ys = list(range(0, h - patch_size + 1, stride))
            xs = list(range(0, w - patch_size + 1, stride))

            stem = os.path.splitext(basename)[0]

            if not ys or not xs:
                patch = rgb.resize((patch_size, patch_size), Image.BILINEAR)
                fname = f'{stem}_full.png'
                patch.save(os.path.join(out_split_dir, label_str, fname))
                total_saved += 1
                continue

            for y in ys:
                for x in xs:
                    if filter_background:
                        if raw[y:y + patch_size, x:x + patch_size].std() < std_threshold:
                            total_skipped += 1
                            continue

                    patch = rgb.crop((x, y, x + patch_size, y + patch_size))
                    fname = f'{stem}_y{y:04d}_x{x:04d}.png'
                    patch.save(os.path.join(out_split_dir, label_str, fname))
                    total_saved += 1

        except Exception as e:
            tqdm.write(f'    [ERROR] {basename}: {e}')

    bg_info = f', skipped {total_skipped:,} background' if filter_background else ''
    print(f'  → saved {total_saved:,} patches{bg_info}')
    return total_saved


def export_drug(
    drug: str,
    cfg: Config,
    output_base: str,
    patch_size: int = 224,
    stride: int = 224,
    std_threshold: float = 1000.0,
    filter_train_background: bool = True,
) -> dict:
    """Export patches cho một drug."""
    
    drug_dir   = cfg.drug_dir(drug)
    output_dir = os.path.join(output_base, drug)

    print(f'\n{"="*60}')
    print(f'  Drug       : {drug}')
    print(f'  Input dir  : {drug_dir}')
    print(f'  Output dir : {output_dir}')
    print(f'  Patch size : {patch_size}×{patch_size}  stride={stride}')
    if filter_train_background:
        print(f'  Std filter : ≥ {std_threshold} (train only)')
    else:
        print(f'  Std filter : tắt trên train (giữ mọi patch)')
    print(f'{"="*60}\n')

    csv_path = cfg.split_csv(drug)
    if not os.path.exists(csv_path):
        print(f'  [ERROR] Không tìm thấy: {csv_path}')
        return {'train': 0, 'val': 0, 'test': 0}

    full_df  = pd.read_csv(csv_path)
    train_df = full_df[full_df['split'] == 'train'].reset_index(drop=True)
    val_df   = full_df[full_df['split'] == 'test'].reset_index(drop=True)
    test_df  = full_df[full_df['split'] == 'val'].reset_index(drop=True)

    summary = {}

    print('── TRAIN ──────────────────────────────────────────────────')
    n = export_split(
        train_df, drug_dir,
        folder='train',
        out_split_dir=os.path.join(output_dir, 'train'),
        patch_size=patch_size, stride=stride,
        std_threshold=std_threshold,
        filter_background=filter_train_background,
    )
    summary['train'] = n

    print('\n── VAL (CSV.test → folder test/) ──────────────────────────')
    n = export_split(
        val_df, drug_dir,
        folder='test',
        out_split_dir=os.path.join(output_dir, 'val'),
        patch_size=patch_size, stride=stride,
        filter_background=False,
    )
    summary['val'] = n

    print('\n── TEST (CSV.val → folder val/) ───────────────────────────')
    n = export_split(
        test_df, drug_dir,
        folder='val',
        out_split_dir=os.path.join(output_dir, 'test'),
        patch_size=patch_size, stride=stride,
        filter_background=False,
    )
    summary['test'] = n

    print(f'\n  Total patches for {drug}: {sum(summary.values()):,}')
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Export Task 2 patches for antibiotic resistance classification',
    )
    parser.add_argument(
        '--drug', type=str, nargs='+', required=True,
        help='Drug(s) to export: AZI, CIP, TE, or "all"',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (default: {data_root}/task2_patches)',
    )
    parser.add_argument(
        '--patch_size', type=int, default=224,
        help='Patch size (default: 224)',
    )
    parser.add_argument(
        '--stride', type=int, default=224,
        help='Stride for sliding window (default: 224, no overlap)',
    )
    parser.add_argument(
        '--std_threshold', type=float, default=1000.0,
        help='Std threshold for filtering background patches in train (default: 1000.0)',
    )
    parser.add_argument(
        '--no_train_filter',
        action='store_true',
        help='Không lọc patch theo std ở tập train (giữ mọi patch, giống val/test).',
    )
    parser.add_argument(
        '--skip_existing', action='store_true',
        help='Skip drugs that already have patches exported',
    )

    args = parser.parse_args()
    cfg  = Config()

    # Determine output directory
    output_base = args.output_dir or os.path.join(cfg.data_root, 'task2_patches')
    os.makedirs(output_base, exist_ok=True)

    # Determine drugs to process
    if 'all' in args.drug:
        drugs = cfg.drugs  # ['AZI', 'CIP', 'TE']
    else:
        drugs = args.drug

    print('╔══════════════════════════════════════════════════════════╗')
    print('║  EXPORT TASK 2 PATCHES                                   ║')
    print('╚══════════════════════════════════════════════════════════╝')
    print(f'  Drugs      : {drugs}')
    print(f'  Output     : {output_base}')
    print(f'  Patch size : {args.patch_size}')
    print(f'  Stride     : {args.stride}')
    if args.no_train_filter:
        print(f'  Train std filter: tắt (--no_train_filter)')
    else:
        print(f'  Train std filter: ≥ {args.std_threshold}')

    all_summary = {}

    for drug in drugs:
        drug_output_dir = os.path.join(output_base, drug)
        
        if args.skip_existing:
            marker = os.path.join(drug_output_dir, '.export_done')
            if os.path.exists(marker):
                print(f'\n[SKIP] {drug} - already exported (found .export_done marker)')
                continue

        summary = export_drug(
            drug=drug,
            cfg=cfg,
            output_base=output_base,
            patch_size=args.patch_size,
            stride=args.stride,
            std_threshold=args.std_threshold,
            filter_train_background=not args.no_train_filter,
        )
        all_summary[drug] = summary

        # Tạo marker file để đánh dấu đã export xong
        marker = os.path.join(drug_output_dir, '.export_done')
        with open(marker, 'w') as f:
            f.write(f'patch_size={args.patch_size}\n')
            f.write(f'stride={args.stride}\n')
            f.write(f'std_threshold={args.std_threshold}\n')
            f.write(f'no_train_filter={args.no_train_filter}\n')
            for split, n in summary.items():
                f.write(f'{split}={n}\n')

    # Final summary
    print('\n' + '═'*60)
    print('  FINAL SUMMARY')
    print('═'*60)
    grand_total = 0
    for drug, summary in all_summary.items():
        total = sum(summary.values())
        grand_total += total
        print(f'  {drug}:')
        for split, n in summary.items():
            print(f'    {split:<6}: {n:,}')
        print(f'    {"TOTAL":<6}: {total:,}')
    print(f'\n  Grand total: {grand_total:,} patches')
    print(f'  Saved to   : {output_base}/')


if __name__ == '__main__':
    main()
