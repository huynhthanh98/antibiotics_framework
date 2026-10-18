"""
Paired One-tailed t-test: Task 1 vs Task 2

Mục tiêu: Kiểm định thống kê rằng Task 1 (masked cells) tổng quát hơn
          Task 2 (original image patches).

Giả thuyết:
    H₀: μ(Task1) ≤ μ(Task2)   [Task1 không tốt hơn Task2]
    H₁: μ(Task1) > μ(Task2)   [Task1 TỐT HƠN Task2]

    → One-tailed (right-tail), Paired (cùng fold index → cùng data split)
    → Paired t-test giảm variance từ data split variability

Cách chạy:
    # Sau khi đã train xong Task1 và Task2:
    python ttest_analysis.py

    # Chỉ định thư mục kết quả:
    python ttest_analysis.py --results_dir /path/to/results

    # Chọn metric cụ thể:
    python ttest_analysis.py --metrics auc f1_macro accuracy

    # Chọn drug và model:
    python ttest_analysis.py --drug AZI --model resnet50
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from itertools import product

warnings.filterwarnings('ignore')


# ─── Metric names hiển thị ────────────────────────────────────────────────────
METRIC_DISPLAY = {
    'recall_resistant':    'Recall    (Resistant)',
    'precision_resistant': 'Precision (Resistant)',
    'f1_resistant':        'F1        (Resistant)',
    'recall_sensitive':    'Recall    (Sensitive)',
    'f1_macro':            'F1 Macro',
    'f1_weighted':         'F1 Weighted',
    'auc':                 'AUC (ROC)',
    'accuracy':            'Accuracy',
    'mcc':                 'MCC',
}

DEFAULT_METRICS = [
    'recall_resistant', 'precision_resistant', 'f1_resistant',
    'recall_sensitive', 'auc', 'accuracy', 'f1_macro',
]
LEVELS          = ['cell_level', 'image_level']


# ─── Core t-test function ─────────────────────────────────────────────────────

def paired_one_tail_ttest(
    task1_values: np.ndarray,
    task2_values: np.ndarray,
    alpha:        float = 0.05,
) -> Dict:
    """
    Paired one-tailed t-test: H₁: Task1 > Task2.

    Args:
        task1_values : Array [fold1, fold2, ..., fold5] của Task 1
        task2_values : Array [fold1, fold2, ..., fold5] của Task 2
        alpha        : Mức ý nghĩa (default=0.05)

    Returns:
        dict với:
            t_stat      : t-statistic
            p_value     : p-value (one-tailed, right-tail)
            significant : True nếu p < alpha
            mean_diff   : mean(Task1) - mean(Task2)
            ci_lower    : Confidence interval lower bound của mean diff
            ci_upper    : Confidence interval upper bound của mean diff
            effect_size : Cohen's d cho paired samples
            interpretation : Chuỗi mô tả kết quả
    """
    task1_values = np.asarray(task1_values, dtype=float)
    task2_values = np.asarray(task2_values, dtype=float)
    n            = len(task1_values)

    if n < 2:
        return {
            't_stat': np.nan, 'p_value': np.nan,
            'significant': False, 'mean_diff': np.nan,
            'effect_size': np.nan, 'interpretation': 'Không đủ data (n<2)',
        }

    diff      = task1_values - task2_values
    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1))

    # Paired t-test hai chiều, rồi chuyển sang một chiều (right-tail)
    t_stat, p_two_tail = stats.ttest_rel(task1_values, task2_values)
    t_stat = float(t_stat)

    # One-tailed p-value (H₁: Task1 > Task2 → right tail)
    if t_stat > 0:
        p_one_tail = float(p_two_tail) / 2.0
    else:
        p_one_tail = 1.0 - float(p_two_tail) / 2.0

    # Cohen's d cho paired samples
    if std_diff > 1e-10:
        cohen_d = mean_diff / std_diff
    else:
        cohen_d = 0.0

    # 95% Confidence Interval của mean difference
    t_critical = stats.t.ppf(1 - alpha, df=n - 1)
    se_diff    = std_diff / np.sqrt(n)
    ci_lower   = mean_diff - t_critical * se_diff
    ci_upper   = mean_diff + t_critical * se_diff

    # Interpret effect size
    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        effect_str = "negligible"
    elif abs_d < 0.5:
        effect_str = "small"
    elif abs_d < 0.8:
        effect_str = "medium"
    else:
        effect_str = "large"

    significant = bool(p_one_tail < alpha) and (t_stat > 0)

    if significant:
        interpretation = (
            f"✅ SIGNIFICANT (p={p_one_tail:.4f} < {alpha}) → "
            f"Task1 tốt hơn Task2 có ý nghĩa thống kê "
            f"(effect={effect_str}, d={cohen_d:.3f})"
        )
    elif mean_diff > 0:
        interpretation = (
            f"⚠️ KHÔNG significant (p={p_one_tail:.4f} ≥ {alpha}) → "
            f"Task1 cao hơn nhưng chưa đủ bằng chứng thống kê "
            f"(effect={effect_str}, d={cohen_d:.3f})"
        )
    else:
        interpretation = (
            f"❌ KHÔNG significant (p={p_one_tail:.4f}) → "
            f"Task2 không kém hơn Task1 "
            f"(effect={effect_str}, d={cohen_d:.3f})"
        )

    return {
        't_stat':        round(t_stat,    4),
        'p_value':       round(p_one_tail, 4),
        'p_two_tail':    round(float(p_two_tail), 4),
        'significant':   significant,
        'mean_task1':    round(float(np.mean(task1_values)), 4),
        'std_task1':     round(float(np.std(task1_values, ddof=1)), 4),
        'mean_task2':    round(float(np.mean(task2_values)), 4),
        'std_task2':     round(float(np.std(task2_values, ddof=1)), 4),
        'mean_diff':     round(mean_diff, 4),
        'std_diff':      round(std_diff,  4),
        'ci_lower':      round(ci_lower,  4),
        'ci_upper':      round(ci_upper,  4),
        'effect_size_d': round(cohen_d,   4),
        'effect_label':  effect_str,
        'n_folds':       n,
        'task1_folds':   task1_values.tolist(),
        'task2_folds':   task2_values.tolist(),
        'diff_folds':    diff.tolist(),
        'interpretation': interpretation,
    }


# ─── Load fold metrics ────────────────────────────────────────────────────────

def load_fold_metrics(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Load master CSV chứa tất cả fold metrics.

    Ưu tiên: all_fold_metrics.csv → merge từ các fold_by_fold.csv riêng lẻ.
    """
    master_path = os.path.join(results_dir, 'all_fold_metrics.csv')

    if os.path.exists(master_path):
        df = pd.read_csv(master_path)
        print(f"  Loaded: {master_path} ({len(df)} rows)")
        return df

    # Fallback: tìm tất cả fold_by_fold.csv
    print(f"  Không tìm thấy all_fold_metrics.csv, tìm fold_by_fold.csv...")
    all_dfs = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == 'fold_by_fold.csv':
                fpath = os.path.join(root, f)
                try:
                    df = pd.read_csv(fpath)
                    all_dfs.append(df)
                    print(f"  Loaded: {fpath}")
                except Exception as e:
                    print(f"  Lỗi đọc {fpath}: {e}")

    if not all_dfs:
        print(f"  ✗ Không tìm thấy fold metrics nào trong {results_dir}")
        return None

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(master_path, index=False)
    print(f"  Merged {len(all_dfs)} files → {master_path}")
    return merged


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_ttest_analysis(
    results_dir:   str,
    drugs:         Optional[List[str]] = None,
    model_names:   Optional[List[str]] = None,
    metrics:       Optional[List[str]] = None,
    levels:        Optional[List[str]] = None,
    alpha:         float = 0.05,
    output_csv:    Optional[str] = None,
    verbose:       bool  = True,
) -> pd.DataFrame:
    """
    Chạy Paired One-tailed t-test cho tất cả combinations.

    So sánh Task1 vs Task2 theo từng (drug, model, metric, level).

    Args:
        results_dir  : Thư mục chứa kết quả (có all_fold_metrics.csv)
        drugs        : List drugs cần analyze. None → tất cả
        model_names  : List models. None → tất cả
        metrics      : List metrics cần test. None → DEFAULT_METRICS
        levels       : ['cell_level', 'image_level']. None → cả hai
        alpha        : Mức ý nghĩa (default=0.05)
        output_csv   : Đường dẫn lưu kết quả CSV. None → tự động
        verbose      : In chi tiết ra màn hình

    Returns:
        DataFrame kết quả t-test
    """
    if metrics is None: metrics = DEFAULT_METRICS
    if levels  is None: levels  = LEVELS

    # Load data
    df = load_fold_metrics(results_dir)
    if df is None:
        raise FileNotFoundError(
            f"Không tìm thấy fold metrics trong {results_dir}.\n"
            f"Hãy chạy training trước: python main.py --task 1 2"
        )

    # Kiểm tra cột bắt buộc
    required = ['task', 'drug', 'model', 'fold', 'level'] + metrics
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️ CSV thiếu cột: {missing}")
        metrics = [m for m in metrics if m in df.columns]

    # Filter
    if drugs       is not None: df = df[df['drug'].isin(drugs)]
    if model_names is not None: df = df[df['model'].isin(model_names)]

    available_drugs  = sorted(df['drug'].unique())
    available_models = sorted(df['model'].unique())

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  PAIRED ONE-TAILED T-TEST: Task1 vs Task2")
        print(f"  H₀: Task1 ≤ Task2  |  H₁: Task1 > Task2  |  α={alpha}")
        print(f"{'═'*70}")
        print(f"  Drugs  : {available_drugs}")
        print(f"  Models : {available_models}")
        print(f"  Metrics: {metrics}")
        print(f"  Levels : {levels}")
        print(f"{'═'*70}\n")

    rows = []

    for drug, model_name, level, metric in product(
        available_drugs, available_models, levels, metrics
    ):
        # Lấy kết quả Task1 và Task2 cho combination này
        mask1 = (
            (df['task']  == 1) &
            (df['drug']  == drug) &
            (df['model'] == model_name) &
            (df['level'] == level)
        )
        mask2 = (
            (df['task']  == 2) &
            (df['drug']  == drug) &
            (df['model'] == model_name) &
            (df['level'] == level)
        )

        df1 = df[mask1].sort_values('fold')
        df2 = df[mask2].sort_values('fold')

        if len(df1) == 0 or len(df2) == 0:
            if verbose:
                print(f"  Bỏ qua: {drug} | {model_name} | {level} | {metric} "
                      f"(Task1={len(df1)}, Task2={len(df2)} folds)")
            continue

        if len(df1) != len(df2):
            # Align theo fold index nếu số fold khác nhau
            common_folds = set(df1['fold'].values) & set(df2['fold'].values)
            df1 = df1[df1['fold'].isin(common_folds)].sort_values('fold')
            df2 = df2[df2['fold'].isin(common_folds)].sort_values('fold')
            if verbose and len(df1) < 5:
                print(f"  ⚠️ Chỉ có {len(df1)} folds cho {drug}|{model_name}|{level}")

        if metric not in df1.columns or metric not in df2.columns:
            continue

        task1_values = df1[metric].values.astype(float)
        task2_values = df2[metric].values.astype(float)

        result = paired_one_tail_ttest(task1_values, task2_values, alpha=alpha)

        row = {
            'drug':        drug,
            'model':       model_name,
            'level':       level,
            'metric':      metric,
            'metric_name': METRIC_DISPLAY.get(metric, metric),
            **result,
        }
        rows.append(row)

        if verbose:
            level_str = 'Cell' if level == 'cell_level' else 'Image'
            metric_str = METRIC_DISPLAY.get(metric, metric)
            print(
                f"  [{drug}|{model_name}|{level_str}|{metric_str:<30}]\n"
                f"    Task1: {result['mean_task1']:.4f}±{result['std_task1']:.4f}  "
                f"| Task2: {result['mean_task2']:.4f}±{result['std_task2']:.4f}\n"
                f"    Δ={result['mean_diff']:+.4f} "
                f"[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]\n"
                f"    t={result['t_stat']:.3f}, p={result['p_value']:.4f} "
                f"(1-tail), d={result['effect_size_d']:.3f}\n"
                f"    {result['interpretation']}\n"
            )

    if not rows:
        print("  ✗ Không có kết quả nào để t-test. Kiểm tra lại training.")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    # ─── In tóm tắt ──────────────────────────────────────────────────────────
    if verbose:
        _print_summary(result_df, alpha)

    # ─── Lưu CSV ──────────────────────────────────────────────────────────────
    if output_csv is None:
        output_csv = os.path.join(results_dir, 'ttest_results.csv')

    # Chọn cột để lưu (không lưu list folds vào CSV chính)
    save_cols = [
        'drug', 'model', 'level', 'metric', 'metric_name',
        'mean_task1', 'std_task1', 'mean_task2', 'std_task2',
        'mean_diff', 'std_diff', 'ci_lower', 'ci_upper',
        't_stat', 'p_value', 'p_two_tail', 'significant',
        'effect_size_d', 'effect_label', 'n_folds', 'interpretation',
    ]
    save_cols = [c for c in save_cols if c in result_df.columns]
    result_df[save_cols].to_csv(output_csv, index=False)

    if verbose:
        print(f"\n  ✓ Kết quả t-test → {output_csv}")

    # Lưu thêm file chi tiết với fold values
    detail_rows = []
    for row in rows:
        for i, (v1, v2, diff) in enumerate(
            zip(row['task1_folds'], row['task2_folds'], row['diff_folds'])
        ):
            detail_rows.append({
                'drug':    row['drug'],
                'model':   row['model'],
                'level':   row['level'],
                'metric':  row['metric'],
                'fold':    i + 1,
                'task1':   v1,
                'task2':   v2,
                'diff':    diff,
            })
    detail_df   = pd.DataFrame(detail_rows)
    detail_path = output_csv.replace('.csv', '_detail.csv')
    detail_df.to_csv(detail_path, index=False)
    if verbose:
        print(f"  ✓ Chi tiết fold-by-fold → {detail_path}")

    return result_df


def _print_summary(result_df: pd.DataFrame, alpha: float):
    """In bảng tóm tắt kết quả t-test."""
    print(f"\n{'═'*70}")
    print(f"  TÓM TẮT T-TEST (α={alpha})")
    print(f"{'═'*70}")

    # Số kết quả significant
    n_total = len(result_df)
    n_sig   = result_df['significant'].sum()
    print(f"  Significant: {n_sig}/{n_total} "
          f"({100*n_sig/max(n_total,1):.1f}% của tất cả tests)")

    # Breakdown theo metric
    print(f"\n  Theo Metric:")
    for metric in result_df['metric'].unique():
        sub = result_df[result_df['metric'] == metric]
        n_s = sub['significant'].sum()
        mean_diff = sub['mean_diff'].mean()
        print(
            f"  {METRIC_DISPLAY.get(metric, metric):<35} "
            f"significant: {n_s}/{len(sub)}  "
            f"Δ mean={mean_diff:+.4f}"
        )

    # Breakdown theo level
    print(f"\n  Theo Level:")
    for level in result_df['level'].unique():
        sub   = result_df[result_df['level'] == level]
        n_s   = sub['significant'].sum()
        lname = 'Cell-level' if level == 'cell_level' else 'Image-level'
        print(f"  {lname:<20} significant: {n_s}/{len(sub)}")

    # Breakdown theo drug
    print(f"\n  Theo Drug:")
    for drug in result_df['drug'].unique():
        sub = result_df[result_df['drug'] == drug]
        n_s = sub['significant'].sum()
        print(f"  {drug:<10} significant: {n_s}/{len(sub)}")

    # Best performers
    if n_sig > 0:
        print(f"\n  Significant tests (Task1 > Task2):")
        sig_df = result_df[result_df['significant']].sort_values('p_value')
        for _, row in sig_df.iterrows():
            level_str = 'Cell' if row['level'] == 'cell_level' else 'Img'
            print(
                f"  ✅ [{row['drug']}|{row['model']}|{level_str}] "
                f"{METRIC_DISPLAY.get(row['metric'], row['metric'])}: "
                f"T1={row['mean_task1']:.4f} vs T2={row['mean_task2']:.4f} "
                f"(Δ={row['mean_diff']:+.4f}, p={row['p_value']:.4f}, "
                f"d={row['effect_size_d']:.3f} [{row['effect_label']}])"
            )

    print(f"{'═'*70}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Paired One-tailed T-test: Task1 vs Task2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--results_dir', type=str,
        default='/home/aisia/Son/Disk/new_mia/results',
        help='Thư mục chứa kết quả training. Default: results/',
    )
    parser.add_argument(
        '--drug', type=str, nargs='+', default=None,
        help='Lọc theo drug. Default: tất cả',
    )
    parser.add_argument(
        '--model', type=str, nargs='+', default=None,
        help='Lọc theo model. Default: tất cả',
    )
    parser.add_argument(
        '--metrics', type=str, nargs='+', default=None,
        choices=list(METRIC_DISPLAY.keys()),
        help=f'Metrics cần test. Default: {DEFAULT_METRICS}',
    )
    parser.add_argument(
        '--level', type=str, nargs='+', default=None,
        choices=['cell_level', 'image_level'],
        help='Level cần test. Default: cả hai',
    )
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='Mức ý nghĩa. Default: 0.05',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Đường dẫn lưu CSV kết quả. Default: {results_dir}/ttest_results.csv',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Không in chi tiết từng test',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    result_df = run_ttest_analysis(
        results_dir  = args.results_dir,
        drugs        = args.drug,
        model_names  = args.model,
        metrics      = args.metrics,
        levels       = args.level,
        alpha        = args.alpha,
        output_csv   = args.output,
        verbose      = not args.quiet,
    )

