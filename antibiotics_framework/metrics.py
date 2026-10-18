"""
Tính toán các metrics đánh giá cho Task 1 và Task 2.

Metrics được tính ở hai mức:
  1. Cell-level (hoặc patch-level cho Task 2): đánh giá từng cell/patch
  2. Image-level: vote từ tất cả cells/patches của cùng một ảnh gốc

Metrics bao gồm:
  - Sensitivity  = Recall của lớp Resistant (label=1)  = TP/(TP+FN)
  - Specificity  = Recall của lớp Sensitive (label=0)   = TN/(TN+FP)
  - Precision    = TP/(TP+FP) [cho lớp Resistant]
  - F1 Score     = 2 * Precision * Sensitivity / (Precision + Sensitivity)
  - AUC          = Area Under ROC Curve
  - Accuracy     = (TP+TN) / (TP+TN+FP+FN)
  - MCC          = Matthews Correlation Coefficient (optional)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# ─── Threshold Calibration ────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = 'f1',
    min_recall: Optional[float] = None,
) -> float:
    """
    Tìm optimal classification threshold dựa vào validation set.

    Args:
        y_true     : Ground truth labels (0/1)
        y_prob     : Predicted probabilities cho lớp 1 (resistant)
        method     : Phương pháp tìm threshold:
                     - 'f1'        : Maximize F1 score của lớp resistant (default)
                     - 'balanced'  : Minimize |Sensitivity - Specificity|
                     - 'min_recall': Giữ recall >= min_recall, trong đó chọn threshold
                                     có precision cao nhất (trade-off recall vs precision)
                     - 'pr_equal'  : Minimize |Precision - Recall| → P ≈ R cho lớp resistant
                     - 'gmean'     : Maximize G-mean = sqrt(Sensitivity × Specificity),
                                     cân bằng tốt nhất khi class imbalance
                     - 'fixed'     : Trả về thẳng min_recall (dùng như hard threshold, truyền giá trị qua min_recall)
        min_recall : Chỉ dùng khi method='min_recall'. Recall tối thiểu cần đạt (e.g. 0.85).
                     Với method='fixed': giá trị hard threshold (e.g. 0.7).
                     Nếu không có threshold nào đạt đủ recall thì chọn threshold cho recall cao nhất.

    Returns:
        float: Optimal threshold trong [0, 1]
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        return 0.5  # Chỉ có 1 lớp → không calibrate được

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if method == 'fixed':
        # Hard threshold cố định, truyền qua tham số min_recall
        fixed_val = min_recall if min_recall is not None else 0.5
        return float(np.clip(fixed_val, 0.0, 1.0))

    elif method == 'f1':
        # Maximize F1 của lớp resistant
        f1_scores = []
        for thresh in thresholds:
            preds = (y_prob >= thresh).astype(int)
            f1    = f1_score(y_true, preds, zero_division=0)
            f1_scores.append(f1)
        best_idx = int(np.argmax(f1_scores))

    elif method == 'balanced':
        # Minimize |Sensitivity - Specificity|
        specificity = 1.0 - fpr
        diff        = np.abs(tpr - specificity)
        best_idx    = int(np.argmin(diff))

    elif method == 'min_recall':
        # Recall đủ cao (>= min_recall), trong đó tối đa precision
        r_min = min_recall if min_recall is not None else 0.85
        best_idx = None
        best_prec = -1.0
        for i, thresh in enumerate(thresholds):
            preds   = (y_prob >= thresh).astype(int)
            rec     = recall_score(y_true, preds, pos_label=1, zero_division=0)
            prec    = precision_score(y_true, preds, pos_label=1, zero_division=0)
            if rec >= r_min and prec > best_prec:
                best_prec = prec
                best_idx  = i
        if best_idx is None:
            # Không có threshold nào đạt recall >= r_min → chọn threshold cho recall cao nhất
            best_idx = int(np.argmax(tpr))

    elif method == 'pr_equal':
        # Minimize |Precision - Recall| của lớp resistant
        # → Dùng precision_recall_curve để có đủ thresholds trên toàn miền [0,1]
        pr_precisions, pr_recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
        # pr_thresholds có length = len(pr_precisions) - 1 (bỏ điểm cuối)
        diffs = np.abs(pr_precisions[:-1] - pr_recalls[:-1])
        best_pr_idx = int(np.argmin(diffs))
        optimal_threshold = float(np.clip(pr_thresholds[best_pr_idx], 0.0, 1.0))
        return optimal_threshold

    elif method == 'gmean':
        # Maximize G-mean = sqrt(Sensitivity × Specificity)
        # Cân bằng tốt nhất khi class imbalance, không bị ảnh hưởng bởi class size
        gmean_scores = np.sqrt(tpr * (1.0 - fpr))
        best_idx = int(np.argmax(gmean_scores))

    else:
        raise ValueError(
            f"method phải là 'f1', 'balanced', 'min_recall', 'pr_equal', 'gmean', hoặc 'fixed'. "
            f"Nhận: '{method}'"
        )

    # Clip threshold về [0, 1]
    optimal_threshold = float(np.clip(thresholds[best_idx], 0.0, 1.0))
    return optimal_threshold


# ─── Core metric computation ──────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Tính đầy đủ các metrics cho bài toán binary classification.

    Args:
        y_true    : Ground truth labels (0 = sensitive, 1 = resistant)
        y_pred    : Predicted labels (argmax của logits/probs) - dùng khi threshold=0.5
        y_prob    : Predicted probabilities cho lớp positive (resistant=1)
                    Shape: (N,) - prob[:,1] nếu đầu ra là 2-class
        threshold : Decision threshold. Default=0.5.
                    Nếu khác 0.5 → y_pred sẽ được recompute từ y_prob.

    Returns:
        dict với tất cả metrics
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Recompute y_pred nếu threshold không phải 0.5
    if abs(threshold - 0.5) > 1e-6:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Chỉ có 1 lớp trong y_true
        tn, fp, fn, tp = 0, 0, 0, 0
        if np.all(y_true == 0):
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
        else:
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))

    # Sensitivity (Recall cho lớp Resistant = 1)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (Recall cho lớp Sensitive = 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision cho lớp Resistant (label=1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Precision cho lớp Sensitive (label=0): TN / (TN + FN)
    precision_sensitive = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # F1 score (cho lớp Resistant)
    f1_resistant = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0 else 0.0
    )

    # F1 score (cho lớp Sensitive)
    f1_sensitive = (
        2 * precision_sensitive * specificity / (precision_sensitive + specificity)
        if (precision_sensitive + specificity) > 0 else 0.0
    )

    # F1 macro (trung bình hai lớp)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # F1 weighted
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # AUC
    try:
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = float('nan')
    except Exception:
        auc = float('nan')

    # MCC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = float('nan')

    # Youden's J = Sensitivity + Specificity - 1  (∈ [-1, 1], cao hơn tốt hơn)
    youden_j = sensitivity + specificity - 1.0

    return {
        # ── Metrics chính của nhãn Resistant (label=1) ──────────────────────
        'recall_resistant':    round(float(sensitivity), 4),  # TP/(TP+FN)
        'precision_resistant': round(float(precision),   4),  # TP/(TP+FP)
        'f1_resistant':        round(float(f1_resistant),4),  # 2*P*R/(P+R)
        # ── Metrics bổ sung ─────────────────────────────────────────────────
        'recall_sensitive':    round(float(specificity),       4),  # TN/(TN+FP)
        'precision_sensitive': round(float(precision_sensitive),4),  # TN/(TN+FN)
        'f1_sensitive':        round(float(f1_sensitive),      4),
        'youden_j':            round(float(youden_j),    4),  # Sens + Spec - 1
        'f1_macro':            round(float(f1_macro),    4),
        'f1_weighted':         round(float(f1_weighted), 4),
        'auc':                 round(float(auc),         4) if not np.isnan(auc) else 0.0,
        'accuracy':            round(float(accuracy),    4),
        'mcc':                 round(float(mcc),         4) if not np.isnan(mcc) else 0.0,
        'threshold':           round(float(threshold),   4),
        # ── Counts ──────────────────────────────────────────────────────────
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
        'n_total':     int(len(y_true)),
        'n_resistant': int(np.sum(y_true == 1)),
        'n_sensitive': int(np.sum(y_true == 0)),
    }


# ─── Image-level voting ───────────────────────────────────────────────────────

def image_level_voting(
    orig_files: List[str],
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_prob:     np.ndarray,
    method:     str   = 'soft',
    threshold:  float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Aggregate cell/patch predictions thành image-level predictions.

    Hai phương pháp:
      - 'soft' : Average probabilities → so sánh với threshold
      - 'hard' : Majority vote từ cell-level predictions (tính lại từ y_prob + threshold)

    Args:
        orig_files : List các original_file tương ứng với mỗi cell/patch
        y_true     : Ground truth labels (N,)
        y_pred     : Predicted labels (N,) - dùng khi threshold=0.5
        y_prob     : Predicted probabilities của lớp 1 (resistant) (N,)
        method     : 'soft' hoặc 'hard'
        threshold  : Decision threshold (dùng cho cả soft và hard vote)

    Returns:
        Tuple (img_y_true, img_y_pred, img_y_prob, img_orig_files)
        Mỗi array có shape (M,) với M = số ảnh gốc duy nhất
    """
    y_true     = np.asarray(y_true)
    y_prob     = np.asarray(y_prob)
    orig_files = np.asarray(orig_files)
    # Luôn tính lại y_pred từ y_prob + threshold để cả soft/hard đều dùng đúng threshold
    y_pred     = (y_prob >= threshold).astype(int)

    unique_imgs   = []
    img_y_true    = []
    img_y_pred    = []
    img_y_prob    = []

    for orig_file in pd.unique(orig_files):  # giữ thứ tự
        mask = (orig_files == orig_file)

        # Ground truth (tất cả cells của cùng 1 ảnh có cùng label)
        gt = y_true[mask]
        unique_labels = np.unique(gt)
        if len(unique_labels) > 1:
            # Cảnh báo nếu có cells với label khác nhau từ cùng 1 ảnh
            print(f"  [metrics] Cảnh báo: ảnh {orig_file} có nhiều labels: {unique_labels}")
        img_gt = int(unique_labels[0])

        if method == 'soft':
            # Average probabilities → so sánh với threshold
            avg_prob   = float(np.mean(y_prob[mask]))
            pred_label = 1 if avg_prob >= threshold else 0
        else:  # 'hard'
            # Majority vote: đếm cell-level predictions đã được tính với đúng threshold
            votes       = y_pred[mask]   # đã được tính lại từ y_prob + threshold ở trên
            n_resistant = int(np.sum(votes == 1))
            n_sensitive = int(np.sum(votes == 0))
            # Tie-breaking: nếu bằng nhau → chọn resistant (conservative)
            pred_label  = 1 if n_resistant >= n_sensitive else 0
            avg_prob    = float(n_resistant / len(votes)) if len(votes) > 0 else 0.0

        unique_imgs.append(orig_file)
        img_y_true.append(img_gt)
        img_y_pred.append(pred_label)
        img_y_prob.append(avg_prob)

    return (
        np.array(img_y_true),
        np.array(img_y_pred),
        np.array(img_y_prob),
        unique_imgs,
    )


# ─── Full evaluation ─────────────────────────────────────────────────────────

def evaluate_full(
    orig_files:    List[str],
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    y_prob:        np.ndarray,
    voting_method: str   = 'soft',
    threshold:     float = 0.5,
) -> Dict[str, Dict]:
    """
    Tính metrics ở cả cell-level và image-level.

    Args:
        orig_files     : Original file name cho mỗi cell/patch
        y_true         : Ground truth labels
        y_pred         : Predicted labels (sẽ bị override nếu threshold != 0.5)
        y_prob         : Predicted probabilities (lớp 1)
        voting_method  : 'soft' hoặc 'hard'
        threshold      : Decision threshold (từ calibration hoặc 0.5 mặc định)

    Returns:
        {
            'cell_level':  {accuracy, sensitivity, specificity, f1, auc, ...},
            'image_level': {accuracy, sensitivity, specificity, f1, auc, ...},
        }
    """
    # Cell/Patch-level metrics (với optimal threshold)
    cell_metrics = compute_metrics(y_true, y_pred, y_prob, threshold=threshold)

    # Image-level metrics
    img_y_true, img_y_pred, img_y_prob, _ = image_level_voting(
        orig_files, y_true, y_pred, y_prob, method=voting_method, threshold=threshold
    )
    image_metrics = compute_metrics(img_y_true, img_y_pred, img_y_prob, threshold=threshold)

    return {
        'cell_level':  cell_metrics,
        'image_level': image_metrics,
    }


# ─── Fold summary ─────────────────────────────────────────────────────────────

def summarize_folds(fold_results: List[Dict],
                    levels: List[str] = ('cell_level', 'image_level')
                    ) -> Dict[str, Dict]:
    """
    Tính mean ± std của các metrics qua tất cả folds.

    Args:
        fold_results : List kết quả từng fold (output của evaluate_full)
        levels       : Các level cần tổng hợp

    Returns:
        Dict với key = level, value = dict {metric: 'mean±std'}
    """
    summary = {}
    metric_keys = [
        'recall_resistant', 'precision_resistant', 'f1_resistant',
        'recall_sensitive', 'precision_sensitive', 'f1_sensitive',
        'youden_j', 'f1_macro', 'f1_weighted', 'auc', 'accuracy', 'mcc',
    ]

    for level in levels:
        level_summary = {}
        for key in metric_keys:
            values = [r[level][key] for r in fold_results
                      if key in r.get(level, {})]
            if values:
                mean = float(np.mean(values))
                std  = float(np.std(values))
                level_summary[key] = {
                    'mean': round(mean, 4),
                    'std':  round(std,  4),
                    'str':  f"{mean:.4f} ± {std:.4f}",
                }
        summary[level] = level_summary

    return summary


# ─── Pretty print ─────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict, title: str = '', indent: int = 4):
    """In metrics ra màn hình theo dạng đẹp."""
    pad = ' ' * indent
    if title:
        print(f"\n{pad}{'─' * 50}")
        print(f"{pad}  {title}")
        print(f"{pad}{'─' * 50}")

    metric_display = [
        ('Recall    (Resistant)',  'recall_resistant'),
        ('Precision (Resistant)',  'precision_resistant'),
        ('F1        (Resistant)',  'f1_resistant'),
        ('Recall    (Sensitive)',  'recall_sensitive'),
        ('Precision (Sensitive)',  'precision_sensitive'),
        ('F1        (Sensitive)',  'f1_sensitive'),
        ("Youden's J",             'youden_j'),
        ('F1 Macro',               'f1_macro'),
        ('AUC',                    'auc'),
        ('Accuracy',               'accuracy'),
        ('MCC',                    'mcc'),
    ]

    for display_name, key in metric_display:
        val = metrics.get(key, 'N/A')
        if isinstance(val, float):
            print(f"{pad}  {display_name:<25}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"{pad}  {display_name:<25}: {val.get('str', 'N/A')}")
        else:
            print(f"{pad}  {display_name:<25}: {val}")

    if 'tp' in metrics:
        print(f"\n{pad}  Confusion Matrix:")
        print(f"{pad}    TP={metrics['tp']:5d}  FP={metrics['fp']:5d}")
        print(f"{pad}    FN={metrics['fn']:5d}  TN={metrics['tn']:5d}")
        print(f"{pad}  Total={metrics['n_total']:5d}  "
              f"(R={metrics['n_resistant']:5d}, S={metrics['n_sensitive']:5d})")


def print_fold_summary(summary: Dict, title: str = 'FOLD SUMMARY'):
    """In tổng hợp các fold ra màn hình."""
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")

    for level, level_data in summary.items():
        if level == 'cell_level':
            level_name = 'Cell-Level'
        elif level == 'image_level_hard':
            level_name = 'Image-Level (Majority Vote)'
        else:
            level_name = 'Image-Level (Soft Vote)'
        print(f"\n  [{level_name}]")
        metric_keys = [
            ('Recall    (Resistant)', 'recall_resistant'),
            ('Precision (Resistant)', 'precision_resistant'),
            ('F1        (Resistant)', 'f1_resistant'),
            ('Recall    (Sensitive)', 'recall_sensitive'),
            ('Precision (Sensitive)', 'precision_sensitive'),
            ('F1        (Sensitive)', 'f1_sensitive'),
            ("Youden's J",            'youden_j'),
            ('F1 Macro',              'f1_macro'),
            ('AUC',                   'auc'),
            ('Accuracy',              'accuracy'),
        ]
        for display_name, key in metric_keys:
            val = level_data.get(key, {})
            if isinstance(val, dict):
                print(f"    {display_name:<25}: {val.get('str', 'N/A')}")

