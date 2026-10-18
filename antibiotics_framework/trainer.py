"""
Training engine với Multi-Seed runs cho Antibiotic Resistance Classification.

Thay thế StratifiedKFold bằng cách dùng trực tiếp train/val/test splits có sẵn
trong *_joblib folders. Chạy N lần với random seed khác nhau → báo cáo mean ± std.

Flow:
  1. Load split_info.csv → tách train_df / val_df / test_df
  2. Lặp qua từng seed trong cfg.seeds (mặc định [0,1,2,3,4]):
     a. set_seed(seed)
     b. Train model trên train_df
     c. Dùng val_df để calibrate threshold (chọn best epoch)
     d. Evaluate trên test_df với threshold tìm được
     e. Lưu kết quả của run này
  3. Sau tất cả seeds: tổng hợp mean ± std, lưu run_metrics.csv
"""

import os
import json
import time
import random
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from datasets import (
    build_datasets, Task1Dataset,
    get_test_transform, get_train_transform,
    get_train_transform_with_erosion_aug,
)
from models import get_model, count_parameters
from losses import get_loss_fn, get_weighted_sampler
from metrics import (
    evaluate_full,
    find_optimal_threshold,
    image_level_voting,
    summarize_folds,
    print_metrics,
    print_fold_summary,
    compute_metrics,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ─── Seed everything ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Cố định tất cả random seeds để reproduce kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Train one epoch ──────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    use_bf16:  bool = False,
) -> Dict[str, float]:
    """Train model cho một epoch. Hỗ trợ BFloat16 autocast."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    # BF16 không cần GradScaler (khác FP16), chỉ cần autocast
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) \
              if use_bf16 and device.type == 'cuda' \
              else torch.autocast(device_type='cuda', enabled=False)

    for batch_idx, (imgs, labels, _, _) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with amp_ctx:
            logits = model(imgs)
            loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return {
        'loss':     total_loss / total if total > 0 else 0.0,
        'accuracy': correct    / total if total > 0 else 0.0,
    }


# ─── Evaluate on a DataLoader ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    use_bf16: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Chạy inference trên toàn bộ DataLoader. Hỗ trợ BFloat16 autocast.

    Returns:
        (y_true, y_pred, y_prob_class1, orig_files, fnames)
    """
    model.eval()
    all_y_true     = []
    all_y_pred     = []
    all_y_prob     = []
    all_orig_files = []
    all_fnames     = []

    softmax  = nn.Softmax(dim=1)
    amp_ctx  = torch.autocast(device_type='cuda', dtype=torch.bfloat16) \
               if use_bf16 and device.type == 'cuda' \
               else torch.autocast(device_type='cuda', enabled=False)

    for imgs, labels, orig_files, fnames in loader:
        imgs = imgs.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(imgs)
        probs = softmax(logits.float())   # cast về FP32 trước softmax
        preds = probs.argmax(dim=1)
        prob1 = probs[:, 1]

        all_y_true.extend(labels.cpu().numpy().tolist())
        all_y_pred.extend(preds.cpu().numpy().tolist())
        all_y_prob.extend(prob1.cpu().numpy().tolist())
        all_orig_files.extend(list(orig_files))
        all_fnames.extend(list(fnames))

    return (
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_prob),
        all_orig_files,
        all_fnames,
    )


def _train_label_col(df: pd.DataFrame) -> str:
    if 'resistant' in df.columns:
        return 'resistant'
    if 'resistance_label' in df.columns:
        return 'resistance_label'
    raise ValueError('train_df thiếu cột resistant hoặc resistance_label')


def task1_downsample_train_df(
    train_df: pd.DataFrame,
    cfg: Config,
    seed: int,
) -> pd.DataFrame:
    """
    Undersample tập train Task 1 (trước khi build Task1Dataset).

    - train_downsample_balance_classes: mỗi lớp = min(n_R, n_S).
    - train_downsample_fraction: mỗi lớp giữ max(1, floor(n * f)).
    Nếu bật balance thì bỏ qua fraction.
    """
    balance = getattr(cfg, 'train_downsample_balance_classes', False)
    frac = getattr(cfg, 'train_downsample_fraction', None)

    if not balance and (frac is None or frac >= 1.0):
        return train_df

    if frac is not None and (frac <= 0 or frac > 1.0):
        raise ValueError(
            f'train_downsample_fraction phải thuộc (0, 1], nhận: {frac}'
        )

    label_col = _train_label_col(train_df)
    df = train_df.copy()
    rng = np.random.default_rng(int(seed))

    if balance:
        sub0 = df[df[label_col] == 0]
        sub1 = df[df[label_col] == 1]
        n = min(len(sub0), len(sub1))
        if n == 0:
            return train_df
        i0 = rng.choice(len(sub0), size=n, replace=False)
        i1 = rng.choice(len(sub1), size=n, replace=False)
        out = pd.concat([sub0.iloc[i0], sub1.iloc[i1]], ignore_index=True)
        return out.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    parts = []
    for y in (0, 1):
        sub = df[df[label_col] == y]
        if len(sub) == 0:
            continue
        n_keep = max(1, int(len(sub) * float(frac)))
        n_keep = min(n_keep, len(sub))
        idx = rng.choice(len(sub), size=n_keep, replace=False)
        parts.append(sub.iloc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


# ─── Single seed run ──────────────────────────────────────────────────────────

def train_one_run(
    run_idx:          int,
    seed:             int,
    task:             int,
    drug:             str,
    model_name:       str,
    train_df:         pd.DataFrame,
    val_df:           pd.DataFrame,
    test_df:          pd.DataFrame,
    cfg:              Config,
    device:           torch.device,
    loss_name:        str = 'focal',
    threshold_method: str = 'f1',
    use_bf16:         bool = False,
) -> Dict:
    """
    Train và evaluate model cho một seed run.

    Args:
        run_idx          : Index của run (0-based)
        seed             : Random seed cho run này
        task             : 1 hoặc 2
        drug             : 'AZI', 'CIP', 'TE'
        model_name       : Tên model trong timm
        train_df         : DataFrame train cells
        val_df           : DataFrame validation cells (calibrate threshold)
        test_df          : DataFrame test cells (evaluate cuối)
        cfg              : Config object
        device           : torch.device
        loss_name        : 'focal' hoặc 'ce'
        threshold_method : 'f1', 'balanced', 'min_recall', 'pr_equal', 'gmean', hoặc 'fixed'

    Returns:
        dict kết quả run với đầy đủ cell-level + image-level metrics
    """
    logger.info(
        f"\n  ╔══ Run {run_idx+1}/{cfg.n_seeds} | Seed={seed} "
        f"| Task{task} | {drug} | {model_name} ══╗"
    )

    drug_dir = cfg.drug_dir(drug)

    train_df_model = train_df
    if task == 1:
        train_df_model = task1_downsample_train_df(train_df, cfg, seed)
        if len(train_df_model) != len(train_df):
            lc = _train_label_col(train_df_model)
            mode = (
                'balance→1:1'
                if getattr(cfg, 'train_downsample_balance_classes', False)
                else f"fraction={getattr(cfg, 'train_downsample_fraction', None)}"
            )
            logger.info(
                f"  [Train downsample:{mode}] {len(train_df):,} → {len(train_df_model):,} cells | "
                f"R={(train_df_model[lc] == 1).sum():,} S={(train_df_model[lc] == 0).sum():,}"
            )

    # ─── Build datasets ───────────────────────────────────────────────────────
    # Task 1: tuỳ chọn erosion augmentation khi train
    _erosion_aug    = getattr(cfg, 'mask_erosion_aug',    False)
    _erosion_max_px = getattr(cfg, 'mask_erosion_max_px', 3)
    if task == 1 and _erosion_aug:
        train_transform = get_train_transform_with_erosion_aug(cfg.img_size, _erosion_max_px)
        logger.info(
            f"  [ErosionAug] RandomMaskErosion(max_px={_erosion_max_px}, p=0.5) bật cho train"
        )
    else:
        train_transform = get_train_transform(cfg.img_size)
    test_transform  = get_test_transform(cfg.img_size)

    swap = getattr(cfg, 'swap_val_test', True)
    val_folder  = 'test' if swap else 'val'
    test_folder = 'val'  if swap else 'test'
    if task == 1:
        from datasets import Task1Dataset
        _img_folder = getattr(cfg, 'task1_image_folder', 'masked')
        train_ds = Task1Dataset(train_df_model, drug_dir, split='train', image_folder=_img_folder, transform=train_transform)
        val_ds   = Task1Dataset(val_df,   drug_dir, split=val_folder,  image_folder=_img_folder, transform=test_transform)
        test_ds  = Task1Dataset(test_df,  drug_dir, split=test_folder, image_folder=_img_folder, transform=test_transform)
    elif task == 2:
        # Kiểm tra có patches đã export không
        patches_dir = getattr(cfg, 'task2_patches_dir', None)
        if patches_dir:
            drug_patches_dir = os.path.join(patches_dir, drug)
        else:
            drug_patches_dir = None

        # Nếu có patches đã export → dùng Task2PatchDataset (nhanh hơn nhiều)
        if drug_patches_dir and os.path.exists(os.path.join(drug_patches_dir, 'train')):
            from datasets import Task2PatchDataset
            logger.info(f"  [Task2] Using precomputed patches from: {drug_patches_dir}")
            train_ds = Task2PatchDataset(drug_patches_dir, split='train', transform=train_transform)
            val_ds   = Task2PatchDataset(drug_patches_dir, split='val',   transform=test_transform)
            test_ds  = Task2PatchDataset(drug_patches_dir, split='test',  transform=test_transform)
        else:
            # Fallback: process patches on-the-fly (chậm hơn)
            from datasets import Task2TrainDataset, Task2TestDataset
            logger.info("  [Task2] Processing patches on-the-fly (no precomputed patches found)")
            train_ds = Task2TrainDataset(
                train_df_model, drug_dir, split='train',
                patch_size=cfg.patch_size,
                content_std_threshold=getattr(cfg, 'task2_content_std_threshold', 1000.0),
                transform=train_transform,
            )
            val_ds = Task2TestDataset(
                val_df, drug_dir, split=val_folder,
                patch_size=cfg.patch_size,
                stride=cfg.test_stride,
                transform=test_transform,
            )
            test_ds = Task2TestDataset(
                test_df, drug_dir, split=test_folder,
                patch_size=cfg.patch_size,
                stride=cfg.test_stride,
                transform=test_transform,
            )
    else:
        raise ValueError(f"Task phải là 1 hoặc 2, nhận: {task}")

    logger.info(
        f"  Samples → Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}"
    )

    # ─── WeightedRandomSampler ────────────────────────────────────────────────
    if hasattr(train_ds, 'samples'):
        train_labels = np.array([s[1] for s in train_ds.samples])
    elif hasattr(train_ds, 'labels'):
        train_labels = np.array(train_ds.labels)
    else:
        train_labels = train_df['resistant'].values.astype(int)
    sampler = get_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.batch_size,
        sampler     = sampler,
        num_workers = cfg.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.batch_size * 2,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg.batch_size * 2,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )

    # ─── Model ────────────────────────────────────────────────────────────────
    set_seed(seed)
    model = get_model(model_name, num_classes=cfg.num_classes, pretrained=True)
    model = model.to(device)

    params = count_parameters(model)
    logger.info(
        f"  Params: {params['trainable']:,} trainable / {params['total']:,} total"
    )

    # ─── Loss function ────────────────────────────────────────────────────────
    label_counts = np.bincount(train_labels.astype(int))
    criterion    = get_loss_fn(
        loss_name    = loss_name,
        class_counts = label_counts,
        device       = device,
        gamma        = getattr(cfg, 'focal_gamma', 2.0),
    )

    # ─── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    # ─── Training loop ────────────────────────────────────────────────────────
    _metric_level    = getattr(cfg, 'best_metric_level', 'cell_level')
    use_image_level  = _metric_level in ('image_level', 'image_level_hard')
    use_hard_level   = _metric_level == 'image_level_hard'
    best_metric_name = getattr(cfg, 'best_metric', 'f1_resistant')
    best_val_score   = -1.0
    patience_counter = 0
    best_ckpt_path   = cfg.checkpoint_path(task, drug, model_name, seed)
    train_history    = []
    best_test_results   = None
    best_opt_threshold  = 0.5
    if _metric_level == 'image_level_hard':
        level_label = "Img-Hard"
    elif _metric_level == 'image_level':
        level_label = "Image-Level"
    else:
        level_label = "Cell-Level"
    if best_metric_name == 'recall_resistant':
        metric_label = "Recall(R)"
    elif best_metric_name == 'f1_macro':
        metric_label = "F1(Macro)"
    elif best_metric_name == 'youden_j':
        metric_label = "YoudenJ"
    else:
        metric_label = "F1(R)"

    logger.info(f"  Best-epoch criterion : Val {metric_label} @ {level_label} ↑")
    logger.info(f"  Checkpoint           : {best_ckpt_path}")
    if use_bf16:
        logger.info("  Precision            : BFloat16 autocast (RTX 4090)")

    for epoch in range(1, cfg.num_epochs + 1):
        t0          = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            use_bf16=use_bf16,
        )
        scheduler.step()

        # ── Evaluate Val: cell-level + image-level ────────────────────────────
        v_true, v_pred, v_prob, v_orig, _ = evaluate_loader(
            model, val_loader, device, use_bf16=use_bf16
        )
        # Calibrate threshold on VAL mỗi epoch (để log Val metric không bị "kẹt" ở 0 do threshold=0.5)
        if threshold_method == 'fixed':
            _thresh_arg_epoch = getattr(cfg, 'threshold_fixed_value', 0.65)
        elif threshold_method == 'min_recall':
            _thresh_arg_epoch = getattr(cfg, 'threshold_min_recall', 0.85)
        else:
            _thresh_arg_epoch = None
        val_opt_threshold = find_optimal_threshold(
            v_true, v_prob, method=threshold_method, min_recall=_thresh_arg_epoch
        )

        val_cell_stats = compute_metrics(v_true, v_pred, v_prob, threshold=val_opt_threshold)
        img_y_true, img_y_pred, img_y_prob, _ = image_level_voting(
            v_orig, v_true, v_pred, v_prob, method='soft', threshold=val_opt_threshold
        )
        val_img_stats = compute_metrics(img_y_true, img_y_pred, img_y_prob, threshold=val_opt_threshold)
        # Hard (majority) vote trên val — dùng khi best_metric_level == 'image_level_hard'
        img_h_true, img_h_pred, img_h_prob, _ = image_level_voting(
            v_orig, v_true, v_pred, v_prob, method='hard', threshold=val_opt_threshold
        )
        val_img_hard_stats = compute_metrics(img_h_true, img_h_pred, img_h_prob, threshold=val_opt_threshold)

        if use_hard_level:
            _active_stats = val_img_hard_stats
        elif use_image_level:
            _active_stats = val_img_stats
        else:
            _active_stats = val_cell_stats

        val_recall_r = _active_stats['recall_resistant']
        val_f1_r     = _active_stats['f1_resistant']
        val_prec_r   = _active_stats['precision_resistant']
        val_auc_val  = _active_stats['auc']
        val_f1_macro = _active_stats['f1_macro']
        val_youden_j = _active_stats['youden_j']

        if best_metric_name == 'recall_resistant':
            val_score = val_recall_r
        elif best_metric_name == 'f1_macro':
            val_score = val_f1_macro
        elif best_metric_name == 'youden_j':
            val_score = val_youden_j
        else:
            val_score = val_f1_r

        elapsed = time.time() - t0
        is_best = val_score > best_val_score
        best_tag = " ★ NEW BEST" if is_best else ""

        logger.info(
            f"  Epoch {epoch:3d}/{cfg.num_epochs} "
            f"| LR: {scheduler.get_last_lr()[0]:.2e} "
            f"| TrainLoss: {train_stats['loss']:.4f} "
            f"| Val[{level_label[:4]}]{metric_label}: {val_score:.4f} "
            f"| ValPrec(R): {val_prec_r:.4f} "
            f"| ValAUC: {val_auc_val:.4f} "
            f"| Thr(val): {val_opt_threshold:.4f} "
            f"| {elapsed:.1f}s{best_tag}"
        )

        train_history.append({
            'epoch':                       epoch,
            'train_loss':                  train_stats['loss'],
            'train_acc':                   train_stats['accuracy'],
            'val_img_recall_resistant':    val_img_stats['recall_resistant'],
            'val_img_precision_resistant': val_img_stats['precision_resistant'],
            'val_img_recall_sensitive':    val_img_stats['recall_sensitive'],
            'val_img_auc':                 val_img_stats['auc'],
            'val_cell_recall_resistant':   val_cell_stats['recall_resistant'],
            'val_cell_auc':                val_cell_stats['auc'],
            'lr':                          scheduler.get_last_lr()[0],
            'is_best':                     is_best,
        })

        if is_best:
            best_val_score   = val_score
            patience_counter = 0

            torch.save({
                'epoch':                             epoch,
                'model_state':                       model.state_dict(),
                'val_img_recall_resistant':          val_img_stats['recall_resistant'],
                'val_img_precision_resistant':       val_img_stats['precision_resistant'],
                'val_img_recall_sensitive':          val_img_stats['recall_sensitive'],
                'val_img_auc':                       val_img_stats['auc'],
                'val_img_hard_recall_resistant':     val_img_hard_stats['recall_resistant'],
                'val_img_hard_precision_resistant':  val_img_hard_stats['precision_resistant'],
                'val_img_hard_recall_sensitive':     val_img_hard_stats['recall_sensitive'],
                'val_img_hard_auc':                  val_img_hard_stats['auc'],
                'val_cell_recall_resistant':         val_cell_stats['recall_resistant'],
                'val_cell_precision_resistant':      val_cell_stats['precision_resistant'],
                'val_cell_recall_sensitive':         val_cell_stats['recall_sensitive'],
                'val_cell_auc':                      val_cell_stats['auc'],
                'best_metric':                       cfg.best_metric,
                'best_metric_level':                 cfg.best_metric_level,
                'seed':                              seed,
                'run_idx':                           run_idx,
                'task':                              task,
                'drug':                              drug,
                'model_name':                        model_name,
            }, best_ckpt_path)

            # ── Threshold calibration trên val ────────────────────────────────
            if threshold_method == 'fixed':
                _thresh_arg = getattr(cfg, 'threshold_fixed_value', 0.65)
            elif threshold_method == 'min_recall':
                _thresh_arg = getattr(cfg, 'threshold_min_recall', 0.85)
            else:
                _thresh_arg = None
            opt_threshold      = find_optimal_threshold(
                v_true, v_prob, method=threshold_method, min_recall=_thresh_arg
            )
            best_opt_threshold = opt_threshold

            # ── Tính metrics trên Test tại best epoch ─────────────────────────
            t_true, t_pred, t_prob, t_orig, _ = evaluate_loader(
                model, test_loader, device, use_bf16=use_bf16
            )
            # Soft vote
            best_test_results = evaluate_full(
                orig_files    = t_orig,
                y_true        = t_true,
                y_pred        = t_pred,
                y_prob        = t_prob,
                voting_method = 'soft',
                threshold     = opt_threshold,
            )
            # Hard (majority) vote
            best_test_results_hard = evaluate_full(
                orig_files    = t_orig,
                y_true        = t_true,
                y_pred        = t_pred,
                y_prob        = t_prob,
                voting_method = 'hard',
                threshold     = opt_threshold,
            )

            cl   = best_test_results['cell_level']
            img  = best_test_results['image_level']
            imgH = best_test_results_hard['image_level']
            logger.info(
                f"\n  ★ Best epoch={epoch} "
                f"| Val[{level_label}]{metric_label}={val_score:.4f} "
                f"| ValPrec(R)={val_prec_r:.4f} "
                f"| Threshold={opt_threshold:.4f} [{threshold_method}]"
                f"\n  ┌─ TEST Cell-Level ─────────────────────────────────────────────"
                f"\n  │  Recall(R)={cl['recall_resistant']:.4f}  "
                f"Prec(R)={cl['precision_resistant']:.4f}  "
                f"F1(R)={cl['f1_resistant']:.4f}"
                f"\n  │  Recall(S)={cl['recall_sensitive']:.4f}  "
                f"Prec(S)={cl['precision_sensitive']:.4f}  "
                f"F1(S)={cl['f1_sensitive']:.4f}"
                f"\n  │  Youden={cl['youden_j']:.4f}  F1(Macro)={cl['f1_macro']:.4f}  "
                f"AUC={cl['auc']:.4f}  Acc={cl['accuracy']:.4f}"
                f"\n  ├─ TEST Image-Level [Soft Vote] ─────────────────────────────────"
                f"\n  │  Recall(R)={img['recall_resistant']:.4f}  "
                f"Prec(R)={img['precision_resistant']:.4f}  "
                f"F1(R)={img['f1_resistant']:.4f}"
                f"\n  │  Recall(S)={img['recall_sensitive']:.4f}  "
                f"Prec(S)={img['precision_sensitive']:.4f}  "
                f"F1(S)={img['f1_sensitive']:.4f}"
                f"\n  │  Youden={img['youden_j']:.4f}  F1(Macro)={img['f1_macro']:.4f}  "
                f"AUC={img['auc']:.4f}  Acc={img['accuracy']:.4f}"
                f"\n  ├─ TEST Image-Level [Hard/Majority Vote] ────────────────────────"
                f"\n  │  Recall(R)={imgH['recall_resistant']:.4f}  "
                f"Prec(R)={imgH['precision_resistant']:.4f}  "
                f"F1(R)={imgH['f1_resistant']:.4f}"
                f"\n  │  Recall(S)={imgH['recall_sensitive']:.4f}  "
                f"Prec(S)={imgH['precision_sensitive']:.4f}  "
                f"F1(S)={imgH['f1_sensitive']:.4f}"
                f"\n  │  Youden={imgH['youden_j']:.4f}  F1(Macro)={imgH['f1_macro']:.4f}  "
                f"AUC={imgH['auc']:.4f}  Acc={imgH['accuracy']:.4f}"
                f"\n  └──────────────────────────────────────────────────────────────\n"
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info(
                    f"  ⚡ Early stopping tại epoch {epoch} "
                    f"(patience={cfg.patience}, "
                    f"best Val[{metric_label}]={best_val_score:.4f})"
                )
                break

    # ─── Tóm tắt run hoàn thành ───────────────────────────────────────────────
    ckpt = torch.load(best_ckpt_path, map_location=device)
    if use_hard_level:
        r, p, s, a = (
            ckpt.get('val_img_hard_recall_resistant', 0),
            ckpt.get('val_img_hard_precision_resistant', 0),
            ckpt.get('val_img_hard_recall_sensitive', 0),
            ckpt.get('val_img_hard_auc', 0),
        )
    elif use_image_level:
        r, p, s, a = (
            ckpt['val_img_recall_resistant'], ckpt['val_img_precision_resistant'],
            ckpt['val_img_recall_sensitive'], ckpt['val_img_auc'],
        )
    else:
        r, p, s, a = (
            ckpt['val_cell_recall_resistant'], ckpt.get('val_cell_precision_resistant', 0),
            ckpt.get('val_cell_recall_sensitive', 0), ckpt.get('val_cell_auc', 0),
        )
    logger.info(
        f"\n  ══ Run {run_idx+1} (seed={seed}) hoàn thành ══"
        f"\n  Best epoch              : {ckpt['epoch']}"
        f"\n  Val [{level_label}] Recall(R)   : {r:.4f}"
        f"\n  Val [{level_label}] Precision(R): {p:.4f}"
        f"\n  Val [{level_label}] Recall(S)   : {s:.4f}"
        f"\n  Val [{level_label}] AUC         : {a:.4f}"
        f"\n  Opt Threshold           : {best_opt_threshold:.4f}"
    )

    print_metrics(
        best_test_results['cell_level'],
        title=f"Run {run_idx+1} (seed={seed}) - TEST Cell-Level (threshold={best_opt_threshold:.3f})"
    )
    print_metrics(
        best_test_results['image_level'],
        title=f"Run {run_idx+1} (seed={seed}) - TEST Image-Level / Soft Vote (threshold={best_opt_threshold:.3f})"
    )
    print_metrics(
        best_test_results_hard['image_level'],
        title=f"Run {run_idx+1} (seed={seed}) - TEST Image-Level / Majority Vote (threshold={best_opt_threshold:.3f})"
    )

    if use_hard_level:
        best_val_recall_r_used  = float(ckpt.get('val_img_hard_recall_resistant', 0.0))
        best_val_prec_r_used    = float(ckpt.get('val_img_hard_precision_resistant', 0.0))
        best_val_recall_s_used  = float(ckpt.get('val_img_hard_recall_sensitive', 0.0))
        best_val_auc_used       = float(ckpt.get('val_img_hard_auc', 0.0))
    elif use_image_level:
        best_val_recall_r_used  = float(ckpt['val_img_recall_resistant'])
        best_val_prec_r_used    = float(ckpt['val_img_precision_resistant'])
        best_val_recall_s_used  = float(ckpt['val_img_recall_sensitive'])
        best_val_auc_used       = float(ckpt.get('val_img_auc', 0.0))
    else:
        best_val_recall_r_used  = float(ckpt['val_cell_recall_resistant'])
        best_val_prec_r_used    = float(ckpt.get('val_cell_precision_resistant', 0.0))
        best_val_recall_s_used  = float(ckpt.get('val_cell_recall_sensitive', 0.0))
        best_val_auc_used       = float(ckpt.get('val_cell_auc', 0.0))

    return {
        'run':                           run_idx + 1,
        'seed':                          seed,
        'best_epoch':                    int(ckpt['epoch']),
        'best_metric_level':             cfg.best_metric_level,
        'best_val_recall_resistant':     best_val_recall_r_used,
        'best_val_precision_resistant':  best_val_prec_r_used,
        'best_val_recall_sensitive':     best_val_recall_s_used,
        'best_val_auc':                  best_val_auc_used,
        'opt_threshold':                 float(best_opt_threshold),
        'cell_level':                    best_test_results['cell_level'],
        'image_level':                   best_test_results['image_level'],
        'image_level_hard':              best_test_results_hard['image_level'],
        'train_history':                 train_history,
        'checkpoint':                    best_ckpt_path,
    }


# ─── Main multi-seed runner ───────────────────────────────────────────────────

def run_multiseed(
    task:             int,
    drug:             str,
    model_name:       str,
    cfg:              Config,
    loss_name:        str = 'focal',
    threshold_method: str = 'f1',
    use_bf16:         bool = False,
) -> Dict:
    """
    Chạy N lần với các random seed khác nhau trên train/val/test split cố định.

    Thay thế StratifiedKFold: splits đã được định nghĩa sẵn trong *_joblib folders.
    Sự khác biệt giữa các runs đến từ: model init, data augmentation, sampler.

    Args:
        task             : 1 hoặc 2
        drug             : 'AZI', 'CIP', 'TE'
        model_name       : Tên model trong timm
        cfg              : Config object
        loss_name        : 'focal' hoặc 'ce'
        threshold_method : 'f1', 'balanced', 'min_recall', 'pr_equal', 'gmean', hoặc 'fixed'
    """
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu'
    )

    logger.info(f"\n{'═'*65}")
    logger.info(
        f"  Task {task} | Drug: {drug} | Model: {model_name} | Loss: {loss_name}"
    )
    logger.info(
        f"  Device: {device} | Seeds: {cfg.seeds} | Threshold: {threshold_method}"
    )
    logger.info(f"{'═'*65}")

    # ─── Load split_info.csv ──────────────────────────────────────────────────
    csv_path = cfg.split_csv(drug)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy: {csv_path}")

    full_df  = pd.read_csv(csv_path)
    train_df = full_df[full_df['split'] == 'train'].reset_index(drop=True)
    swap = getattr(cfg, 'swap_val_test', True)
    if swap:
        # Hoán đổi vai trò: CSV split='test' → Val | CSV split='val' → Test
        val_df  = full_df[full_df['split'] == 'test'].reset_index(drop=True)
        test_df = full_df[full_df['split'] == 'val' ].reset_index(drop=True)
        val_src, test_src = 'CSV.test', 'CSV.val'
    else:
        val_df  = full_df[full_df['split'] == 'val' ].reset_index(drop=True)
        test_df = full_df[full_df['split'] == 'test'].reset_index(drop=True)
        val_src, test_src = 'CSV.val', 'CSV.test'

    logger.info(f"  Loaded CSV: {len(full_df):,} rows từ {csv_path}")
    if swap:
        logger.info("  [Swap] Val ← CSV.test | Test ← CSV.val")
    else:
        logger.info("  [No swap] Val ← CSV.val | Test ← CSV.test")

    # Đếm nhãn theo tên cột phù hợp
    label_col = 'resistant' if 'resistant' in full_df.columns else 'resistance_label'
    logger.info(
        f"  Train: {len(train_df):,} cells "
        f"(R={(train_df[label_col]==1).sum():,}, "
        f"S={(train_df[label_col]==0).sum():,})"
    )
    logger.info(
        f"  Val:   {len(val_df):,} cells (từ {val_src}) "
        f"(R={(val_df[label_col]==1).sum():,}, "
        f"S={(val_df[label_col]==0).sum():,})"
    )
    logger.info(
        f"  Test:  {len(test_df):,} cells (từ {test_src}) "
        f"(R={(test_df[label_col]==1).sum():,}, "
        f"S={(test_df[label_col]==0).sum():,})"
    )
    if task == 1 and (
        getattr(cfg, 'train_downsample_balance_classes', False)
        or (
            getattr(cfg, 'train_downsample_fraction', None) is not None
            and getattr(cfg, 'train_downsample_fraction', 1.0) < 1.0
        )
    ):
        logger.info(
            "  [Train downsample] Bật trong train_one_run (mỗi seed một subset) — "
            "số train ở trên là CSV gốc."
        )

    # ─── Multi-seed runs ──────────────────────────────────────────────────────
    run_results = []

    for run_idx, seed in enumerate(cfg.seeds):
        run_result = train_one_run(
            run_idx          = run_idx,
            seed             = seed,
            task             = task,
            drug             = drug,
            model_name       = model_name,
            train_df         = train_df,
            val_df           = val_df,
            test_df          = test_df,
            cfg              = cfg,
            device           = device,
            loss_name        = loss_name,
            threshold_method = threshold_method,
            use_bf16         = use_bf16,
        )
        run_results.append(run_result)

    # ─── Tổng hợp kết quả tất cả seeds ───────────────────────────────────────
    summary = summarize_folds(
        run_results,
        levels=('cell_level', 'image_level', 'image_level_hard'),
    )
    print_fold_summary(
        summary,
        title=f"TỔNG HỢP {cfg.n_seeds} SEEDS | Task{task} | {drug} | {model_name}"
    )

    # ─── Lưu kết quả ─────────────────────────────────────────────────────────
    result_dir = cfg.result_dir(task, drug, model_name)

    _save_run_metrics_csv(run_results, task, drug, model_name, result_dir)

    final_output = {
        'task':             task,
        'drug':             drug,
        'model_name':       model_name,
        'seeds':            cfg.seeds,
        'n_seeds':          cfg.n_seeds,
        'loss':             loss_name,
        'threshold_method': threshold_method,
        'num_epochs':       cfg.num_epochs,
        'run_results':      run_results,
        'summary':          summary,
    }
    _save_json(final_output, os.path.join(result_dir, 'results.json'))
    logger.info(f"  ✓ results.json saved → {result_dir}")

    return final_output


# ─── Per-run CSV saver ────────────────────────────────────────────────────────

def _save_run_metrics_csv(
    run_results: List[Dict],
    task:        int,
    drug:        str,
    model_name:  str,
    result_dir:  str,
):
    """
    Lưu per-run metrics vào CSV (mỗi run = 1 seed, 2 rows: cell_level + image_level).

    Format:
        task | drug | model | run | seed | level | threshold |
        recall_resistant | precision_resistant | f1_resistant |
        recall_sensitive | f1_macro | f1_weighted | auc | accuracy | mcc
    """
    rows = []
    for rr in run_results:
        for level in ['cell_level', 'image_level']:
            m = rr[level]
            row = {
                'task':                        task,
                'drug':                        drug,
                'model':                       model_name,
                'run':                         rr['run'],
                'seed':                        rr['seed'],
                'level':                       level,
                'best_epoch':                  rr['best_epoch'],
                'best_val_recall_resistant':   rr['best_val_recall_resistant'],
                'best_val_precision_resistant':rr['best_val_precision_resistant'],
                'best_val_recall_sensitive':   rr['best_val_recall_sensitive'],
                'best_val_auc':                rr['best_val_auc'],
                'opt_threshold':               rr['opt_threshold'],
                'threshold':                   m.get('threshold', 0.5),
                'recall_resistant':            m['recall_resistant'],
                'precision_resistant':         m['precision_resistant'],
                'f1_resistant':                m['f1_resistant'],
                'recall_sensitive':            m['recall_sensitive'],
                'youden_j':                    m['youden_j'],
                'f1_macro':                    m['f1_macro'],
                'f1_weighted':                 m['f1_weighted'],
                'auc':                         m['auc'],
                'accuracy':                    m['accuracy'],
                'mcc':                         m['mcc'],
                'tp':                          m['tp'],
                'tn':                          m['tn'],
                'fp':                          m['fp'],
                'fn':                          m['fn'],
                'n_total':                     m['n_total'],
                'n_resistant':                 m['n_resistant'],
                'n_sensitive':                 m['n_sensitive'],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(result_dir, 'run_metrics.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ run_metrics.csv saved → {csv_path}")
    return df


# ─── JSON saver (handle numpy types) ─────────────────────────────────────────

def _save_json(obj, path: str):
    def _convert(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return o

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_convert)


# ─── Run toàn bộ experiment ───────────────────────────────────────────────────

def run_experiment(
    cfg:              Config,
    tasks:            Optional[List[int]] = None,
    drugs:            Optional[List[str]] = None,
    model_names:      Optional[List[str]] = None,
    loss_name:        str = 'focal',
    threshold_method: str = 'f1',
    use_bf16:         bool = False,
) -> Dict:
    """
    Chạy toàn bộ experiment cho nhiều combinations (task × drug × model).

    Mỗi combination chạy n_seeds lần → mean ± std trên test set.
    """
    if tasks       is None: tasks       = [1, 2]
    if drugs       is None: drugs       = cfg.drugs
    if model_names is None: model_names = cfg.model_names

    all_results = {}
    total   = len(tasks) * len(drugs) * len(model_names)
    counter = 0

    logger.info(f"\n{'█'*65}")
    logger.info(f"  ANTIBIOTIC RESISTANCE CLASSIFICATION FRAMEWORK")
    logger.info(f"  Tasks: {tasks} | Drugs: {drugs}")
    logger.info(f"  Models: {model_names}")
    logger.info(f"  Seeds: {cfg.seeds} ({cfg.n_seeds} runs/combination)")
    logger.info(f"  Loss: {loss_name} | Threshold: {threshold_method}")
    logger.info(f"  Tổng số combinations: {total} ({total * cfg.n_seeds} runs)")
    logger.info(f"{'█'*65}")

    for task in tasks:
        for drug in drugs:
            for model_name in model_names:
                counter += 1
                key = f"task{task}_{drug}_{model_name}"
                logger.info(f"\n[{counter}/{total}] Bắt đầu: {key}")

                try:
                    result = run_multiseed(
                        task             = task,
                        drug             = drug,
                        model_name       = model_name,
                        cfg              = cfg,
                        loss_name        = loss_name,
                        threshold_method = threshold_method,
                        use_bf16         = use_bf16,
                    )
                    all_results[key] = result
                    logger.info(f"  ✓ Hoàn thành: {key}")

                except Exception as e:
                    logger.error(f"  ✗ Lỗi khi chạy {key}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[key] = {'error': str(e)}

    os.makedirs(cfg.output_dir, exist_ok=True)
    _save_json(all_results, os.path.join(cfg.output_dir, 'all_results.json'))

    _merge_run_csvs(cfg, tasks, drugs, model_names)

    logger.info(f"\n{'█'*65}")
    logger.info(f"  ✓ Hoàn thành tất cả {total} combinations")
    logger.info(f"  ✓ Kết quả: {cfg.output_dir}")
    logger.info(f"{'█'*65}")

    return all_results


def _merge_run_csvs(
    cfg:         Config,
    tasks:       List[int],
    drugs:       List[str],
    model_names: List[str],
):
    """Gộp tất cả run_metrics.csv vào một file master CSV."""
    all_dfs = []
    for task in tasks:
        for drug in drugs:
            for model_name in model_names:
                csv_path = os.path.join(
                    cfg.result_dir(task, drug, model_name), 'run_metrics.csv'
                )
                if os.path.exists(csv_path):
                    all_dfs.append(pd.read_csv(csv_path))

    if all_dfs:
        master_df   = pd.concat(all_dfs, ignore_index=True)
        master_path = os.path.join(cfg.output_dir, 'all_run_metrics.csv')
        master_df.to_csv(master_path, index=False)
        logger.info(f"  ✓ Master run metrics → {master_path}")
        logger.info(f"     Tổng {len(master_df)} rows "
                    f"({len(master_df)//2} run-results)")
