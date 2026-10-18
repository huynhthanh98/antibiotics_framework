"""
CLI Entry Point - Antibiotic Resistance Classification Framework

Dùng *_joblib folders với train/val/test splits có sẵn.
Chạy N seeds (mặc định 5) → tính mean ± std trên test set.

Usage examples:
  # Chạy Task 1, tất cả drugs, tất cả models:
  python main.py --task 1

  # Chạy Task 2, chỉ drug AZI, model ResNet50:
  python main.py --task 2 --drug AZI --model resnet50

  # Chạy cả Task 1 và 2, drug CIP, model EfficientNet-B0:
  python main.py --task 1 2 --drug CIP --model efficientnet_b0

  # Chạy toàn bộ (tất cả task, drug, model):
  python main.py

  # Tuỳ chỉnh hyperparameters:
  python main.py --task 1 --drug AZI --model resnet50 \\
                 --epochs 50 --batch_size 64 --lr 5e-5 \\
                 --seeds 0 1 2 3 4 --device cuda

  # Xem danh sách model trong timm:
  python main.py --list_models resnet
"""

import argparse
import sys
import os

# Đảm bảo import từ cùng thư mục
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from trainer import run_experiment, run_multiseed, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Antibiotic Resistance Classification Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ─── Task và Data ─────────────────────────────────────────────────────────
    parser.add_argument(
        '--task', type=int, nargs='+', choices=[1, 2], default=[1, 2],
        help='Task(s) cần chạy: 1 (masked cells) và/hoặc 2 (original 224x224 crops). '
             'Default: 1 2 (cả hai)',
    )
    parser.add_argument(
        '--drug', type=str, nargs='+',
        choices=['AZI', 'CIP', 'TE', 'all'],
        default=['all'],
        help='Loại thuốc cần chạy. Default: all (AZI, CIP, TE)',
    )
    parser.add_argument(
        '--model', type=str, nargs='+',
        default=None,
        help='Tên model(s) trong timm. '
             'Ví dụ: resnet50 vgg16 convnext_small densenet121 efficientnet_b0. '
             'Default: dùng danh sách trong config',
    )

    # ─── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument('--epochs',     type=int,   default=None,
                        help='Số epoch. Default: 30')
    parser.add_argument('--batch_size', type=int,   default=None,
                        help='Batch size. Default: 32')
    parser.add_argument('--lr',         type=float, default=None,
                        help='Learning rate. Default: 1e-4')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay. Default: 1e-4')
    parser.add_argument('--patience',   type=int,   default=None,
                        help='Early stopping patience (epochs). Default: 5')
    parser.add_argument('--seeds',      type=int,   nargs='+', default=None,
                        help='Danh sách random seeds. Default: 0 1 2 3 4')
    parser.add_argument('--num_workers', type=int,  default=None,
                        help='DataLoader workers. Default: 4')

    # ─── Task 2 specific ──────────────────────────────────────────────────────
    parser.add_argument('--task2_content_std_threshold', type=float, default=None,
                        help='[Task2] Nguong std dev (uint16) de loc patch background. '
                             'Patch co std < nguong bi bo qua. Default: 1000.0')
    parser.add_argument('--test_stride', type=int, default=None,
                        help='[Task2] Stride sliding window khi test (224=no overlap). Default: 224')
    parser.add_argument('--task2_patches_dir', type=str, default=None,
                        help='[Task2] Thu muc chua patches da export (e.g., /path/task2_patches). '
                             'Neu chi dinh, se load patches tu day thay vi process on-the-fly.')

    # ─── Paths ────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--data_root', type=str, default=None,
        help='Thư mục gốc chứa AZI_joblib*/, CIP_joblib*/, TE_joblib*/. '
             'Default: thư mục gốc của repo (cha của antibiotics_framework/)',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Thư mục lưu kết quả. Default: {data_root}/results',
    )
    parser.add_argument(
        '--joblib_suffix', type=str, default=None,
        help='Suffix cho drug folders (e.g., "_joblib", "_joblib_cells"). Default: _joblib',
    )
    parser.add_argument(
        '--task1_image_folder', type=str, default=None,
        help='Folder chứa ảnh Task1 (e.g., "masked", "cell"). Default: masked',
    )

    # ─── Device ───────────────────────────────────────────────────────────────
    parser.add_argument(
        '--device', type=str, choices=['cuda', 'cpu'], default=None,
        help='Device: cuda hoặc cpu. Default: cuda',
    )

    # ─── Imbalance handling ───────────────────────────────────────────────────
    parser.add_argument(
        '--loss', type=str, choices=['focal', 'ce'], default='focal',
        help='Loss function: focal (Focal Loss) hoặc ce (CrossEntropy). Default: focal',
    )
    parser.add_argument(
        '--threshold_method', type=str,
        choices=['f1', 'balanced', 'min_recall', 'pr_equal', 'gmean', 'fixed'], default='f1',
        help='Threshold: f1, balanced, min_recall, pr_equal (P≈R), gmean, fixed. Default: f1',
    )
    parser.add_argument(
        '--threshold_fixed_value', type=float, default=None,
        help='Chỉ dùng khi --threshold_method fixed. Giá trị hard threshold (e.g. 0.7). Default: 0.7',
    )
    parser.add_argument(
        '--threshold_min_recall', type=float, default=None,
        help='Chỉ dùng khi --threshold_method min_recall. Recall tối thiểu (e.g. 0.85). Default: 0.85',
    )
    parser.add_argument(
        '--focal_gamma', type=float, default=2.0,
        help='Focal Loss gamma. Default: 2.0',
    )
    parser.add_argument(
        '--best_metric', type=str,
        choices=['f1_resistant', 'recall_resistant', 'f1_macro', 'youden_j'], default=None,
        help='Tiêu chí chọn best epoch trên val: f1_resistant | recall_resistant | f1_macro | youden_j. Default: f1_resistant',
    )
    parser.add_argument(
        '--best_metric_level', type=str,
        choices=['cell_level', 'image_level', 'image_level_hard'], default=None,
        help='Level tính best metric: cell_level, image_level (soft vote) hoặc image_level_hard (majority vote). Default: cell_level',
    )
    parser.add_argument(
        '--train_downsample_fraction', type=float, default=None,
        help='[Task1] Stratified: mỗi lớp giữ ~fraction×số cell (0<f≤1). Mỗi seed một subset ngẫu nhiên. '
             'Nếu kèm --train_downsample_balance_classes thì chỉ áp dụng balance.',
    )
    parser.add_argument(
        '--train_downsample_balance_classes', action='store_true',
        help='[Task1] Undersample lớp đa số: R và S cùng = min(n_R, n_S) cells. Mỗi seed một subset.',
    )
    parser.add_argument(
        '--no_swap_val_test', action='store_true',
        help='Không hoán đổi val/test: Val ← CSV.val, Test ← CSV.test (mặc định: hoán đổi)',
    )

    # ─── T-test ───────────────────────────────────────────────────────────────
    parser.add_argument(
        '--run_ttest', action='store_true',
        help='Chạy t-test analysis sau khi train xong',
    )
    parser.add_argument(
        '--ttest_only', action='store_true',
        help='Chỉ chạy t-test (không train), dùng kết quả đã có',
    )
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='Mức ý nghĩa cho t-test. Default: 0.05',
    )

    # ─── Utilities ────────────────────────────────────────────────────────────
    parser.add_argument(
        '--list_models', type=str, default=None, metavar='PATTERN',
        help='Liệt kê các model trong timm khớp với pattern. Ví dụ: --list_models resnet',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Chỉ kiểm tra cấu hình và data, không train',
    )
    parser.add_argument(
        '--pretrained', action='store_true', default=True,
        help='Dùng ImageNet pretrained weights. Default: True',
    )
    parser.add_argument(
        '--bf16', action='store_true', default=False,
        help='Bật BFloat16 autocast (RTX 30xx/40xx, A100+). Tăng tốc ~1.5-2x, giảm VRAM ~40%%. Default: False',
    )
    parser.add_argument(
        '--mask_erosion_aug', action='store_true', default=False,
        help='[Task1] Bật erosion augmentation khi train — mô phỏng mask kém để tăng robustness. Default: False',
    )
    parser.add_argument(
        '--mask_erosion_max_px', type=int, default=3,
        help='[Task1] Max erosion pixels khi dùng --mask_erosion_aug (ngẫu nhiên 1–max_px, p=0.5). Default: 3',
    )

    return parser.parse_args()


def build_config(args) -> Config:
    """Tạo Config từ command line arguments."""
    cfg = Config()

    # Override với args nếu được chỉ định
    if args.data_root    is not None: cfg.data_root    = args.data_root
    if args.output_dir   is not None: cfg.output_dir   = args.output_dir
    if args.joblib_suffix is not None: cfg.joblib_suffix = args.joblib_suffix
    if args.task1_image_folder is not None: cfg.task1_image_folder = args.task1_image_folder
    if args.epochs       is not None: cfg.num_epochs   = args.epochs
    if args.batch_size   is not None: cfg.batch_size   = args.batch_size
    if args.lr           is not None: cfg.lr           = args.lr
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.patience     is not None: cfg.patience     = args.patience
    if args.seeds        is not None: cfg.seeds        = args.seeds
    if args.num_workers  is not None: cfg.num_workers  = args.num_workers
    if args.device       is not None: cfg.device       = args.device
    if getattr(args, 'task2_content_std_threshold', None) is not None:
        cfg.task2_content_std_threshold = args.task2_content_std_threshold
    if args.test_stride is not None: cfg.test_stride = args.test_stride
    if getattr(args, 'task2_patches_dir', None) is not None:
        cfg.task2_patches_dir = args.task2_patches_dir

    # Focal Loss gamma
    cfg.focal_gamma = args.focal_gamma
    if getattr(args, 'best_metric', None) is not None:
        cfg.best_metric = args.best_metric
    if getattr(args, 'best_metric_level', None) is not None:
        cfg.best_metric_level = args.best_metric_level
    if getattr(args, 'threshold_fixed_value', None) is not None:
        cfg.threshold_fixed_value = args.threshold_fixed_value
    if getattr(args, 'threshold_min_recall', None) is not None:
        cfg.threshold_min_recall = args.threshold_min_recall

    # Models
    if args.model is not None:
        cfg.model_names = args.model

    # BF16
    cfg.use_bf16 = getattr(args, 'bf16', False)

    # Erosion augmentation (Task 1)
    cfg.mask_erosion_aug    = getattr(args, 'mask_erosion_aug',    False)
    cfg.mask_erosion_max_px = getattr(args, 'mask_erosion_max_px', 3)

    if getattr(args, 'train_downsample_fraction', None) is not None:
        cfg.train_downsample_fraction = args.train_downsample_fraction
    if getattr(args, 'train_downsample_balance_classes', False):
        cfg.train_downsample_balance_classes = True

    cfg.swap_val_test = not getattr(args, 'no_swap_val_test', False)

    # Output dir default
    if args.output_dir is None:
        cfg.output_dir = os.path.join(cfg.data_root, 'results')

    return cfg


def main():
    args = parse_args()

    # ─── List models mode ─────────────────────────────────────────────────────
    if args.list_models is not None:
        from models import list_available_models
        models = list_available_models(args.list_models)
        print(f"\nModels trong timm khớp với '{args.list_models}':")
        for m in models[:50]:
            print(f"  {m}")
        if len(models) > 50:
            print(f"  ... và {len(models)-50} models khác")
        return

    # ─── Build config ─────────────────────────────────────────────────────────
    cfg = build_config(args)
    set_seed(cfg.seeds[0] if cfg.seeds else 0)

    # ─── Resolve drugs ────────────────────────────────────────────────────────
    if 'all' in args.drug:
        drugs = cfg.drugs  # ['AZI', 'CIP', 'TE']
    else:
        drugs = args.drug

    tasks       = args.task
    model_names = cfg.model_names

    # ─── T-test only mode ────────────────────────────────────────────────────
    if args.ttest_only:
        print("\n[T-TEST ONLY] Đang chạy t-test analysis trên kết quả đã có...\n")
        from ttest_analysis import run_ttest_analysis
        run_ttest_analysis(
            results_dir = cfg.output_dir,
            drugs       = None if 'all' in args.drug else drugs,
            model_names = model_names if args.model else None,
            alpha       = args.alpha,
        )
        return

    # ─── Print config ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ANTIBIOTIC RESISTANCE CLASSIFICATION FRAMEWORK")
    print("═" * 60)
    print(f"  Data root   : {cfg.data_root}")
    print(f"  Output dir  : {cfg.output_dir}")
    print(f"  Dataset     : *{cfg.joblib_suffix} folders, Task1 images in '{cfg.task1_image_folder}/'")
    print(f"  Tasks       : {tasks}")
    print(f"  Drugs       : {drugs}")
    print(f"  Models      : {model_names}")
    print(f"  Seeds       : {cfg.seeds} ({cfg.n_seeds} runs)")
    print(f"  Epochs      : {cfg.num_epochs}")
    print(f"  Batch size  : {cfg.batch_size}")
    print(f"  LR          : {cfg.lr}")
    print(f"  Device      : {cfg.device}")
    print(f"  Img size    : {cfg.img_size}")
    print(f"  Loss        : {args.loss} (γ={args.focal_gamma})")
    print(f"  Best metric : {cfg.best_metric} @ {cfg.best_metric_level}")
    print(f"  Threshold   : {args.threshold_method}")
    print(f"  Precision   : {'BFloat16 (bf16)' if cfg.use_bf16 else 'Float32 (fp32)'}")
    print(f"  ErosionAug  : {cfg.mask_erosion_aug} (max_px={cfg.mask_erosion_max_px})")
    print(f"  Task2 std threshold  : {getattr(cfg, 'task2_content_std_threshold', 1000.0)}")
    print(f"  Task2 stride  (test) : {cfg.test_stride}")
    _ds_bal = getattr(cfg, 'train_downsample_balance_classes', False)
    _ds_fr = getattr(cfg, 'train_downsample_fraction', None)
    if _ds_bal:
        print("  Train downsample     : balance_classes (1:1 min count per seed)")
    elif _ds_fr is not None:
        print(f"  Train downsample     : fraction={_ds_fr} (stratified per seed)")
    else:
        print("  Train downsample     : off")
    print("═" * 60)

    # ─── Dry run: kiểm tra data ───────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Kiểm tra cấu hình và dữ liệu...\n")
        _dry_run_check(cfg, tasks, drugs, model_names)
        return

    # ─── Run experiments ──────────────────────────────────────────────────────
    run_experiment(
        cfg              = cfg,
        tasks            = tasks,
        drugs            = drugs,
        model_names      = model_names,
        loss_name        = args.loss,
        threshold_method = args.threshold_method,
        use_bf16         = cfg.use_bf16,
    )

    # ─── T-test sau khi train xong ────────────────────────────────────────────
    if args.run_ttest and 1 in tasks and 2 in tasks:
        print("\n" + "═"*60)
        print("  CHẠY T-TEST ANALYSIS...")
        print("═"*60)
        from ttest_analysis import run_ttest_analysis
        run_ttest_analysis(
            results_dir = cfg.output_dir,
            drugs       = None if 'all' in args.drug else drugs,
            model_names = model_names if args.model else None,
            alpha       = args.alpha,
        )


def _dry_run_check(cfg: Config, tasks: list, drugs: list, model_names: list):
    """Kiểm tra tất cả paths và in thống kê data."""
    import pandas as pd
    import torch

    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")

    try:
        import timm
        print(f"  timm version    : {timm.__version__}")
    except ImportError:
        print("  timm            : NOT installed (sẽ dùng torchvision)")

    print()

    for drug in drugs:
        drug_dir = cfg.drug_dir(drug)
        csv_path = cfg.split_csv(drug)

        print(f"  [{drug}] Drug dir: {drug_dir}")
        print(f"         CSV path: {csv_path}")
        print(f"         Dir exists: {os.path.exists(drug_dir)}")
        print(f"         CSV exists: {os.path.exists(csv_path)}")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            label_col = 'resistant' if 'resistant' in df.columns else 'resistance_label'
            orig_col  = 'original_tiff' if 'original_tiff' in df.columns else 'original_file'
            for sp in ['train', 'val', 'test']:
                sp_df = df[df['split'] == sp]
                if len(sp_df) == 0:
                    continue
                print(f"         {sp:5s} cells: {len(sp_df):,} "
                      f"(R={(sp_df[label_col]==1).sum():,}, "
                      f"S={(sp_df[label_col]==0).sum():,}) "
                      f"| {sp_df[orig_col].nunique()} unique imgs")

        # Kiểm tra thư mục
        for split in ['train', 'val', 'test']:
            for label in ['resistant', 'sensitive']:
                for folder in [cfg.task1_image_folder, 'original']:
                    p = os.path.join(drug_dir, split, label, folder)
                    exists = os.path.exists(p)
                    if exists:
                        n = len([f for f in os.listdir(p)
                                 if os.path.isfile(os.path.join(p, f))])
                        print(f"         {split}/{label}/{folder}: {n:,} files")
                    else:
                        print(f"         {split}/{label}/{folder}: NOT FOUND")
        print()

    # Kiểm tra models
    print("  Models:")
    from models import get_model
    for m in model_names:
        try:
            model = get_model(m, num_classes=2, pretrained=False)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"    ✓ {m:<30} ({n_params:.1f}M params)")
            del model
        except Exception as e:
            print(f"    ✗ {m:<30} ERROR: {e}")


if __name__ == '__main__':
    main()

