"""
Configuration cho Antibiotic Resistance Classification Framework.
Hỗ trợ Task 1 (masked cell images) và Task 2 (original images với 224x224 crops).

Sử dụng *_joblib folders với train/val/test splits có sẵn.
Thay vì StratifiedKFold, chạy N_SEEDS lần với random seed khác nhau → mean ± std.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Thư mục gốc của repo (cha của antibiotics_framework/); dữ liệu mặc định nằm ở đây.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Default seeds ────────────────────────────────────────────────────────────
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


@dataclass
class Config:
    """Toàn bộ cấu hình của framework."""

    # ─── Paths ────────────────────────────────────────────────────────────────
    data_root: str = REPO_ROOT
    output_dir: str = os.path.join(REPO_ROOT, 'results')
    
    # ─── Dataset structure ───────────────────────────────────────────────────
    joblib_suffix: str = '_joblib'          # Suffix cho drug folders (e.g., '_joblib', '_joblib_cells')
    task1_image_folder: str = 'masked'      # Folder chứa ảnh Task1 (e.g., 'masked', 'cell')

    # ─── Multi-seed runs (thay cho KFold) ─────────────────────────────────────
    # Chạy n_seeds lần với seed khác nhau, mỗi lần dùng cùng train/val/test split
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    # ─── Training hyperparameters ─────────────────────────────────────────────
    num_epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    img_size: int = 224                                                                                                                                                                                                                           
    num_workers: int = 4
    patience: int = 5      # Early stopping: số epoch không cải thiện metric


    # ─── Device ───────────────────────────────────────────────────────────────
    device: str = 'cuda'        # 'cuda' hoặc 'cpu'

    # ─── Mixed precision ──────────────────────────────────────────────────────
    use_bf16: bool = False      # BFloat16 autocast (RTX 30xx/40xx, A100+)

    # ─── Erosion augmentation (Task 1) ────────────────────────────────────────
    mask_erosion_aug: bool = False   # Bật erosion aug khi train Task 1
    mask_erosion_max_px: int = 3     # Max erosion pixels (random 1–max_px, p=0.5)

    # ─── Task 2 specific ──────────────────────────────────────────────────────
    patch_size: int = 224
    test_stride: int = 224
    # Ngưỡng std dev (uint16) để lọc patch background khi build train dataset.
    # Anh 16-bit: ~1000 | Anh 8-bit RGB: ~5
    task2_content_std_threshold: float = 1000.0
    # Thư mục chứa patches đã export (nếu None → process on-the-fly)
    task2_patches_dir: str = None

    # ─── Drugs ────────────────────────────────────────────────────────────────
    drugs: List[str] = field(default_factory=lambda: ['AZI', 'CIP', 'TE'])

    # ─── Models ───────────────────────────────────────────────────────────────
    model_names: List[str] = field(default_factory=lambda: [
        'resnet50',
        'vgg16',
        'convnext_small',
        'densenet121',
        'efficientnet_b0',
    ])

    # ─── Labels ───────────────────────────────────────────────────────────────
    num_classes: int = 2
    class_names: List[str] = field(default_factory=lambda: ['sensitive', 'resistant'])

    # ─── Best-model criterion ─────────────────────────────────────────────────
    best_metric: str = 'f1_resistant'       # 'recall_resistant' | 'f1_resistant' | 'f1_macro' | 'youden_j'
    best_metric_level: str = 'image_level'   # 'image_level' | 'image_level_hard' | 'cell_level'

    # ─── Threshold calibration ────────────────────────────────────────────────
    threshold_fixed_value: float = 0.5
    threshold_min_recall: float  = 0.85

    # ─── Task 1: undersample tập train (thử giảm lệch lớp / kích thước) ───────
    # balance_classes=True → mỗi lớp giữ min(n_R, n_S) cell (1:1), seed quyết subset.
    # fraction ∈ (0,1] → mỗi lớp giữ floor(n * fraction); nếu cả balance thì ưu tiên balance.
    train_downsample_balance_classes: bool = False
    train_downsample_fraction: Optional[float] = None

    # ─── Val/Test split mapping ───────────────────────────────────────────────
    # True (mặc định): Val ← CSV.test, Test ← CSV.val (hoán đổi vai trò)
    # False: Val ← CSV.val, Test ← CSV.test (theo tên split gốc)
    swap_val_test: bool = True

    # ─── Helper methods ───────────────────────────────────────────────────────
    def drug_dir(self, drug: str) -> str:
        """Trả về thư mục gốc của từng loại thuốc (dùng joblib_suffix)."""
        return os.path.join(self.data_root, f'{drug}{self.joblib_suffix}')

    def split_csv(self, drug: str) -> str:
        """Trả về đường dẫn file split_info.csv của từng loại thuốc."""
        return os.path.join(self.drug_dir(drug), 'split_info.csv')

    def result_dir(self, task: int, drug: str, model_name: str) -> str:
        """Tạo và trả về thư mục lưu kết quả."""
        path = os.path.join(self.output_dir, f'task{task}', drug, model_name)
        os.makedirs(path, exist_ok=True)
        return path

    def checkpoint_path(self, task: int, drug: str, model_name: str, seed: int) -> str:
        """Đường dẫn lưu model weights tốt nhất cho mỗi run."""
        d = self.result_dir(task, drug, model_name)
        return os.path.join(d, f'seed{seed}_best.pth')

