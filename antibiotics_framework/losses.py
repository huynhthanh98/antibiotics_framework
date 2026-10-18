"""
Loss functions cho Antibiotic Resistance Classification Framework.

Bao gồm:
  1. FocalLoss        - focus vào hard/minority examples, γ=2 (mặc định)
  2. WeightedFocalLoss - Focal Loss với class weights α
  3. get_loss_fn()    - factory function chọn loss theo tên
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss cho binary/multiclass classification.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Tác dụng:
      - (1 - p_t)^γ : giảm contribution của easy examples (p_t cao)
      - Focus model vào hard examples và minority class
      - γ=0 → equivalent với CrossEntropyLoss thông thường
      - γ=2 : giá trị chuẩn từ paper "Focal Loss for Dense Object Detection"

    Args:
        alpha    : Class weights tensor shape (num_classes,) hoặc None.
                   None → không dùng class weights.
        gamma    : Focusing parameter. Default=2.0
        reduction: 'mean', 'sum', hoặc 'none'. Default='mean'

    References:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        alpha:     Optional[torch.Tensor] = None,
        gamma:     float = 2.0,
        reduction: str   = 'mean',
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs  : Logits (unnormalized), shape (N, C)
            targets : Class indices, shape (N,)
        Returns:
            Scalar loss
        """
        # Bước 1: CE loss per sample, không reduction
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight    = self.alpha,
            reduction = 'none',
        )  # shape: (N,)

        # Bước 2: p_t = exp(-CE) = xác suất của lớp đúng
        p_t = torch.exp(-ce_loss)

        # Bước 3: Focal weight = (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma

        # Bước 4: Focal loss
        focal_loss = focal_weight * ce_loss  # shape: (N,)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss  # 'none'

    def extra_repr(self) -> str:
        return f'gamma={self.gamma}, reduction={self.reduction}'


# ─── Factory function ─────────────────────────────────────────────────────────

def get_loss_fn(
    loss_name:     str,
    class_counts:  Optional[np.ndarray] = None,
    device:        torch.device = torch.device('cpu'),
    gamma:         float = 2.0,
) -> nn.Module:
    """
    Tạo loss function theo tên.

    Args:
        loss_name    : 'focal' hoặc 'ce' (cross entropy)
        class_counts : Array [n_class0, n_class1] để tính class weights.
                       None → không dùng class weights.
        device       : torch.device
        gamma        : Focal loss γ parameter (chỉ dùng khi loss_name='focal')

    Returns:
        nn.Module loss function

    Examples:
        >>> loss_fn = get_loss_fn('focal', class_counts=np.array([18715, 20938]))
        >>> loss_fn = get_loss_fn('ce',    class_counts=np.array([18715, 20938]))
        >>> loss_fn = get_loss_fn('focal')  # không dùng class weights
    """
    # Tính class weights từ inverse frequency
    alpha = None
    if class_counts is not None:
        class_counts = np.asarray(class_counts, dtype=np.float64)
        n_total      = class_counts.sum()
        n_classes    = len(class_counts)
        # weight_i = N_total / (n_classes * count_i)
        weights      = n_total / (n_classes * class_counts)
        alpha        = torch.tensor(weights, dtype=torch.float32).to(device)

    loss_name_lower = loss_name.lower()

    if loss_name_lower == 'focal':
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        if alpha is not None:
            w_str = ', '.join([f'{w:.3f}' for w in alpha.cpu().numpy()])
            print(f"  [loss] FocalLoss(γ={gamma}, α=[{w_str}])")
        else:
            print(f"  [loss] FocalLoss(γ={gamma}, α=None)")

    elif loss_name_lower in ('ce', 'crossentropy', 'cross_entropy'):
        loss_fn = nn.CrossEntropyLoss(weight=alpha, reduction='mean')
        if alpha is not None:
            w_str = ', '.join([f'{w:.3f}' for w in alpha.cpu().numpy()])
            print(f"  [loss] CrossEntropyLoss(α=[{w_str}])")
        else:
            print(f"  [loss] CrossEntropyLoss(α=None)")

    else:
        raise ValueError(
            f"Loss không hợp lệ: '{loss_name}'. "
            f"Chọn: 'focal' hoặc 'ce'"
        )

    return loss_fn


# ─── WeightedRandomSampler helper ─────────────────────────────────────────────

def get_weighted_sampler(labels: np.ndarray) -> torch.utils.data.WeightedRandomSampler:
    """
    Tạo WeightedRandomSampler để oversample minority class trong mỗi batch.

    Mỗi sample có weight = 1 / count_of_its_class
    → Mỗi class được sample với tần suất bằng nhau.

    Args:
        labels : Array các labels (0/1) của toàn bộ training set

    Returns:
        WeightedRandomSampler
    """
    from torch.utils.data import WeightedRandomSampler

    labels       = np.asarray(labels, dtype=int)
    class_counts = np.bincount(labels, minlength=2)  # [n_class0, n_class1]

    # Tránh chia cho 0 nếu một class không xuất hiện (có thể xảy ra nếu split lỗi).
    if (class_counts == 0).any():
        print(f"  [sampler] WARNING: class_counts={class_counts.tolist()} contains 0; "
              f"fall back to uniform sampling (no WeightedRandomSampler).")
        return None

    class_weights = 1.0 / class_counts           # [w0, w1]
    sample_weights = class_weights[labels]       # weight cho mỗi sample

    sampler = WeightedRandomSampler(
        weights     = torch.FloatTensor(sample_weights),
        num_samples = len(sample_weights),
        replacement = True,
    )

    w0, w1 = class_weights
    print(
        f"  [sampler] WeightedRandomSampler: "
        f"sensitive(0) w={w0:.8f}, resistant(1) w={w1:.8f} | "
        f"counts: S={int(class_counts[0]):,}, R={int(class_counts[1]):,} | "
        f"n={len(labels):,} samples"
    )
    return sampler


