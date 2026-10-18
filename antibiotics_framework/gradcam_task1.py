"""
GradCAM visualization cho Task 1 - DenseNet169 trên ảnh cell CIP.

Usage:
    python gradcam_task1.py
    python gradcam_task1.py --drug AZI --seed 0 --n_cells 8
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms

from models import get_model
from datasets import get_test_transform


# ─── GradCAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    GradCAM cho bất kỳ model CNN nào.
    Tự động tìm lớp conv cuối cùng nếu không chỉ định.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """
        Tính GradCAM map.

        Returns
        -------
        cam : np.ndarray  shape (H, W), values in [0, 1]
        pred_class : int
        pred_prob  : float  (prob của resistant class)
        """
        self.model.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x  # (1, C, H, W)

        # Forward
        logits = self.model(x)
        probs  = F.softmax(logits, dim=1)

        pred_class = logits.argmax(dim=1).item()
        pred_prob  = probs[0, 1].item()  # prob của class 1 (resistant)

        if class_idx is None:
            class_idx = pred_class

        # Backward
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Weights = global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()  # (h, w)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_class, pred_prob


def get_densenet_last_conv(model) -> torch.nn.Module:
    """Lấy lớp conv cuối cùng của DenseNet169 (features.denseblock4)."""
    # timm DenseNet: model.features.denseblock4
    try:
        return model.features.denseblock4
    except AttributeError:
        pass
    # Fallback: tìm Conv2d cuối cùng
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


# ─── Preprocessing ────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def tight_crop(img: Image.Image, padding: int = 6) -> Image.Image:
    """
    Crop chặt quanh vùng sáng (cell) để loại bỏ background đen.
    Nếu toàn ảnh là background thì trả về ảnh gốc.
    """
    arr  = np.array(img.convert('L'))           # grayscale
    mask = arr > 10                             # pixel sáng hơn ngưỡng
    if mask.sum() < 50:                         # gần như toàn đen
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    h, w   = arr.shape
    r0 = max(0, r0 - padding)
    r1 = min(h, r1 + padding)
    c0 = max(0, c0 - padding)
    c1 = min(w, c1 + padding)
    return img.crop((c0, r0, c1, r1))


def preprocess(img_path: str, img_size: int = 224) -> torch.Tensor:
    """Load ảnh → crop cell → tensor chuẩn hoá."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert('RGB')
    img = tight_crop(img)                       # crop sát cell
    return tf(img)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Đưa tensor về ảnh hiển thị [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# ─── Collect cell paths ───────────────────────────────────────────────────────

def collect_cells(drug_dir: str, split: str = 'test',
                  n_per_class: int = 4, seed: int = 0):
    """
    Lấy n_per_class ảnh từ mỗi nhãn (resistant / sensitive).
    """
    random.seed(seed)
    cells = []
    for label_int, label_str in [(1, 'resistant'), (0, 'sensitive')]:
        folder = os.path.join(drug_dir, split, label_str, 'masked')
        if not os.path.exists(folder):
            print(f'[WARN] Không tìm thấy: {folder}')
            continue
        files = [os.path.join(folder, f)
                 for f in os.listdir(folder) if f.endswith('.png')]
        chosen = random.sample(files, min(n_per_class, len(files)))
        for path in chosen:
            cells.append({'path': path, 'label': label_int, 'label_str': label_str})
    return cells


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_gradcam(cells, gradcam_fn, device, img_size,
                 output_path: str, drug: str, model_name: str):
    """
    Vẽ lưới ảnh:  [Original | GradCAM overlay | Heatmap]  cho mỗi cell.
    """
    n      = len(cells)
    ncols  = 3
    nrows  = n

    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.2))
    fig.suptitle(
        f'GradCAM — {model_name} | Drug: {drug} | Task 1 (Cell-level)',
        fontsize=13, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(nrows, ncols, hspace=0.45, wspace=0.15)

    col_titles = ['Original Cell', 'GradCAM Overlay', 'Heatmap']
    for c, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, c])
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
        ax.axis('off')

    for row, cell in enumerate(cells):
        tensor = preprocess(cell['path'], img_size).to(device)

        # GradCAM
        cam, pred_class, pred_prob = gradcam_fn(tensor, class_idx=1)  # target: resistant

        # Resize cam về img_size
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_resized.resize((img_size, img_size), Image.BILINEAR)
        cam_np      = np.array(cam_resized) / 255.0

        # Original img (denormalized)
        orig_np = denormalize(tensor.cpu())

        # Overlay: heatmap màu + original, mask background đen
        heatmap  = plt.cm.jet(cam_np)[:, :, :3]  # (H, W, 3)
        # Tạo mask: pixel sáng (cell) thì blend, pixel tối (background) thì giữ nguyên
        gray     = orig_np.mean(axis=2)           # (H, W)
        cell_mask = (gray > 0.05).astype(float)   # 1=cell, 0=background
        cell_mask_3ch = np.stack([cell_mask]*3, axis=2)
        overlay  = cell_mask_3ch * (0.45 * orig_np + 0.55 * heatmap) \
                 + (1 - cell_mask_3ch) * orig_np
        overlay  = np.clip(overlay, 0, 1)

        # Labels
        true_str = cell['label_str']
        pred_str = 'resistant' if pred_class == 1 else 'sensitive'
        correct  = '✓' if pred_class == cell['label'] else '✗'
        row_title = (f"True: {true_str}  |  Pred: {pred_str} "
                     f"(p={pred_prob:.2f})  {correct}")

        # Col 0: Original
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(orig_np)
        ax0.set_ylabel(row_title, fontsize=7.5, rotation=0,
                       labelpad=110, va='center')
        ax0.set_xticks([]); ax0.set_yticks([])

        # Col 1: Overlay
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(overlay)
        ax1.set_xticks([]); ax1.set_yticks([])

        # Col 2: Heatmap
        ax2 = fig.add_subplot(gs[row, 2])
        im = ax2.imshow(cam_np, cmap='jet', vmin=0, vmax=1)
        ax2.set_xticks([]); ax2.set_yticks([])

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Activation intensity')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {output_path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--drug',       default='CIP', choices=['AZI', 'CIP', 'TE'])
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--n_cells',    type=int, default=4,
                   help='số cell mỗi class (resistant / sensitive)')
    p.add_argument('--img_size',   type=int, default=224)
    p.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    p.add_argument('--output_dir', default='/home/son/new_mia/gradcam_outputs')
    return p.parse_args()


def main():
    args      = parse_args()
    drug      = args.drug
    seed      = args.seed
    n_cells   = args.n_cells
    img_size  = args.img_size
    model_name = 'densenet169'

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = f'/home/son/new_mia/results/task1/{drug}/{model_name}/seed{seed}_best.pth'
    drug_dir  = f'/home/son/new_mia/{drug}_joblib'
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint không tồn tại: {ckpt_path}')

    print(f'Loading {model_name} from {ckpt_path}')
    model = get_model(model_name, num_classes=2, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    # ── Target layer ──────────────────────────────────────────────────────────
    target_layer = get_densenet_last_conv(model)
    print(f'Target layer: {type(target_layer).__name__}')

    # ── GradCAM ───────────────────────────────────────────────────────────────
    gradcam = GradCAM(model, target_layer)

    # ── Collect cells ─────────────────────────────────────────────────────────
    cells = collect_cells(drug_dir, split=args.split,
                          n_per_class=n_cells, seed=seed)
    print(f'Collected {len(cells)} cells '
          f'({sum(c["label"]==1 for c in cells)} resistant, '
          f'{sum(c["label"]==0 for c in cells)} sensitive)')

    # ── Plot ──────────────────────────────────────────────────────────────────
    output_path = os.path.join(
        args.output_dir,
        f'gradcam_{model_name}_{drug}_seed{seed}_{args.split}.png'
    )
    plot_gradcam(cells, gradcam, device, img_size,
                 output_path, drug, model_name)

    print('Done.')


if __name__ == '__main__':
    main()
