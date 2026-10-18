"""
Tự động chọn các GradCAM đẹp nhất và xuất ảnh sạch (không label, không colorbar).

Tiêu chí chọn ảnh tốt:
  1. Dự đoán đúng (correct prediction)
  2. Confidence cao (prob resistant > 0.7 hoặc < 0.3)
  3. Heatmap tập trung vào vùng cell (overlap score cao)
"""

import os, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms

from models import get_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── GradCAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model     = model
        self.grads     = None
        self.acts      = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def __call__(self, x, class_idx=1):
        self.model.eval()
        x      = x.unsqueeze(0) if x.dim() == 3 else x
        logits = self.model(x)
        probs  = F.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).item()
        prob1  = probs[0, 1].item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        w   = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred, prob1


# ─── Helpers ──────────────────────────────────────────────────────────────────

def tight_crop(img: Image.Image, pad: int = 8) -> Image.Image:
    arr  = np.array(img.convert('L'))
    mask = arr > 10
    if mask.sum() < 30:
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    h, w   = arr.shape
    return img.crop((max(0,c0-pad), max(0,r0-pad),
                     min(w,c1+pad), min(h,r1+pad)))


def to_tensor(img: Image.Image, sz: int = 224) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img)


def denorm(t: torch.Tensor) -> np.ndarray:
    m = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    s = torch.tensor(IMAGENET_STD).view(3,1,1)
    return (t.cpu()*s+m).clamp(0,1).permute(1,2,0).numpy()


def cell_overlap_score(cam: np.ndarray, orig: np.ndarray) -> float:
    """Tỉ lệ activation nằm trên vùng cell (không phải background đen)."""
    gray = orig.mean(axis=2)          # (H, W)
    cell = (gray > 0.05).astype(float)
    if cell.sum() == 0:
        return 0.0
    return (cam * cell).sum() / (cam.sum() + 1e-8)


def quality_score(cam, orig, pred, label, prob1):
    """Điểm tổng hợp để xếp hạng ảnh GradCAM."""
    correct   = 1.0 if pred == label else 0.0
    conf      = abs(prob1 - 0.5) * 2          # [0,1], 1 = rất tự tin
    overlap   = cell_overlap_score(cam, orig)
    act_peak  = float(cam.max())               # activation có rõ không
    return correct * (0.4*conf + 0.4*overlap + 0.2*act_peak)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(drug='CIP', seed=0, candidates_per_class=60,
        top_per_class=4, img_size=224):

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = f'/home/son/new_mia/results/task1/{drug}/densenet169/seed{seed}_best.pth'
    drug_dir  = f'/home/son/new_mia/{drug}_joblib'
    out_dir   = '/home/son/new_mia/gradcam_outputs'
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = get_model('densenet169', num_classes=2, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    target_layer = model.features.denseblock4
    gradcam      = GradCAM(model, target_layer)

    # ── Chọn ảnh tốt nhất cho mỗi class ──────────────────────────────────────
    best = {0: [], 1: []}   # label → list of (score, orig_np, cam_np, overlay_np)

    random.seed(seed)
    for label_int, label_str in [(1,'resistant'), (0,'sensitive')]:
        folder = os.path.join(drug_dir, 'test', label_str, 'masked')
        files  = [os.path.join(folder, f)
                  for f in os.listdir(folder) if f.endswith('.png')]
        random.shuffle(files)
        sample = files[:candidates_per_class]

        print(f'Scoring {len(sample)} {label_str} cells...')
        scored = []
        for path in sample:
            try:
                img_pil = tight_crop(Image.open(path).convert('RGB'))
                tensor  = to_tensor(img_pil, img_size).to(device)
                cam, pred, prob1 = gradcam(tensor, class_idx=1)

                # Resize cam to img_size
                cam_rs = np.array(
                    Image.fromarray((cam*255).astype(np.uint8))
                    .resize((img_size, img_size), Image.BILINEAR)
                ) / 255.0

                orig = denorm(tensor)
                score = quality_score(cam_rs, orig, pred, label_int, prob1)
                scored.append((score, orig, cam_rs, label_int))
            except Exception as e:
                print(f'  skip {os.path.basename(path)}: {e}')

        scored.sort(key=lambda x: x[0], reverse=True)
        best[label_int] = scored[:top_per_class]

    # ── Build overlay ──────────────────────────────────────────────────────────
    def make_overlay(orig, cam):
        hm      = plt.cm.jet(cam)[:,:,:3]
        gray    = orig.mean(axis=2)
        mask    = (gray > 0.05).astype(float)
        mask3   = np.stack([mask]*3, axis=2)
        overlay = mask3*(0.4*orig + 0.6*hm) + (1-mask3)*orig
        return np.clip(overlay, 0, 1)

    # ── Plot layout ───────────────────────────────────────────────────────────
    # Mỗi sample = [original | GradCAM], xếp thành top_per_class cột
    # 2 hàng: Resistant (trên) / Sensitive (dưới)
    # Giữa 2 hàng có khoảng trắng + dòng kẻ phân cách
    cols    = top_per_class
    cell_sz = 2.8          # inch mỗi ô vuông
    gap     = 0.9          # khoảng trắng giữa 2 nhóm (inch)
    pad_top = 0.55         # chỗ cho label nhóm

    fig_w = cols * cell_sz * 2 + (cols - 1) * 0.12
    fig_h = 2 * (cell_sz + pad_top) + gap

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    class_labels = {1: 'Resistant', 0: 'Sensitive'}
    label_colors = {1: '#c0392b',   0: '#2980b9'}

    for row_idx, label_int in enumerate([1, 0]):
        entries = best[label_int]

        # Top-left corner của nhóm này (normalized figure coords)
        group_h  = cell_sz + pad_top
        top_norm = 1.0 - row_idx * (group_h + gap) / fig_h - 0.01
        bot_norm = top_norm - group_h / fig_h
        lft_norm = 0.01
        rgt_norm = 0.99

        # GridSpec cho nhóm này: 1 row, cols*2 cols (orig | overlay)
        inner_gs = gridspec.GridSpec(
            1, cols * 2,
            figure=fig,
            left=lft_norm, right=rgt_norm,
            bottom=bot_norm, top=top_norm - pad_top/fig_h,
            wspace=0.05
        )

        # Nhãn nhóm
        fig.text(
            lft_norm, top_norm - 0.005,
            class_labels[label_int],
            ha='left', va='top',
            fontsize=11, fontweight='bold',
            color=label_colors[label_int],
            transform=fig.transFigure
        )

        for col_idx, (score, orig, cam_rs, _) in enumerate(entries):
            overlay = make_overlay(orig, cam_rs)

            ax_o = fig.add_subplot(inner_gs[0, col_idx * 2])
            ax_o.imshow(orig)
            ax_o.axis('off')

            ax_v = fig.add_subplot(inner_gs[0, col_idx * 2 + 1])
            ax_v.imshow(overlay)
            ax_v.axis('off')

    out_path = os.path.join(out_dir,
                f'gradcam_best_densenet169_{drug}_seed{seed}.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'\nSaved → {out_path}')


def export_clean(drug='CIP', seed=0, candidates_per_class=80,
                 top_per_class=3, img_size=224):
    """
    Xuất chỉ GradCAM overlay (không original), n_per_class ảnh mỗi nhóm.
    Layout: 1 row resistant | 1 row sensitive.  400 DPI.
    """
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = f'/home/son/new_mia/results/task1/{drug}/densenet169/seed{seed}_best.pth'
    drug_dir  = f'/home/son/new_mia/{drug}_joblib'
    out_dir   = '/home/son/new_mia/gradcam_outputs'
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = get_model('densenet169', num_classes=2, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    gradcam = GradCAM(model, model.features.denseblock4)

    # ── Chọn ảnh tốt nhất ─────────────────────────────────────────────────────
    best = {0: [], 1: []}
    random.seed(seed)
    for label_int, label_str in [(1, 'resistant'), (0, 'sensitive')]:
        folder = os.path.join(drug_dir, 'test', label_str, 'masked')
        files  = [os.path.join(folder, f)
                  for f in os.listdir(folder) if f.endswith('.png')]
        random.shuffle(files)
        scored = []
        for path in files[:candidates_per_class]:
            try:
                img_pil = tight_crop(Image.open(path).convert('RGB'))
                tensor  = to_tensor(img_pil, img_size).to(device)
                cam, pred, prob1 = gradcam(tensor, class_idx=1)
                cam_rs = np.array(
                    Image.fromarray((cam*255).astype(np.uint8))
                    .resize((img_size, img_size), Image.BILINEAR)
                ) / 255.0
                orig  = denorm(tensor)
                score = quality_score(cam_rs, orig, pred, label_int, prob1)
                scored.append((score, orig, cam_rs, label_int))
            except Exception:
                pass
        scored.sort(key=lambda x: x[0], reverse=True)
        best[label_int] = scored[:top_per_class]
        print(f'{label_str}: {len(scored)} scored, kept top {top_per_class}')

    def make_overlay(orig, cam):
        hm   = plt.cm.jet(cam)[:, :, :3]
        gray = orig.mean(axis=2)
        mask = np.stack([(gray > 0.05).astype(float)] * 3, axis=2)
        ov   = mask * (0.4*orig + 0.6*hm) + (1-mask)*orig
        return np.clip(ov, 0, 1)

    # ── Plot: 2 rows × top_per_class cols, overlay only ──────────────────────
    cols   = top_per_class
    rows   = 2
    cell_w = 2.0    # inch
    cell_h = 2.0
    pad    = 0.08   # khoảng cách giữa các ô

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * cell_w + (cols-1)*pad,
                 rows * cell_h + (rows-1)*pad),
        facecolor='black',
        gridspec_kw=dict(wspace=pad/cell_w, hspace=pad/cell_h)
    )

    for row_idx, label_int in enumerate([1, 0]):
        for col_idx, (score, orig, cam_rs, _) in enumerate(best[label_int]):
            ax = axes[row_idx, col_idx]
            ax.imshow(make_overlay(orig, cam_rs))
            ax.axis('off')
            # Viền màu mỏng phân biệt nhóm
            for spine in ax.spines.values():
                spine.set_edgecolor('#c0392b' if label_int == 1 else '#2980b9')
                spine.set_linewidth(2)
                spine.set_visible(True)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=pad/cell_w, hspace=pad/cell_h)

    out_path = os.path.join(out_dir,
                f'gradcam_clean_{drug}_seed{seed}_top{top_per_class}.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    export_clean(drug='CIP', seed=0, candidates_per_class=80, top_per_class=3)
