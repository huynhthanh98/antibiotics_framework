#!/usr/bin/env python3
"""
Grad-CAM figures per drug (ResNet50 / ConvNeXt / DenseNet) with intact-cell filtering.

Usage:
  # Per-drug (2 columns, strict intact cells):
  python scripts/plot_gradcam_three_drugs.py --model resnet50 --drug AZI --seed 13 --n_cells 2 --output_suffix best

  # Combined paper figure (AZI seed13, CIP seed7, TE seed21):
  python scripts/plot_gradcam_three_drugs.py --model resnet50 --compose_paper
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label as ndimage_label
from scipy.spatial import ConvexHull
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "antibiotics_framework"
sys.path.insert(0, str(FRAMEWORK))

from models import get_model  # noqa: E402
from datasets import open_as_rgb  # noqa: E402

DRUGS = ["AZI", "CIP", "TE"]
DRUG_FULL = {"AZI": "Azithromycin (AZI)", "CIP": "Ciprofloxacin (CIP)", "TE": "Tetracycline (TE)"}
PAPER_SEEDS = {"AZI": 13, "CIP": 7, "TE": 21}
OUT = ROOT / "reports" / "figures"

CMAP_CAM = LinearSegmentedColormap.from_list(
    "saliency", [(0, 0, 0, 0), (0.55, 0.05, 0.0, 0.55), (1.0, 0.35, 0.0, 0.9), (1.0, 0.95, 0.3, 1.0)]
)
ROW_STYLE = {
    1: {"name": "Resistant", "color": "#C62828", "bg": "#FFF5F5", "cmap": CMAP_CAM},
    0: {"name": "Sensitive", "color": "#1565C0", "bg": "#F0F8FF", "cmap": CMAP_CAM},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
})


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(
            lambda _m, _i, o: setattr(self, "activations", o.detach())
        )
        self.target_layer.register_full_backward_hook(
            lambda _m, _gi, go: setattr(self, "gradients", go[0].detach())
        )

    def __call__(self, x: torch.Tensor, class_idx: int) -> tuple[np.ndarray, int, float, float]:
        self.model.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device, non_blocking=self.device.type == "cuda")
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits[0, class_idx].backward()
        probs = F.softmax(logits.detach(), dim=1)
        cam = F.relu((self.gradients.mean(dim=(2, 3), keepdim=True) * self.activations).sum(dim=1))
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.ndim == 0:
            cam = np.array([[float(cam)]])
        return cam, int(logits.argmax(dim=1).item()), float(probs[0, 1].item()), float(probs[0, 0].item())


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if "densenet" in model_name:
        try:
            return model.features.denseblock4
        except AttributeError:
            pass
    if "resnet" in model_name:
        try:
            return model.layer4
        except AttributeError:
            pass
    if "convnext" in model_name:
        try:
            return model.stages[-1].blocks[-1]
        except AttributeError:
            try:
                return model.stages[-1]
            except AttributeError:
                pass
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError(f"No Conv2d layer found for {model_name}")
    return last_conv


def tight_crop(img: Image.Image, padding: int = 10) -> Image.Image:
    arr = np.array(img.convert("L"))
    mask = arr > 12
    if mask.sum() < 50:
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    h, w = arr.shape
    r0, c0 = max(0, r0 - padding), max(0, c0 - padding)
    r1, c1 = min(h, r1 + padding), min(w, c1 + padding)
    return img.crop((c0, r0, c1, r1))


def load_tensor(path: str, img_size: int = 224) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(tight_crop(open_as_rgb(path)))


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def to_luminance(orig: np.ndarray) -> np.ndarray:
    return (0.299 * orig[..., 0] + 0.587 * orig[..., 1] + 0.114 * orig[..., 2]).astype(np.float32)


def cell_mask_from_orig(orig: np.ndarray, thr: float = 0.06) -> np.ndarray:
    return to_luminance(orig) > thr


def cell_integrity(orig: np.ndarray, mask: np.ndarray) -> dict | None:
    """Reject fragmented, border-cut, hollow-ring, or tiny cells."""
    gray = to_luminance(orig)
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) < 40:
        return None
    area = float(mask.sum())
    area_frac = area / (h * w)

    border = np.zeros_like(mask, dtype=bool)
    m = 4
    border[:m, :] = border[-m:, :] = border[:, :m] = border[:, -m:] = True
    border_frac = float((mask & border).sum() / area)

    try:
        hull = ConvexHull(np.column_stack([xs, ys]))
        solidity = area / (float(hull.volume) + 1e-6)
    except Exception:
        solidity = 0.0

    cy, cx = ys.mean(), xs.mean()
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_max = dist.max() + 1e-6
    inner = dist < 0.38 * r_max
    outer = dist > 0.62 * r_max
    hollow_score = 1.0
    if inner.sum() > 8 and outer.sum() > 8:
        inner_m = float(gray[ys[inner], xs[inner]].mean())
        outer_m = float(gray[ys[outer], xs[outer]].mean())
        hollow_score = outer_m / (inner_m + 1e-6)

    bh = float(ys.max() - ys.min() + 1)
    bw = float(xs.max() - xs.min() + 1)
    aspect = max(bh, bw) / (min(bh, bw) + 1e-6)

    labeled, n_comp = ndimage_label(mask)
    n_significant = 0
    if n_comp > 0:
        comp_areas = sorted(((labeled == i).sum() for i in range(1, n_comp + 1)), reverse=True)
        n_significant = sum(1 for a in comp_areas if a >= max(40, 0.08 * area))

    return {
        "area_frac": area_frac,
        "border_frac": border_frac,
        "solidity": solidity,
        "hollow_score": hollow_score,
        "aspect": aspect,
        "brightness": float(gray[mask].mean()),
        "n_components": n_significant,
    }


def passes_integrity(
    m: dict | None,
    min_solidity: float,
    max_border_frac: float,
    max_hollow: float,
    min_area_frac: float,
    min_brightness: float,
    min_aspect: float,
    max_aspect: float,
    max_components: int = 1,
) -> bool:
    if m is None:
        return False
    return (
        m["solidity"] >= min_solidity
        and m["border_frac"] <= max_border_frac
        and m["hollow_score"] <= max_hollow
        and m["area_frac"] >= min_area_frac
        and m["brightness"] >= min_brightness
        and min_aspect <= m["aspect"] <= max_aspect
        and m["n_components"] <= max_components
    )


def grayscale_display(orig: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = to_luminance(orig)
    vals = gray[mask] if mask.any() else gray.ravel()
    if vals.size < 20:
        out = np.clip(gray * 2.2, 0, 1)
    else:
        lo, hi = np.percentile(vals, (1, 99))
        out = np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1) if hi > lo else np.clip(gray * 2.2, 0, 1)
        out = np.power(out, 0.82)
    return np.stack([out, out, out], axis=-1)


def prepare_cam_display(cam_raw: np.ndarray, img_size: int, mask: np.ndarray, sigma: float = 2.8):
    cam = cam_raw.astype(np.float64)
    if cam.max() > 1e-10:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.power(cam, 0.65)
    cam_np = gaussian_filter(
        np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((img_size, img_size), Image.BICUBIC)) / 255.0,
        sigma=sigma,
    )
    mask_d = binary_dilation(mask, iterations=12)
    lo, hi = np.percentile(cam_np.ravel(), (1, 99))
    cam_disp = np.power(np.clip((cam_np - lo) / (hi - lo + 1e-6), 0, 1), 0.5)
    peak = float(cam_disp[mask_d].max()) if mask_d.any() else float(cam_disp.max())
    visible_frac = float((cam_disp[mask_d] > 0.12).mean()) if mask_d.any() else 0.0
    return cam_disp, mask_d, peak, visible_frac


def overlay_on_gray(orig_gray, cam_disp, mask, cmap, alpha=0.70):
    hm_rgba = cmap(cam_disp)
    hm_a = hm_rgba[..., 3:4] * alpha
    m3 = np.repeat(mask[..., None], 3, axis=2)
    thr = float(np.percentile(cam_disp[mask], 8)) if mask.any() else 0.02
    salient = cam_disp[..., None] > max(thr, 0.02)
    blended = orig_gray * (1 - hm_a) + hm_rgba[..., :3] * hm_a
    return np.clip(np.where(m3, np.where(salient, blended, orig_gray), orig_gray * 0.10), 0, 1)


def cam_focus_score(cam: np.ndarray) -> float:
    flat = cam.ravel().astype(np.float64)
    if flat.max() < 1e-10:
        return 0.0
    flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)
    top = np.sort(flat)[-max(1, int(0.05 * flat.size)) :]
    return float(top.mean() / (flat.mean() + 1e-6))


def panel_quality(tensor, cam_raw, img_size):
    orig = denormalize(tensor)
    mask = cell_mask_from_orig(orig)
    cam_disp, mask_d, peak, visible_frac = prepare_cam_display(cam_raw, img_size, mask)
    gray = grayscale_display(orig, mask)
    overlay = overlay_on_gray(gray, cam_disp, mask_d, CMAP_CAM)
    diff = np.abs(overlay - gray).mean(axis=2)[mask_d].mean() if mask_d.any() else 0.0
    return {"peak": peak, "visible_frac": visible_frac, "color_energy": float(diff)}


def resolve_device(request: str) -> torch.device:
    req = request.lower().strip()
    if req == "auto":
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        dev = torch.device(req if ":" in req else "cuda:0")
    elif req == "cpu":
        dev = torch.device("cpu")
    else:
        raise ValueError(req)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  GPU: {torch.cuda.get_device_name(dev)}")
    return dev


def image_id_from_path(path: str) -> str:
    name = Path(path).stem
    return name.split("_")[0] if "_" in name else name


def list_cells(drug_dir: Path, split: str, label_str: str, image_folder: str) -> list[str]:
    folder = drug_dir / split / label_str / image_folder
    if not folder.is_dir():
        return []
    exts = {".png", ".tiff", ".tif", ".jpg"}
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in exts)


def integrity_for_label(args, true_label: int) -> dict:
    """Stricter morphology for Sensitive (solid rods, no ring/hollow)."""
    if true_label == 0:
        return dict(
            min_solidity=args.min_solidity_s,
            max_border_frac=args.max_border_frac_s,
            max_hollow=args.max_hollow_s,
            min_area_frac=args.min_area_frac,
            min_brightness=args.min_brightness,
            max_components=args.max_components,
        )
    return dict(
        min_solidity=args.min_solidity_r,
        max_border_frac=args.max_border_frac,
        max_hollow=args.max_hollow_r,
        min_area_frac=args.min_area_frac,
        min_brightness=args.min_brightness,
        max_components=args.max_components,
    )


def build_candidate_pool(
    paths, gradcam, device, img_size, true_label, seed, pool_size,
    min_focus, min_peak, min_visible_frac, min_color_energy,
    min_aspect, max_aspect, integrity_kw,
) -> list[dict]:
    rng = random.Random(seed)
    pool = paths.copy()
    rng.shuffle(pool)
    pool = pool[: min(pool_size, len(pool))]

    prelim = []
    target_cls = true_label
    for p in pool:
        t = load_tensor(p, img_size).to(device)
        orig = denormalize(t)
        mask = cell_mask_from_orig(orig)
        integ = cell_integrity(orig, mask)
        if not passes_integrity(integ, min_aspect=min_aspect, max_aspect=max_aspect, **integrity_kw):
            continue
        cam_raw, pred, pr, ps = gradcam(t, target_cls)
        focus = cam_focus_score(cam_raw)
        if focus < min_focus:
            continue
        conf = pr if true_label == 1 else ps
        prelim.append((focus * (0.3 + conf) * integ["brightness"] * integ["solidity"], {
            "path": p, "tensor": t.cpu(), "cam_raw": cam_raw,
            "pred": pred, "prob_r": pr, "prob_s": ps,
            "true_label": true_label, "target_cls": target_cls,
            "focus": focus, "image_id": image_id_from_path(p),
            **integ,
        }))

    prelim.sort(key=lambda x: x[0], reverse=True)
    shortlist = [x[1] for x in prelim[: max(100, 30)]]
    other_cls = 1 - target_cls
    scored = []
    for item in shortlist:
        t = item["tensor"].to(device)
        cam_other, _, _, _ = gradcam(t, other_cls)
        disc = item["focus"] / (cam_focus_score(cam_other) + 0.3)
        q = panel_quality(item["tensor"], item["cam_raw"], img_size)
        item.update(disc=disc, **q)
        if q["peak"] < min_peak * 0.45 or q["visible_frac"] < min_visible_frac * 0.45:
            continue
        if q["color_energy"] < min_color_energy * 0.45:
            continue
        conf = item["prob_r"] if true_label == 1 else item["prob_s"]
        morph = item["solidity"] * (1.0 / (item["hollow_score"] + 0.15))
        item["score"] = (
            item["focus"] * q["peak"] * q["visible_frac"] * (q["color_energy"] + 0.03)
            * (0.3 + conf) * (0.7 + disc) * morph
        )
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def pick_matched_pairs_split_integrity(
    r_paths, s_paths, gradcam, device, img_size, n, seed, aspect_tol, pool_kw_base,
    r_integrity, s_integrity,
) -> tuple[list, list]:
    r_kw = {**pool_kw_base, "integrity_kw": r_integrity}
    s_kw = {**pool_kw_base, "integrity_kw": s_integrity}
    r_pool = build_candidate_pool(r_paths, gradcam, device, img_size, 1, seed, **r_kw)
    s_pool = build_candidate_pool(s_paths, gradcam, device, img_size, 0, seed + 31, **s_kw)
    return _match_pairs_from_pools(r_pool, s_pool, n, aspect_tol)


def pick_matched_pairs(r_paths, s_paths, gradcam, device, img_size, n, seed, aspect_tol, pool_kw) -> tuple[list, list]:
    r_pool = build_candidate_pool(r_paths, gradcam, device, img_size, 1, seed, **pool_kw)
    s_pool = build_candidate_pool(s_paths, gradcam, device, img_size, 0, seed + 31, **pool_kw)
    return _match_pairs_from_pools(r_pool, s_pool, n, aspect_tol)


def _match_pairs_from_pools(r_pool, s_pool, n, aspect_tol) -> tuple[list, list]:
    resistant, sensitive = [], []
    used_r, used_s, used_ir, used_is = set(), set(), set(), set()

    for tol in (aspect_tol, aspect_tol * 1.4, aspect_tol * 2.2):
        resistant.clear()
        sensitive.clear()
        used_r.clear()
        used_s.clear()
        used_ir.clear()
        used_is.clear()
        for r_item in r_pool:
            if len(resistant) >= n:
                break
            if r_item["path"] in used_r:
                continue
            if r_item["image_id"] in used_ir and len(used_ir) < n:
                continue
            ra = r_item["aspect"]
            best_s, best_val = None, -1.0
            for s_item in s_pool:
                if s_item["path"] in used_s:
                    continue
                ad = abs(s_item["aspect"] - ra)
                if ad > tol:
                    continue
                val = s_item["score"] / (1.0 + ad)
                if val > best_val:
                    best_val, best_s = val, s_item
            if best_s is None:
                continue
            resistant.append(r_item)
            sensitive.append(best_s)
            used_r.add(r_item["path"])
            used_s.add(best_s["path"])
            used_ir.add(r_item["image_id"])
            used_is.add(best_s["image_id"])
        if len(resistant) >= n:
            break
    return resistant[:n], sensitive[:n]


def render_panel(ax_in, ax_gc, ex, style, img_size, show_titles=True, letter="a"):
    orig = denormalize(ex["tensor"])
    mask = cell_mask_from_orig(orig)
    gray = grayscale_display(orig, mask)
    cam_disp, mask_d, _, _ = prepare_cam_display(ex["cam_raw"], img_size, mask)
    overlay = overlay_on_gray(gray, cam_disp, mask_d, style["cmap"])
    ax_in.imshow(gray)
    ax_gc.imshow(overlay)
    for ax, lw in ((ax_in, 1.4), (ax_gc, 2.0)):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#0D0D0D")
        for sp in ax.spines.values():
            sp.set_color(style["color"])
            sp.set_linewidth(lw)
    if show_titles:
        ax_in.set_title(f"({letter}) Input", fontsize=8, color="#424242", pad=3)
        ax_gc.set_title(f"({letter}) Grad-CAM", fontsize=8, color="#424242", pad=3)


def plot_drug_figure(drug, resistant, sensitive, n, img_size, output_stem, show_col_titles=True):
    label_w, panel_w = 0.55, 1.35
    fig_w = label_w + n * panel_w * 2 + 0.25
    fig = plt.figure(figsize=(fig_w, 4.2), facecolor="#FAFAFA")
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.20, top=0.96, bottom=0.06, left=0.05, right=0.98)
    for row_i, label in enumerate((1, 0)):
        style = ROW_STYLE[label]
        examples = resistant if label == 1 else sensitive
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 1 + n * 2, subplot_spec=outer[row_i],
            width_ratios=[label_w] + [1, 1] * n, wspace=0.06,
        )
        ax_lbl = fig.add_subplot(inner[0])
        ax_lbl.set_facecolor(style["bg"])
        ax_lbl.text(0.5, 0.5, style["name"], rotation=90, va="center", ha="center",
                    fontsize=11, fontweight="bold", color=style["color"])
        ax_lbl.axis("off")
        for j, ex in enumerate(examples):
            ax_in = fig.add_subplot(inner[1 + j * 2])
            ax_gc = fig.add_subplot(inner[2 + j * 2])
            render_panel(ax_in, ax_gc, ex, style, img_size,
                         show_titles=show_col_titles and row_i == 0, letter=chr(ord("a") + j))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{output_stem}.{ext}", dpi=300, facecolor="#FAFAFA", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  saved {output_stem}.png")


def plot_compose_paper(
    all_data: dict[str, tuple[list, list]], model_name: str, img_size: int, n: int, output_suffix: str,
):
    """Stack AZI / CIP / TE (2 rows each) in one figure."""
    label_w, panel_w, drug_w = 0.45, 1.25, 0.35
    fig_w = drug_w + label_w + n * panel_w * 2 + 0.2
    fig_h = len(DRUGS) * 2 * 1.05 + 0.3
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#FAFAFA")
    total_rows = len(DRUGS) * 2
    outer = gridspec.GridSpec(total_rows, 1, figure=fig, hspace=0.28, wspace=0.04,
                              top=0.98, bottom=0.02, left=0.04, right=0.98)

    row_idx = 0
    for drug in DRUGS:
        resistant, sensitive = all_data[drug]
        for label, examples in ((1, resistant), (0, sensitive)):
            style = ROW_STYLE[label]
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2 + n * 2, subplot_spec=outer[row_idx],
                width_ratios=[drug_w, label_w] + [1, 1] * n, wspace=0.05,
            )
            ax_d = fig.add_subplot(inner[0])
            ax_d.axis("off")
            if label == 1:
                ax_d.text(0.5, 0.5, drug, rotation=90, va="center", ha="center",
                          fontsize=12, fontweight="bold", color="#333333")
            ax_lbl = fig.add_subplot(inner[1])
            ax_lbl.set_facecolor(style["bg"])
            ax_lbl.text(0.5, 0.5, style["name"], rotation=90, va="center", ha="center",
                        fontsize=10, fontweight="bold", color=style["color"])
            ax_lbl.axis("off")
            for j, ex in enumerate(examples):
                ax_in = fig.add_subplot(inner[2 + j * 2])
                ax_gc = fig.add_subplot(inner[3 + j * 2])
                render_panel(ax_in, ax_gc, ex, style, img_size,
                             show_titles=(row_idx == 0), letter=chr(ord("a") + j))
            row_idx += 1

    stem = f"fig_gradcam_three_drugs_{model_name}_{output_suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, facecolor="#FAFAFA", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"\n  saved {stem}.png / .pdf")


def load_model_and_gradcam(model_name, drug, device, results_dir, ckpt_seed):
    ckpt = Path(results_dir) / "task1" / drug / model_name / f"seed{ckpt_seed}_best.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model = get_model(model_name, 2, False)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model = model.to(device).eval()
    return GradCAM(model, get_target_layer(model, model_name), device)


def process_drug(drug, args, device, seed_override: int | None = None) -> tuple[list, list]:
    seed = seed_override if seed_override is not None else args.seed + (hash(drug) % 997)
    gradcam = load_model_and_gradcam(args.model, drug, device, args.results_dir, args.ckpt_seed)
    drug_dir = ROOT / f"{drug}{args.joblib_suffix}"
    r_paths = list_cells(drug_dir, args.split, "resistant", args.image_folder)
    s_paths = list_cells(drug_dir, args.split, "sensitive", args.image_folder)
    print(f"\n[{drug}] seed={seed}  pool R={len(r_paths)} S={len(s_paths)}")

    r_integrity = integrity_for_label(args, 1)
    s_integrity = integrity_for_label(args, 0)
    pool_kw_base = dict(
        pool_size=args.pool_size, min_focus=args.min_focus, min_peak=args.min_peak,
        min_visible_frac=args.min_visible_frac, min_color_energy=args.min_color_energy,
        min_aspect=args.min_aspect, max_aspect=args.max_aspect,
    )
    resistant, sensitive = pick_matched_pairs_split_integrity(
        r_paths, s_paths, gradcam, device, args.img_size, args.n_cells,
        seed=seed, aspect_tol=args.aspect_tol, pool_kw_base=pool_kw_base,
        r_integrity=r_integrity, s_integrity=s_integrity,
    )
    if len(resistant) < args.n_cells or len(sensitive) < args.n_cells:
        raise RuntimeError(f"{drug}: only {len(resistant)} R, {len(sensitive)} S (need {args.n_cells})")

    for tag, items in [("R", resistant), ("S", sensitive)]:
        aspects = [round(x["aspect"], 2) for x in items]
        solid = [round(x["solidity"], 2) for x in items]
        hollow = [round(x["hollow_score"], 2) for x in items]
        comps = [x["n_components"] for x in items]
        print(f"  {tag}: aspect={aspects}, solidity={solid}, hollow={hollow}, n_comp={comps}")

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    if not args.compose_paper:
        plot_drug_figure(drug, resistant, sensitive, args.n_cells, args.img_size,
                         f"fig_gradcam_{drug}_{args.model}{suffix}")
    return resistant, sensitive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", nargs="+", default=DRUGS, choices=DRUGS)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_seed", type=int, default=0)
    ap.add_argument("--n_cells", type=int, default=2, help="2 columns = compact paper figure")
    ap.add_argument("--output_suffix", default="best")
    ap.add_argument("--compose_paper", action="store_true", help="one figure: AZI/CIP/TE stacked")
    ap.add_argument("--pool_size", type=int, default=600)
    ap.add_argument("--min_focus", type=float, default=1.1)
    ap.add_argument("--min_peak", type=float, default=0.22)
    ap.add_argument("--min_visible_frac", type=float, default=0.08)
    ap.add_argument("--min_color_energy", type=float, default=0.08)
    ap.add_argument("--aspect_tol", type=float, default=0.28)
    ap.add_argument("--min_aspect", type=float, default=1.15)
    ap.add_argument("--max_aspect", type=float, default=2.6)
    ap.add_argument("--min_solidity", type=float, default=0.78, help="legacy default for R if --min_solidity_r unset")
    ap.add_argument("--min_solidity_r", type=float, default=None, help="Resistant: reject fragmented cells")
    ap.add_argument("--min_solidity_s", type=float, default=0.86, help="Sensitive: require solid rods")
    ap.add_argument("--max_border_frac", type=float, default=0.06, help="reject cut-off at edge (R)")
    ap.add_argument("--max_border_frac_s", type=float, default=0.04, help="stricter border for Sensitive")
    ap.add_argument("--max_hollow", type=float, default=2.2, help="legacy default for R if --max_hollow_r unset")
    ap.add_argument("--max_hollow_r", type=float, default=None, help="Resistant ring/hollow limit")
    ap.add_argument("--max_hollow_s", type=float, default=1.75, help="Sensitive: reject ring/hollow cells")
    ap.add_argument("--max_components", type=int, default=1, help="max connected blobs in mask")
    ap.add_argument("--min_area_frac", type=float, default=0.035)
    ap.add_argument("--min_brightness", type=float, default=0.07)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--split", default="train")
    ap.add_argument("--image_folder", default="masked")
    ap.add_argument("--results_dir", default=str(ROOT / "results_cells"))
    ap.add_argument("--joblib_suffix", default="_joblib")
    args = ap.parse_args()
    if args.min_solidity_r is None:
        args.min_solidity_r = args.min_solidity
    if args.max_hollow_r is None:
        args.max_hollow_r = args.max_hollow

    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"Warning: {exc} → CPU")
        device = torch.device("cpu")
    print(f"Device: {device} | model={args.model} | n_cells={args.n_cells} | strict intact cells")

    if args.compose_paper:
        all_data = {}
        for drug in DRUGS:
            seed = PAPER_SEEDS.get(drug, args.seed)
            all_data[drug] = process_drug(drug, args, device, seed_override=seed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        plot_compose_paper(all_data, args.model, args.img_size, args.n_cells, args.output_suffix)
    else:
        for drug in args.drug:
            seed = PAPER_SEEDS.get(drug, args.seed) if args.output_suffix == "best" else args.seed
            process_drug(drug, args, device, seed_override=seed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
