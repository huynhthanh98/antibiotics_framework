"""
Kiểm tra nhanh xem các model trong run_all_models.sh có load được không.
Chỉ thử tạo model (pretrained=False) với num_classes=2, không cần GPU hay data.
"""

import sys
import time

try:
    import timm
except ImportError:
    print("[ERROR] timm chưa được cài. Chạy: pip install timm")
    sys.exit(1)

import torch
import torchvision.models as tv_models

# ─── Danh sách model (phải khớp với run_all_models.sh) ────────────────────
MODELS = [
    # ResNet
    "resnet18", "resnet50", "resnet101", "resnet152",
    # VGG
    "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    # ConvNeXt v1
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    # ConvNeXt v2
    "convnextv2_tiny", "convnextv2_small", "convnextv2_base",
    # DenseNet
    "densenet121", "densenet169", "densenet201",
    # EfficientNet
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b7",
    # Swin Transformer
    "swin_tiny", "swin_small", "swin_base",
    # Vision Transformer
    "vit_small", "vit_base", "vit_large",
]

# Alias map (giống models.py)
TIMM_MODEL_ALIASES = {
    "resnet18":   "resnet18",
    "resnet50":   "resnet50",
    "resnet101":  "resnet101",
    "resnet152":  "resnet152",
    "vgg16":      "vgg16",
    "vgg16_bn":   "vgg16_bn",
    "vgg19":      "vgg19",
    "vgg19_bn":   "vgg19_bn",
    "convnext_tiny":  "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base":  "convnext_base",
    "convnext_large": "convnext_large",
    "densenet121":    "densenet121",
    "densenet169":    "densenet169",
    "densenet201":    "densenet201",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "efficientnet_b7": "efficientnet_b7",
    "swin_tiny":  "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base":  "swin_base_patch4_window7_224",
    "vit_small":  "vit_small_patch16_224",
    "vit_base":   "vit_base_patch16_224",
    "vit_large":  "vit_large_patch16_224",
    # convnextv2 không có trong alias → dùng thẳng tên timm
    "convnextv2_tiny":  "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_small": "convnextv2_small.fcmae_ft_in22k_in1k",
    "convnextv2_base":  "convnextv2_base.fcmae_ft_in22k_in1k",
}


def check_model(name: str) -> tuple[bool, str, float]:
    """Trả về (ok, message, elapsed_ms)."""
    timm_name = TIMM_MODEL_ALIASES.get(name, name)
    t0 = time.time()
    try:
        model = timm.create_model(timm_name, pretrained=False, num_classes=2)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        elapsed = (time.time() - t0) * 1000
        return True, f"{params:.1f}M params", elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        # Thử fallback tên gốc nếu tên alias thất bại
        if timm_name != name:
            try:
                model = timm.create_model(name, pretrained=False, num_classes=2)
                params = sum(p.numel() for p in model.parameters()) / 1e6
                elapsed = (time.time() - t0) * 1000
                return True, f"{params:.1f}M params (dùng tên gốc)", elapsed
            except Exception as e2:
                return False, str(e2)[:80], elapsed
        return False, str(e)[:80], elapsed


def main():
    print("=" * 65)
    print("  KIỂM TRA MODEL AVAILABILITY")
    print(f"  timm version : {timm.__version__}")
    print(f"  torch version: {torch.__version__}")
    print(f"  Tổng model   : {len(MODELS)}")
    print("=" * 65)

    ok_list   = []
    fail_list = []

    for i, name in enumerate(MODELS, 1):
        sys.stdout.write(f"  [{i:02d}/{len(MODELS)}] {name:<30} ... ")
        sys.stdout.flush()
        ok, msg, ms = check_model(name)
        if ok:
            print(f"OK  ({msg}, {ms:.0f}ms)")
            ok_list.append(name)
        else:
            print(f"FAIL  → {msg}")
            fail_list.append((name, msg))

    # ─── Tóm tắt ──────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  KẾT QUẢ: {len(ok_list)} OK / {len(fail_list)} FAIL / {len(MODELS)} tổng")
    print("=" * 65)

    if fail_list:
        print("\n  [FAIL] Các model không load được:")
        for name, reason in fail_list:
            print(f"    ✗ {name:<30} {reason}")
        print()
        print("  → Hãy xóa hoặc thay thế các model FAIL trong run_all_models.sh")
    else:
        print("\n  Tất cả model đều load được! Script sẵn sàng chạy.")

    print()
    if ok_list:
        print("  [OK] Dán vào run_all_models.sh (chỉ model chạy được):")
        # In theo nhóm 5 model mỗi dòng
        chunks = [ok_list[i:i+5] for i in range(0, len(ok_list), 5)]
        for chunk in chunks:
            print("    " + " ".join(chunk))


if __name__ == "__main__":
    main()
