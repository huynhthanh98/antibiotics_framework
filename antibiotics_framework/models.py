"""
Model factory cho Antibiotic Resistance Classification Framework.

Sử dụng thư viện `timm` (PyTorch Image Models) để load các backbone:
    - ResNet:       resnet18, resnet50, resnet101, resnet152
    - VGG:          vgg16, vgg16_bn, vgg19, vgg19_bn
    - ConvNeXt:     convnext_tiny, convnext_small, convnext_base, convnext_large
    - DenseNet:     densenet121, densenet169, densenet201
    - EfficientNet: efficientnet_b0 ... efficientnet_b7
    - Swin:         swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
    - ViT:          vit_base_patch16_224, vit_small_patch16_224
    - ... bất kỳ model nào có trong timm

Tất cả model được fine-tune với 2 class đầu ra (sensitive / resistant).
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[models.py] Cảnh báo: timm không được cài đặt. Chuyển sang torchvision.")

import torchvision.models as tv_models


# ─── Mapping tên model ngắn gọn → tên đầy đủ trong timm ──────────────────────
TIMM_MODEL_ALIASES = {
    # ResNet
    'resnet18':   'resnet18',
    'resnet50':   'resnet50',
    'resnet101':  'resnet101',
    'resnet152':  'resnet152',
    # VGG
    'vgg16':      'vgg16',
    'vgg16_bn':   'vgg16_bn',
    'vgg19':      'vgg19',
    'vgg19_bn':   'vgg19_bn',
    # ConvNeXt
    'convnext_tiny':  'convnext_tiny',
    'convnext_small': 'convnext_small',
    'convnext_base':  'convnext_base',
    'convnext_large': 'convnext_large',
    # DenseNet
    'densenet121': 'densenet121',
    'densenet169': 'densenet169',
    'densenet201': 'densenet201',
    # EfficientNet
    'efficientnet_b0': 'efficientnet_b0',
    'efficientnet_b1': 'efficientnet_b1',
    'efficientnet_b2': 'efficientnet_b2',
    'efficientnet_b3': 'efficientnet_b3',
    'efficientnet_b4': 'efficientnet_b4',
    'efficientnet_b5': 'efficientnet_b5',
    'efficientnet_b7': 'efficientnet_b7',
    # Swin Transformer
    'swin_tiny':  'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base':  'swin_base_patch4_window7_224',
    # Vision Transformer
    'vit_small':  'vit_small_patch16_224',
    'vit_base':   'vit_base_patch16_224',
    'vit_large':  'vit_large_patch16_224',
}


def get_model(model_name: str, num_classes: int = 2,
              pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    """
    Tạo model với backbone đã chọn, fine-tune cho num_classes lớp.

    Args:
        model_name  : Tên model (xem TIMM_MODEL_ALIASES hoặc bất kỳ tên timm)
        num_classes : Số lớp đầu ra (2: sensitive/resistant)
        pretrained  : Dùng pretrained ImageNet weights
        drop_rate   : Dropout rate cho classifier head

    Returns:
        nn.Module: Model sẵn sàng để train

    Examples:
        >>> model = get_model('resnet50', num_classes=2, pretrained=True)
        >>> model = get_model('efficientnet_b0')
        >>> model = get_model('convnext_small')
    """
    # Resolve alias
    resolved_name = TIMM_MODEL_ALIASES.get(model_name, model_name)

    if TIMM_AVAILABLE:
        return _build_timm_model(resolved_name, num_classes, pretrained, drop_rate)
    else:
        return _build_torchvision_model(model_name, num_classes, pretrained)


def _build_timm_model(model_name: str, num_classes: int,
                      pretrained: bool, drop_rate: float) -> nn.Module:
    """Tạo model từ thư viện timm."""
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        print(f"  [models] Đã tạo '{model_name}' từ timm "
              f"(pretrained={pretrained}, num_classes={num_classes})")
        return model
    except Exception as e:
        raise ValueError(
            f"Không thể tạo model '{model_name}' từ timm.\n"
            f"Lỗi: {e}\n"
            f"Kiểm tra danh sách model: timm.list_models('{model_name}*')"
        )


def _build_torchvision_model(model_name: str, num_classes: int,
                              pretrained: bool) -> nn.Module:
    """
    Fallback: tạo model từ torchvision nếu timm không available.
    Chỉ hỗ trợ một số model cơ bản.
    """
    weights_arg = 'DEFAULT' if pretrained else None
    model_name_lower = model_name.lower()

    if 'resnet50' in model_name_lower:
        model = tv_models.resnet50(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'resnet18' in model_name_lower:
        model = tv_models.resnet18(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'resnet101' in model_name_lower:
        model = tv_models.resnet101(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vgg16' in model_name_lower:
        model = tv_models.vgg16(weights=weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'vgg19' in model_name_lower:
        model = tv_models.vgg19(weights=weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'densenet121' in model_name_lower:
        model = tv_models.densenet121(weights=weights_arg)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'efficientnet_b0' in model_name_lower:
        model = tv_models.efficientnet_b0(weights=weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'efficientnet_b3' in model_name_lower:
        model = tv_models.efficientnet_b3(weights=weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        # Fallback cuối cùng: ResNet50
        print(f"  [models] Model '{model_name}' không được hỗ trợ trong torchvision. "
              f"Dùng ResNet50 thay thế.")
        model = tv_models.resnet50(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    print(f"  [models] Đã tạo '{model_name}' từ torchvision (pretrained={pretrained})")
    return model


def count_parameters(model: nn.Module) -> dict:
    """Đếm số lượng parameters của model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total':     total,
        'trainable': trainable,
        'frozen':    total - trainable,
    }


def list_available_models(pattern: str = '') -> list:
    """Liệt kê các model có sẵn trong timm."""
    if TIMM_AVAILABLE:
        return timm.list_models(f'*{pattern}*')
    else:
        return list(TIMM_MODEL_ALIASES.keys())

