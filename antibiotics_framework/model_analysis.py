"""
Script để phân tích thông số params và FLOPS của các model
cho cả Task 1 (masked cell images) và Task 2 (original images với 224x224 crops).

Models cần phân tích:
- convnextv2_tiny
- convnext_tiny  
- vgg19
- resnet50
- mobilenet (mobilenetv3_small)
- densenet121 (densnet16 -> densenet121)
- efficientnet_b3
"""

import torch
import torch.nn as nn
from models import get_model, count_parameters
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("fvcore không có sẵn. Cài đặt: pip install fvcore")

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("ptflops không có sẵn. Cài đặt: pip install ptflops")

try:
    from torchprofile import profile_macs
    TORCHPROFILE_AVAILABLE = True
except ImportError:
    TORCHPROFILE_AVAILABLE = False
    print("torchprofile không có sẵn. Cài đặt: pip install torchprofile")


def get_flops_fvcore(model: nn.Module, input_shape: Tuple[int, int, int]) -> float:
    """Tính FLOPS bằng fvcore."""
    if not FVCORE_AVAILABLE:
        return -1
    
    try:
        model.eval()
        batch_size = 1
        inputs = torch.randn(batch_size, *input_shape)
        
        with torch.no_grad():
            flops_dict, _ = flop_count(model, (inputs,), supported_ops=None)
            total_flops = sum(flops_dict.values())
        
        return total_flops
    except Exception as e:
        print(f"Lỗi fvcore: {e}")
        return -1


def get_flops_ptflops(model: nn.Module, input_shape: Tuple[int, int, int]) -> Tuple[float, float]:
    """Tính FLOPS và params bằng ptflops."""
    if not PTFLOPS_AVAILABLE:
        return -1, -1
    
    try:
        model.eval()
        macs, params = get_model_complexity_info(
            model, 
            input_shape, 
            print_per_layer_stat=False,
            as_strings=False
        )
        # MACs (Multiply-Accumulate Operations) ≈ FLOPS/2
        flops = 2 * macs
        return flops, params
    except Exception as e:
        print(f"Lỗi ptflops: {e}")
        return -1, -1


def get_flops_torchprofile(model: nn.Module, input_shape: Tuple[int, int, int]) -> float:
    """Tính MACs bằng torchprofile."""
    if not TORCHPROFILE_AVAILABLE:
        return -1
    
    try:
        model.eval()
        batch_size = 1
        inputs = torch.randn(batch_size, *input_shape)
        
        with torch.no_grad():
            macs = profile_macs(model, inputs)
        
        return 2 * macs  # MACs → FLOPS
    except Exception as e:
        print(f"Lỗi torchprofile: {e}")
        return -1


def analyze_model(model_name: str, input_shape: Tuple[int, int, int], 
                  num_classes: int = 2) -> Dict:
    """Phân tích một model và trả về thông số."""
    print(f"\n=== Phân tích {model_name} với input shape {input_shape} ===")
    
    try:
        # Tạo model
        model = get_model(model_name, num_classes=num_classes, pretrained=False)
        model.eval()
        
        # Đếm parameters
        param_info = count_parameters(model)
        total_params = param_info['total']
        trainable_params = param_info['trainable']
        
        print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # Tính FLOPS bằng nhiều phương pháp
        flops_fvcore = get_flops_fvcore(model, input_shape)
        flops_ptflops, params_ptflops = get_flops_ptflops(model, input_shape)
        flops_torchprofile = get_flops_torchprofile(model, input_shape)
        
        # Chọn FLOPS tốt nhất
        flops_values = [f for f in [flops_fvcore, flops_ptflops, flops_torchprofile] if f > 0]
        best_flops = np.mean(flops_values) if flops_values else -1
        
        print(f"FLOPS (fvcore): {flops_fvcore:,.0f}")
        print(f"FLOPS (ptflops): {flops_ptflops:,.0f}")  
        print(f"FLOPS (torchprofile): {flops_torchprofile:,.0f}")
        print(f"FLOPS (trung bình): {best_flops:,.0f}")
        
        return {
            'model_name': model_name,
            'input_shape': input_shape,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'flops_fvcore': flops_fvcore,
            'flops_ptflops': flops_ptflops,
            'flops_torchprofile': flops_torchprofile,
            'flops_mean': best_flops,
            'params_ptflops': params_ptflops,
        }
        
    except Exception as e:
        print(f"Lỗi khi phân tích {model_name}: {e}")
        return {
            'model_name': model_name,
            'input_shape': input_shape,
            'error': str(e)
        }


def main():
    """Phân tích tất cả models cho cả Task 1 và Task 2."""
    
    # Danh sách models cần phân tích
    models_to_analyze = [
        'convnextv2_tiny',    # ConvNeXt V2 Tiny
        'convnext_tiny',      # ConvNeXt Tiny
        'vgg19',              # VGG19
        'resnet50',           # ResNet50
        'mobilenetv3_small_100',  # MobileNet V3 Small
        'densenet121',        # DenseNet121 (thay cho densnet16)
        'efficientnet_b3',    # EfficientNet B3
    ]
    
    # Input shapes cho 2 tasks
    # Task 1: Masked cell images (thường nhỏ hơn)
    # Task 2: Original images với 224x224 crops
    input_shapes = {
        'Task 1': (3, 224, 224),  # Giả sử cũng resize về 224x224
        'Task 2': (3, 224, 224),  # 224x224 crops từ ảnh gốc
    }
    
    results = []
    
    for task_name, input_shape in input_shapes.items():
        print(f"\n{'='*60}")
        print(f"PHÂN TÍCH CHO {task_name.upper()}")
        print(f"Input shape: {input_shape}")
        print(f"{'='*60}")
        
        for model_name in models_to_analyze:
            result = analyze_model(model_name, input_shape)
            result['task'] = task_name
            results.append(result)
    
    # Tạo bảng tổng kết
    print(f"\n{'='*80}")
    print("BẢNG TỔNG KẾT")
    print(f"{'='*80}")
    
    df_results = []
    for result in results:
        if 'error' not in result:
            df_results.append({
                'Task': result['task'],
                'Model': result['model_name'],
                'Input Shape': f"{result['input_shape']}",
                'Total Params (M)': f"{result['total_params']/1e6:.2f}",
                'Trainable Params (M)': f"{result['trainable_params']/1e6:.2f}",
                'FLOPS (G)': f"{result['flops_mean']/1e9:.2f}" if result['flops_mean'] > 0 else "N/A",
                'FLOPS (fvcore)': f"{result['flops_fvcore']/1e9:.2f}" if result['flops_fvcore'] > 0 else "N/A",
                'FLOPS (ptflops)': f"{result['flops_ptflops']/1e9:.2f}" if result['flops_ptflops'] > 0 else "N/A",
                'FLOPS (torchprofile)': f"{result['flops_torchprofile']/1e9:.2f}" if result['flops_torchprofile'] > 0 else "N/A",
            })
    
    if df_results:
        df = pd.DataFrame(df_results)
        print(df.to_string(index=False))
        
        # Lưu ra file CSV
        output_file = '/home/son/new_mia/antibiotics_framework/model_analysis_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nKết quả đã được lưu vào: {output_file}")
        
        # Tạo bảng so sánh riêng cho mỗi task
        for task in ['Task 1', 'Task 2']:
            task_df = df[df['Task'] == task].copy()
            task_df = task_df.drop('Task', axis=1)
            print(f"\n{task.upper()}:")
            print(task_df.to_string(index=False))
    
    # In thông tin chi tiết
    print(f"\n{'='*80}")
    print("CHI TIẾT KẾT QUẢ")
    print(f"{'='*80}")
    
    for result in results:
        if 'error' not in result:
            print(f"\n{result['task']} - {result['model_name']}:")
            print(f"  • Total Parameters: {result['total_params']:,}")
            print(f"  • Trainable Parameters: {result['trainable_params']:,}")
            if result['flops_mean'] > 0:
                print(f"  • FLOPS (trung bình): {result['flops_mean']:,.0f}")
            print(f"  • FLOPS (fvcore): {result['flops_fvcore']:,.0f}")
            print(f"  • FLOPS (ptflops): {result['flops_ptflops']:,.0f}")
            print(f"  • FLOPS (torchprofile): {result['flops_torchprofile']:,.0f}")
        else:
            print(f"\n{result['task']} - {result['model_name']}: LỖI - {result['error']}")


if __name__ == '__main__':
    main()