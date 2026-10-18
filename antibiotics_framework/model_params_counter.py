"""
Script đơn giản để đếm parameters của các model và ước tính FLOPS cơ bản.
"""

import torch
import torch.nn as nn
from models import get_model, count_parameters
import pandas as pd
from typing import Dict, List


def estimate_flops_basic(model: nn.Module, input_shape: tuple) -> float:
    """Ước tính FLOPS cơ bản dựa trên kiến trúc model."""
    total_flops = 0
    
    # Đếm Conv2d layers
    conv_flops = 0
    linear_flops = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Ước tính cho Conv2d: kernel_size^2 * in_channels * out_channels * output_size^2
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            # Giả sử output size trung bình là input_size / stride
            if hasattr(module, 'stride'):
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            else:
                stride = 1
            
            # Ước tính output size (đơn giản)
            output_size = input_shape[1] // stride  # Giả sử square input
            output_elements = module.out_channels * output_size * output_size
            
            conv_flops += kernel_ops * output_elements
            
        elif isinstance(module, nn.Linear):
            # FLOPS cho Linear: in_features * out_features
            linear_flops += module.in_features * module.out_features
    
    total_flops = conv_flops + linear_flops
    return total_flops


def analyze_model_simple(model_name: str, task_name: str) -> Dict:
    """Phân tích model đơn giản."""
    print(f"Đang phân tích {model_name} cho {task_name}...")
    
    try:
        # Tạo model
        model = get_model(model_name, num_classes=2, pretrained=False)
        
        # Đếm parameters
        param_info = count_parameters(model)
        
        # Ước tính FLOPS cơ bản
        input_shape = (3, 224, 224)  # Cả 2 task đều dùng 224x224
        estimated_flops = estimate_flops_basic(model, input_shape)
        
        result = {
            'task': task_name,
            'model_name': model_name,
            'total_params': param_info['total'],
            'trainable_params': param_info['trainable'],
            'estimated_flops': estimated_flops,
            'success': True
        }
        
        print(f"  ✓ {param_info['total']:,} params, ~{estimated_flops:,.0f} FLOPS")
        return result
        
    except Exception as e:
        print(f"  ✗ Lỗi: {e}")
        return {
            'task': task_name,
            'model_name': model_name,
            'error': str(e),
            'success': False
        }


def get_model_complexity_manual(model_name: str) -> Dict:
    """
    Lấy thông số model theo cách thủ công dựa trên kiến thức về kiến trúc.
    Dùng khi không thể tạo model được.
    """
    
    # Thông số tham khảo từ các paper và documentation
    model_specs = {
        'resnet50': {
            'params': 25_557_032,  # ~25.6M parameters
            'flops': 4_089_184_256,  # ~4.1G FLOPS cho 224x224
        },
        'vgg19': {
            'params': 143_667_240,  # ~144M parameters
            'flops': 19_632_428_032,  # ~19.6G FLOPS
        },
        'densenet121': {
            'params': 7_978_856,  # ~8M parameters
            'flops': 2_834_378_752,  # ~2.8G FLOPS
        },
        'efficientnet_b3': {
            'params': 12_233_232,  # ~12.2M parameters
            'flops': 1_827_000_000,  # ~1.8G FLOPS
        },
        'convnext_tiny': {
            'params': 28_589_128,  # ~28.6M parameters
            'flops': 4_460_000_000,  # ~4.5G FLOPS
        },
        'mobilenetv3_small_100': {
            'params': 2_542_856,  # ~2.5M parameters
            'flops': 56_000_000,  # ~56M FLOPS
        }
    }
    
    return model_specs.get(model_name, {
        'params': 0,
        'flops': 0
    })


def main():
    """Chạy phân tích chính."""
    
    print("="*70)
    print("PHÂN TÍCH PARAMETERS VÀ FLOPS CHO CÁC MODEL")
    print("="*70)
    
    # Danh sách models cần phân tích
    models_to_analyze = [
        'convnext_tiny',      # ConvNeXt Tiny
        'vgg19',              # VGG19
        'resnet50',           # ResNet50
        'densenet121',        # DenseNet121
        'efficientnet_b3',    # EfficientNet B3
        'mobilenetv3_small_100',  # MobileNet V3 Small
    ]
    
    tasks = ['Task 1', 'Task 2']
    
    all_results = []
    
    # Phân tích từng model
    for model_name in models_to_analyze:
        print(f"\n{'-'*50}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'-'*50}")
        
        # Thử phân tích bằng code
        success = False
        for task in tasks:
            result = analyze_model_simple(model_name, task)
            if result['success']:
                all_results.append(result)
                success = True
            else:
                # Nếu lỗi, dùng thông số thủ công
                manual_specs = get_model_complexity_manual(model_name)
                if manual_specs['params'] > 0:
                    manual_result = {
                        'task': task,
                        'model_name': model_name,
                        'total_params': manual_specs['params'],
                        'trainable_params': manual_specs['params'],  # Giả sử tất cả trainable
                        'estimated_flops': manual_specs['flops'],
                        'success': True,
                        'manual': True
                    }
                    all_results.append(manual_result)
                    print(f"  ✓ {task} (thủ công): {manual_specs['params']:,} params, {manual_specs['flops']:,.0f} FLOPS")
                    success = True
        
        if not success:
            print(f"  ✗ Không thể phân tích {model_name}")
    
    # Tạo bảng kết quả
    if all_results:
        print(f"\n{'='*80}")
        print("BẢNG TỔNG KẾT")
        print(f"{'='*80}")
        
        # Tạo DataFrame
        df_data = []
        for result in all_results:
            df_data.append({
                'Task': result['task'],
                'Model': result['model_name'],
                'Total Params': f"{result['total_params']:,}",
                'Params (M)': f"{result['total_params']/1e6:.2f}M",
                'Estimated FLOPS': f"{result['estimated_flops']:,}",
                'FLOPS (G)': f"{result['estimated_flops']/1e9:.2f}G",
                'Source': 'Manual' if result.get('manual', False) else 'Computed'
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Lưu kết quả
        output_file = '/home/son/new_mia/antibiotics_framework/model_complexity_analysis.csv'
        df.to_csv(output_file, index=False)
        print(f"\nKết quả đã lưu vào: {output_file}")
        
        # Tạo bảng tóm tắt theo model (loại bỏ trùng lặp giữa Task 1 và Task 2)
        print(f"\n{'='*60}")
        print("BẢNG TÓM TẮT THEO MODEL")
        print(f"{'='*60}")
        
        summary_data = []
        unique_models = df['Model'].unique()
        
        for model in unique_models:
            model_data = df[df['Model'] == model].iloc[0]  # Lấy dòng đầu tiên
            summary_data.append({
                'Model': model_data['Model'],
                'Parameters': model_data['Total Params'],
                'Params (M)': model_data['Params (M)'],
                'FLOPS': model_data['Estimated FLOPS'],
                'FLOPS (G)': model_data['FLOPS (G)'],
                'Source': model_data['Source']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Lưu bảng tóm tắt
        summary_file = '/home/son/new_mia/antibiotics_framework/model_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nBảng tóm tắt đã lưu vào: {summary_file}")
        
        print(f"\n{'='*60}")
        print("LƯU Ý")
        print(f"{'='*60}")
        print("• FLOPS cho Task 1 và Task 2 giống nhau vì cả hai đều dùng input 224x224")
        print("• Các con số FLOPS là ước tính, có thể khác với thực tế tùy implementation")
        print("• Parameters là số lượng trọng số của model (không đổi giữa các task)")
        print("• Source 'Manual' = lấy từ specification, 'Computed' = tính toán từ model")


if __name__ == '__main__':
    main()