"""
Script đơn giản để phân tích params và ước tính FLOPS của các model
cho Task 1 và Task 2 mà không cần thư viện phức tạp.
"""

import torch
import torch.nn as nn
from models import get_model, count_parameters
import pandas as pd
from typing import Dict, List, Tuple


def estimate_flops_simple(model: nn.Module, input_shape: Tuple[int, int, int]) -> float:
    """
    Ước tính FLOPS đơn giản bằng cách đếm operations trong forward pass.
    Không chính xác 100% nhưng cho kết quả gần đúng.
    """
    model.eval()
    
    # Hook để đếm operations
    flops_count = [0]
    
    def conv_flop_count(module, input, output):
        if isinstance(module, nn.Conv2d):
            # FLOPS = output_elements * (kernel_size * input_channels + bias)
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            
            if len(output_tensor.shape) == 4:
                batch_size, output_channels, output_height, output_width = output_tensor.shape
            elif len(output_tensor.shape) == 3:
                batch_size, output_channels, output_size = output_tensor.shape
                output_height = output_width = int(output_size ** 0.5)
            else:
                return
                
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_elements = batch_size * output_channels * output_height * output_width
            flops_count[0] += output_elements * kernel_flops
    
    def linear_flop_count(module, input, output):
        if isinstance(module, nn.Linear):
            # FLOPS = input_features * output_features * batch_size
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input
            batch_size = input_tensor.shape[0]
            flops_count[0] += module.in_features * module.out_features * batch_size
    
    # Đăng ký hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_flop_count))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flop_count))
    
    # Forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        _ = model(dummy_input)
    
    # Xóa hooks
    for hook in hooks:
        hook.remove()
    
    return flops_count[0]


def get_model_info(model_name: str, input_shape: Tuple[int, int, int], 
                   num_classes: int = 2) -> Dict:
    """Lấy thông tin về model."""
    print(f"Phân tích {model_name}...")
    
    try:
        # Tạo model
        model = get_model(model_name, num_classes=num_classes, pretrained=False)
        
        # Đếm parameters
        param_info = count_parameters(model)
        
        # Ước tính FLOPS
        estimated_flops = estimate_flops_simple(model, input_shape)
        
        return {
            'model_name': model_name,
            'total_params': param_info['total'],
            'trainable_params': param_info['trainable'],
            'estimated_flops': estimated_flops,
            'success': True
        }
        
    except Exception as e:
        print(f"Lỗi với {model_name}: {e}")
        return {
            'model_name': model_name,
            'error': str(e),
            'success': False
        }


def main():
    """Phân tích các models."""
    
    # Danh sách models
    models_to_test = [
        'convnext_tiny',      # ConvNeXt Tiny (convnextv2_tiny có thể không có trong timm cũ)
        'vgg19',              # VGG19
        'resnet50',           # ResNet50
        'densenet121',        # DenseNet121
        'efficientnet_b3',    # EfficientNet B3
    ]
    
    # Thêm MobileNet nếu có
    try:
        # Thử các tên MobileNet khác nhau
        mobilenet_names = ['mobilenetv3_small_100', 'mobilenet_v3_small', 'mobilenetv2_100']
        for name in mobilenet_names:
            try:
                test_model = get_model(name, num_classes=2, pretrained=False)
                models_to_test.append(name)
                print(f"Tìm thấy MobileNet: {name}")
                break
            except:
                continue
    except:
        print("Không tìm thấy MobileNet phù hợp")
    
    # Input shapes cho 2 tasks
    tasks = {
        'Task 1': (3, 224, 224),  # Masked cell images
        'Task 2': (3, 224, 224),  # Original images với 224x224 crops
    }
    
    all_results = []
    
    print("="*60)
    print("PHÂN TÍCH MODELS CHO ANTIBIOTIC RESISTANCE CLASSIFICATION")
    print("="*60)
    
    for task_name, input_shape in tasks.items():
        print(f"\n{task_name} - Input shape: {input_shape}")
        print("-" * 40)
        
        task_results = []
        
        for model_name in models_to_test:
            result = get_model_info(model_name, input_shape)
            result['task'] = task_name
            result['input_shape'] = input_shape
            
            if result['success']:
                task_results.append(result)
                print(f"✓ {model_name}: {result['total_params']:,} params, "
                      f"{result['estimated_flops']:,.0f} FLOPS")
            else:
                print(f"✗ {model_name}: {result['error']}")
        
        all_results.extend(task_results)
    
    # Tạo bảng tổng kết
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
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Lưu kết quả
        output_file = '/home/son/new_mia/antibiotics_framework/model_params_flops.csv'
        df.to_csv(output_file, index=False)
        print(f"\nKết quả đã lưu vào: {output_file}")
        
        # So sánh giữa Task 1 và Task 2
        print(f"\n{'='*60}")
        print("SO SÁNH GIỮA CÁC TASKS")
        print(f"{'='*60}")
        
        for task in ['Task 1', 'Task 2']:
            task_df = df[df['Task'] == task]
            print(f"\n{task}:")
            task_summary = task_df[['Model', 'Params (M)', 'FLOPS (G)']].copy()
            print(task_summary.to_string(index=False))
        
        # Thống kê
        print(f"\n{'='*60}")
        print("THỐNG KÊ TỔNG QUAN")
        print(f"{'='*60}")
        
        unique_models = df['Model'].unique()
        for model in unique_models:
            model_data = df[df['Model'] == model]
            if len(model_data) >= 2:
                params = model_data.iloc[0]['Total Params']
                flops_task1 = model_data[model_data['Task'] == 'Task 1']['Estimated FLOPS'].iloc[0]
                flops_task2 = model_data[model_data['Task'] == 'Task 2']['Estimated FLOPS'].iloc[0]
                
                print(f"\n{model}:")
                print(f"  • Parameters: {params}")
                print(f"  • FLOPS Task 1: {flops_task1}")
                print(f"  • FLOPS Task 2: {flops_task2}")
                print(f"  • FLOPS giống nhau: {'Có' if flops_task1 == flops_task2 else 'Không'}")


if __name__ == '__main__':
    main()