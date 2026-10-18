"""
Script chính xác để đếm FLOPS sử dụng thư viện torchinfo hoặc fvcore.
"""

import torch
import torch.nn as nn
from models import get_model, count_parameters
import pandas as pd

try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("torchinfo không có sẵn. Cài đặt: pip install torchinfo")

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("thop không có sẵn. Cài đặt: pip install thop")


def get_flops_thop(model, input_shape):
    """Tính FLOPS bằng thop library."""
    if not THOP_AVAILABLE:
        return -1, -1
    
    try:
        model.eval()
        input_tensor = torch.randn(1, *input_shape)
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        # Chuyển đổi định dạng
        macs_str, params_str = clever_format([macs, params], "%.3f")
        
        return macs, params
    except Exception as e:
        print(f"Lỗi thop: {e}")
        return -1, -1


def get_model_summary(model, input_shape):
    """Lấy thông tin model bằng torchinfo."""
    if not TORCHINFO_AVAILABLE:
        return None
    
    try:
        model_stats = summary(
            model, 
            input_size=(1, *input_shape),
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "mult_adds"]
        )
        return model_stats
    except Exception as e:
        print(f"Lỗi torchinfo: {e}")
        return None


def analyze_model_accurate(model_name: str) -> dict:
    """Phân tích model với độ chính xác cao."""
    print(f"\nPhân tích {model_name}...")
    
    try:
        # Tạo model
        model = get_model(model_name, num_classes=2, pretrained=False)
        model.eval()
        
        # Input shape cho cả 2 task
        input_shape = (3, 224, 224)
        
        # Đếm parameters bằng function có sẵn
        param_info = count_parameters(model)
        
        # Tính FLOPS bằng thop
        macs_thop, params_thop = get_flops_thop(model, input_shape)
        flops_thop = 2 * macs_thop if macs_thop > 0 else -1
        
        # Lấy summary bằng torchinfo
        model_stats = get_model_summary(model, input_shape)
        
        # Thông tin từ torchinfo
        torchinfo_flops = -1
        if model_stats is not None:
            try:
                torchinfo_flops = model_stats.total_mult_adds * 2  # MACs -> FLOPS
            except:
                torchinfo_flops = -1
        
        result = {
            'model_name': model_name,
            'total_params': param_info['total'],
            'trainable_params': param_info['trainable'],
            'params_thop': params_thop,
            'macs_thop': macs_thop,
            'flops_thop': flops_thop,
            'flops_torchinfo': torchinfo_flops,
            'success': True
        }
        
        print(f"  ✓ Parameters: {param_info['total']:,}")
        if flops_thop > 0:
            print(f"  ✓ FLOPS (thop): {flops_thop:,.0f}")
        if torchinfo_flops > 0:
            print(f"  ✓ FLOPS (torchinfo): {torchinfo_flops:,.0f}")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Lỗi: {e}")
        return {
            'model_name': model_name,
            'error': str(e),
            'success': False
        }


def get_reference_specs():
    """Thông số tham khảo từ các paper chính thức."""
    return {
        'resnet50': {
            'params': 25_557_032,
            'flops': 4_089_184_256,  # 4.1G FLOPS
        },
        'vgg19': {
            'params': 143_667_240,
            'flops': 19_632_428_032,  # 19.6G FLOPS
        },
        'densenet121': {
            'params': 7_978_856,
            'flops': 2_834_378_752,  # 2.8G FLOPS
        },
        'efficientnet_b3': {
            'params': 12_233_232,
            'flops': 1_827_000_000,  # 1.8G FLOPS
        },
        'convnext_tiny': {
            'params': 28_589_128,
            'flops': 4_460_000_000,  # 4.5G FLOPS
        },
        'mobilenetv3_small_100': {
            'params': 2_542_856,
            'flops': 56_000_000,  # 56M FLOPS
        }
    }


def main():
    """Chạy phân tích chính xác."""
    
    print("="*70)
    print("PHÂN TÍCH CHÍNH XÁC PARAMETERS VÀ FLOPS")
    print("="*70)
    
    models_to_analyze = [
        'convnext_tiny',
        'vgg19', 
        'resnet50',
        'densenet121',
        'efficientnet_b3',
        'mobilenetv3_small_100'
    ]
    
    results = []
    reference_specs = get_reference_specs()
    
    for model_name in models_to_analyze:
        result = analyze_model_accurate(model_name)
        if result['success']:
            results.append(result)
    
    # Tạo bảng so sánh
    if results:
        print(f"\n{'='*100}")
        print("BẢNG SO SÁNH KẾT QUẢ")
        print(f"{'='*100}")
        
        df_data = []
        for result in results:
            model_name = result['model_name']
            ref_spec = reference_specs.get(model_name, {})
            
            # Chọn FLOPS tốt nhất
            flops_computed = result.get('flops_thop', -1)
            if flops_computed <= 0:
                flops_computed = result.get('flops_torchinfo', -1)
            
            flops_reference = ref_spec.get('flops', -1)
            
            # Chọn FLOPS để hiển thị
            if flops_computed > 0:
                final_flops = flops_computed
                flops_source = "Computed"
            else:
                final_flops = flops_reference
                flops_source = "Reference"
            
            df_data.append({
                'Model': model_name,
                'Parameters': f"{result['total_params']:,}",
                'Params (M)': f"{result['total_params']/1e6:.2f}M",
                'FLOPS': f"{final_flops:,}" if final_flops > 0 else "N/A",
                'FLOPS (G)': f"{final_flops/1e9:.2f}G" if final_flops > 0 else "N/A",
                'FLOPS Source': flops_source,
                'Reference FLOPS (G)': f"{flops_reference/1e9:.2f}G" if flops_reference > 0 else "N/A",
                'Computed FLOPS (G)': f"{flops_computed/1e9:.2f}G" if flops_computed > 0 else "N/A",
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Lưu kết quả
        output_file = '/home/son/new_mia/antibiotics_framework/accurate_model_analysis.csv'
        df.to_csv(output_file, index=False)
        print(f"\nKết quả đã lưu vào: {output_file}")
        
        # Tạo bảng đơn giản cho Task 1 và Task 2
        print(f"\n{'='*80}")
        print("BẢNG KẾT QUẢ CHO TASK 1 VÀ TASK 2")
        print(f"{'='*80}")
        
        simple_data = []
        for _, row in df.iterrows():
            # Task 1
            simple_data.append({
                'Task': 'Task 1',
                'Model': row['Model'],
                'Parameters': row['Parameters'],
                'Params (M)': row['Params (M)'],
                'FLOPS': row['FLOPS'],
                'FLOPS (G)': row['FLOPS (G)']
            })
            # Task 2 (giống Task 1)
            simple_data.append({
                'Task': 'Task 2', 
                'Model': row['Model'],
                'Parameters': row['Parameters'],
                'Params (M)': row['Params (M)'],
                'FLOPS': row['FLOPS'],
                'FLOPS (G)': row['FLOPS (G)']
            })
        
        simple_df = pd.DataFrame(simple_data)
        print(simple_df.to_string(index=False))
        
        # Lưu bảng đơn giản
        simple_output = '/home/son/new_mia/antibiotics_framework/task_model_analysis.csv'
        simple_df.to_csv(simple_output, index=False)
        print(f"\nBảng Task đã lưu vào: {simple_output}")
        
        print(f"\n{'='*60}")
        print("GHI CHÚ")
        print(f"{'='*60}")
        print("• FLOPS cho Task 1 và Task 2 giống nhau (cùng input size 224x224)")
        print("• 'Computed' = tính từ model thực tế")
        print("• 'Reference' = từ paper/documentation chính thức")
        print("• Các con số có thể khác nhau tùy implementation và cách tính")


if __name__ == '__main__':
    main()