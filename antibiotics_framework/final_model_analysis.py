"""
Script cuối cùng để tạo bảng thông số đầy đủ cho tất cả models yêu cầu.
Bao gồm: convnextv2_tiny, convnext_tiny, vgg19, resnet50, mobilenet, densenet121, efficientnet_b3
"""

import torch
import pandas as pd
from models import get_model, count_parameters


def get_all_model_specs():
    """Lấy thông số tất cả models từ nhiều nguồn."""
    
    # Thông số chính thức từ papers và timm documentation
    specs = {
        'convnextv2_tiny': {
            'params': 28_636_196,  # ConvNeXt V2 Tiny
            'flops': 4_470_000_000,  # ~4.47G FLOPS
            'note': 'ConvNeXt V2 Tiny (improved version)'
        },
        'convnext_tiny': {
            'params': 28_589_128,  # ConvNeXt Tiny
            'flops': 4_460_000_000,  # ~4.46G FLOPS  
            'note': 'ConvNeXt Tiny (original)'
        },
        'vgg19': {
            'params': 143_667_240,  # VGG19
            'flops': 19_632_428_032,  # ~19.63G FLOPS
            'note': 'VGG19 with BatchNorm'
        },
        'resnet50': {
            'params': 25_557_032,  # ResNet50
            'flops': 4_089_184_256,  # ~4.09G FLOPS
            'note': 'ResNet50 standard'
        },
        'mobilenetv3_small_100': {
            'params': 2_542_856,  # MobileNet V3 Small
            'flops': 56_000_000,  # ~56M FLOPS
            'note': 'MobileNet V3 Small (efficient)'
        },
        'densenet121': {
            'params': 7_978_856,  # DenseNet121
            'flops': 2_834_378_752,  # ~2.83G FLOPS
            'note': 'DenseNet121 (densnet16 equivalent)'
        },
        'efficientnet_b3': {
            'params': 12_233_232,  # EfficientNet B3
            'flops': 1_827_000_000,  # ~1.83G FLOPS
            'note': 'EfficientNet B3'
        }
    }
    
    return specs


def try_get_actual_params(model_name):
    """Thử lấy thông số thực tế từ model nếu có thể."""
    try:
        model = get_model(model_name, num_classes=2, pretrained=False)
        param_info = count_parameters(model)
        return param_info['total'], True
    except Exception as e:
        print(f"Không thể tạo {model_name}: {e}")
        return None, False


def create_final_analysis():
    """Tạo phân tích cuối cùng."""
    
    print("="*80)
    print("PHÂN TÍCH CUỐI CÙNG - TẤT CẢ MODELS YÊU CẦU")
    print("="*80)
    
    # Lấy thông số tham khảo
    all_specs = get_all_model_specs()
    
    # Danh sách models theo yêu cầu
    requested_models = [
        'convnextv2_tiny',
        'convnext_tiny', 
        'vgg19',
        'resnet50',
        'mobilenetv3_small_100',  # MobileNet
        'densenet121',  # DenseNet (thay cho densnet16)
        'efficientnet_b3'
    ]
    
    results = []
    
    print("\nKiểm tra từng model:")
    print("-" * 50)
    
    for model_name in requested_models:
        print(f"\n{model_name}:")
        
        # Lấy thông số tham khảo
        ref_spec = all_specs.get(model_name, {})
        ref_params = ref_spec.get('params', 0)
        ref_flops = ref_spec.get('flops', 0)
        note = ref_spec.get('note', '')
        
        # Thử lấy thông số thực tế
        actual_params, success = try_get_actual_params(model_name)
        
        # Chọn thông số cuối cùng
        if success and actual_params:
            final_params = actual_params
            params_source = "Computed"
            print(f"  ✓ Computed params: {actual_params:,}")
        else:
            final_params = ref_params
            params_source = "Reference"
            print(f"  ✗ Using reference params: {ref_params:,}")
        
        print(f"  • Reference FLOPS: {ref_flops:,}")
        print(f"  • Note: {note}")
        
        # Thêm vào kết quả
        results.append({
            'model_name': model_name,
            'final_params': final_params,
            'reference_params': ref_params,
            'reference_flops': ref_flops,
            'params_source': params_source,
            'note': note,
            'success': success
        })
    
    return results


def create_output_tables(results):
    """Tạo các bảng kết quả."""
    
    # Bảng tổng hợp
    print(f"\n{'='*100}")
    print("BẢNG TỔNG HỢP THÔNG SỐ MODELS")
    print(f"{'='*100}")
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model_name'],
            'Parameters': f"{result['final_params']:,}",
            'Params (M)': f"{result['final_params']/1e6:.2f}M",
            'FLOPS': f"{result['reference_flops']:,}",
            'FLOPS (G)': f"{result['reference_flops']/1e9:.2f}G",
            'Params Source': result['params_source'],
            'Note': result['note']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Bảng cho Task 1 và Task 2
    print(f"\n{'='*80}")
    print("BẢNG CHO TASK 1 VÀ TASK 2")
    print(f"{'='*80}")
    
    task_data = []
    for result in results:
        # Task 1
        task_data.append({
            'Task': 'Task 1 (Masked Cell Images)',
            'Model': result['model_name'],
            'Parameters': f"{result['final_params']:,}",
            'Params (M)': f"{result['final_params']/1e6:.2f}M",
            'FLOPS': f"{result['reference_flops']:,}",
            'FLOPS (G)': f"{result['reference_flops']/1e9:.2f}G"
        })
        
        # Task 2  
        task_data.append({
            'Task': 'Task 2 (Original Images 224x224)',
            'Model': result['model_name'],
            'Parameters': f"{result['final_params']:,}",
            'Params (M)': f"{result['final_params']/1e6:.2f}M",
            'FLOPS': f"{result['reference_flops']:,}",
            'FLOPS (G)': f"{result['reference_flops']/1e9:.2f}G"
        })
    
    task_df = pd.DataFrame(task_data)
    print(task_df.to_string(index=False))
    
    # Lưu files
    output_dir = '/home/son/new_mia/antibiotics_framework/'
    
    summary_file = output_dir + 'final_model_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    task_file = output_dir + 'final_task_analysis.csv'
    task_df.to_csv(task_file, index=False)
    
    print(f"\nFiles đã lưu:")
    print(f"• Tổng hợp: {summary_file}")
    print(f"• Task 1&2: {task_file}")
    
    return summary_df, task_df


def print_model_ranking(summary_df):
    """In bảng xếp hạng models."""
    
    print(f"\n{'='*60}")
    print("XẾP HẠNG MODELS")
    print(f"{'='*60}")
    
    # Sắp xếp theo parameters (tăng dần)
    print("\nTheo Parameters (ít → nhiều):")
    params_ranking = summary_df.copy()
    params_ranking['params_numeric'] = params_ranking['Parameters'].str.replace(',', '').astype(int)
    params_ranking = params_ranking.sort_values('params_numeric')
    
    for i, (_, row) in enumerate(params_ranking.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['Params (M)']} ({row['Note']})")
    
    # Sắp xếp theo FLOPS (tăng dần)
    print("\nTheo FLOPS (ít → nhiều):")
    flops_ranking = summary_df.copy()
    flops_ranking['flops_numeric'] = flops_ranking['FLOPS'].str.replace(',', '').astype(int)
    flops_ranking = flops_ranking.sort_values('flops_numeric')
    
    for i, (_, row) in enumerate(flops_ranking.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['FLOPS (G)']} ({row['Note']})")


def main():
    """Chạy phân tích cuối cùng."""
    
    # Phân tích
    results = create_final_analysis()
    
    # Tạo bảng
    summary_df, task_df = create_output_tables(results)
    
    # Xếp hạng
    print_model_ranking(summary_df)
    
    # Thông tin bổ sung
    print(f"\n{'='*60}")
    print("THÔNG TIN BỔ SUNG")
    print(f"{'='*60}")
    print("• Task 1: Masked cell images (ảnh tế bào đã mask)")
    print("• Task 2: Original images với 224x224 crops (ảnh gốc cắt 224x224)")
    print("• FLOPS giống nhau cho cả 2 task vì cùng input size")
    print("• Parameters không đổi giữa các task")
    print("• convnextv2_tiny là phiên bản cải tiến của convnext_tiny")
    print("• densenet121 được dùng thay cho densnet16 (không tồn tại)")
    print("• mobilenetv3_small_100 là MobileNet hiệu quả nhất")


if __name__ == '__main__':
    main()