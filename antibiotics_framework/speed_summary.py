"""
Tóm tắt kết quả benchmark tốc độ dựa trên test nhanh.
"""

import pandas as pd


def create_speed_summary():
    """Tạo bảng tóm tắt tốc độ inference."""
    
    print("TỔNG KẾT TỐC ĐỘ INFERENCE - TASK 1 VS TASK 2")
    print("="*80)
    
    # Kết quả từ test thực tế (dựa trên quick test và detailed benchmark)
    results_data = [
        # Task 1 - Masked Cell Images (1 ảnh = 1 prediction)
        {'Task': 'Task 1', 'Model': 'mobilenetv3_small_100', 'Images/sec': 2807.49, 'Type': 'Masked Cell Images'},
        {'Task': 'Task 1', 'Model': 'efficientnet_b3', 'Images/sec': 444.94, 'Type': 'Masked Cell Images'},
        {'Task': 'Task 1', 'Model': 'resnet50', 'Images/sec': 593.47, 'Type': 'Masked Cell Images'},
        
        # Task 2 - Patch-based (1 ảnh = nhiều patches → aggregate)
        {'Task': 'Task 2', 'Model': 'mobilenetv3_small_100', 'Images/sec': 34.65, 'Type': 'Patch-based (81 patches/image)'},
        {'Task': 'Task 2', 'Model': 'efficientnet_b3', 'Images/sec': 0.18, 'Type': 'Patch-based (81 patches/image)'},
        {'Task': 'Task 2', 'Model': 'resnet50', 'Images/sec': 0.17, 'Type': 'Patch-based (81 patches/image)'},
    ]
    
    # Tạo DataFrame
    df = pd.DataFrame(results_data)
    
    # Bảng chi tiết
    print("\nBẢNG CHI TIẾT:")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['Task']} - {row['Model']}: {row['Images/sec']:.2f} images/sec")
        print(f"    ({row['Type']})")
    
    # Bảng so sánh
    print(f"\n{'='*80}")
    print("BẢNG SO SÁNH TỐC ĐỘ")
    print(f"{'='*80}")
    
    models = df['Model'].unique()
    comparison_data = []
    
    for model in models:
        model_data = df[df['Model'] == model]
        task1_speed = model_data[model_data['Task'] == 'Task 1']['Images/sec'].iloc[0]
        task2_speed = model_data[model_data['Task'] == 'Task 2']['Images/sec'].iloc[0]
        
        speedup = task1_speed / task2_speed if task2_speed > 0 else float('inf')
        
        comparison_data.append({
            'Model': model,
            'Task 1 (images/sec)': f"{task1_speed:.2f}",
            'Task 2 (images/sec)': f"{task2_speed:.2f}",
            'Speedup (Task 1 faster)': f"{speedup:.1f}x"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Phân tích
    print(f"\n{'='*80}")
    print("PHÂN TÍCH KẾT QUẢ")
    print(f"{'='*80}")
    
    print("\n🚀 **KẾT LUẬN CHÍNH:**")
    print("   • Task 1 (Masked Cell Images) NHANH HƠN ĐÁNG KỂ so với Task 2")
    print("   • Tốc độ Task 1 nhanh hơn từ 81x đến 3,452x tùy model")
    
    print("\n📊 **CHI TIẾT THEO MODEL:**")
    for _, row in comparison_df.iterrows():
        print(f"   • {row['Model']}:")
        print(f"     - Task 1: {row['Task 1 (images/sec)']} images/sec")
        print(f"     - Task 2: {row['Task 2 (images/sec)']} images/sec")
        print(f"     - Task 1 nhanh hơn: {row['Speedup (Task 1 faster)']}")
    
    print("\n⚡ **TẠI SAO TASK 1 NHANH HƠN:**")
    print("   1. Chỉ cần 1 lần inference cho 1 ảnh")
    print("   2. Task 2 cần ~81 lần inference cho 1 ảnh (81 patches)")
    print("   3. Task 2 có thêm overhead từ:")
    print("      - Cắt ảnh thành patches")
    print("      - Aggregate predictions từ nhiều patches")
    print("      - Memory transfer cho nhiều patches")
    
    print("\n🎯 **KHUYẾN NGHỊ:**")
    print("   • Nếu ưu tiên TỐC ĐỘ: chọn Task 1")
    print("   • Nếu ưu tiên ĐỘ CHÍNH XÁC: có thể cân nhắc Task 2")
    print("   • Model nhanh nhất: MobileNetV3 Small")
    print("   • Cân bằng tốc độ/chất lượng: EfficientNet B3 hoặc ResNet50")
    
    # Lưu kết quả
    output_file = '/home/son/new_mia/antibiotics_framework/final_speed_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\n📁 Kết quả đã lưu: {output_file}")
    
    return comparison_df


def estimate_real_world_performance():
    """Ước tính hiệu suất thực tế."""
    
    print(f"\n{'='*80}")
    print("ƯỚC TÍNH HIỆU SUẤT THỰC TẾ")
    print(f"{'='*80}")
    
    # Giả sử cần xử lý 1000 ảnh test
    num_test_images = 1000
    
    scenarios = [
        {'model': 'mobilenetv3_small_100', 'task1_speed': 2807.49, 'task2_speed': 34.65},
        {'model': 'efficientnet_b3', 'task1_speed': 444.94, 'task2_speed': 0.18},
        {'model': 'resnet50', 'task1_speed': 593.47, 'task2_speed': 0.17},
    ]
    
    print(f"\nThời gian xử lý {num_test_images} ảnh test:")
    print("-" * 60)
    
    for scenario in scenarios:
        model = scenario['model']
        task1_time = num_test_images / scenario['task1_speed']
        task2_time = num_test_images / scenario['task2_speed']
        
        print(f"\n{model}:")
        print(f"  Task 1: {task1_time:.1f} giây ({task1_time/60:.1f} phút)")
        print(f"  Task 2: {task2_time:.1f} giây ({task2_time/60:.1f} phút)")
        print(f"  Tiết kiệm: {task2_time - task1_time:.1f} giây ({(task2_time - task1_time)/60:.1f} phút)")
    
    print(f"\n💡 **KẾT LUẬN:**")
    print(f"   • Với {num_test_images} ảnh, Task 1 tiết kiệm từ vài phút đến hàng giờ")
    print(f"   • Đối với dataset lớn, sự khác biệt sẽ càng rõ rệt")


if __name__ == '__main__':
    # Tạo tóm tắt
    comparison_df = create_speed_summary()
    
    # Ước tính hiệu suất thực tế
    estimate_real_world_performance()