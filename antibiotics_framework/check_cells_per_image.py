"""
Kiểm tra số lượng cell per image để hiểu rõ Task 1 vs Task 2.
"""

import pandas as pd
import numpy as np
from config import Config


def analyze_cells_per_image(drug='AZI'):
    """Phân tích số cell per image."""
    
    print(f"PHÂN TÍCH SỐ CELL PER IMAGE - {drug}")
    print("="*60)
    
    config = Config()
    
    # Load split info
    split_csv = config.split_csv(drug)
    df = pd.read_csv(split_csv)
    
    print(f"Total records: {len(df)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    
    # Focus on test split
    test_df = df[df['split'] == 'test'].copy()
    print(f"\nTest records: {len(test_df)}")
    
    # Group by original image
    original_col = 'original_tiff' if 'original_tiff' in test_df.columns else 'original_file'
    grouped = test_df.groupby(original_col)
    
    cells_per_image = []
    image_info = []
    
    for orig_file, group in grouped:
        num_cells = len(group)
        cells_per_image.append(num_cells)
        image_info.append({
            'original_file': orig_file,
            'num_cells': num_cells,
            'labels': group['resistant'].value_counts().to_dict()
        })
    
    # Statistics
    cells_array = np.array(cells_per_image)
    
    print(f"\n📊 THỐNG KÊ CELLS PER IMAGE:")
    print(f"  Tổng số ảnh test: {len(cells_per_image)}")
    print(f"  Tổng số cells: {sum(cells_per_image)}")
    print(f"  Trung bình: {np.mean(cells_array):.1f} cells/image")
    print(f"  Median: {np.median(cells_array):.1f} cells/image")
    print(f"  Min: {np.min(cells_array)} cells/image")
    print(f"  Max: {np.max(cells_array)} cells/image")
    print(f"  Std: {np.std(cells_array):.1f}")
    
    # Distribution
    print(f"\n📈 PHÂN PHỐI:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(cells_array, p)
        print(f"  {p}th percentile: {val:.1f} cells")
    
    # Show some examples
    print(f"\n📋 VÍ DỤ (10 ảnh đầu):")
    for i, info in enumerate(image_info[:10]):
        basename = info['original_file'].split('/')[-1] if '/' in info['original_file'] else info['original_file']
        print(f"  {i+1}. {basename}: {info['num_cells']} cells {info['labels']}")
    
    # Comparison with Task 2
    patches_per_image = 81  # Task 2 fixed
    avg_cells = np.mean(cells_array)
    
    print(f"\n⚖️  SO SÁNH VỚI TASK 2:")
    print(f"  Task 1 (avg): {avg_cells:.1f} cells/image")
    print(f"  Task 2 (fixed): {patches_per_image} patches/image")
    
    if avg_cells > patches_per_image:
        ratio = avg_cells / patches_per_image
        print(f"  → Task 1 có NHIỀU HƠN {ratio:.1f}x operations/image")
    else:
        ratio = patches_per_image / avg_cells
        print(f"  → Task 2 có NHIỀU HƠN {ratio:.1f}x operations/image")
    
    # Estimate processing time
    print(f"\n⏱️  ƯỚC TÍNH THỜI GIAN (giả sử 1000 images/sec per operation):")
    task1_time_per_image = avg_cells / 1000  # seconds
    task2_time_per_image = patches_per_image / 1000  # seconds
    
    print(f"  Task 1: {task1_time_per_image:.3f}s/image ({1/task1_time_per_image:.1f} images/sec)")
    print(f"  Task 2: {task2_time_per_image:.3f}s/image ({1/task2_time_per_image:.1f} images/sec)")
    
    return {
        'total_images': len(cells_per_image),
        'total_cells': sum(cells_per_image),
        'avg_cells': avg_cells,
        'min_cells': np.min(cells_array),
        'max_cells': np.max(cells_array),
        'patches_per_image': patches_per_image,
        'cells_vs_patches_ratio': avg_cells / patches_per_image
    }


def quick_speed_estimate():
    """Ước tính nhanh tốc độ dựa trên số operations."""
    
    print(f"\n{'='*60}")
    print("ƯỚC TÍNH TỐC ĐỘ DỰA TRÊN SỐ OPERATIONS")
    print(f"{'='*60}")
    
    # Analyze data
    stats = analyze_cells_per_image('AZI')
    
    # Assumptions
    base_inference_speed = 3000  # operations/sec (from previous benchmark)
    
    print(f"\n🔢 TÍNH TOÁN:")
    print(f"  Base inference speed: {base_inference_speed} operations/sec")
    print(f"  Task 1 avg operations/image: {stats['avg_cells']:.1f}")
    print(f"  Task 2 operations/image: {stats['patches_per_image']}")
    
    # Calculate speeds
    task1_images_per_sec = base_inference_speed / stats['avg_cells']
    task2_images_per_sec = base_inference_speed / stats['patches_per_image']
    
    print(f"\n📊 KẾT QUẢ ƯỚC TÍNH:")
    print(f"  Task 1: {task1_images_per_sec:.2f} images/sec")
    print(f"  Task 2: {task2_images_per_sec:.2f} images/sec")
    
    if task1_images_per_sec > task2_images_per_sec:
        speedup = task1_images_per_sec / task2_images_per_sec
        print(f"  → Task 1 nhanh hơn {speedup:.2f}x")
    else:
        speedup = task2_images_per_sec / task1_images_per_sec
        print(f"  → Task 2 nhanh hơn {speedup:.2f}x")
    
    print(f"\n💡 KẾT LUẬN:")
    if stats['avg_cells'] > stats['patches_per_image']:
        print(f"  • Task 1 cần xử lý NHIỀU HƠN ({stats['avg_cells']:.1f} vs {stats['patches_per_image']})")
        print(f"  • Vậy Task 1 sẽ CHẬM HƠN Task 2")
    else:
        print(f"  • Task 2 cần xử lý NHIỀU HƠN ({stats['patches_per_image']} vs {stats['avg_cells']:.1f})")
        print(f"  • Vậy Task 2 sẽ CHẬM HƠN Task 1")


if __name__ == '__main__':
    quick_speed_estimate()