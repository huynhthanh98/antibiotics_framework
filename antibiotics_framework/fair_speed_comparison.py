"""
So sánh công bằng tốc độ giữa Task 1 và Task 2.

Task 1: 1 ảnh gốc → detect/load tất cả cells → predict từng cell → vote
Task 2: 1 ảnh gốc → cắt 81 patches → predict từng patch → vote
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict, Counter

from models import get_model
from datasets import Task2TestDataset, get_test_transform
from config import Config


def count_cells_per_image(drug_dir: str, test_df: pd.DataFrame):
    """Đếm số cell per image từ split_info.csv."""
    
    print("Đang đếm số cell per image...")
    
    # Group by original image
    original_col = 'original_tiff' if 'original_tiff' in test_df.columns else 'original_file'
    grouped = test_df.groupby(original_col)
    
    cells_per_image = {}
    total_cells = 0
    
    for orig_file, group in grouped:
        num_cells = len(group)
        cells_per_image[orig_file] = num_cells
        total_cells += num_cells
        
    avg_cells = total_cells / len(cells_per_image)
    
    print(f"Tổng số ảnh: {len(cells_per_image)}")
    print(f"Tổng số cells: {total_cells}")
    print(f"Trung bình cells/ảnh: {avg_cells:.1f}")
    print(f"Min cells/ảnh: {min(cells_per_image.values())}")
    print(f"Max cells/ảnh: {max(cells_per_image.values())}")
    
    return cells_per_image, avg_cells


def simulate_task1_fair(model, test_df, drug_dir, test_transform, device, cells_per_image):
    """
    Simulate Task 1 công bằng: 
    Load tất cả cells của 1 ảnh → predict → aggregate
    """
    
    print(f"\n=== TASK 1 SIMULATION (FAIR) ===")
    
    # Group cells by original image
    original_col = 'original_tiff' if 'original_tiff' in test_df.columns else 'original_file'
    grouped = test_df.groupby(original_col)
    
    image_times = []
    total_images = 0
    total_cells_processed = 0
    
    model.eval()
    
    with torch.no_grad():
        for orig_file, group in grouped:
            if total_images >= 10:  # Limit for quick test
                break
                
            start_time = time.time()
            
            # Load all cells for this image
            cell_predictions = []
            cells_loaded = 0
            
            for _, row in group.iterrows():
                label = int(row['resistant'])
                cell_file = row['cell_file']
                
                # Build path to masked cell
                label_str = 'resistant' if label == 1 else 'sensitive'
                cell_path = os.path.join(drug_dir, 'test', label_str, 'masked', cell_file)
                
                if os.path.exists(cell_path):
                    try:
                        # Load and preprocess cell
                        from PIL import Image
                        cell_img = Image.open(cell_path).convert('RGB')
                        cell_tensor = test_transform(cell_img).unsqueeze(0).to(device)
                        
                        # Predict
                        output = model(cell_tensor)
                        prob = torch.softmax(output, dim=1)
                        cell_predictions.append(prob.cpu().numpy()[0])
                        cells_loaded += 1
                        
                    except Exception as e:
                        print(f"Error loading {cell_path}: {e}")
                        continue
            
            # Aggregate predictions (average probabilities)
            if cell_predictions:
                avg_probs = np.mean(cell_predictions, axis=0)
                final_pred = np.argmax(avg_probs)
            else:
                final_pred = 0
            
            end_time = time.time()
            image_time = end_time - start_time
            
            image_times.append(image_time)
            total_images += 1
            total_cells_processed += cells_loaded
            
            print(f"  Image {total_images}: {cells_loaded} cells in {image_time:.3f}s "
                  f"({cells_loaded/image_time:.1f} cells/s)")
    
    # Calculate metrics
    total_time = sum(image_times)
    avg_time_per_image = np.mean(image_times)
    images_per_sec = total_images / total_time
    cells_per_sec = total_cells_processed / total_time
    
    results = {
        'total_images': total_images,
        'total_cells': total_cells_processed,
        'avg_cells_per_image': total_cells_processed / total_images,
        'total_time': total_time,
        'avg_time_per_image': avg_time_per_image,
        'images_per_sec': images_per_sec,
        'cells_per_sec': cells_per_sec
    }
    
    print(f"\nTask 1 Results:")
    print(f"  Total images: {total_images}")
    print(f"  Total cells: {total_cells_processed}")
    print(f"  Avg cells/image: {total_cells_processed/total_images:.1f}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Speed: {images_per_sec:.2f} images/sec")
    print(f"  Speed: {cells_per_sec:.2f} cells/sec")
    
    return results


def benchmark_task2_fair(model, test_df, drug_dir, test_transform, device):
    """Task 2 như cũ nhưng chỉ test với cùng số ảnh như Task 1."""
    
    print(f"\n=== TASK 2 (SAME IMAGES) ===")
    
    # Create dataset (limit to same images as Task 1)
    original_col = 'original_tiff' if 'original_tiff' in test_df.columns else 'original_file'
    unique_images = test_df[original_col].unique()[:10]  # Same 10 images
    test_df_limited = test_df[test_df[original_col].isin(unique_images)]
    
    task2_dataset = Task2TestDataset(
        test_df_limited, drug_dir, split='test',
        patch_size=224, stride=224,
        transform=test_transform
    )
    
    dataloader = DataLoader(task2_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    total_patches = len(task2_dataset)
    num_images = len(unique_images)
    patches_per_image = total_patches / num_images
    
    print(f"Dataset: {total_patches} patches from {num_images} images")
    print(f"Patches per image: {patches_per_image:.1f}")
    
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels, _, _) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    images_per_sec = num_images / total_time
    patches_per_sec = total_patches / total_time
    
    results = {
        'total_images': num_images,
        'total_patches': total_patches,
        'patches_per_image': patches_per_image,
        'total_time': total_time,
        'images_per_sec': images_per_sec,
        'patches_per_sec': patches_per_sec
    }
    
    print(f"\nTask 2 Results:")
    print(f"  Total images: {num_images}")
    print(f"  Total patches: {total_patches}")
    print(f"  Patches per image: {patches_per_image:.1f}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Speed: {images_per_sec:.2f} images/sec")
    print(f"  Speed: {patches_per_sec:.2f} patches/sec")
    
    return results


def fair_comparison():
    """So sánh công bằng giữa Task 1 và Task 2."""
    
    print("SO SÁNH CÔNG BẰNG: TASK 1 VS TASK 2")
    print("="*80)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    drug = 'AZI'
    split_csv = config.split_csv(drug)
    df = pd.read_csv(split_csv)
    test_df = df[df['split'] == 'test'].copy()
    
    drug_dir = config.drug_dir(drug)
    test_transform = get_test_transform(224)
    
    # Count cells per image
    cells_per_image, avg_cells = count_cells_per_image(drug_dir, test_df)
    
    # Test with lightweight model
    model_name = 'mobilenetv3_small_100'
    print(f"\nTesting with: {model_name}")
    
    model = get_model(model_name, num_classes=2, pretrained=False)
    model = model.to(device)
    
    # Warm up
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    # Run fair comparison
    task1_results = simulate_task1_fair(model, test_df, drug_dir, test_transform, device, cells_per_image)
    task2_results = benchmark_task2_fair(model, test_df, drug_dir, test_transform, device)
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"\nTask 1 (Cell-based):")
    print(f"  Images/sec: {task1_results['images_per_sec']:.2f}")
    print(f"  Avg cells/image: {task1_results['avg_cells_per_image']:.1f}")
    print(f"  Cells/sec: {task1_results['cells_per_sec']:.2f}")
    
    print(f"\nTask 2 (Patch-based):")
    print(f"  Images/sec: {task2_results['images_per_sec']:.2f}")
    print(f"  Patches/image: {task2_results['patches_per_image']:.1f}")
    print(f"  Patches/sec: {task2_results['patches_per_sec']:.2f}")
    
    # Speed comparison
    task1_speed = task1_results['images_per_sec']
    task2_speed = task2_results['images_per_sec']
    
    if task1_speed > task2_speed:
        speedup = task1_speed / task2_speed
        winner = "Task 1"
    else:
        speedup = task2_speed / task1_speed
        winner = "Task 2"
    
    print(f"\n🏆 WINNER: {winner}")
    print(f"📊 Speedup: {speedup:.2f}x")
    
    print(f"\n💡 ANALYSIS:")
    print(f"  • Task 1 processes ~{task1_results['avg_cells_per_image']:.0f} cells per image")
    print(f"  • Task 2 processes {task2_results['patches_per_image']:.0f} patches per image")
    
    if task1_results['avg_cells_per_image'] > task2_results['patches_per_image']:
        print(f"  • Task 1 has MORE operations per image ({task1_results['avg_cells_per_image']:.0f} vs {task2_results['patches_per_image']:.0f})")
    else:
        print(f"  • Task 2 has MORE operations per image ({task2_results['patches_per_image']:.0f} vs {task1_results['avg_cells_per_image']:.0f})")


if __name__ == '__main__':
    fair_comparison()