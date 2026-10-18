"""
Benchmark chi tiết tốc độ inference cho Task 1 vs Task 2.
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

from models import get_model
from datasets import Task1Dataset, Task2TestDataset, get_test_transform
from config import Config


def detailed_benchmark(drug='AZI', models=['mobilenetv3_small_100', 'efficientnet_b3', 'resnet50'], 
                      max_images=20, batch_size=32):
    """
    Benchmark chi tiết với số lượng ảnh giới hạn để có kết quả chính xác.
    """
    
    print(f"DETAILED SPEED BENCHMARK")
    print(f"Drug: {drug}")
    print(f"Max images per test: {max_images}")
    print(f"Batch size: {batch_size}")
    print("="*80)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test data
    split_csv = config.split_csv(drug)
    if not os.path.exists(split_csv):
        print(f"Split CSV not found: {split_csv}")
        return
    
    df = pd.read_csv(split_csv)
    test_df = df[df['split'] == 'test'].copy()
    
    # Limit to unique images
    original_col = 'original_tiff' if 'original_tiff' in test_df.columns else 'original_file'
    unique_images = test_df[original_col].unique()[:max_images]
    test_df_limited = test_df[test_df[original_col].isin(unique_images)].copy()
    
    print(f"Total test images: {len(unique_images)}")
    print(f"Total test samples (Task 1): {len(test_df_limited)}")
    
    drug_dir = config.drug_dir(drug)
    test_transform = get_test_transform(224)
    
    all_results = []
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model = get_model(model_name, num_classes=2, pretrained=False)
            model = model.to(device)
            model.eval()
            
            # Warm up
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Test Task 1
            print(f"\n--- TASK 1: Masked Cell Images ---")
            task1_results = test_task1(model, test_df_limited, drug_dir, test_transform, 
                                     device, batch_size)
            task1_results['model'] = model_name
            task1_results['task'] = 1
            all_results.append(task1_results)
            
            # Test Task 2  
            print(f"\n--- TASK 2: Patch-based ---")
            task2_results = test_task2(model, test_df_limited, drug_dir, test_transform,
                                     device, batch_size, len(unique_images))
            task2_results['model'] = model_name
            task2_results['task'] = 2
            all_results.append(task2_results)
            
            # Comparison
            print(f"\n--- COMPARISON ---")
            task1_speed = task1_results['images_per_sec']
            task2_speed = task2_results['images_per_sec']
            speedup = task1_speed / task2_speed if task2_speed > 0 else float('inf')
            
            print(f"Task 1: {task1_speed:.2f} images/sec")
            print(f"Task 2: {task2_speed:.2f} images/sec")
            print(f"Speedup: {speedup:.1f}x (Task 1 faster)")
            
            # Clear memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue
    
    # Create summary
    create_summary_report(all_results, drug)
    
    return all_results


def test_task1(model, test_df, drug_dir, test_transform, device, batch_size):
    """Test Task 1: Masked cell images."""
    
    # Create dataset
    task1_dataset = Task1Dataset(test_df, drug_dir, 'test', transform=test_transform)
    task1_loader = DataLoader(task1_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset size: {len(task1_dataset)} samples")
    
    # Benchmark
    times = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels, _, _) in enumerate(task1_loader):
            
            start_time = time.time()
            
            # Move to device
            images = images.to(device)
            
            # Inference
            outputs = model(images)
            
            # Synchronize
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_time = end_time - start_time
            batch_size_actual = images.size(0)
            
            times.append(batch_time)
            total_samples += batch_size_actual
            
            print(f"  Batch {batch_idx+1}: {batch_size_actual} samples in {batch_time:.3f}s "
                  f"({batch_size_actual/batch_time:.1f} samples/s)")
    
    # Calculate metrics
    total_time = sum(times)
    avg_time_per_batch = np.mean(times)
    samples_per_sec = total_samples / total_time
    
    results = {
        'total_samples': total_samples,
        'total_images': total_samples,  # Task 1: 1 sample = 1 image
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'samples_per_sec': samples_per_sec,
        'images_per_sec': samples_per_sec,
        'batches': len(times)
    }
    
    print(f"Results:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Speed: {samples_per_sec:.2f} images/sec")
    
    return results


def test_task2(model, test_df, drug_dir, test_transform, device, batch_size, num_images):
    """Test Task 2: Patch-based."""
    
    # Create dataset
    task2_dataset = Task2TestDataset(
        test_df, drug_dir, split='test',
        patch_size=224, stride=224,
        transform=test_transform
    )
    task2_loader = DataLoader(task2_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    total_patches = len(task2_dataset)
    patches_per_image = total_patches / num_images
    
    print(f"Dataset size: {total_patches} patches from {num_images} images")
    print(f"Patches per image: {patches_per_image:.1f}")
    
    # Benchmark
    times = []
    total_processed = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels, orig_files, _) in enumerate(task2_loader):
            
            start_time = time.time()
            
            # Move to device
            images = images.to(device)
            
            # Inference
            outputs = model(images)
            
            # Synchronize
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_time = end_time - start_time
            batch_size_actual = images.size(0)
            
            times.append(batch_time)
            total_processed += batch_size_actual
            
            print(f"  Batch {batch_idx+1}: {batch_size_actual} patches in {batch_time:.3f}s "
                  f"({batch_size_actual/batch_time:.1f} patches/s)")
    
    # Calculate metrics
    total_time = sum(times)
    avg_time_per_batch = np.mean(times)
    patches_per_sec = total_processed / total_time
    images_per_sec = patches_per_sec / patches_per_image
    
    results = {
        'total_patches': total_processed,
        'total_images': num_images,
        'patches_per_image': patches_per_image,
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'patches_per_sec': patches_per_sec,
        'images_per_sec': images_per_sec,
        'batches': len(times)
    }
    
    print(f"Results:")
    print(f"  Total patches: {total_processed}")
    print(f"  Total images: {num_images}")
    print(f"  Patches per image: {patches_per_image:.1f}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Speed: {patches_per_sec:.2f} patches/sec")
    print(f"  Speed: {images_per_sec:.2f} images/sec")
    
    return results


def create_summary_report(all_results, drug):
    """Tạo báo cáo tổng kết."""
    
    print(f"\n{'='*100}")
    print(f"SUMMARY REPORT - {drug.upper()}")
    print(f"{'='*100}")
    
    # Create DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model'],
            'Task': result['task'],
            'Total Images': result['total_images'],
            'Images/sec': f"{result['images_per_sec']:.2f}",
            'Total Time (s)': f"{result['total_time']:.2f}",
            'Batches': result['batches'],
            'Avg Batch Time (s)': f"{result['avg_time_per_batch']:.3f}",
        })
        
        if result['task'] == 2:
            summary_data[-1]['Patches/Image'] = f"{result['patches_per_image']:.1f}"
            summary_data[-1]['Patches/sec'] = f"{result['patches_per_sec']:.2f}"
        else:
            summary_data[-1]['Patches/Image'] = "1.0"
            summary_data[-1]['Patches/sec'] = f"{result['samples_per_sec']:.2f}"
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Speed comparison table
    print(f"\n{'='*80}")
    print("SPEED COMPARISON TABLE")
    print(f"{'='*80}")
    
    models = df['Model'].unique()
    comparison_data = []
    
    for model in models:
        model_data = df[df['Model'] == model]
        task1_data = model_data[model_data['Task'] == 1]
        task2_data = model_data[model_data['Task'] == 2]
        
        if len(task1_data) > 0 and len(task2_data) > 0:
            task1_speed = float(task1_data['Images/sec'].iloc[0])
            task2_speed = float(task2_data['Images/sec'].iloc[0])
            speedup = task1_speed / task2_speed if task2_speed > 0 else float('inf')
            
            comparison_data.append({
                'Model': model,
                'Task 1 (img/s)': f"{task1_speed:.2f}",
                'Task 2 (img/s)': f"{task2_speed:.2f}",
                'Speedup': f"{speedup:.1f}x",
                'Winner': 'Task 1' if speedup > 1 else 'Task 2'
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save results
    output_file = f'/home/son/new_mia/antibiotics_framework/speed_benchmark_{drug}.csv'
    df.to_csv(output_file, index=False)
    
    comparison_file = f'/home/son/new_mia/antibiotics_framework/speed_comparison_{drug}.csv'
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Detailed: {output_file}")
    print(f"  Comparison: {comparison_file}")


if __name__ == '__main__':
    # Run detailed benchmark
    results = detailed_benchmark(
        drug='AZI',
        models=['mobilenetv3_small_100', 'efficientnet_b3', 'resnet50'],
        max_images=10,  # Test với 10 ảnh để có kết quả nhanh
        batch_size=16
    )