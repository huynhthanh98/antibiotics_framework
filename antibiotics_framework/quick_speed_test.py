"""
Script đơn giản để test tốc độ inference nhanh.
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from models import get_model
from datasets import Task1Dataset, Task2TestDataset, get_test_transform
from config import Config


def quick_speed_test(drug='AZI', model_name='mobilenetv3_small_100', max_samples=100):
    """Test tốc độ nhanh với số lượng sample giới hạn."""
    
    print(f"Quick Speed Test: {model_name} on {drug}")
    print("="*50)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = get_model(model_name, num_classes=2, pretrained=False)
    model = model.to(device)
    model.eval()
    
    # Load test data
    split_csv = config.split_csv(drug)
    if not os.path.exists(split_csv):
        print(f"Split CSV not found: {split_csv}")
        return
    
    df = pd.read_csv(split_csv)
    test_df = df[df['split'] == 'test'].copy()
    
    # Limit samples for quick test
    if len(test_df) > max_samples:
        test_df = test_df.head(max_samples)
    
    print(f"Test samples: {len(test_df)}")
    
    drug_dir = config.drug_dir(drug)
    test_transform = get_test_transform(224)
    
    results = {}
    
    # Test Task 1
    print(f"\n--- TASK 1 ---")
    try:
        task1_dataset = Task1Dataset(test_df, drug_dir, 'test', transform=test_transform)
        task1_loader = DataLoader(task1_dataset, batch_size=16, shuffle=False)
        
        print(f"Task 1 samples: {len(task1_dataset)}")
        
        start_time = time.time()
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels, _, _) in enumerate(task1_loader):
                images = images.to(device)
                outputs = model(images)
                total_samples += images.size(0)
                
                if batch_idx >= 5:  # Limit to first few batches for quick test
                    break
        
        end_time = time.time()
        task1_time = end_time - start_time
        task1_speed = total_samples / task1_time
        
        results['task1'] = {
            'samples': total_samples,
            'time': task1_time,
            'speed': task1_speed
        }
        
        print(f"  Processed: {total_samples} samples")
        print(f"  Time: {task1_time:.3f}s")
        print(f"  Speed: {task1_speed:.2f} samples/sec")
        
    except Exception as e:
        print(f"  Error in Task 1: {e}")
        results['task1'] = {'error': str(e)}
    
    # Test Task 2
    print(f"\n--- TASK 2 ---")
    try:
        task2_dataset = Task2TestDataset(
            test_df, drug_dir, split='test',
            patch_size=224, stride=224,
            transform=test_transform
        )
        task2_loader = DataLoader(task2_dataset, batch_size=16, shuffle=False)
        
        print(f"Task 2 patches: {len(task2_dataset)}")
        
        # Count unique images
        original_col = 'original_tiff' if 'original_tiff' in test_df.columns else 'original_file'
        unique_images = test_df[original_col].nunique()
        patches_per_image = len(task2_dataset) / unique_images
        
        start_time = time.time()
        total_patches = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels, _, _) in enumerate(task2_loader):
                images = images.to(device)
                outputs = model(images)
                total_patches += images.size(0)
                
                if batch_idx >= 10:  # Process more batches for Task 2
                    break
        
        end_time = time.time()
        task2_time = end_time - start_time
        task2_patch_speed = total_patches / task2_time
        task2_image_speed = task2_patch_speed / patches_per_image
        
        results['task2'] = {
            'patches': total_patches,
            'images': total_patches / patches_per_image,
            'patches_per_image': patches_per_image,
            'time': task2_time,
            'patch_speed': task2_patch_speed,
            'image_speed': task2_image_speed
        }
        
        print(f"  Processed: {total_patches} patches ({total_patches/patches_per_image:.1f} images)")
        print(f"  Patches per image: {patches_per_image:.1f}")
        print(f"  Time: {task2_time:.3f}s")
        print(f"  Speed: {task2_patch_speed:.2f} patches/sec")
        print(f"  Speed: {task2_image_speed:.2f} images/sec")
        
    except Exception as e:
        print(f"  Error in Task 2: {e}")
        results['task2'] = {'error': str(e)}
    
    # Comparison
    print(f"\n--- COMPARISON ---")
    if 'error' not in results.get('task1', {}) and 'error' not in results.get('task2', {}):
        task1_img_speed = results['task1']['speed']
        task2_img_speed = results['task2']['image_speed']
        
        print(f"Task 1 speed: {task1_img_speed:.2f} images/sec")
        print(f"Task 2 speed: {task2_img_speed:.2f} images/sec")
        
        if task1_img_speed > task2_img_speed:
            speedup = task1_img_speed / task2_img_speed
            print(f"Task 1 is {speedup:.2f}x faster than Task 2")
        else:
            speedup = task2_img_speed / task1_img_speed
            print(f"Task 2 is {speedup:.2f}x faster than Task 1")
    
    return results


def compare_models():
    """So sánh tốc độ của nhiều models."""
    
    models = [
        'mobilenetv3_small_100',
        'efficientnet_b3', 
        'resnet50'
    ]
    
    print("COMPARING MODELS")
    print("="*60)
    
    all_results = {}
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
        try:
            results = quick_speed_test('AZI', model_name, max_samples=50)
            all_results[model_name] = results
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    
    print(f"{'Model':<25} {'Task 1 (img/s)':<15} {'Task 2 (img/s)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"{model_name:<25} {'ERROR':<15} {'ERROR':<15} {'N/A':<10}")
            continue
        
        task1_speed = results.get('task1', {}).get('speed', 0)
        task2_speed = results.get('task2', {}).get('image_speed', 0)
        
        if task1_speed > 0 and task2_speed > 0:
            speedup = task1_speed / task2_speed
            print(f"{model_name:<25} {task1_speed:<15.2f} {task2_speed:<15.2f} {speedup:<10.2f}x")
        else:
            print(f"{model_name:<25} {task1_speed:<15.2f} {task2_speed:<15.2f} {'N/A':<10}")


if __name__ == '__main__':
    # Quick test single model
    # quick_speed_test('AZI', 'mobilenetv3_small_100')
    
    # Compare multiple models
    compare_models()