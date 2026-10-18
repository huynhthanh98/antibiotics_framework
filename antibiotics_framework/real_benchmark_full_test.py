"""
Benchmark thực tế trên toàn bộ tập test để có kết quả chính xác.
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from PIL import Image

from models import get_model
from datasets import Task2TestDataset, get_test_transform
from config import Config


def benchmark_task1_real_full(model, test_df, drug_dir, test_transform, device, max_images=None):
    """
    Benchmark Task 1 thực tế trên toàn bộ (hoặc một phần) test set.
    Xử lý từng ảnh gốc: load tất cả cells → predict → aggregate.
    """
    
    print(f"\n=== TASK 1 REAL BENCHMARK (FULL TEST) ===")
    
    # Group by original image
    original_col = 'original_tiff' if 'original_tiff' in test_df.columns else 'original_file'
    grouped = test_df.groupby(original_col)
    
    if max_images:
        grouped = list(grouped)[:max_images]
        print(f"Testing on first {max_images} images")
    else:
        grouped = list(grouped)
        print(f"Testing on ALL {len(grouped)} images")
    
    model.eval()
    
    image_results = []
    total_start_time = time.time()
    
    with torch.no_grad():
        for img_idx, (orig_file, group) in enumerate(grouped):
            
            img_start_time = time.time()
            
            # Load all cells for this image
            cell_tensors = []
            cell_labels = []
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
                        cell_img = Image.open(cell_path).convert('RGB')
                        cell_tensor = test_transform(cell_img)
                        cell_tensors.append(cell_tensor)
                        cell_labels.append(label)
                        cells_loaded += 1
                        
                    except Exception as e:
                        print(f"    Error loading {cell_path}: {e}")
                        continue
            
            if not cell_tensors:
                print(f"  Image {img_idx+1}: No cells loaded, skipping")
                continue
            
            # Batch inference for all cells of this image
            inference_start = time.time()
            
            # Process in batches to avoid memory issues
            batch_size = 32
            all_probs = []
            
            for i in range(0, len(cell_tensors), batch_size):
                batch_tensors = torch.stack(cell_tensors[i:i+batch_size]).to(device)
                outputs = model(batch_tensors)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
            
            # Combine all probabilities
            all_probs = np.vstack(all_probs)
            
            inference_end = time.time()
            
            # Aggregate predictions (average probabilities)
            avg_probs = np.mean(all_probs, axis=0)
            final_pred = np.argmax(avg_probs)
            true_label = cell_labels[0]  # Should be same for all cells from same image
            
            img_end_time = time.time()
            
            # Record results
            img_time = img_end_time - img_start_time
            inference_time = inference_end - inference_start
            
            image_results.append({
                'image_idx': img_idx,
                'orig_file': orig_file,
                'num_cells': cells_loaded,
                'total_time': img_time,
                'inference_time': inference_time,
                'preprocessing_time': img_time - inference_time,
                'prediction': final_pred,
                'true_label': true_label,
                'correct': final_pred == true_label
            })
            
            print(f"  Image {img_idx+1}/{len(grouped)}: {cells_loaded} cells in {img_time:.3f}s "
                  f"(inference: {inference_time:.3f}s, {cells_loaded/img_time:.1f} cells/s)")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Calculate overall metrics
    total_images = len(image_results)
    total_cells = sum(r['num_cells'] for r in image_results)
    total_inference_time = sum(r['inference_time'] for r in image_results)
    
    avg_cells_per_image = total_cells / total_images
    images_per_sec = total_images / total_time
    cells_per_sec = total_cells / total_time
    accuracy = sum(r['correct'] for r in image_results) / total_images
    
    results = {
        'task': 1,
        'total_images': total_images,
        'total_cells': total_cells,
        'avg_cells_per_image': avg_cells_per_image,
        'total_time': total_time,
        'total_inference_time': total_inference_time,
        'images_per_sec': images_per_sec,
        'cells_per_sec': cells_per_sec,
        'accuracy': accuracy,
        'image_details': image_results
    }
    
    print(f"\n📊 TASK 1 RESULTS:")
    print(f"  Total images: {total_images}")
    print(f"  Total cells: {total_cells}")
    print(f"  Avg cells/image: {avg_cells_per_image:.1f}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Inference time: {total_inference_time:.2f}s")
    print(f"  Images/sec: {images_per_sec:.2f}")
    print(f"  Cells/sec: {cells_per_sec:.2f}")
    print(f"  Accuracy: {accuracy:.3f}")
    
    return results


def benchmark_task2_real_full(model, test_df, drug_dir, test_transform, device, max_images=None):
    """
    Benchmark Task 2 thực tế trên cùng số ảnh như Task 1.
    """
    
    print(f"\n=== TASK 2 REAL BENCHMARK (SAME IMAGES) ===")
    
    # Get same images as Task 1
    original_col = 'original_tiff' if 'original_tiff' in test_df.columns else 'original_file'
    unique_images = test_df[original_col].unique()
    
    if max_images:
        unique_images = unique_images[:max_images]
        print(f"Testing on first {max_images} images")
    else:
        print(f"Testing on ALL {len(unique_images)} images")
    
    # Filter test_df to same images
    test_df_filtered = test_df[test_df[original_col].isin(unique_images)]
    
    # Create Task 2 dataset
    task2_dataset = Task2TestDataset(
        test_df_filtered, drug_dir, split='test',
        patch_size=224, stride=224,
        transform=test_transform
    )
    
    dataloader = DataLoader(task2_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    total_patches = len(task2_dataset)
    num_images = len(unique_images)
    patches_per_image = total_patches / num_images
    
    print(f"Dataset: {total_patches} patches from {num_images} images")
    print(f"Patches per image: {patches_per_image:.1f}")
    
    model.eval()
    
    start_time = time.time()
    processed_patches = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels, orig_files, patch_ids) in enumerate(dataloader):
            
            # Move to device and inference
            images = images.to(device)
            outputs = model(images)
            
            # Synchronize for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            processed_patches += images.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = processed_patches / elapsed
                print(f"  Processed {processed_patches}/{total_patches} patches "
                      f"({speed:.1f} patches/s)")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    images_per_sec = num_images / total_time
    patches_per_sec = processed_patches / total_time
    
    results = {
        'task': 2,
        'total_images': num_images,
        'total_patches': processed_patches,
        'patches_per_image': patches_per_image,
        'total_time': total_time,
        'images_per_sec': images_per_sec,
        'patches_per_sec': patches_per_sec
    }
    
    print(f"\n📊 TASK 2 RESULTS:")
    print(f"  Total images: {num_images}")
    print(f"  Total patches: {processed_patches}")
    print(f"  Patches per image: {patches_per_image:.1f}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Images/sec: {images_per_sec:.2f}")
    print(f"  Patches/sec: {patches_per_sec:.2f}")
    
    return results


def run_full_benchmark(max_images=None):
    """
    Chạy benchmark thực tế trên toàn bộ (hoặc một phần) test set.
    """
    
    print("REAL BENCHMARK ON FULL TEST SET")
    print("="*80)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test data
    drug = 'AZI'
    split_csv = config.split_csv(drug)
    df = pd.read_csv(split_csv)
    test_df = df[df['split'] == 'test'].copy()
    
    drug_dir = config.drug_dir(drug)
    test_transform = get_test_transform(224)
    
    # Load model
    model_name = 'mobilenetv3_small_100'  # Fast model for testing
    print(f"Model: {model_name}")
    
    model = get_model(model_name, num_classes=2, pretrained=False)
    model = model.to(device)
    
    # Warm up
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    # Run benchmarks
    task1_results = benchmark_task1_real_full(model, test_df, drug_dir, test_transform, device, max_images)
    task2_results = benchmark_task2_real_full(model, test_df, drug_dir, test_transform, device, max_images)
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    
    task1_speed = task1_results['images_per_sec']
    task2_speed = task2_results['images_per_sec']
    
    print(f"Task 1: {task1_speed:.2f} images/sec ({task1_results['avg_cells_per_image']:.1f} cells/image)")
    print(f"Task 2: {task2_speed:.2f} images/sec ({task2_results['patches_per_image']:.1f} patches/image)")
    
    if task1_speed > task2_speed:
        speedup = task1_speed / task2_speed
        winner = "Task 1"
    else:
        speedup = task2_speed / task1_speed
        winner = "Task 2"
    
    print(f"\n🏆 WINNER: {winner}")
    print(f"📊 Speedup: {speedup:.2f}x")
    
    # Save results
    summary = {
        'model': model_name,
        'device': str(device),
        'task1_images_per_sec': task1_speed,
        'task2_images_per_sec': task2_speed,
        'task1_avg_cells': task1_results['avg_cells_per_image'],
        'task2_patches': task2_results['patches_per_image'],
        'winner': winner,
        'speedup': speedup
    }
    
    output_file = '/home/son/new_mia/antibiotics_framework/real_benchmark_results.txt'
    with open(output_file, 'w') as f:
        f.write("REAL BENCHMARK RESULTS\n")
        f.write("="*50 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return task1_results, task2_results


if __name__ == '__main__':
    # Test with subset first (5 images)
    print("Testing with 5 images first...")
    run_full_benchmark(max_images=5)
    
    # Uncomment to test with all images
    # print("\n" + "="*80)
    # print("Testing with ALL images...")
    # run_full_benchmark(max_images=None)