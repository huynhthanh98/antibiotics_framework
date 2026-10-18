"""
Benchmark tốc độ inference cho Task 1 và Task 2 trên tập test.

Task 1: 1 ảnh cell → 1 prediction (fast)
Task 2: 1 ảnh gốc → nhiều patches → nhiều predictions → aggregate (slower)
"""

import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import psutil
import gc

from models import get_model
from datasets import Task1Dataset, Task2TestDataset, get_test_transform
from config import Config


class InferenceSpeedBenchmark:
    """Class để benchmark tốc độ inference."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Memory tracking
        self.process = psutil.Process()
        
    def load_test_data(self, drug: str, task: int):
        """Load test dataset."""
        print(f"\nLoading test data for {drug} Task {task}...")
        
        # Load split info
        split_csv = self.config.split_csv(drug)
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        
        df = pd.read_csv(split_csv)
        test_df = df[df['split'] == 'test'].copy()
        
        print(f"Test samples: {len(test_df)}")
        
        # Create dataset
        drug_dir = self.config.drug_dir(drug)
        test_transform = get_test_transform(self.config.img_size)
        
        if task == 1:
            test_dataset = Task1Dataset(test_df, drug_dir, 'test', transform=test_transform)
        else:  # task == 2
            test_dataset = Task2TestDataset(
                test_df, drug_dir, split='test',
                patch_size=self.config.patch_size,
                stride=self.config.test_stride,
                transform=test_transform
            )
        
        print(f"Dataset size: {len(test_dataset)} samples")
        return test_dataset, test_df
    
    def benchmark_task1(self, model: nn.Module, test_dataset, batch_size: int = 32):
        """Benchmark Task 1: 1 ảnh → 1 prediction."""
        print(f"\n=== BENCHMARKING TASK 1 ===")
        print(f"Batch size: {batch_size}")
        
        model.eval()
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Metrics
        total_samples = len(test_dataset)
        total_images = total_samples  # Task 1: 1 sample = 1 image
        
        times = {
            'data_loading': 0,
            'inference': 0,
            'postprocessing': 0,
            'total': 0
        }
        
        memory_usage = []
        predictions = []
        labels = []
        
        print(f"Processing {total_samples} samples in {len(dataloader)} batches...")
        
        start_total = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, batch_labels, orig_files, filenames) in enumerate(dataloader):
                
                # Data loading time (already loaded)
                start_data = time.time()
                images = images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                end_data = time.time()
                times['data_loading'] += (end_data - start_data)
                
                # Inference time
                start_inference = time.time()
                outputs = model(images)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_inference = time.time()
                times['inference'] += (end_inference - start_inference)
                
                # Postprocessing time
                start_post = time.time()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
                end_post = time.time()
                times['postprocessing'] += (end_post - start_post)
                
                # Memory tracking
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
                else:
                    memory_usage.append(self.process.memory_info().rss / 1024**2)  # MB
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * batch_size}/{total_samples} samples")
        
        end_total = time.time()
        times['total'] = end_total - start_total
        
        # Calculate metrics
        throughput_samples = total_samples / times['total']
        throughput_images = total_images / times['total']
        
        results = {
            'task': 1,
            'total_samples': total_samples,
            'total_images': total_images,
            'batch_size': batch_size,
            'times': times,
            'throughput_samples_per_sec': throughput_samples,
            'throughput_images_per_sec': throughput_images,
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'predictions': predictions,
            'labels': labels
        }
        
        return results
    
    def benchmark_task2(self, model: nn.Module, test_dataset, test_df, batch_size: int = 32):
        """Benchmark Task 2: nhiều patches → aggregate predictions."""
        print(f"\n=== BENCHMARKING TASK 2 ===")
        print(f"Batch size: {batch_size}")
        
        model.eval()
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Group patches by original image
        unique_images = test_df['original_file'].unique()
        total_images = len(unique_images)
        total_samples = len(test_dataset)
        
        print(f"Processing {total_samples} patches from {total_images} images...")
        
        times = {
            'data_loading': 0,
            'inference': 0,
            'postprocessing': 0,
            'aggregation': 0,
            'total': 0
        }
        
        memory_usage = []
        patch_predictions = []
        patch_labels = []
        patch_orig_files = []
        
        start_total = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, batch_labels, orig_files, patch_ids) in enumerate(dataloader):
                
                # Data loading time
                start_data = time.time()
                images = images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                end_data = time.time()
                times['data_loading'] += (end_data - start_data)
                
                # Inference time
                start_inference = time.time()
                outputs = model(images)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_inference = time.time()
                times['inference'] += (end_inference - start_inference)
                
                # Postprocessing time
                start_post = time.time()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Store patch-level results
                patch_predictions.extend(probs.cpu().numpy())
                patch_labels.extend(batch_labels.cpu().numpy())
                patch_orig_files.extend(orig_files)
                end_post = time.time()
                times['postprocessing'] += (end_post - start_post)
                
                # Memory tracking
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
                else:
                    memory_usage.append(self.process.memory_info().rss / 1024**2)  # MB
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * batch_size}/{total_samples} patches")
        
        # Aggregation time
        start_agg = time.time()
        image_predictions, image_labels = self._aggregate_predictions(
            patch_predictions, patch_labels, patch_orig_files, test_df
        )
        end_agg = time.time()
        times['aggregation'] = end_agg - start_agg
        
        end_total = time.time()
        times['total'] = end_total - start_total
        
        # Calculate metrics
        throughput_samples = total_samples / times['total']
        throughput_images = total_images / times['total']
        
        results = {
            'task': 2,
            'total_samples': total_samples,
            'total_images': total_images,
            'patches_per_image': total_samples / total_images,
            'batch_size': batch_size,
            'times': times,
            'throughput_samples_per_sec': throughput_samples,
            'throughput_images_per_sec': throughput_images,
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'predictions': image_predictions,
            'labels': image_labels
        }
        
        return results
    
    def _aggregate_predictions(self, patch_probs, patch_labels, patch_orig_files, test_df):
        """Aggregate patch predictions to image-level predictions."""
        
        # Group by original file
        image_data = defaultdict(list)
        
        for prob, label, orig_file in zip(patch_probs, patch_labels, patch_orig_files):
            image_data[orig_file].append({
                'prob': prob,
                'label': label
            })
        
        image_predictions = []
        image_labels = []
        
        for orig_file in test_df['original_file'].unique():
            if orig_file in image_data:
                patches = image_data[orig_file]
                
                # Average probabilities (soft voting)
                avg_probs = np.mean([p['prob'] for p in patches], axis=0)
                pred_label = np.argmax(avg_probs)
                
                # True label (should be same for all patches from same image)
                true_label = patches[0]['label']
                
                image_predictions.append(pred_label)
                image_labels.append(true_label)
        
        return image_predictions, image_labels
    
    def run_benchmark(self, drug: str, model_name: str, batch_sizes: list = [16, 32, 64]):
        """Run complete benchmark for both tasks."""
        
        print(f"\n{'='*80}")
        print(f"BENCHMARKING {model_name.upper()} ON {drug.upper()}")
        print(f"{'='*80}")
        
        # Load model
        print(f"Loading model: {model_name}")
        model = get_model(model_name, num_classes=self.config.num_classes, pretrained=False)
        model = model.to(self.device)
        
        # Warm up GPU
        if torch.cuda.is_available():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)
            torch.cuda.synchronize()
        
        all_results = []
        
        for task in [1, 2]:
            print(f"\n{'-'*60}")
            print(f"TASK {task}")
            print(f"{'-'*60}")
            
            # Load test data
            test_dataset, test_df = self.load_test_data(drug, task)
            
            for batch_size in batch_sizes:
                print(f"\n--- Batch Size: {batch_size} ---")
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                try:
                    if task == 1:
                        results = self.benchmark_task1(model, test_dataset, batch_size)
                    else:
                        results = self.benchmark_task2(model, test_dataset, test_df, batch_size)
                    
                    results['drug'] = drug
                    results['model'] = model_name
                    all_results.append(results)
                    
                    # Print summary
                    self._print_results(results)
                    
                except Exception as e:
                    print(f"Error with batch size {batch_size}: {e}")
                    continue
        
        return all_results
    
    def _print_results(self, results):
        """Print benchmark results."""
        print(f"\nResults for Task {results['task']}:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Total images: {results['total_images']}")
        if results['task'] == 2:
            print(f"  Patches per image: {results['patches_per_image']:.1f}")
        
        print(f"\nTiming (seconds):")
        for key, value in results['times'].items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nThroughput:")
        print(f"  Samples/sec: {results['throughput_samples_per_sec']:.2f}")
        print(f"  Images/sec: {results['throughput_images_per_sec']:.2f}")
        
        print(f"\nMemory Usage:")
        print(f"  Average: {results['avg_memory_mb']:.1f} MB")
        print(f"  Peak: {results['max_memory_mb']:.1f} MB")


def main():
    """Run speed benchmark."""
    
    config = Config()
    benchmark = InferenceSpeedBenchmark(config)
    
    # Test parameters
    drugs = ['AZI']  # Start with one drug
    models = ['mobilenetv3_small_100', 'efficientnet_b3', 'resnet50']  # Fast to slow
    batch_sizes = [16, 32, 64]
    
    all_results = []
    
    for drug in drugs:
        for model_name in models:
            try:
                results = benchmark.run_benchmark(drug, model_name, batch_sizes)
                all_results.extend(results)
            except Exception as e:
                print(f"Error benchmarking {model_name} on {drug}: {e}")
                continue
    
    # Create summary report
    if all_results:
        print(f"\n{'='*100}")
        print("SUMMARY REPORT")
        print(f"{'='*100}")
        
        # Create DataFrame for analysis
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Drug': result['drug'],
                'Model': result['model'],
                'Task': result['task'],
                'Batch Size': result['batch_size'],
                'Total Images': result['total_images'],
                'Images/sec': f"{result['throughput_images_per_sec']:.2f}",
                'Total Time (s)': f"{result['times']['total']:.2f}",
                'Inference Time (s)': f"{result['times']['inference']:.2f}",
                'Memory (MB)': f"{result['avg_memory_mb']:.1f}",
                'Patches/Image': f"{result.get('patches_per_image', 1):.1f}"
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Save results
        output_file = '/home/son/new_mia/antibiotics_framework/inference_speed_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Speed comparison
        print(f"\n{'='*60}")
        print("SPEED COMPARISON (Images/sec)")
        print(f"{'='*60}")
        
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            print(f"\n{model}:")
            for _, row in model_data.iterrows():
                print(f"  Task {row['Task']} (BS={row['Batch Size']}): {row['Images/sec']} images/sec")


if __name__ == '__main__':
    main()