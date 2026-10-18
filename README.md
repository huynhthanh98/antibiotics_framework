# antibiotics_framework

Code for the paper *From Image Patches to Individual Cells: Deep Learning for Image-Based Antimicrobial Susceptibility Testing* (Son T. Huynh, Tuan-Anh Tran, Hien D. Nguyen, Binh T. Nguyen).

The framework classifies *Escherichia coli* high-content fluorescence microscopy images as resistant (R) or susceptible (S) to azithromycin (AZI), ciprofloxacin (CIP) and tetracycline (TE). It implements two pipelines that share the same image-level data split, backbones, aggregation and metrics:

- **Task 1, individual-level.** Segmented single-cell crops are classified one by one, and the per-cell resistant probabilities of each source image are averaged (soft vote) into an image-level prediction.
- **Task 2, patch-based.** Each 2048 × 2048 image is tiled into non-overlapping 224 × 224 patches. Background patches (raw intensity std below 1,000 on the uint16 scale) are dropped, and the patch probabilities are averaged per image.

Cell segmentation (HTC-ConvNeXtV2-Tiny-CBAM with SAHI, from the FluoEcoli-Instance work) runs upstream of this repository. The framework starts from its output: cell crops and masks organised per drug.

## Installation

Tested with Python 3.10, PyTorch 2.7 (CUDA 12.6) and timm 1.0.15.

```bash
conda create -n amr python=3.10
conda activate amr
pip install -r requirements.txt
```

## Data layout

By default the code looks for data in the repository root (the parent of `antibiotics_framework/`). Use `--data_root` to point elsewhere.

```
<data_root>/
  {DRUG}_joblib/           masked cell images (Task 1, --task1_image_folder masked)
  {DRUG}_joblib_cells/     original cell crops as TIFF (Task 1, --task1_image_folder cell)
      split_info.csv       train/val/test assignment per cell
      train/
          resistant/
              masked/ or cell/   cell images (Task 1)
              original/          full microscopy TIFFs (Task 2)
          sensitive/
      val/
      test/
  task2_patches/           optional, pre-exported patches (see export_patches.py)
```

`split_info.csv` columns: `split, drug, isolate, iso_id, anti_id, image_name, label, resistant, cell_file, mask_file, original_tiff, original_mask`.

The split is defined at the level of the original microscopy image, so every cell and patch inherits the split and label of its source image. With the default `swap_val_test=True`, the CSV `test` split is used for validation and the CSV `val` split for testing; pass `--no_swap_val_test` to use the names as written.

## Usage

Run the commands from inside `antibiotics_framework/`.

```bash
cd antibiotics_framework

# Check config and data without training
python main.py --task 1 --drug CIP --model resnet50 --seeds 0 1 2 --dry_run

# Task 1 (cells), settings used in the paper
python main.py \
  --task 1 --drug AZI CIP TE \
  --model efficientnet_b5 densenet121 --seeds 0 1 2 3 4 \
  --loss focal --threshold_method pr_equal \
  --best_metric recall_resistant --best_metric_level image_level \
  --joblib_suffix _joblib_cells --task1_image_folder cell \
  --output_dir ../results_cells --bf16

# Task 2 (patches): export once, then train from disk
python export_patches.py --drug all --output_dir ../task2_patches
python main.py --task 2 --drug CIP --model resnet50 --task2_patches_dir ../task2_patches

# List timm models matching a pattern
python main.py --list_models efficientnet
```

`python main.py --help` lists every option. The full sweeps over all drugs and backbones are in `run_all_models_cells.sh`, `run_all_models_tasks2.sh` and `run_all_models_task1_erosion_aug.sh`; each sets a batch size per backbone.

Results go to `<output_dir>/task{N}/{DRUG}/{MODEL}/`: one checkpoint per seed (`seed{N}_best.pth`), `results.json` and `run_metrics.csv`. `all_run_metrics.csv` in `<output_dir>` merges every configuration.

## Method details

| File | Role |
|------|------|
| `config.py` | `Config` dataclass with all paths and hyperparameters |
| `main.py` | Command-line entry point, calls `run_experiment()` |
| `trainer.py` | Training loop, early stopping, multi-seed runs, result aggregation |
| `datasets.py` | Task 1 and Task 2 datasets, 16-bit TIFF to 8-bit RGB conversion, augmentation |
| `models.py` | `get_model()` wrapping timm, with short aliases in `TIMM_MODEL_ALIASES` |
| `losses.py` | Focal loss, class-weighted loss factory, `WeightedRandomSampler` helper |
| `metrics.py` | Threshold selection, metrics, image-level soft and hard voting |

- **Preprocessing.** 16-bit intensities are clipped at the 1st and 99th percentiles, rescaled to [0, 255], replicated to three channels and normalised with ImageNet statistics.
- **Augmentation.** Resize to 256 then random crop to 224, horizontal and vertical flips, rotation within ±15°, mild color jitter. `--mask_erosion_aug` adds random 1 to 3 px mask erosion for Task 1.
- **Class imbalance.** Focal loss (γ = 2) with inverse-frequency class weights, plus a weighted random sampler. `--train_downsample_balance_classes` undersamples to a 1:1 R:S ratio.
- **Optimisation.** AdamW (lr 5e-4, weight decay 1e-4), cosine annealing to 1e-6, up to 30 epochs, early stopping with patience 5, optional BF16.
- **Model selection and threshold.** The best checkpoint is chosen on validation (`--best_metric`, at cell or image level). The decision threshold is then calibrated on validation with `f1`, `balanced`, `min_recall`, `pr_equal`, `gmean` or `fixed`, and frozen before testing.
- **Evaluation.** Five seeds on a fixed split, reported as mean ± std. Resistant is the positive class. Recall, precision, F1, accuracy and ROC-AUC are computed at cell/patch level and at image level.

## Analysis tools

```bash
# Logs or result folders to an Excel summary
python parse_log.py --from-results ../results --task 1 --drug CIP --out summary.xlsx

# Grad-CAM on single cells
python gradcam_task1.py --drug CIP --seed 0 --n_cells 8

# Robustness of Task 1 models to degraded masks (occlusion, erosion, bg_noise)
python degraded_mask_eval.py --drug CIP --model efficientnet_b5 --mode erosion --seeds 0 1 2 3 4

# Paired one-sided t-tests between Task 1 and Task 2
python ttest_analysis.py --help
python task1_vs_task2_stats_seedavg.py --help
```

`scripts/` holds the comparison reports and figures from the paper (`compare_task1_task2_logs.py`, `export_softvote_*.py`, `plot_cell_vs_patch_recall.py`, `plot_gradcam_three_drugs.py`) and `resplit_joblib_cells.py`, which re-splits a drug folder at image level.

Some analysis and benchmark scripts still contain absolute paths from the original workstation. Edit the path variables near the top of a script, or pass the corresponding argument, before running it.

## Citation

If you use this code, please cite the paper above. Funded by Vietnam National University Ho Chi Minh City (VNU-HCM) under grant number A2025-18-02.
