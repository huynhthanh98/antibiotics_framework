"""
Seed-averaged paired statistical test (most conservative).

For each drug and for each seed:
  - compute Task1 metric = mean over all common models
  - compute Task2 metric = mean over all common models
Then run paired one-sided tests across seeds (n = number of seeds).

Implements:
  1) Wilcoxon signed-rank test, one-sided (H1: Task1 > Task2)
  2) Paired t-test, one-sided (H1: Task1 > Task2)

Output:
  - per-drug CSV (and optionally Excel) for each metric.

Example:
  python task1_vs_task2_stats_seedavg.py \
    --task1_results /home/aisia/new_mia/results_erosion_aug \
    --task2_results /home/aisia/new_mia/results \
    --drug AZI CIP TE \
    --level image_level \
    --out /home/aisia/new_mia/task1_vs_task2_seedavg_stats
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


METRIC_LIST_DEFAULT = [
    "f1_resistant",
    "recall_resistant",
    "precision_resistant",
    "auc",
    "accuracy",
]


def _one_sided_p_from_ttest_rel(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (t_stat, one_sided_p) for H1: x > y."""
    t_stat, p_two = stats.ttest_rel(x, y, nan_policy="omit")
    if np.isnan(t_stat) or np.isnan(p_two):
        return np.nan, np.nan
    if t_stat > 0:
        return float(t_stat), float(p_two) / 2.0
    return float(t_stat), 1.0 - float(p_two) / 2.0


def _load_metric_values_for_model(
    run_metrics_path: str,
    level: str,
    metric: str,
) -> Dict[int, float]:
    if not os.path.exists(run_metrics_path):
        return {}
    df = pd.read_csv(run_metrics_path)
    df = df[df["level"] == level].copy()
    if df.empty or metric not in df.columns:
        return {}
    # There should be one row per seed
    out = {}
    for _, r in df.iterrows():
        seed = int(r["seed"])
        val = float(r[metric])
        out[seed] = val
    return out


def _seed_avg_paired_tests(
    task1_results: str,
    task2_results: str,
    drug: str,
    metric: str,
    level: str,
    seed_list: List[int],
    common_models: List[str],
    alpha: float,
) -> Dict:
    task1_seed_vals = []
    task2_seed_vals = []

    for seed in seed_list:
        t1_model_vals = []
        t2_model_vals = []
        for model in common_models:
            p1 = os.path.join(task1_results, "task1", drug, model, "run_metrics.csv")
            p2 = os.path.join(task2_results, "task2", drug, model, "run_metrics.csv")
            d1 = _load_metric_values_for_model(p1, level=level, metric=metric)
            d2 = _load_metric_values_for_model(p2, level=level, metric=metric)
            if seed in d1:
                t1_model_vals.append(d1[seed])
            if seed in d2:
                t2_model_vals.append(d2[seed])

        # Only keep seeds where we have at least 1 model value on both sides
        if len(t1_model_vals) == 0 or len(t2_model_vals) == 0:
            continue
        task1_seed_vals.append(float(np.mean(t1_model_vals)))
        task2_seed_vals.append(float(np.mean(t2_model_vals)))

    x = np.asarray(task1_seed_vals, dtype=float)
    y = np.asarray(task2_seed_vals, dtype=float)

    n = len(x)
    if n < 2:
        return {
            "drug": drug,
            "metric": metric,
            "n_seeds": n,
            "task1_mean": np.nan,
            "task2_mean": np.nan,
            "delta_mean": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_p_one_sided": np.nan,
            "ttest_t_stat": np.nan,
            "ttest_p_one_sided": np.nan,
            "significant_wilcoxon": False,
            "significant_ttest": False,
        }

    # Wilcoxon signed-rank test (paired)
    try:
        w_stat, w_p = stats.wilcoxon(x, y, alternative="greater", zero_method="wilcox")
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception:
        w_stat, w_p = np.nan, np.nan

    # Paired t-test (one-sided)
    t_stat, t_p = _one_sided_p_from_ttest_rel(x, y)

    d = x - y
    return {
        "drug": drug,
        "metric": metric,
        "n_seeds": n,
        "task1_mean": float(np.mean(x)),
        "task2_mean": float(np.mean(y)),
        "delta_mean": float(np.mean(d)),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_one_sided": w_p,
        "ttest_t_stat": float(t_stat),
        "ttest_p_one_sided": float(t_p),
        "significant_wilcoxon": bool((not np.isnan(w_p)) and (w_p < alpha)),
        "significant_ttest": bool((not np.isnan(t_p)) and (t_p < alpha)),
    }


def _get_common_models(task_results: str, drug: str) -> List[str]:
    d = os.path.join(task_results, "task1", drug)
    if not os.path.isdir(d):
        # task1/task2 swapped by caller; we just need a directory listing
        d = os.path.join(task_results, "task1", drug)
    if not os.path.isdir(d):
        return []
    return []


def _common_models_between_tasks(task1_results: str, task2_results: str, drug: str) -> List[str]:
    d1 = os.path.join(task1_results, "task1", drug)
    d2 = os.path.join(task2_results, "task2", drug)
    if not os.path.isdir(d1) or not os.path.isdir(d2):
        return []
    models = sorted(set(os.listdir(d1)).intersection(set(os.listdir(d2))))
    models = [m for m in models if os.path.isdir(os.path.join(d1, m)) and os.path.isdir(os.path.join(d2, m))]
    return models


def _get_seed_list_for_metric(
    task_results: str,
    task_id: int,
    drug: str,
    model: str,
    metric: str,
    level: str,
) -> List[int]:
    p = os.path.join(task_results, f"task{task_id}", drug, model, "run_metrics.csv")
    if not os.path.exists(p):
        return []
    df = pd.read_csv(p)
    df = df[df["level"] == level].copy()
    if df.empty or metric not in df.columns:
        return []
    if "seed" not in df.columns:
        return []
    seeds = sorted([int(s) for s in df["seed"].unique().tolist()])
    return seeds


def main():
    ap = argparse.ArgumentParser(description="Seed-averaged paired one-sided stats Task1 vs Task2")
    ap.add_argument("--task1_results", type=str, required=True, help="results folder containing task1/")
    ap.add_argument("--task2_results", type=str, required=True, help="results folder containing task2/")
    ap.add_argument("--drug", nargs="+", default=["AZI", "CIP", "TE"])
    ap.add_argument("--level", type=str, default="image_level", choices=["image_level", "cell_level"])
    ap.add_argument("--metric", nargs="+", default=METRIC_LIST_DEFAULT, help="Metrics to test")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=str, default=None, help="CSV output path prefix")
    args = ap.parse_args()

    task1_results = args.task1_results
    task2_results = args.task2_results
    alpha = float(args.alpha)

    metrics = args.metric
    drugs = args.drug
    level = args.level

    all_rows = []

    for drug in drugs:
        common_models = _common_models_between_tasks(task1_results, task2_results, drug)
        if not common_models:
            continue

        # Determine seed list from the first common model where metric exists
        seed_list = []
        for model in common_models:
            seeds1 = _get_seed_list_for_metric(task1_results, 1, drug, model, metric=metrics[0], level=level)
            seeds2 = _get_seed_list_for_metric(task2_results, 2, drug, model, metric=metrics[0], level=level)
            seed_list = sorted(set(seeds1).intersection(set(seeds2)))
            if len(seed_list) > 0:
                break

        if len(seed_list) < 2:
            # Not enough seeds shared
            continue

        for metric in metrics:
            row = _seed_avg_paired_tests(
                task1_results=task1_results,
                task2_results=task2_results,
                drug=drug,
                metric=metric,
                level=level,
                seed_list=seed_list,
                common_models=common_models,
                alpha=alpha,
            )
            all_rows.append(row)

    if not all_rows:
        raise SystemExit("No rows produced. Check paths/models/metrics/level.")

    df = pd.DataFrame(all_rows).sort_values(["metric", "drug"])

    if args.out is None:
        out_csv = os.path.join(os.getcwd(), f"task1_vs_task2_seedavg_stats_{level}.csv")
    else:
        if args.out.endswith(".csv"):
            out_csv = args.out
        else:
            out_csv = f"{args.out}_{level}_seedavg.csv"

    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

