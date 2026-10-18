import os
import pandas as pd

BASE = "/home/aisia/new_mia"

# Metrics: per-drug pooled comparisons (Task1 erosion-augmented vs Task2)
FILES = {
    "F1": {
        "drug": os.path.join(BASE, "task1_erosion_aug_vs_task2_stats_per_drug.csv"),
    },
    "Recall": {
        "drug": os.path.join(BASE, "task1_erosion_aug_vs_task2_recall_per_drug.csv"),
    },
    "Precision": {
        "drug": os.path.join(BASE, "task1_erosion_aug_vs_task2_precision_per_drug.csv"),
    },
    "AUC": {
        "drug": os.path.join(BASE, "task1_erosion_aug_vs_task2_auc_per_drug.csv"),
    },
    "Accuracy": {
        "drug": os.path.join(BASE, "task1_erosion_aug_vs_task2_accuracy_per_drug.csv"),
    },
}


def load_metric(metric_name: str, path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file for metric={metric_name}: {path}")
    df = pd.read_csv(path)

    keep = [
        "drug",
        "task1_mean",
        "task2_mean",
        "delta_mean",
        "wilcoxon_p_one_sided",
        "ttest_p_one_sided",
    ]
    for c in keep:
        if c not in df.columns:
            raise ValueError(f"{path} missing column {c}")

    df = df[keep].copy()
    df = df.rename(
        columns={
            "task1_mean": f"{metric_name}_task1_mean",
            "task2_mean": f"{metric_name}_task2_mean",
            "delta_mean": f"{metric_name}_delta_mean",
            "wilcoxon_p_one_sided": f"{metric_name}_wilcoxon_p_one_sided",
            "ttest_p_one_sided": f"{metric_name}_ttest_p_one_sided",
        }
    )
    return df


def main() -> None:
    # Only load metrics for which files exist
    active = []
    for m, fs in FILES.items():
        if os.path.exists(fs["drug"]):
            active.append(m)
        else:
            print(f"[Skip] Missing {m} csv: {fs['drug']}")

    if not active:
        raise SystemExit("Không có metric nào để tạo proof xlsx.")

    dfs = []
    for m in active:
        dfs.append(load_metric(m, FILES[m]["drug"]))

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="drug", how="inner")

    metric_delta_cols = [f"{m}_delta_mean" for m in active]
    out["n_metrics_task1_better"] = (out[metric_delta_cols] > 0).sum(axis=1)
    out["task1_better_all_metrics"] = (out[metric_delta_cols] > 0).all(axis=1)

    # Sort drugs for readability
    drug_order = ["AZI", "CIP", "TE"]
    out["drug"] = pd.Categorical(out["drug"], categories=drug_order, ordered=True)
    out = out.sort_values("drug").reset_index(drop=True)

    out_xlsx = os.path.join(
        BASE, "task1_task2_proof_F1RecallPrecisionAUCAccuracy_perDrug_pooled.xlsx"
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # Keep only essentials for “proof”
        cols = ["drug", "n_metrics_task1_better", "task1_better_all_metrics"]
        for m in active:
            cols += [
                f"{m}_task1_mean",
                f"{m}_task2_mean",
                f"{m}_delta_mean",
                f"{m}_wilcoxon_p_one_sided",
            ]
        out[cols].to_excel(writer, sheet_name="PerDrug_proof", index=False)

    print("Saved:", out_xlsx)
    print(out[["drug", "n_metrics_task1_better", "task1_better_all_metrics"] + metric_delta_cols].to_string(index=False))


if __name__ == "__main__":
    main()

