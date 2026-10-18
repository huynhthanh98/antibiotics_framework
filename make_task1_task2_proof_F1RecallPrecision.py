import os
import pandas as pd


BASE = "/home/aisia/new_mia"

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
}


def _load_metric(metric_name: str, path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
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
            raise ValueError(f"File {path} missing column: {c}")

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
    active = {}
    for metric_name, d in FILES.items():
        p = d["drug"]
        if os.path.exists(p):
            active[metric_name] = p
        else:
            raise FileNotFoundError(f"Missing required file for metric={metric_name}: {p}")

    dfs = []
    for metric_name, p in active.items():
        dfs.append(_load_metric(metric_name, p))

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="drug", how="inner")

    metric_delta_cols = [f"{m}_delta_mean" for m in active.keys()]
    out["n_metrics_task1_better"] = (out[metric_delta_cols] > 0).sum(axis=1)
    out["task1_better_all_metrics"] = (out[metric_delta_cols] > 0).all(axis=1)

    order = ["AZI", "CIP", "TE"]
    out["drug"] = pd.Categorical(out["drug"], categories=order, ordered=True)
    out = out.sort_values("drug").reset_index(drop=True)

    out_xlsx = os.path.join(
        BASE, "task1_task2_proof_F1RecallPrecision_perDrug_pooled.xlsx"
    )
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="PerDrug_pooled_proof", index=False)

    print("Saved:", out_xlsx)
    print(
        out[
            ["drug", "n_metrics_task1_better", "task1_better_all_metrics"]
            + metric_delta_cols
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

