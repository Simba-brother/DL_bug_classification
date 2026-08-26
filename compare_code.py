import os

import numpy as np
import pandas as pd
from scipy import stats
from cliffs_delta import cliffs_delta

LABEL_NUMS = tuple(range(6))
CLASS_ID2NAME = {
    0: "model",
    1: "tensor",
    2: "training",
    3: "gpu",
    4: "api",
    5: "Others",
}
METRICS = ("acc", "f1", "auc")
WTL_ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147


def iter_metric_columns():
    for label_num in LABEL_NUMS:
        for metric in METRICS:
            yield label_num, CLASS_ID2NAME[label_num], metric, f"{metric}_{label_num}"

    for metric in METRICS:
        yield "all", "all", metric, f"{metric}_all"


def validate_all_res_df(df: pd.DataFrame, df_name: str) -> None:
    if df.empty:
        raise ValueError(f"{df_name} is empty.")

    missing_columns = [
        column
        for _, _, _, column in iter_metric_columns()
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"{df_name} missing columns: {missing_columns}")


def get_paired_metric_values(
    withcode_df: pd.DataFrame,
    nocode_df: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    paired_df = pd.DataFrame({
        "withcode": pd.to_numeric(withcode_df[column], errors="coerce"),
        "nocode": pd.to_numeric(nocode_df[column], errors="coerce"),
    }).dropna()

    if paired_df.empty:
        raise ValueError(f"No valid paired values for column: {column}")

    return (
        paired_df["withcode"].to_numpy(dtype=float),
        paired_df["nocode"].to_numpy(dtype=float),
    )


def calc_wtl(
    withcode_values: np.ndarray,
    nocode_values: np.ndarray,
) -> tuple[str, float, float]:
    """
    Return W/T/L for with-code versus no-code.

    W means with-code is significantly better than no-code.
    L means with-code is significantly worse than no-code.
    T means tied or the effect is too small.
    """
    if len(withcode_values) != len(nocode_values):
        raise ValueError(
            "WTL requires paired samples with the same length: "
            f"withcode={len(withcode_values)}, nocode={len(nocode_values)}"
        )

    if np.allclose(withcode_values, nocode_values):
        return "T", 1.0, 0.0

    p_value = stats.wilcoxon(withcode_values, nocode_values).pvalue
    delta, _ = cliffs_delta(
        sorted(withcode_values.tolist()),
        sorted(nocode_values.tolist()),
    )

    if p_value < WTL_ALPHA and delta > CLIFFS_DELTA_THRESHOLD:
        return "W", float(p_value), float(delta)

    if p_value < WTL_ALPHA and delta < -CLIFFS_DELTA_THRESHOLD:
        return "L", float(p_value), float(delta)

    return "T", float(p_value), float(delta)


def build_code_comparison_df(
    withcode_df: pd.DataFrame,
    nocode_df: pd.DataFrame,
) -> pd.DataFrame:
    validate_all_res_df(withcode_df, "withcode_df")
    validate_all_res_df(nocode_df, "nocode_df")

    if len(withcode_df) != len(nocode_df):
        raise ValueError(
            "withcode_df and nocode_df must have the same number of rows "
            "because WTL is computed as a paired comparison. "
            f"withcode={len(withcode_df)}, nocode={len(nocode_df)}"
        )

    rows = []
    for class_id, class_name, metric, column in iter_metric_columns():
        withcode_values, nocode_values = get_paired_metric_values(
            withcode_df,
            nocode_df,
            column,
        )
        wtl, p_value, delta = calc_wtl(withcode_values, nocode_values)
        withcode_mean = float(np.mean(withcode_values))
        nocode_mean = float(np.mean(nocode_values))

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "metric": metric,
            "withcode_mean": withcode_mean,
            "nocode_mean": nocode_mean,
            "mean_diff": withcode_mean - nocode_mean,
            "wtl": wtl,
            "p_value": p_value,
            "cliffs_delta": delta,
            "n_pairs": len(withcode_values),
        })

    return pd.DataFrame(rows)


def main():
    withcode_df = pd.read_csv(os.path.join(exp_root_dir,"exp/sobert_res/all_res.csv"))
    nocode_df = pd.read_csv(os.path.join(exp_root_dir,"exp_nocode/sobert_res/all_res.csv"))
    comparison_df = build_code_comparison_df(withcode_df, nocode_df)

    save_path = "result_code_vs_nocode.csv"
    comparison_df.round(6).to_csv(save_path, index=False)
    print(f"with-code vs no-code comparison saved to: {save_path}")

if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    main()
