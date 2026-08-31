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
BASELINE_CLFS = ("LR", "DT", "RF", "SVM", "KNN")
SINGLE_MODEL_METHODS = ("sobert", "robert", "codebert")
GROUPED_CLF_METHODS = ("tfidf", "word2vec")
# METHODS_TO_COMPARE = ("sobert", "tfidf", "word2vec")
METHODS_TO_COMPARE = ("robert",)
WTL_ALPHA = 0.05
CLIFFS_DELTA_THRESHOLD = 0.147


def build_method_configs() -> dict[str, dict]:
    configs = {
        method_name: {
            "res_dir": f"{method_name}_res",
            "metric_groups": ((method_name, ""),),
        }
        for method_name in SINGLE_MODEL_METHODS
    }
    for method_name in GROUPED_CLF_METHODS:
        configs[method_name] = {
            "res_dir": f"{method_name}_res",
            "metric_groups": tuple(
                (clf_name, f"{clf_name}_")
                for clf_name in BASELINE_CLFS
            ),
        }
    return configs


def iter_metric_columns(prefix: str = ""):
    for label_num in LABEL_NUMS:
        for metric in METRICS:
            yield label_num, CLASS_ID2NAME[label_num], metric, f"{prefix}{metric}_{label_num}"

    for metric in METRICS:
        yield "all", "all", metric, f"{prefix}{metric}_all"


def iter_group_metric_columns(metric_groups: tuple[tuple[str, str], ...]):
    for group_name, prefix in metric_groups:
        yield from (
            (group_name, class_id, class_name, metric, column)
            for class_id, class_name, metric, column
            in iter_metric_columns(prefix=prefix)
        )


def validate_all_res_df(
    df: pd.DataFrame,
    df_name: str,
    metric_groups: tuple[tuple[str, str], ...],
) -> None:
    if df.empty:
        raise ValueError(f"{df_name} is empty.")

    missing_columns = [
        column
        for _, _, _, _, column in iter_group_metric_columns(metric_groups)
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

    p_value = stats.wilcoxon(
        withcode_values,
        nocode_values,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        mode="auto",
    ).pvalue
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
    method_name: str,
    withcode_df: pd.DataFrame,
    nocode_df: pd.DataFrame,
    metric_groups: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    validate_all_res_df(withcode_df, "withcode_df", metric_groups)
    validate_all_res_df(nocode_df, "nocode_df", metric_groups)

    if len(withcode_df) != len(nocode_df):
        raise ValueError(
            "withcode_df and nocode_df must have the same number of rows "
            "because WTL is computed as a paired comparison. "
            f"withcode={len(withcode_df)}, nocode={len(nocode_df)}"
    )

    rows = []
    for group_name, class_id, class_name, metric, column in iter_group_metric_columns(
        metric_groups
    ):
        withcode_values, nocode_values = get_paired_metric_values(
            withcode_df,
            nocode_df,
            column,
        )
        wtl, p_value, delta = calc_wtl(withcode_values, nocode_values)
        withcode_mean = float(np.mean(withcode_values))
        nocode_mean = float(np.mean(nocode_values))

        rows.append({
            "method": method_name,
            "group": group_name,
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


def compare_method_code_vs_nocode(method_name: str, method_config: dict) -> pd.DataFrame:
    res_dir = method_config["res_dir"]
    metric_groups = method_config["metric_groups"]
    withcode_df = pd.read_csv(os.path.join(exp_root_dir, "exp", res_dir, "all_res.csv"))
    nocode_df = pd.read_csv(os.path.join(exp_root_dir, "exp_nocode", res_dir, "all_res.csv"))
    comparison_df = build_code_comparison_df(
        method_name,
        withcode_df,
        nocode_df,
        metric_groups,
    )

    save_path = f"result_{method_name}_code_vs_nocode.csv"
    comparison_df.round(6).to_csv(save_path, index=False)
    print(f"{method_name} with-code vs no-code comparison saved to: {save_path}")
    return comparison_df


def main():
    method_configs = build_method_configs()
    comparison_dfs = []
    for method_name in METHODS_TO_COMPARE:
        if method_name not in method_configs:
            raise ValueError(f"Unsupported method_name: {method_name}")
        comparison_dfs.append(
            compare_method_code_vs_nocode(method_name, method_configs[method_name])
        )

    all_comparison_df = pd.concat(comparison_dfs, ignore_index=True)
    save_path = "result_code_vs_nocode.csv"
    all_comparison_df.round(6).to_csv(save_path, index=False)
    print(f"combined with-code vs no-code comparison saved to: {save_path}")

if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    main()
