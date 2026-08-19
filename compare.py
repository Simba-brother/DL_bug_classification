"""
读取各个方法的 all_res.csv，并生成最终对比表 result.csv和result_wtl.csv。
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from cliffs_delta import cliffs_delta

BASELINE_CLFS = ("LR", "DT", "RF", "SVM", "KNN")
BERT_METHODS = ("sobert", "robert", "codebert")
BASELINE_METHODS = ("tfidf", "word2vec")
# LLM_METHODS = ("chatgpt", "claude")
LLM_METHODS = ("chatgpt",)
LABEL_NUMS = tuple(range(6))
CLASS_ID2NAME = {
    0: "model",
    1: "tensor",
    2: "training",
    3: "gpu",
    4: "api",
    5: "Others",
}


def get_all_res_path(method_name:str) -> str:
    if method_name in BERT_METHODS:
        return os.path.join(exp_root_dir, f"{method_name}_res", "all_res.csv")

    if method_name in BASELINE_METHODS:
        return os.path.join(exp_root_dir, f"{method_name}_res", "all_res.csv")

    if method_name in LLM_METHODS:
        return os.path.join(exp_root_dir, f"{method_name}_prob_res", "all_res.csv")

    raise ValueError(f"Unsupported method_name:{method_name}")


def load_all_res(method_name:str) -> pd.DataFrame:
    all_res_path = get_all_res_path(method_name)
    if not os.path.exists(all_res_path):
        raise FileNotFoundError(f"指标文件不存在:{all_res_path}")
    all_res_df = pd.read_csv(all_res_path)
    if all_res_df.shape[0] != 15:
        raise ValueError(f"{all_res_path} 应该包含15次实验结果，实际行数:{all_res_df.shape[0]}")
    return all_res_df


def mean_column(all_res_df:pd.DataFrame, column:str) -> float:
    if column not in all_res_df.columns:
        raise ValueError(f"all_res.csv缺少列:{column}")
    return float(np.nanmean(pd.to_numeric(all_res_df[column], errors="coerce")))


def build_result_row(all_res_df:pd.DataFrame, prefix:str="") -> dict[str, float]:
    row = {
        "all_acc": mean_column(all_res_df, f"{prefix}acc_all"),
        "all_f1": mean_column(all_res_df, f"{prefix}f1_all"),
        "all_auc": mean_column(all_res_df, f"{prefix}auc_all"),
    }

    for label_num in LABEL_NUMS:
        class_name = CLASS_ID2NAME[label_num]
        row[f"{class_name}_acc"] = mean_column(all_res_df, f"{prefix}acc_{label_num}")
        row[f"{class_name}_f1"] = mean_column(all_res_df, f"{prefix}f1_{label_num}")
        row[f"{class_name}_auc"] = mean_column(all_res_df, f"{prefix}auc_{label_num}")

    return row


def add_single_method_rows(rows:dict[str, dict[str, float]], method_name:str) -> None:
    all_res_df = load_all_res(method_name)
    rows[method_name] = build_result_row(all_res_df)


def add_baseline_method_rows(rows:dict[str, dict[str, float]], method_name:str) -> None:
    all_res_df = load_all_res(method_name)
    for clf_name in BASELINE_CLFS:
        rows[f"{method_name}_{clf_name}"] = build_result_row(
            all_res_df,
            prefix=f"{clf_name}_",
        )


def build_result_df() -> pd.DataFrame:
    rows = {}

    for method_name in BERT_METHODS:
        add_single_method_rows(rows, method_name)

    for method_name in BASELINE_METHODS:
        add_baseline_method_rows(rows, method_name)

    for method_name in LLM_METHODS:
        add_single_method_rows(rows, method_name)

    result_df = pd.DataFrame.from_dict(rows, orient="index")
    result_df = result_df[
        ["all_acc", "all_f1", "all_auc"]
        + [
            metric_col
            for label_num in LABEL_NUMS
            for metric_col in (
                f"{CLASS_ID2NAME[label_num]}_acc",
                f"{CLASS_ID2NAME[label_num]}_f1",
                f"{CLASS_ID2NAME[label_num]}_auc",
            )
        ]
    ]
    result_df.index.name = "row_name"
    return result_df


def get_compare_metric_columns() -> list[tuple[str, str]]:
    metric_columns = [
        ("acc_all", "all_acc"),
        ("f1_all", "all_f1"),
        ("auc_all", "all_auc"),
    ]
    for label_num in LABEL_NUMS:
        class_name = CLASS_ID2NAME[label_num]
        metric_columns.extend([
            (f"acc_{label_num}", f"{class_name}_acc"),
            (f"f1_{label_num}", f"{class_name}_f1"),
            (f"auc_{label_num}", f"{class_name}_auc"),
        ])
    return metric_columns


def get_method_metric_lists(method_name:str) -> dict[str, dict[str, list[float]]]:
    all_res_df = load_all_res(method_name)
    metric_columns = get_compare_metric_columns()

    if method_name in BASELINE_METHODS:
        method_metric_lists = {}
        for clf_name in BASELINE_CLFS:
            row_name = f"{method_name}_{clf_name}"
            method_metric_lists[row_name] = {}
            for raw_metric_col, result_metric_col in metric_columns:
                source_col = f"{clf_name}_{raw_metric_col}"
                if source_col not in all_res_df.columns:
                    raise ValueError(f"{method_name} all_res.csv缺少列:{source_col}")
                method_metric_lists[row_name][result_metric_col] = (
                    pd.to_numeric(all_res_df[source_col], errors="coerce")
                    .astype(float)
                    .tolist()
                )
        return method_metric_lists

    method_metric_lists = {method_name: {}}
    for raw_metric_col, result_metric_col in metric_columns:
        if raw_metric_col not in all_res_df.columns:
            raise ValueError(f"{method_name} all_res.csv缺少列:{raw_metric_col}")
        method_metric_lists[method_name][result_metric_col] = (
            pd.to_numeric(all_res_df[raw_metric_col], errors="coerce")
            .astype(float)
            .tolist()
        )
    return method_metric_lists


def wtl(our_data_list,baseline_data_list):
    # data list 是正向指标
    p_value = stats.wilcoxon(our_data_list, baseline_data_list).pvalue
    sorted_our_data_list = sorted(our_data_list)
    sorted_baseline_data_list = sorted(baseline_data_list)
    delta,info = cliffs_delta(sorted_our_data_list, sorted_baseline_data_list)
    if p_value < 0.05 and delta > 0.147:
        # ours和baseline有差异，且我们的值偏大
        return 'W'
    elif p_value < 0.05 and delta < 0.147:
        # ours和baseline有差异，且我们的值偏小
        return 'L'
    else:
        return 'T'

def eval_wtl():
    '''
    获取 sobert 和其他方法在每个指标下的15次实验结果 list。
    tfidf/word2vec 会进一步按 LR/DT/RF/SVM/KNN 拆分。
    '''
    sobert_metric_lists = get_method_metric_lists("sobert")["sobert"]
    baseline_metric_lists = {}
    # method_name_list = ("robert", "codebert", "tfidf", "word2vec", "chatgpt", "claude")
    method_name_list = ("robert", "codebert", "tfidf", "word2vec", "chatgpt")
    for method_name in method_name_list:
        baseline_metric_lists.update(get_method_metric_lists(method_name))

    wtl_rows = {}
    for baseline_name, metric_lists in baseline_metric_lists.items():
        wtl_rows[baseline_name] = {}
        for metric_name, sobert_values in sobert_metric_lists.items():
            baseline_values = metric_lists[metric_name]
            wtl_rows[baseline_name][metric_name] = wtl(sobert_values, baseline_values)

    wtl_df = pd.DataFrame.from_dict(wtl_rows, orient="index")
    wtl_df.index.name = "row_name"
    wtl_df.to_csv("result_wtl.csv", index=True)
    return sobert_metric_lists, baseline_metric_lists, wtl_df


def main():
    result_df = build_result_df()
    result_df.round(3).to_csv("result.csv", index=True)


if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification/time5_3" # time5_3|time_15seed|random_15seed
    main()
    # eval_wtl()
