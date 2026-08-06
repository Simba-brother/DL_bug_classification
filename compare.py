"""
读取各个方法的评价指标，并生成最终对比表。
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd


BASELINE_CLFS = ("LR", "DT", "RF", "SVM", "KNN")
BERT_METHODS = ("sobert", "codebert", "robert")
LLM_METHODS = ("chatgpt", "claude")
BASELINE_METHODS = ("tfidf", "word2vec")


def load_res(method_name:str) -> dict:
    return joblib.load(os.path.join(exp_root_dir, f"{method_name}_res", "res.joblib"))


def get_class_id2name() -> dict[int, str]:
    test_df = pd.read_csv(test_csv_path)
    return dict(
        test_df[["LabelNum", "Label"]]
        .drop_duplicates()
        .sort_values("LabelNum")
        .itertuples(index=False, name=None)
    )


def get_method_reports(method_name:str, res:dict) -> dict[str, list[dict]]:
    if method_name in BERT_METHODS:
        return {method_name: [res[seed] for seed in range(42, 42 + 15)]}

    if method_name in LLM_METHODS:
        return {method_name: [res[repeat] for repeat in range(1, 1 + 15)]}

    if method_name in BASELINE_METHODS:
        return {
            f"{method_name}_{clf_name}": [
                res[seed][clf_name]
                for seed in range(42, 42 + 15)
            ]
            for clf_name in BASELINE_CLFS
        }

    raise ValueError(f"Unsupported method_name: {method_name}")


def mean_metric(reports:list[dict], path:list[str]) -> float:
    values = []
    for report in reports: # 15次实验报告
        value = report
        for key in path:
            value = value[key]
        values.append(value)
    return float(np.mean(values))


def build_result_row(reports:list[dict], class_id2name:dict[int, str]) -> dict[str, float]:
    row = {
        "all_acc": mean_metric(reports, ["accuracy"]),
        "all_f1": mean_metric(reports, ["macro avg", "f1-score"]),
    }

    for class_id, class_name in class_id2name.items():
        class_key = str(class_id)
        row[f"{class_name}_acc"] = mean_metric(reports, [class_key, "precision"])
        row[f"{class_name}_f1"] = mean_metric(reports, [class_key, "f1-score"])

    return row


def build_result_df(method_names:list[str]) -> pd.DataFrame:
    class_id2name = get_class_id2name()
    rows = {}

    for method_name in method_names:
        res = load_res(method_name)
        method_reports = get_method_reports(method_name, res)
        for row_name, reports in method_reports.items():
            rows[row_name] = build_result_row(reports, class_id2name)

    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    method_names = [
        "sobert",
        "codebert",
        "robert",
        "chatgpt",
        "claude",
        "tfidf",
        "word2vec",
    ]
    df = build_result_df(method_names)
    df = df.round(3)
    df.index.name = "row_name"
    df.to_csv("result.csv", index=True)


if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification/"
    test_csv_path = "reconstruct_dataset/test_dataset.csv"
    main()
