'''
基于时间对dataset.csv进行切分出train/eval/test
'''
import pandas as pd

def choose_split_time(
    test_ratio=0.15,
    size_tolerance_ratio=0.05,
    top_k=10,
):
    """
    选择 test 集的时间切分点。

    createTime 晚于切分点的数据属于 test 集。候选切分点需满足：
    1. test 集大小接近 test_ratio；
    2. test 集包含全部 6 个类别；
    3. 优先让 6 个类别的样本数最大差值最小。

    返回最佳切分时间（pandas.Timestamp）。
    """
    df = pd.read_csv(dataset_csv_path)

    required_columns = {"createTime", "LabelNum"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"dataset.csv 缺少字段：{sorted(missing_columns)}"
        )

    # 转换为时间类型；无法解析的值会变成 NaT。
    df["createTime"] = pd.to_datetime(
        df["createTime"],
        errors="coerce",
    )
    invalid_time_mask = df["createTime"].isna()
    if invalid_time_mask.any():
        invalid_ids = (
            df.loc[invalid_time_mask, "Id"].tolist()
            if "Id" in df.columns
            else df.index[invalid_time_mask].tolist()
        )
        raise ValueError(
            f"有 {invalid_time_mask.sum()} 条 createTime 无法解析，"
            f"对应 Id/行号：{invalid_ids}"
        )

    expected_labels = set(range(6))
    actual_labels = set(df["LabelNum"].unique())
    if actual_labels != expected_labels:
        raise ValueError(
            f"LabelNum 应为 0-5，实际为：{sorted(actual_labels)}"
        )

    total_size = len(df)
    target_test_size = round(total_size * test_ratio)
    tolerance = round(total_size * size_tolerance_ratio)
    min_test_size = max(6, target_test_size - tolerance)
    max_test_size = min(total_size - 1, target_test_size + tolerance)

    candidates = []
    for split_time in sorted(df["createTime"].unique()):
        # 严格晚于 split_time 的帖子进入 test 集。
        test_df = df[df["createTime"] > split_time]
        test_size = len(test_df)

        if not min_test_size <= test_size <= max_test_size:
            continue

        class_counts = (
            test_df["LabelNum"]
            .value_counts()
            .reindex(range(6), fill_value=0)
        )

        # test 集必须包含全部类别，否则无法计算完整的六分类指标。
        if (class_counts == 0).any():
            continue

        max_gap = int(class_counts.max() - class_counts.min())
        candidates.append(
            {
                "split_time": pd.Timestamp(split_time),
                "test_size": test_size,
                "size_difference": abs(test_size - target_test_size),
                "max_class_gap": max_gap,
                "class_counts": class_counts.tolist(),
            }
        )

    if not candidates:
        raise ValueError(
            "没有找到满足测试集规模且包含全部 6 类的时间切分点。"
            "请增大 size_tolerance_ratio。"
        )

    # 先保证类别数量均衡，再考虑测试集大小是否接近目标值；
    # 若仍相同，则选择更晚的切分点。
    candidates.sort(
        key=lambda item: (
            item["max_class_gap"],
            item["size_difference"],
            -item["split_time"].value,
        )
    )

    print(
        f"目标 test 大小：{target_test_size}，"
        f"允许范围：[{min_test_size}, {max_test_size}]"
    )
    print("排名靠前的候选时间切分点：")
    for index, candidate in enumerate(candidates[:top_k], start=1):
        print(
            f"{index}. split_time={candidate['split_time']}, "
            f"test_size={candidate['test_size']}, "
            f"各类别数量={candidate['class_counts']}, "
            f"最大类别差值={candidate['max_class_gap']}"
        )

    best = candidates[0]
    print(
        f"最佳切分点：{best['split_time']}；"
        f"test_size={best['test_size']}；"
        f"各类别数量={best['class_counts']}"
    )
    return best["split_time"]




def construct_testset(best_split_time):
    # 下面按照这个时间之后切分出 test。
    df = pd.read_csv(dataset_csv_path)
    df["createTime"] = pd.to_datetime(
        df["createTime"],
        errors="coerce",
    )

    # 与 choose_split_time() 保持一致，严格晚于切分点的数据进入 test。
    test_df = df[df["createTime"] > best_split_time].copy()
    if test_df.empty:
        raise ValueError(
            f"切分时间 {best_split_time} 之后没有测试数据。"
        )

    # 按创建时间排序并保存，不写入 DataFrame 的额外索引列。
    test_df.sort_values("createTime", inplace=True)
    test_df.to_csv(test_dataset_csv_path, index=False)

    class_counts = (
        test_df["LabelNum"]
        .value_counts()
        .reindex(range(6), fill_value=0)
        .sort_index()
    )
    print(
        f"已基于切分时间 {best_split_time} 生成 "
        f"{test_dataset_csv_path}"
    )
    print(f"test 集样本数：{len(test_df)}")
    print(f"test 集各类别数量：{class_counts.to_dict()}")

def construct_trainvalset():
    all_df = pd.read_csv(dataset_csv_path)
    test_df = pd.read_csv(test_dataset_csv_path)
    # 请基于ID字段，从all_df中抽取出来trainval_df，即trainval_df = all_df - test_df
    if "Id" not in all_df.columns or "Id" not in test_df.columns:
        raise ValueError("dataset.csv 和 test_dataset.csv 都必须包含 Id 字段。")

    test_ids = set(test_df["Id"])
    all_ids = set(all_df["Id"])
    unknown_test_ids = test_ids - all_ids
    if unknown_test_ids:
        raise ValueError(
            "test_dataset.csv 中存在不属于 dataset.csv 的 Id："
            f"{sorted(unknown_test_ids)}"
        )

    trainval_df = all_df[~all_df["Id"].isin(test_ids)].copy()
    if trainval_df.empty:
        raise ValueError("去除 test 数据后，trainval_df 为空。")

    # 保持与时间切分逻辑一致，并按时间先后排列训练验证数据。
    if "createTime" in trainval_df.columns:
        trainval_df["createTime"] = pd.to_datetime(
            trainval_df["createTime"],
            errors="coerce",
        )
        trainval_df.sort_values("createTime", inplace=True)

    trainval_df.to_csv(trainval_dataset_csv_path, index=False)

    class_counts = (
        trainval_df["LabelNum"]
        .value_counts()
        .reindex(range(6), fill_value=0)
        .sort_index()
    )
    print(f"已生成 {trainval_dataset_csv_path}")
    print(f"trainval 集样本数：{len(trainval_df)}")
    print(f"trainval 集各类别数量：{class_counts.to_dict()}")



def main():
    stage = 0 # 贯穿整个流程
    if stage == 0:
        best_split_time = choose_split_time()
        construct_testset(best_split_time)
        construct_trainvalset()
    elif stage == 1:
        # 构建出测试集 
        best_split_time = choose_split_time()
        construct_testset(best_split_time)
    elif stage == 2:
        # 构建出训练集
        construct_trainvalset()

if __name__ == "__main__":
    dataset_csv_path = "./dataset.csv"
    test_dataset_csv_path = "./reconstruct_dataset/test_dataset.csv"
    trainval_dataset_csv_path = "./reconstruct_dataset/trainval_dataset.csv"
    main()

