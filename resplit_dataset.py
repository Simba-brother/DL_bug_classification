'''
基于时间对dataset.csv进行切分出train/eval/test
'''
import os
from sklearn.model_selection import train_test_split
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


def split_latest_90_by_create_time(
    test_size=90,
    output_dir="./reconstruct_dataset/time",
):
    """
    基于 dataset.csv 的 createTime 字段切分数据集。

    createTime 最新的 test_size 条样本作为 test，其余样本作为 trainval。
    结果保存到 reconstruct_dataset/time/test_dataset.csv 和
    reconstruct_dataset/time/trainval_dataset.csv。
    """
    df = pd.read_csv(dataset_csv_path)
    required_columns = {"createTime", "Label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"dataset.csv 缺少字段：{sorted(missing_columns)}")

    if len(df) <= test_size:
        raise ValueError(
            f"dataset.csv 样本数必须大于 test_size={test_size}，"
            f"当前样本数：{len(df)}"
        )

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

    sort_columns = ["createTime"]
    ascending = [False]
    if "Id" in df.columns:
        sort_columns.append("Id")
        ascending.append(True)

    sorted_df = df.sort_values(sort_columns, ascending=ascending)
    test_df = sorted_df.head(test_size).copy()
    trainval_df = sorted_df.iloc[test_size:].copy()

    # 保存时按时间升序排列，便于人工检查时间范围。
    test_df.sort_values("createTime", inplace=True)
    trainval_df.sort_values("createTime", inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    test_csv_path = os.path.join(output_dir, "test_dataset.csv")
    trainval_csv_path = os.path.join(output_dir, "trainval_dataset.csv")
    test_df.to_csv(test_csv_path, index=False)
    trainval_df.to_csv(trainval_csv_path, index=False)

    test_label_counts = test_df["Label"].value_counts().sort_index()
    trainval_label_counts = trainval_df["Label"].value_counts().sort_index()

    print(f"已生成 {test_csv_path}")
    print(f"test 集样本数：{len(test_df)}")
    print(f"test 集 createTime 范围：{test_df['createTime'].min()} ~ {test_df['createTime'].max()}")
    print(f"test 集各 Label 数量：{test_label_counts.to_dict()}")
    print(f"已生成 {trainval_csv_path}")
    print(f"trainval 集样本数：{len(trainval_df)}")
    print(
        f"trainval 集 createTime 范围："
        f"{trainval_df['createTime'].min()} ~ {trainval_df['createTime'].max()}"
    )
    print(f"trainval 集各 Label 数量：{trainval_label_counts.to_dict()}")
    return trainval_df, test_df


def split_train_val_test_by_create_time(
    test_size=90,
    val_size=90,
    output_dir="./reconstruct_dataset/time_tvt90",
):
    """
    基于 dataset.csv 的 createTime 字段切分数据集。

    createTime 最新的 test_size 条样本作为 test，
    次新的 val_size 条样本作为 val，
    剩余样本作为 train。
    结果保存到 reconstruct_dataset/time/test_dataset.csv、
    val_dataset.csv 和 train_dataset.csv。
    """
    df = pd.read_csv(dataset_csv_path)
    required_columns = {"createTime", "Label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"dataset.csv 缺少字段：{sorted(missing_columns)}")

    if len(df) <= test_size + val_size:
        raise ValueError(
            f"dataset.csv 样本数必须大于 test_size+val_size="
            f"{test_size + val_size}，当前样本数：{len(df)}"
        )

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

    sort_columns = ["createTime"]
    ascending = [False]
    if "Id" in df.columns:
        sort_columns.append("Id")
        ascending.append(True)

    sorted_df = df.sort_values(sort_columns, ascending=ascending)
    test_df = sorted_df.head(test_size).copy()
    val_df = sorted_df.iloc[test_size:test_size + val_size].copy()
    train_df = sorted_df.iloc[test_size + val_size:].copy()

    # 保存时按时间升序排列，便于人工检查时间范围。
    test_df.sort_values("createTime", inplace=True)
    val_df.sort_values("createTime", inplace=True)
    train_df.sort_values("createTime", inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    test_csv_path = os.path.join(output_dir, "test_dataset.csv")
    val_csv_path = os.path.join(output_dir, "val_dataset.csv")
    train_csv_path = os.path.join(output_dir, "train_dataset.csv")
    test_df.to_csv(test_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    train_df.to_csv(train_csv_path, index=False)

    for name, part_df, csv_path in (
        ("test", test_df, test_csv_path),
        ("val", val_df, val_csv_path),
        ("train", train_df, train_csv_path),
    ):
        label_counts = part_df["Label"].value_counts().sort_index()
        print(f"已生成 {csv_path}")
        print(f"{name} 集样本数：{len(part_df)}")
        print(
            f"{name} 集 createTime 范围："
            f"{part_df['createTime'].min()} ~ {part_df['createTime'].max()}"
        )
        print(f"{name} 集各 Label 数量：{label_counts.to_dict()}")

    return train_df, val_df, test_df



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

def split_nocode_dataset():
    test_df = pd.read_csv(test_dataset_csv_path)
    trainval_df = pd.read_csv(trainval_dataset_csv_path)
    if "Id" not in test_df.columns or "Id" not in trainval_df.columns:
        raise ValueError(
            "test_dataset.csv 和 trainval_dataset.csv 都必须包含 Id 字段。"
        )

    nocode_df = pd.read_csv(nocode_dataset_csv_path)
    if "Id" not in nocode_df.columns:
        raise ValueError("nocode_dataset.csv 必须包含 Id 字段。")

    test_ids = list(test_df["Id"])
    trainval_ids = list(trainval_df["Id"])
    test_id_set = set(test_ids)
    trainval_id_set = set(trainval_ids)
    nocode_id_set = set(nocode_df["Id"])

    overlap_ids = test_id_set & trainval_id_set
    if overlap_ids:
        raise ValueError(
            "test_dataset.csv 和 trainval_dataset.csv 存在重复 Id："
            f"{sorted(overlap_ids)}"
        )

    missing_test_ids = test_id_set - nocode_id_set
    missing_trainval_ids = trainval_id_set - nocode_id_set
    if missing_test_ids or missing_trainval_ids:
        raise ValueError(
            "nocode_dataset.csv 缺少切分所需的 Id。"
            f"test 缺失：{sorted(missing_test_ids)}；"
            f"trainval 缺失：{sorted(missing_trainval_ids)}"
        )

    nocode_test_df = pd.DataFrame({"Id": test_ids}).merge(
        nocode_df,
        on="Id",
        how="left",
        sort=False,
    )
    nocode_trainval_df = pd.DataFrame({"Id": trainval_ids}).merge(
        nocode_df,
        on="Id",
        how="left",
        sort=False,
    )

    nocode_output_dir = "./reconstruct_dataset/nocode"
    os.makedirs(nocode_output_dir, exist_ok=True)
    nocode_test_dataset_csv_path = os.path.join(
        nocode_output_dir,
        "test_dataset.csv",
    )
    nocode_trainval_dataset_csv_path = os.path.join(
        nocode_output_dir,
        "trainval_dataset.csv",
    )

    nocode_test_df.to_csv(nocode_test_dataset_csv_path, index=False)
    nocode_trainval_df.to_csv(nocode_trainval_dataset_csv_path, index=False)

    print(f"已生成 {nocode_test_dataset_csv_path}")
    print(f"nocode test 集样本数：{len(nocode_test_df)}")
    print(f"已生成 {nocode_trainval_dataset_csv_path}")
    print(f"nocode trainval 集样本数：{len(nocode_trainval_df)}")


def pre_split(split_mode:str):
    df = pd.read_csv('dataset.csv')
    test_size = df.shape[0] * 0.15
    val_size = test_size
    if split_mode == "random_5-3":
        output_dir = "./random_5-3"
        split_seeds = list(range(42,42+5)) # [42-46]
        repeats = 3
        id = 1
        for seed in split_seeds:
            for repeat in range(repeats):
                
                output_dir = os.path.join(output_dir,f"{id}")
                os.makedirs(output_dir,exist_ok=True)


                Id_trainval, Id_test, y_trainval, y_test = train_test_split(list(df['Id']),
                                                                    list(df['LabelNum']),
                                                                    test_size=test_size,
                                                                    stratify=df['LabelNum'],
                                                                    random_state=seed)
                
                Id_train, Id_val, y_train, y_val = train_test_split(Id_trainval,
                                                                    y_trainval,
                                                                    test_size=val_size, 
                                                                    stratify=y_train, 
                                                                    random_state=seed)

    elif split_mode == "random_15":
        split_seeds = list(range(42,42+15)) # [42-56]
    elif split_mode == "time_5-3":
        split_seeds = list(range(42,42+5)) # [42-46]
        repeat = 3
    elif split_mode == "time_15":
        split_seeds = list(range(42,42+15)) # [42-56]
    elif split_mode == 'time_tvt':
        test_size = 90
        val_size = 90
        





if __name__ == "__main__":
    dataset_csv_path = "./dataset.csv"
    # split_latest_90_by_create_time()
    split_train_val_test_by_create_time()

    # test_dataset_csv_path = "./reconstruct_dataset/test_dataset.csv"
    # trainval_dataset_csv_path = "./reconstruct_dataset/trainval_dataset.csv"
    # main()
    # nocode_dataset_csv_path = "dataset_nocode.csv"
    # split_nocode_dataset()
