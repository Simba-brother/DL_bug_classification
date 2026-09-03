
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from eval import build_all_res_row_from_infer_df
from compare import wtl


def selectLongIds(dataset_df):
    longIds = []
    seed = 42
    repeat = 1
    trained_model_dir = os.path.join(exp_root_dir,"exp","trained_models","sobert",f"ft_model_{seed}_{repeat}")
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir, use_fast=True)
    for row_id,row in dataset_df.iterrows():
        Id = row['Id']
        text = row['Text']
        token_ids = tokenizer.encode(text,add_special_tokens=False,truncation=False)
        if len(token_ids) > 510: # 要给两个特殊token留地方
            longIds.append(Id)
    return longIds


def print_distribution(df:pd.DataFrame,all_long_ids:list):
    print(f"数据集总数量:{df.shape[0]}")
    print("数据集类别数量分布:")
    print(df["True"].value_counts().sort_index())
    selected_ids = set(df["Id"]) & set(all_long_ids)
    print(f"测试数据集(>512)数量:{len(selected_ids)}/{df.shape[0]}")
    if len(selected_ids) > 0:
        long_df = df[df['Id'].isin(selected_ids)]
        print("测试数据集(>512)类别数量分布:")
        print(long_df["True"].value_counts().sort_index())


def main_1():
    '''
    head与head+tail测试集性能指标对比
    '''
    head_df = pd.read_csv(os.path.join(exp_root_dir,"exp", "sobert_res", "all_res.csv"))
    headTail_df = pd.read_csv(os.path.join(exp_root_dir,"exp", "sobert_res_truncHeadTail", "all_res.csv"))
    col_name_list = head_df.columns.tolist()
    for col_name in col_name_list:
        head_list = head_df[col_name].tolist()
        headTail_list = headTail_df[col_name].tolist()
        h = wtl(head_list,headTail_list)
        head_avg = round(np.mean(head_list),4)
        headTail_avg = round(np.mean(headTail_list),4)
        print(f"{col_name}|head_avg:{head_avg}|headTail_avg:{headTail_avg}|{h}")

def main_2():
    '''
    head与head+tail测试集（>512）性能指标对比
    '''
    dataset_df = pd.read_csv("dataset.csv")
    longIds = selectLongIds(dataset_df)
    long_head_rows = []
    long_headTail_rows = []
    for seed in range(42,42+5):
        for repeat in range(1,1+3):
            print(f"{seed}_{repeat}")
            test_headpred_df =  pd.read_csv(os.path.join(exp_root_dir,"exp","sobert_res", f"seed_{seed}_{repeat}","sobert.csv"))
            test_headTailpred_df =  pd.read_csv(os.path.join(exp_root_dir,"exp","sobert_res_truncHeadTail", f"seed_{seed}_{repeat}","sobert.csv"))
            long_test_headpred_df = test_headpred_df[test_headpred_df['Id'].isin(longIds)]
            long_test_headTailpred_df = test_headTailpred_df[test_headTailpred_df['Id'].isin(longIds)]
            if long_test_headpred_df.shape[0] <= 0:
                print(f"long的数量为0,跳过这个切分")
                continue
            print(f"long的数量:{long_test_headpred_df.shape[0]}/{test_headpred_df.shape[0]}")
            print("long的数据类别分布")
            print(long_test_headpred_df["True"].value_counts().sort_index())
            long_head_rows.append(build_all_res_row_from_infer_df(long_test_headpred_df))
            long_headTail_rows.append(build_all_res_row_from_infer_df(long_test_headTailpred_df))
    long_head_res_df = pd.DataFrame(long_head_rows)
    long_headtail_res_df = pd.DataFrame(long_headTail_rows)
    ordered_columns = []
    for label_num in list(range(6)):
        ordered_columns.extend([f"acc_{label_num}", f"f1_{label_num}", f"auc_{label_num}"])
    ordered_columns.extend(["acc_all", "f1_all", "auc_all"])
    long_head_res_df = long_head_res_df[ordered_columns]
    long_headtail_res_df = long_headtail_res_df[ordered_columns]
    for col_name in ordered_columns:
        head_list = long_head_res_df[col_name].tolist()
        headTail_list = long_headtail_res_df[col_name].tolist()
        h = wtl(head_list,headTail_list)
        head_mean = round(np.nanmean(head_list),4)
        headTail_mean = round(np.nanmean(headTail_list),4)
        print(f"LongText:{col_name}|head:{head_mean}|headTail:{headTail_mean}|{h}")

def main_3():
    '''
    长文数据分布
    '''
    # 整体数据集
    print("="*50)
    print("整体数据集类别分布情况:")
    print("="*50)
    dataset_df = pd.read_csv("dataset.csv")
    print(f"数据集总数量:{dataset_df.shape[0]}")
    print("数据集类别数量分布:")
    print(dataset_df["LabelNum"].value_counts().sort_index())
    longIds = selectLongIds(dataset_df)
    print(f"数据集(>512)数量:{len(longIds)}/{dataset_df.shape[0]}")
    sub_dataset_df = dataset_df[dataset_df['Id'].isin(longIds)]
    print("数据集(>512)类别数量分布:")
    print(sub_dataset_df["LabelNum"].value_counts().sort_index())

    # 测试数据集
    print("="*50)
    print("测试数据集类别分布情况:")
    print("="*50)
    seed = 45
    repeat = 1
    test_df =  pd.read_csv(os.path.join(exp_root_dir,"exp","sobert_res", 
                                        f"seed_{seed}_{repeat}","sobert.csv"))
    print_distribution(test_df,longIds)

    # misclassified testset
    print("="*50)
    print("测试数据集(misclassified)类别分布情况:")
    print("="*50)
    misclassified_ids = []
    for row_id,row in test_df.iterrows():
        if row["True"] != row["pred"]:
            misclassified_ids.append(int(row["Id"]))
    misclassified_df = test_df[test_df['Id'].isin(misclassified_ids)]
    print_distribution(misclassified_df,longIds)


    # misclassified testset(head+tail)
    test_df =  pd.read_csv(os.path.join(exp_root_dir,"exp","sobert_res_truncHeadTail", 
                                        f"seed_{seed}_{repeat}","sobert.csv"))
    print("="*50)
    print("测试数据集(misclassified_head+tail)类别分布情况:")
    print("="*50)
    misclassified_ids = []
    for row_id,row in test_df.iterrows():
        if row["True"] != row["pred"]:
            misclassified_ids.append(int(row["Id"]))
    misclassified_df = test_df[test_df['Id'].isin(misclassified_ids)]
    print_distribution(misclassified_df,longIds)
if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    # main_1()
    main_2()
    # main_3()
