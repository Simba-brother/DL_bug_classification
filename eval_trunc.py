
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from eval import build_all_res_row_from_infer_df
from compare import wtl
from collections import defaultdict


def selectLongIds(dataset_df):
    '''
    从数据集中选择token>512的数据ID
    '''
    longIds = []
    seed = 42
    repeat = 1
    trained_model_dir = os.path.join(exp_root_dir,"exp_random5-3_code","trained_models","sobert",f"ft_model_{seed}_{repeat}")
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir, use_fast=True)
    for row_id,row in dataset_df.iterrows():
        Id = row['Id']
        text = row['Text']
        token_ids = tokenizer.encode(text,add_special_tokens=False,truncation=False)
        if len(token_ids) > 510: # 要给两个特殊token留地方
            longIds.append(Id)
    return longIds


def data_distribution(df:pd.DataFrame,all_long_ids:list):
    dis_res = {}
    dis_res["total"] = df.shape[0]
    dis_res["distribution"] = {}
    count = df["True"].value_counts().sort_index()
    longids = set(df["Id"]) & set(all_long_ids)
    if len(longids) > 0:
        long_df = df[df['Id'].isin(longids)]
        long_count = long_df["True"].value_counts().sort_index()
    else:
        long_count = pd.Series(dtype=int)

    for class_i in range(6):
        if class_i in count:
            class_count = count[class_i]
        else:
            class_count = 0
        if class_i in long_count:
            class_long_count = long_count[class_i]
        else:
            class_long_count = 0
        dis_res["distribution"][class_i] = {"count":class_count,"long_count":class_long_count}
    return dis_res


def get_misclassification_df(df:pd.DataFrame) -> pd.DataFrame:
    misclassified_ids = []
    for row_id,row in df.iterrows():
        if row["True"] != row["pred"]:
            misclassified_ids.append(int(row["Id"]))
    misclassified_df = df[df['Id'].isin(misclassified_ids)]
    return misclassified_df

def main_1():
    '''
    head与head+tail测试集性能指标对比
    '''
    head_df = pd.read_csv(os.path.join(exp_root_dir,"exp_random5-3_code", "sobert_res", "all_res.csv"))
    # headTail_df  = pd.read_csv(os.path.join(exp_root_dir,"exp_finetune_headtail", "sobert_res", "all_res.csv"))
    longformert =  pd.read_csv(os.path.join(exp_root_dir,"exp_random5-3_code", "longformer_res", "all_res.csv"))

    df_base = head_df
    df_improve = longformert

    col_name_list = df_base.columns.tolist()
    for col_name in col_name_list:
        base_list = df_base[col_name].tolist()
        improve_list = df_improve[col_name].tolist()
        h = wtl(base_list,improve_list)
        base_avg = round(np.mean(base_list),4)
        improve_avg = round(np.mean(improve_list),4)
        print(f"{col_name}|base_avg:{base_avg}|improve_avg:{improve_avg}|{h}")

def main_2():
    '''
    base与improve测试集（>512）性能指标对比
    '''
    dataset_df = pd.read_csv("dataset.csv")
    longIds = selectLongIds(dataset_df)
    long_base_rows = []
    long_improve_rows = []
    for seed in range(42,42+5):
        for repeat in range(1,1+3):
            print(f"{seed}_{repeat}")
            test_basepred_df =  pd.read_csv(os.path.join(exp_root_dir,"exp_random5-3_code","sobert_res", f"seed_{seed}_{repeat}","sobert.csv"))
            test_improvepred_df =  pd.read_csv(os.path.join(exp_root_dir,"exp_random5-3_code","longformer_res", f"seed_{seed}_{repeat}","longformer.csv"))
            long_test_basepred_df = test_basepred_df[test_basepred_df['Id'].isin(longIds)]
            long_test_improvepred_df = test_improvepred_df[test_improvepred_df['Id'].isin(longIds)]
            if long_test_basepred_df.shape[0] <= 0:
                print(f"long的数量为0,跳过这个切分")
                continue
            print(f"long的数量:{long_test_basepred_df.shape[0]}/{test_basepred_df.shape[0]}")
            print("long的数据类别分布")
            print(long_test_basepred_df["True"].value_counts().sort_index())
            long_base_rows.append(build_all_res_row_from_infer_df(long_test_basepred_df))
            long_improve_rows.append(build_all_res_row_from_infer_df(long_test_improvepred_df))
    long_base_res_df = pd.DataFrame(long_base_rows)
    long_improve_res_df = pd.DataFrame(long_improve_rows)
    ordered_columns = []
    for label_num in list(range(6)):
        ordered_columns.extend([f"acc_{label_num}", f"f1_{label_num}", f"auc_{label_num}"])
    ordered_columns.extend(["acc_all", "f1_all", "auc_all"])
    long_base_res_df = long_base_res_df[ordered_columns]
    long_improve_res_df = long_improve_res_df[ordered_columns]
    for col_name in ordered_columns:
        base_list = long_base_res_df[col_name].tolist()
        improve_list = long_improve_res_df[col_name].tolist()
        h = wtl(base_list,improve_list)
        base_mean = round(np.nanmean(base_list),4)
        improve_mean = round(np.nanmean(improve_list),4)
        print(f"LongText:{col_name}|base:{base_mean}|improve:{improve_mean}|{h}")





def collection_test_distribution(sobert_res_dir,longIds,modelname):
    repeat_collections = []
    for seed in range(42,42+5):
        for repeat in range(1,1+3):
            test_df =  pd.read_csv(os.path.join(sobert_res_dir,f"seed_{seed}_{repeat}",f"{modelname}.csv"))
            test_distribution = data_distribution(test_df,longIds)
            misclassified_testdf = get_misclassification_df(test_df)
            misclassified_test_distribution = data_distribution(misclassified_testdf,longIds)
            repeat_collections.append({
                "test_distribution":test_distribution,
                "misclassified_test_distribution":misclassified_test_distribution
            })
    return repeat_collections
    

def test_distribution(distribution_list):
    rate_res = defaultdict(list)
    count_res = defaultdict(list)
    longcount_res = defaultdict(list)
    for i in range(15):
        total_count = 0
        total_longcount = 0
        distribution = distribution_list[i]
        for class_i in range(6):
            count = distribution["test_distribution"]["distribution"][class_i]["count"]
            long_count = distribution["test_distribution"]["distribution"][class_i]["long_count"]
            if count > 0:
                rate = round(long_count/count,4)
            else:
                rate = np.nan
            rate_res[class_i].append(rate)
            count_res[class_i].append(count)
            longcount_res[class_i].append(long_count)
            total_count += count
            total_longcount += long_count
        if total_count > 0:
            rate_res["all"].append(round(total_longcount/total_count,4))
        else:
            rate_res["all"].append(np.nan)
        count_res["all"].append(total_count)
        longcount_res["all"].append(total_longcount)
    return rate_res,count_res,longcount_res

def misclassification_distribution(distribution_list):
    rate_res = defaultdict(list)
    count_res = defaultdict(list)
    longcount_res = defaultdict(list)
    for i in range(15):
        total_count = 0
        total_longcount = 0
        distribution = distribution_list[i]
        for class_i in range(6):
            count = distribution["misclassified_test_distribution"]["distribution"][class_i]["count"]
            long_count = distribution["misclassified_test_distribution"]["distribution"][class_i]["long_count"]
            if count > 0:
                rate = round(long_count/count,4)
            else:
                rate = np.nan
            rate_res[class_i].append(rate)
            count_res[class_i].append(count)
            longcount_res[class_i].append(long_count)
            total_count += count
            total_longcount += long_count
        if total_count > 0:
            rate_res["all"].append(round(total_longcount/total_count,4))
        else:
            rate_res["all"].append(np.nan)
        count_res["all"].append(total_count)
        longcount_res["all"].append(total_longcount)
    return rate_res,count_res,longcount_res


def nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return np.nan
    return round(np.nanmean(values), 4)


def wtl_without_nan(head_rate_list, headtail_rate_list):
    valid_pairs = [
        (head_rate, headtail_rate)
        for head_rate, headtail_rate in zip(head_rate_list, headtail_rate_list)
        if not np.isnan(head_rate) and not np.isnan(headtail_rate)
    ]
    if len(valid_pairs) == 0:
        return "NA"
    valid_head_rate_list = [head_rate for head_rate, _ in valid_pairs]
    valid_headtail_rate_list = [headtail_rate for _, headtail_rate in valid_pairs]
    if np.allclose(valid_head_rate_list, valid_headtail_rate_list):
        return "T"
    return wtl(valid_head_rate_list, valid_headtail_rate_list, sign=-1)


def aggregate_rate_or_nan(longcount_list, count_list):
    mean_longcount, mean_count = aggregate_mean_counts(longcount_list, count_list)
    if mean_count <= 0:
        return np.nan
    return round(mean_longcount / mean_count, 4)


def aggregate_counts(longcount_list, count_list):
    return int(np.sum(longcount_list)), int(np.sum(count_list))


def aggregate_mean_counts(longcount_list, count_list):
    mean_longcount = round(np.mean(longcount_list), 1)
    mean_count = round(np.mean(count_list), 1)
    return mean_longcount, mean_count


def compare_aggregate_rate(head_rate, headtail_rate):
    if np.isnan(head_rate) or np.isnan(headtail_rate):
        return "NA"
    if headtail_rate < head_rate:
        return "headtail_better"
    if headtail_rate > head_rate:
        return "head_better"
    return "tie"


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
    print("收集测试数据集类别分布情况:")
    print("="*50)
    base_res_dir = os.path.join(exp_root_dir,"exp_random5-3_code/sobert_res")

    base_distribution_list = collection_test_distribution(base_res_dir,longIds, modelname="sobert")
    improve_res_dir = os.path.join(exp_root_dir,"exp_random5-3_code/longformer_res")
    improve_distribution_list = collection_test_distribution(improve_res_dir,longIds,modelname="longformer")

    test_rate_res,test_count_res,test_longcount_res = test_distribution(base_distribution_list)
    for class_i in [0,1,2,3,4,5,"all"]:
        print(f"class:{class_i}")
        test_count = round(np.nanmean(test_count_res[class_i]),1)
        test_longcount = round(np.nanmean(test_longcount_res[class_i]),1) 
        print(f"long/count:{test_longcount}/{test_count}")

    base_rate_res,base_count_res,base_longcount_res = misclassification_distribution(base_distribution_list)
    improve_rate_res,improve_count_res,improve_longcount_res = misclassification_distribution(improve_distribution_list)

    print("="*50)
    print("收集misclassified测试数据集类别分布情况:")
    print("="*50)
    for class_i in [0,1,2,3,4,5,"all"]:
        print(f"class:{class_i}")
        print("=====Rate=====")
        base_rate_list = base_rate_res[class_i]
        improve_rate_list = improve_rate_res[class_i]
        base_rate_mean = nanmean_or_nan(base_rate_list)
        improve_rate_mean = nanmean_or_nan(improve_rate_list)
        h = wtl_without_nan(base_rate_list, improve_rate_list) # 负向指标
        # print(f"base_list:{base_rate_list}")
        # print(f"improve_rate_list:{improve_rate_list}")
        print(f"baserate:{base_rate_mean}|improverate:{improve_rate_mean}|wtl:{h}")

        print("=====Count=====")
        base_count_list = base_count_res[class_i]
        improve_count_list = improve_count_res[class_i]
        base_count_mean = nanmean_or_nan(base_count_list)
        improve_count_mean = nanmean_or_nan(improve_count_list)
        # h = wtl_without_nan(base_count_list, improve_count_list) # 负向指标


        base_longcount_list = base_longcount_res[class_i]
        improve_longcount_list = improve_longcount_res[class_i]
        base_longcount_mean = nanmean_or_nan(base_longcount_list)
        improve_longcount_mean = nanmean_or_nan(improve_longcount_list)

        print(f"basecount:{base_longcount_mean}/{base_count_mean}|improverate:{improve_longcount_mean}/{improve_count_mean}")


    '''
    print("="*50)
    print("聚合比例统计结果")
    print("="*50)
    for class_i in [0,1,2,3,4,5,"all"]:
        base_count_list = base_count_res[class_i]
        improve_count_list = improve_count_res[class_i]
        base_longcount_list = base_longcount_res[class_i]
        improve_longcount_list = improve_longcount_res[class_i]

        base_aggregate_rate = aggregate_rate_or_nan(base_longcount_list, base_count_list)
        improve_aggregate_rate = aggregate_rate_or_nan(
            improve_longcount_list,
            improve_count_list,
        )
        compare_res = compare_aggregate_rate(base_aggregate_rate, headtail_aggregate_rate)
        head_mean_longcount, head_mean_count = aggregate_mean_counts(
            head_longcount_list,
            head_count_list,
        )
        headtail_mean_longcount, headtail_mean_count = aggregate_mean_counts(
            headtail_longcount_list,
            headtail_count_list,
        )

        print(f"class:{class_i}")
        print(f"head:{head_mean_longcount}/{head_mean_count}")
        print(f"headtail:{headtail_mean_longcount}/{headtail_mean_count}")
        print(
            f"head_aggregate_rate:{head_aggregate_rate}|"
            f"headtail_aggregate_rate:{headtail_aggregate_rate}|"
            f"compare:{compare_res}"
        )
    '''

if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    # main_1() # head与head+tail测试集性能指标对比
    # main_2() # head与head+tail测试集（>512）性能指标对比
    main_3() # 长文数据分布
