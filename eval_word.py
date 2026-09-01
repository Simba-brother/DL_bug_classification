import os
import time
import numpy as np
from collections import defaultdict
from eval import infer_trained_model
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score


def build_prediction_df(df:pd.DataFrame, gt_labels:list, p_labels:list) -> pd.DataFrame:
    predict_data = {
        "True": gt_labels,
        "pred": p_labels,
    }
    if "Id" in df.columns:
        predict_data = {"Id": list(df["Id"]), **predict_data}
    return pd.DataFrame(predict_data)

def get_csv_files(directory):
    """遍历目录下所有CSV文件并返回文件名列表"""
    csv_files = []
    
    # 遍历目录
    for file in os.listdir(directory):
        # 检查是否为CSV文件
        if file.endswith('.csv'):
            csv_files.append(file)
    return csv_files

def extract_rm_words(csv_dir)->list:
    rm_words = [] # 例如：rm_words = ["tensorflow","model"]
    csv_filename_list = get_csv_files(csv_dir)
    for csv_filename in csv_filename_list:
        pattern = r'remove_(.*)\.csv$'
        match = re.search(pattern, csv_filename)
        if match:
            rm_words.append(match.group(1))
        else:
            raise Exception("提取rm word的正则匹配错误")
    assert len(rm_words) == len(csv_filename_list), "csv数量与rm word数量不一致。"
    rm_words.sort() # replace sort一下
    return rm_words

def eval_single_word():
    stage = 3
    if stage == 1:
        '''
        阶段1：获得预测结果
        '''
        s_time = time.monotonic()
        # load model
        trained_model_dir = os.path.join(exp_data_dir,"exp","trained_models",model_name,f"ft_model_{model_id}")
        # load dataset
        # origin dataset
        print(f"预测[origin]...")
        df_origin = pd.read_csv("dataset.csv") # 原始数据集
        df_origin = df_origin[df_origin["Label"] != "Others"].reset_index(drop=True) # 如果不写 drop=True，旧索引会变成一列叫 index 的新列
        gt_labels, p_labels, probs = infer_trained_model(trained_model_dir,df_origin,device)
        predict_df_origin = build_prediction_df(df_origin,gt_labels,p_labels)

        save_dir = os.path.join(exp_data_dir,"remove_word_res")
        os.makedirs(save_dir,exist_ok=True)
        predict_df_origin.to_csv(os.path.join(save_dir,"origin.csv"),index=False)

        # 抽取所有的rm_words
        csv_dir = os.path.join(exp_data_dir,"remove_word")
        rm_words = extract_rm_words(csv_dir) # sorted
        words_len = len(rm_words)
        print(f"rm words总共有:{words_len}")

        mid = words_len // 2
        # 前半段
        # rm_words = rm_words[:mid]
        # print(f"本次处理前[0:{mid})的word index,共{mid}个word")

        # 后半段
        rm_words = rm_words[mid:]
        print(f"本次处理后[{mid}:{words_len})的word index,共{words_len - mid}个word")

        for rm_word in rm_words:
            print(f"预测[{rm_word}]...")
            df_rm = pd.read_csv(os.path.join(exp_data_dir,"remove_word",f"remove_{rm_word}.csv"))
            df_rm = df_rm[df_rm["Label"] != "Others"].reset_index(drop=True) # 如果不写 drop=True，旧索引会变成一列叫 index 的新列
            gt_labels, p_labels, probs = infer_trained_model(trained_model_dir,df_rm,device)
            predict_df_rm = build_prediction_df(df_origin,gt_labels,p_labels)
            predict_df_rm.to_csv(os.path.join(save_dir,f"rm_{rm_word}.csv"),index=False)
        print(f"预测结果保存在:{save_dir}")
        cost_time = int(time.monotonic() - s_time)
        hours, remainder = divmod(cost_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"总耗时: {hours:02d}小时 "
            f"{minutes:02d}分钟 {seconds:02d}秒",
            flush=True,
        )
    if stage == 2:
        '''
        阶段2：基于阶段1的预测结果获得评价指标
        '''
        res_dir = os.path.join(exp_data_dir,"remove_word_res")
        origin_predict_df = pd.read_csv(os.path.join(res_dir,"origin.csv"))
        y_true = origin_predict_df["True"]
        y_pred = origin_predict_df["pred"]
        origin_report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        origin_f1 = {}
        origin_f1["all"] = origin_report["macro avg"]["f1-score"]
        for label_idx in range(5):
            origin_f1[label_idx] = origin_report[str(label_idx)]["f1-score"]

        save_dir = os.path.join(res_dir,"f1")
        os.makedirs(save_dir,exist_ok=True)
        origin_f1_df = pd.DataFrame([origin_f1])
        origin_f1_df.to_csv(os.path.join(save_dir,"origin.csv"),index=False)

        rm_f1 = []
        # 抽取所有的rm_words
        csv_dir = os.path.join(exp_data_dir,"remove_word")
        rm_words = extract_rm_words(csv_dir)

        for rm_word in rm_words:
            row = {}
            row["word"] = rm_word
            rm_predict_df = pd.read_csv(os.path.join(res_dir,f"rm_{rm_word}.csv"))
            y_true = rm_predict_df["True"]
            y_pred = rm_predict_df["pred"]
            rm_report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            )
            row["all"] = rm_report["macro avg"]["f1-score"]
            for label_idx in range(5):
                row[label_idx] = rm_report[str(label_idx)]["f1-score"]
            rm_f1.append(row)
        rm_f1_df = pd.DataFrame(rm_f1)
        rm_f1_df.to_csv(os.path.join(save_dir,"rm.csv"),index=False)
        print(f"F1-score结果保存在:{save_dir}")

    if stage == 3:
        '''
        阶段3：基于阶段2的指标,统计论文结果
        '''
        f1_dir = os.path.join(exp_data_dir,"remove_word_res","f1")
        orgin_f1_df = pd.read_csv(os.path.join(f1_dir,"origin.csv"))
        rm_f1 = pd.read_csv(os.path.join(f1_dir,"rm.csv"))
        for col in [0,1,2,3,4,"all"]:
            increase_dict = {}
            decrease_dict = {}
            min_v = float('inf')
            max_v = float('-inf')
            origin_f1 = orgin_f1_df[str(col)].tolist()[-1]
            origin_f1 = round(origin_f1,4)
            for row_id,row in rm_f1.iterrows():
                word = row["word"]
                value = row[str(col)]
                value = round(value,4)
                min_v = min(min_v,value)
                max_v = max(max_v,value)
                if value > origin_f1:
                    increase_dict[word] = value
                elif value < origin_f1:
                    decrease_dict[word] = value
            # decrease_list 中的item按照 value从小到大排序
            sorted_words = sorted(decrease_dict, key=lambda k: decrease_dict[k])
            # assert len(sorted_words) > 10, "decrease word少于10个"
            print(f"{col}|Increase Count:{len(increase_dict)},Decrease Count:{len(decrease_dict)},Max:{max_v},Min:{min_v},Original:{origin_f1}")
            if len(sorted_words) < 10:
                print(f"Top10(<10) decrease:{sorted_words}")
            else:
                print(f"Top10 decrease:{sorted_words[:10]}")
            print("="*30)




def eval_combinword():
    stage = 1
    if stage == 1:
        '''
        阶段1：获得预测结果
        '''
        s_time = time.monotonic()
        # load model
        trained_model_dir = os.path.join(exp_data_dir,"exp","trained_models",model_name,f"ft_model_{model_id}")
        # load dataset
        # origin dataset
        print(f"预测[origin]...")
        df_origin = pd.read_csv("dataset.csv") # 原始数据集
        df_origin = df_origin[df_origin["Label"] != "Others"].reset_index(drop=True) # 如果不写 drop=True，旧索引会变成一列叫 index 的新列
        gt_labels, p_labels, probs = infer_trained_model(trained_model_dir,df_origin,device)
        predict_df_origin = build_prediction_df(df_origin,gt_labels,p_labels)

        save_dir = os.path.join(exp_data_dir,"remove_combineword_res")
        os.makedirs(save_dir,exist_ok=True)
        predict_df_origin.to_csv(os.path.join(save_dir,"origin.csv"),index=False)

        # 抽取所有的rm_combines
        combinwords_df = pd.read_csv(os.path.join(exp_data_dir,"remove_combineword", "combinewords.csv"))
        combin_ids = combinwords_df["ID"].tolist()
        print(f"rm combins总共有:{len(combin_ids)}")
        for combin_id in combin_ids:
            print(f"预测[{combin_id}]...")
            df_rm = pd.read_csv(os.path.join(exp_data_dir,"remove_combineword","datasets",f"{combin_id}_dataset.csv"))
            df_rm = df_rm[df_rm["Label"] != "Others"].reset_index(drop=True) # 如果不写 drop=True，旧索引会变成一列叫 index 的新列
            gt_labels, p_labels, probs = infer_trained_model(trained_model_dir,df_rm,device)
            predict_df_rm = build_prediction_df(df_origin,gt_labels,p_labels)
            predict_df_rm.to_csv(os.path.join(save_dir,f"rm_{combin_id}.csv"),index=False)
        print(f"预测结果保存在:{save_dir}")
        cost_time = int(time.monotonic() - s_time)
        hours, remainder = divmod(cost_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"总耗时: {hours:02d}小时 "
            f"{minutes:02d}分钟 {seconds:02d}秒",
            flush=True,
        )
    if stage == 2:
        '''
        阶段2：基于阶段1的预测结果获得评价指标
        '''
        res_dir = os.path.join(exp_data_dir,"remove_combineword_res")
        origin_predict_df = pd.read_csv(os.path.join(res_dir,"origin.csv"))
        y_true = origin_predict_df["True"]
        y_pred = origin_predict_df["pred"]
        origin_report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        origin_f1 = {}
        origin_f1["all"] = origin_report["macro avg"]["f1-score"]
        for label_idx in range(5):
            origin_f1[label_idx] = origin_report[str(label_idx)]["f1-score"]

        save_dir = os.path.join(res_dir,"f1")
        os.makedirs(save_dir,exist_ok=True)
        origin_f1_df = pd.DataFrame([origin_f1])
        origin_f1_df.to_csv(os.path.join(save_dir,"origin.csv"),index=False)

        rm_f1 = []
        # 抽取所有的rm_combines
        combinwords_df = pd.read_csv(os.path.join(exp_data_dir,"remove_combineword", "combinewords.csv"))
        combin_ids = combinwords_df["ID"].tolist()
        print(f"rm combins总共有:{len(combin_ids)}")

        for combin_id in combin_ids:
            row = {}
            row["ID"] = combin_id
            rm_predict_df = pd.read_csv(os.path.join(res_dir,f"rm_{combin_id}.csv"))
            y_true = rm_predict_df["True"]
            y_pred = rm_predict_df["pred"]
            rm_report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            )
            row["all"] = rm_report["macro avg"]["f1-score"]
            for label_idx in range(5):
                row[label_idx] = rm_report[str(label_idx)]["f1-score"]
            rm_f1.append(row)
        rm_f1_df = pd.DataFrame(rm_f1)
        rm_f1_df.to_csv(os.path.join(save_dir,"rm.csv"),index=False)
        print(f"F1-score结果保存在:{save_dir}")

    if stage == 3:
        '''
        阶段3：基于阶段2的指标,统计论文结果
        '''
        combinwords_df = pd.read_csv(os.path.join(exp_data_dir,"remove_combineword", "combinewords.csv"))
        combinId2words = {}
        for row_id,row in combinwords_df.iterrows():
            combinId2words[int(row["ID"])] = row["词组"]
        f1_dir = os.path.join(exp_data_dir,"remove_combineword_res","f1")
        orgin_f1_df = pd.read_csv(os.path.join(f1_dir,"origin.csv"))
        rm_f1 = pd.read_csv(os.path.join(f1_dir,"rm.csv"))
        for col in [0,1,2,3,4,"all"]:
            increase_dict = {}
            decrease_dict = {}
            min_v = float('inf')
            max_v = float('-inf')
            origin_f1 = orgin_f1_df[str(col)].tolist()[-1]
            origin_f1 = round(origin_f1,4)
            for row_id,row in rm_f1.iterrows():
                id = row["ID"]
                value = row[str(col)]
                value = round(value,4)
                min_v = min(min_v,value)
                max_v = max(max_v,value)
                if value > origin_f1:
                    increase_dict[id] = value
                elif value < origin_f1:
                    decrease_dict[id] = value
            # decrease_list 中的item按照 value从小到大排序
            sorted_ids = sorted(decrease_dict, key=lambda k: decrease_dict[k])
            sorted_words = []
            for id in sorted_ids:
                sorted_words.append(combinId2words[id])
            # assert len(sorted_words) > 10, "decrease word少于10个"
            print(f"{col}|Increase Count:{len(increase_dict)},Decrease Count:{len(decrease_dict)},Max:{max_v},Min:{min_v},Original:{origin_f1}")
            if len(sorted_words) < 10:
                print(f"Top10(<10) decrease:{sorted_words}")
            else:
                print(f"Top10 decrease:{sorted_words[:10]}")
            print("="*30)




if __name__ == "__main__":
    pid = os.getpid()
    print(f"PID:{pid}")
    model_name = "sobert" # 模型名称
    model_id = "42_1" # 一个代表模型
    device = "cuda:4" # 推理设备
    exp_data_dir = "/data/mml/DL_bug_classification" # 项目实验根目录
    # eval_single_word()
    eval_combinword()