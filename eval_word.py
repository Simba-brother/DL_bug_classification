import os
import numpy as np
from collections import defaultdict
from eval import infer_trained_model
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score


def build_prediction_df(df:pd.DataFrame, gt_labels:list, p_labels:list) -> pd.DataFrame:
    predict_data = {
        "True": gt_labels,
        "pred": p_labels,
    }
    if "Id" in df.columns:
        predict_data = {"Id": list(df["Id"]), **predict_data}
    return pd.DataFrame(predict_data)

def main():
    stage = 2
    if stage == 1:
        '''
        阶段1：获得预测结果
        '''
        # load model
        trained_model_dir = os.path.join(exp_data_dir,"exp","trained_models",model_name,f"ft_model_{model_id}")
        # load dataset
        # origin dataset
        df_origin = pd.read_csv("dataset.csv")
        df_origin = df_origin[df_origin["Label"] != "Others"].reset_index(drop=True) # 如果不写 drop=True，旧索引会变成一列叫 index 的新列
        gt_labels, p_labels, probs = infer_trained_model(trained_model_dir,df_origin,device)
        predict_df_origin = build_prediction_df(df_origin,gt_labels,p_labels)

        save_dir = os.path.join(exp_data_dir,"remove_word_res")
        os.makedirs(save_dir,exist_ok=True)
        predict_df_origin.to_csv(os.path.join(save_dir,"origin.csv"),index=False)

        rm_words = ["tensorflow","model"] 
        for rm_word in rm_words:
            print(f"预测{rm_word}...")
            df_rm = pd.read_csv(os.path.join(exp_data_dir,"remove_word",f"remove_{rm_word}.csv"))
            df_rm = df_rm[df_rm["Label"] != "Others"].reset_index(drop=True) # 如果不写 drop=True，旧索引会变成一列叫 index 的新列
            gt_labels, p_labels, probs = infer_trained_model(trained_model_dir,df_rm,device)
            predict_df_rm = build_prediction_df(df_origin,gt_labels,p_labels)
            predict_df_rm.to_csv(os.path.join(save_dir,f"rm_{rm_word}.csv"),index=False)

        print(f"预测结果保存在:{save_dir}")


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
        rm_words = ["tensorflow","model"]
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
            origin_f1 = orgin_f1_df[col][-1]
            for row in rm_f1.iterrows():
                word = row["word"]
                value = row[col]
                min_v = min(min_v,value)
                max_v = max(max_v,value)
                if value > origin_f1:
                    increase_dict[word] = value
                elif value < origin_f1:
                    decrease_dict[word] = value
            # decrease_list 中的item按照 value从小到大排序
            sorted_words = sorted(decrease_dict, key=lambda k: decrease_dict[k])
            assert len(sorted_words) > 10, "decrease word少于10个"
            print(f"{col}|Increase Count:{len(increase_dict)},Decrease Count:{len(decrease_dict)},Max:{max_v},Min:{min_v},Original:{origin_f1}")
            print(f"Top10 decrease:{sorted_words[:10]}")

if __name__ == "__main__":
    model_name = "sobert"
    model_id = "42_1"
    device = "cuda:0"
    exp_data_dir = "/data/mml/DL_bug_classification"
    main()