'''
基于trained model/测试集上的预测，对性能指标进行评估
'''
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch
from torch.optim import AdamW
import pandas as pd
import time
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import shutil
import joblib

LABEL_NUMS = list(range(6))

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length # Maximum length of each sentence(text)

    def __len__(self):
        return len(self.texts) # Number of sentences(text) in the dataset.

    def __getitem__(self, idx):
        text = self.texts[idx] # Obtain the text based on the idx.
        label = self.labels[idx] # Obtain the label based on the idx.
        # Tokenize the text.
        inputs = self.tokenizer(
            str(text),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        '''
        随后，DataLoader 会把多条样本组合成一个批次。假设 batch_size=32，结果大致为：
        batch['input_ids'].shape       # [32, 512]
        batch['attention_mask'].shape  # [32, 512]
        batch['labels'].shape          # [32]
        '''
        return {
            'input_ids': inputs['input_ids'].squeeze(), # tensor([[101, 2769, 4638, 102, 0]])，每个token在词表中的编号.这里的 .squeeze() 用来去掉 tokenizer 添加的大小为1的批次维度：
            'attention_mask': inputs['attention_mask'].squeeze(), # 1 表示真实 token，0 表示补齐的 padding。
            'labels': torch.tensor(label) # 类别标签
        }

def build_test_df(dataset_split_method:str, rs:int) -> pd.DataFrame:
    """
    按照 mmltrain.py 的 dataset_split_method 构建 test set。
    eval_model 只在这里返回的 X_test/y_test 上生成预测 CSV。
    """
    if dataset_split_method == "time":
        return pd.read_csv("reconstruct_dataset/test_dataset.csv").reset_index(drop=True)

    if dataset_split_method == "random":
        df = pd.read_csv("dataset.csv")
        test_size = int(df.shape[0] * 0.15)
        _, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["LabelNum"],
            random_state=int(rs),
        )
        return test_df.reset_index(drop=True)

    raise ValueError("dataset_split_method 只能是 random 或 time")


def add_one_vs_rest_accuracy(report:dict, gt_labels:list, p_labels:list, label_nums:list) -> dict:
    for label_num in label_nums:
        label_key = str(label_num)
        gt_binary = [label == label_num for label in gt_labels]
        pred_binary = [label == label_num for label in p_labels]
        if label_key in report:
            report[label_key]["accuracy"] = accuracy_score(gt_binary, pred_binary)
    return report


def testing(trained_model_dir:str,df:pd.DataFrame,rs=42, device='cuda:0'):
    '''
    测试函数
    '''
    # 加载回来tokenizer
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir, use_fast=True)
    # 加载回model
    model = AutoModelForSequenceClassification.from_pretrained(trained_model_dir) 
    model.to(device)

    X_test, y_test = list(df["Text"]), list(df["LabelNum"])
    print(f"测试集大小:{len(X_test)}")
    # 测试集加载器
    test_loader = DataLoader(TextDataset(X_test, y_test, tokenizer), batch_size=32, shuffle=False)

    # 模型进入评估模式
    model.eval()
    
    p_labels = [] # 预测 class idx list
    gt_labels = [] # 真值class idx list
    probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1) # 预测class idx
            batch_probs = torch.softmax(logits, dim=-1) # logits -> probs

            p_labels.extend(preds.cpu().numpy()) 
            gt_labels.extend(labels.cpu().numpy())
            probs.extend(batch_probs.cpu().numpy().tolist())
    # 统计指标
    label_nums = LABEL_NUMS
    res = classification_report(
        gt_labels,
        p_labels,
        labels=label_nums,
        output_dict=True,
        zero_division=0,
    )
    res = add_one_vs_rest_accuracy(res, gt_labels, p_labels, label_nums)
    predict_df = pd.DataFrame({
        "True": gt_labels,
        "Pred": p_labels,
        "Probs": probs,
    })
    if "Id" in df.columns:
        predict_df.insert(0, "Id", list(df["Id"]))
    # 返回统计指标
    return res, predict_df

def eval_bert(model_name:str,device:str,dataset_split_method:str):
    '''
    model_name:sobert|codebert|robert
    device:'cuda:0'
    dataset_split_method:random|time
    '''
    assert model_name in ["sobert","codebert","robert"], "model_name 传参错误"
    save_dir = os.path.join(exp_data_dir,f"{model_name}_res")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    all_res = {}
    for rs in range(42,42+15):
        print(f"随机数种子:{rs}")
        test_df = build_test_df(dataset_split_method, rs)
        trained_model_dir = os.path.join(exp_data_dir,"trained_models",model_name,f"ft_model_{rs}")
        res, predict_df = testing(trained_model_dir,test_df,rs=rs, device=device)
        all_res[rs] = res

        predict_save_dir = os.path.join(save_dir, f"seed_{rs}")
        os.makedirs(predict_save_dir, exist_ok=True)
        predict_save_path = os.path.join(predict_save_dir, f"{model_name}.csv")
        predict_df.to_csv(predict_save_path, index=False)
        print(f"{model_name} seed {rs} 测试集预测结果保存在:{predict_save_path}")

    joblib.dump(all_res,save_path)
    print(f"{model_name}实验指标保存在:{save_path}")

def eval_tfidf_and_word2vec(method_name):
    assert method_name in ["tfidf","word2vec"], "method_name 传参错误"
    save_dir = os.path.join(exp_data_dir,f"{method_name}_res")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    all_res = {}
    for rs in range(42,42+15):
        print(f"随机数种子:{rs}")
        all_res[rs] = {}
        predict_dir = os.path.join(exp_data_dir,f"trained_{method_name}",f"seed_{rs}")
        for clf_name in ["LR","DT","RF","SVM","KNN"]:
            print(f"分类器名称:{clf_name}")
            cls_df = pd.read_csv(os.path.join(predict_dir,f"{clf_name}.csv"))
            gt_labels = list(cls_df["True"])
            p_labels = list(cls_df["Pred"])
            label_nums = LABEL_NUMS
            cls_res = classification_report(
                gt_labels,
                p_labels,
                labels=label_nums,
                output_dict=True,
                zero_division=0,
            )
            cls_res = add_one_vs_rest_accuracy(cls_res, gt_labels, p_labels, label_nums)
            all_res[rs][clf_name] = cls_res
    joblib.dump(all_res,save_path)
    print(f"{method_name}实验指标保存在:{save_path}")

def convert_llmlabelname2labelname(llm_labelname_list):
    res = []
    llm_labelname2labelname = {
        "Model":"model",
        "Tensors&Inputs":"tensor",
        "Training":"training",
        "GPU Usage":"gpu",
        "API":"api",
        "Others":"Others"
    }
    for llm_labelname in llm_labelname_list:
        llm_labelname = llm_labelname.replace("\n","")
        # labelname = llm_labelname2labelname[llm_labelname]
        labelname = llm_labelname2labelname.get(llm_labelname,"Others")
        res.append(labelname)
    assert len(res) == len(llm_labelname_list), "llmName转换出错"
    return res

def convert_labelname2labelnum(labelname_list, labelname_to_labelnum:dict):
    res = []
    for labelname in labelname_list:
        labelnum = labelname_to_labelnum[labelname]
        res.append(labelnum)
    assert len(res) == len(labelname_list), "nametonum转换出错"
    return res

def eval_llm(llm_name:str):
    '''
    llm_name:claude|chatgpt
    '''
    assert llm_name in ["claude","chatgpt"], "llm_name 传参错误"
    save_dir = os.path.join(exp_data_dir,f"{llm_name}_res")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    all_res = {}

    labelname_to_labelnum = {}
    test_df = pd.read_csv("reconstruct_dataset/test_dataset.csv")
    for row_id,row in test_df.iterrows():
        labelname_to_labelnum[row["Label"]] = row["LabelNum"]

    for repeat in range(1,16):
        print(f"重复id:{repeat}")
        all_res[repeat] = {}
        llm_df = pd.read_csv(os.path.join(exp_data_dir,f"{llm_name}_res",f"repeat_{repeat}",f"{llm_name}.csv"))
        gt_labelnames = list(llm_df["Label"])
        llm_labelnames = list(llm_df["Answer"])
        # print(set(llm_labelnames))
        gt_labels = convert_labelname2labelnum(gt_labelnames,labelname_to_labelnum)
        p_labels = convert_labelname2labelnum(convert_llmlabelname2labelname(llm_labelnames),labelname_to_labelnum)
        assert len(gt_labels) == len(p_labels), "label转换出错了"
        label_nums = LABEL_NUMS
        llm_res = classification_report(
            gt_labels,
            p_labels,
            labels=label_nums,
            output_dict=True,
            zero_division=0,
        )
        llm_res = add_one_vs_rest_accuracy(llm_res, gt_labels, p_labels, label_nums)
        all_res[repeat] = llm_res
    joblib.dump(all_res,save_path)
    print(f"{llm_name}实验指标保存在:{save_path}")


def eval_xwj():
    label_names = ["model", "tensor", "training", "gpu", "api", "Others"]
    label_nums = list(range(len(label_names)))
    reports = []
    class_accuracy_rows = []

    for seed in [42,43,44,45,46]:
        for repeat in [1,2,3]:
            df = pd.read_csv(f"results/sobert/{seed}_{repeat}.csv")
            gt_labels = list(df["True"])
            p_labels = list(df["pred"])
            report = classification_report(
                gt_labels,
                p_labels,
                labels=label_nums,
                target_names=label_names,
                output_dict=True,
                zero_division=0,
            )
            reports.append(report)

            class_accuracy = {}
            for label_num, label_name in zip(label_nums, label_names):
                gt_binary = [label == label_num for label in gt_labels]
                pred_binary = [label == label_num for label in p_labels]
                class_accuracy[label_name] = accuracy_score(gt_binary, pred_binary)
            class_accuracy_rows.append(class_accuracy)

    print("xwj 5*3共15次实验指标均值:")
    print(f"all accuracy mean: {np.mean([report['accuracy'] for report in reports]):.4f}")
    print(f"all f1 mean: {np.mean([report['macro avg']['f1-score'] for report in reports]):.4f}")
    print("各类别 accuracy/precision/recall/f1 mean:")
    for label_name in label_names:
        accuracy_mean = np.mean([row[label_name] for row in class_accuracy_rows])
        precision_mean = np.mean([report[label_name]["precision"] for report in reports])
        recall_mean = np.mean([report[label_name]["recall"] for report in reports])
        f1_mean = np.mean([report[label_name]["f1-score"] for report in reports])
        print(
            f"{label_name}: "
            f"accuracy={accuracy_mean:.4f}, "
            f"precision={precision_mean:.4f}, "
            f"recall={recall_mean:.4f}, "
            f"f1={f1_mean:.4f}"
        )

def eval_xwj_from_all_res():
    label_names = ["model", "tensor", "training", "gpu", "api", "Others"]
    all_res_path = "results/sobert/all_res.csv"
    all_res_df = pd.read_csv(all_res_path)

    required_columns = ["acc_all", "f1_all"]
    for label_num in range(len(label_names)):
        required_columns.extend([f"acc_{label_num}", f"f1_{label_num}", f"auc_{label_num}"])
    missing_columns = [col for col in required_columns if col not in all_res_df.columns]
    if missing_columns:
        raise ValueError(f"{all_res_path} 缺少必要列: {missing_columns}")

    print(f"xwj 基于 {all_res_path} 的指标均值:")
    print(f"实验次数: {len(all_res_df)}")
    print(f"all accuracy mean: {all_res_df['acc_all'].mean():.4f}")
    print(f"all f1 mean: {all_res_df['f1_all'].mean():.4f}")
    print("注意: all_res.csv 中单类别 acc_* 实际是基于概率的 AUC, auc_* 实际是 one-vs-rest accuracy")
    print("各类别 accuracy/f1 mean:")
    for label_num, label_name in enumerate(label_names):
        accuracy_mean = all_res_df[f"auc_{label_num}"].mean()
        f1_mean = all_res_df[f"f1_{label_num}"].mean()
        print(
            f"{label_name}: "
            f"accuracy={accuracy_mean:.4f}, "
            f"f1={f1_mean:.4f}"
        )

def main():
    # device = "cuda:4"
    # bertname = "sobert" # sobert|codebert|robert
    # dataset_split_method = "random" # random|time
    # eval_bert(bertname, device, dataset_split_method)

    # eval_tfidf_and_word2vec("word2vec") # tfidf|word2vec

    eval_llm("claude") # chatgpt|claude

    # eval_xwj()
    # eval_xwj_from_all_res()
if __name__ == "__main__":
    exp_data_dir = "/data/mml/DL_bug_classification/xwj_reproduction"
    main()
