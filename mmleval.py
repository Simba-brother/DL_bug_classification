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
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
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


def build_report(gt_labels:list, p_labels:list) -> dict:
    label_nums = LABEL_NUMS
    res = classification_report(
        gt_labels,
        p_labels,
        labels=label_nums,
        output_dict=True,
        zero_division=0,
    )
    return add_one_vs_rest_accuracy(res, gt_labels, p_labels, label_nums)


def build_prediction_df(df:pd.DataFrame, gt_labels:list, p_labels:list, probs:list) -> pd.DataFrame:
    probs_array = np.asarray(probs)
    predict_data = {
        "True": gt_labels,
        "pred": p_labels,
    }
    if "Id" in df.columns:
        predict_data = {"Id": list(df["Id"]), **predict_data}

    for label_num in LABEL_NUMS:
        predict_data[f"prob_{label_num}"] = probs_array[:, label_num]

    return pd.DataFrame(predict_data)


def infer_trained_model(trained_model_dir:str, df:pd.DataFrame, device='cuda:0'):
    '''
    使用 trained model 对 df 推理，返回真值、预测类别和每类概率。
    '''
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(trained_model_dir)
    model.to(device)

    X_test, y_test = list(df["Text"]), list(df["LabelNum"])
    print(f"测试集大小:{len(X_test)}")
    test_loader = DataLoader(TextDataset(X_test, y_test, tokenizer), batch_size=32, shuffle=False)

    model.eval()

    p_labels = []
    gt_labels = []
    probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            batch_probs = torch.softmax(logits, dim=-1)

            p_labels.extend(preds.cpu().numpy().tolist())
            gt_labels.extend(labels.cpu().numpy().tolist())
            probs.extend(batch_probs.cpu().numpy().tolist())

    return gt_labels, p_labels, probs


def testing(trained_model_dir:str,df:pd.DataFrame,rs=42, device='cuda:0'):
    '''
    测试函数
    '''
    gt_labels, p_labels, probs = infer_trained_model(trained_model_dir, df, device=device)
    res = build_report(gt_labels, p_labels)
    predict_df = build_prediction_df(df, gt_labels, p_labels, probs)
    # 返回统计指标
    return res, predict_df

def build_all_res_row_from_infer_df(infer_df:pd.DataFrame) -> dict:
    gt_labels = list(infer_df["True"])
    pred_col = "pred" if "pred" in infer_df.columns else "Pred"
    p_labels = list(infer_df[pred_col])
    prob_cols = [f"prob_{label_num}" for label_num in LABEL_NUMS]
    missing_prob_cols = [col for col in prob_cols if col not in infer_df.columns]
    if missing_prob_cols:
        raise ValueError(f"推理CSV缺少概率列:{missing_prob_cols}")

    probs = infer_df[prob_cols].to_numpy()
    gt_binary = label_binarize(gt_labels, classes=LABEL_NUMS)
    pred_binary = label_binarize(p_labels, classes=LABEL_NUMS)
    report = classification_report(
        gt_labels,
        p_labels,
        labels=LABEL_NUMS,
        output_dict=True,
        zero_division=0,
    )

    row = {}
    for label_num in LABEL_NUMS:
        row[f"acc_{label_num}"] = accuracy_score(
            gt_binary[:, label_num],
            pred_binary[:, label_num],
        )
        row[f"f1_{label_num}"] = f1_score(
            gt_binary[:, label_num],
            pred_binary[:, label_num],
            zero_division=0,
        )
        try:
            row[f"auc_{label_num}"] = roc_auc_score(
                gt_binary[:, label_num],
                probs[:, label_num],
            )
        except ValueError:
            row[f"auc_{label_num}"] = np.nan

    row["acc_all"] = report["accuracy"]
    row["f1_all"] = report["macro avg"]["f1-score"]
    try:
        row["auc_all"] = roc_auc_score(
            gt_binary,
            probs,
            average="macro",
        )
    except ValueError:
        row["auc_all"] = np.nan
    return row


def build_llm_pseudo_probs(p_labels:list, confidences:np.ndarray) -> np.ndarray:
    '''
    方案B: LLM 只有 pred + Confidence 时，构造近似概率分布。
    P(pred)=Confidence，其他类别均分 1-Confidence。
    '''
    class_nums = len(LABEL_NUMS)
    probs = np.zeros((len(p_labels), class_nums))
    for row_idx, pred_label in enumerate(p_labels):
        other_prob = (1 - confidences[row_idx]) / (class_nums - 1)
        probs[row_idx, :] = other_prob
        probs[row_idx, int(pred_label)] = confidences[row_idx]
    return probs


def build_all_res_row_from_llm_df(llm_df:pd.DataFrame) -> dict:
    '''
    基于 LLM 自评 Confidence 结果生成 all_res.csv 的一行指标。
    Confidence 表示模型认为 pred 正确的概率；非预测类的分数用剩余概率均分近似。
    '''
    required_columns = ["True", "pred", "Confidence"]
    missing_columns = [col for col in required_columns if col not in llm_df.columns]
    if missing_columns:
        raise ValueError(f"LLM CSV缺少必要列:{missing_columns}")

    if "Status" in llm_df.columns:
        failed_df = llm_df[llm_df["Status"].astype(str).str.lower() != "success"]
        if not failed_df.empty:
            failed_ids = list(failed_df["Id"]) if "Id" in failed_df.columns else list(failed_df.index)
            raise ValueError(f"LLM CSV仍有失败样本，无法生成完整指标，失败ID:{failed_ids[:20]}")

    gt_labels = pd.to_numeric(llm_df["True"], errors="coerce")
    p_labels = pd.to_numeric(llm_df["pred"], errors="coerce")
    confidences = pd.to_numeric(llm_df["Confidence"], errors="coerce")

    invalid_gt_mask = gt_labels.isna() | ~gt_labels.isin(LABEL_NUMS)
    if invalid_gt_mask.any():
        raise ValueError(f"True列存在非0-5标签，行号:{list(llm_df.index[invalid_gt_mask])[:20]}")

    invalid_pred_mask = p_labels.isna() | ~p_labels.isin(LABEL_NUMS)
    if invalid_pred_mask.any():
        raise ValueError(f"pred列存在非0-5标签，行号:{list(llm_df.index[invalid_pred_mask])[:20]}")

    invalid_confidence_mask = confidences.isna() | (confidences < 0) | (confidences > 1)
    if invalid_confidence_mask.any():
        raise ValueError(
            f"Confidence列存在非0-1数值，行号:{list(llm_df.index[invalid_confidence_mask])[:20]}"
        )

    gt_labels = gt_labels.astype(int).tolist()
    p_labels = p_labels.astype(int).tolist()
    confidences = confidences.astype(float).to_numpy()

    probs = build_llm_pseudo_probs(p_labels, confidences)

    gt_binary = label_binarize(gt_labels, classes=LABEL_NUMS)
    pred_binary = label_binarize(p_labels, classes=LABEL_NUMS)
    report = classification_report(
        gt_labels,
        p_labels,
        labels=LABEL_NUMS,
        output_dict=True,
        zero_division=0,
    )

    row = {}
    for label_num in LABEL_NUMS:
        row[f"acc_{label_num}"] = accuracy_score(
            gt_binary[:, label_num],
            pred_binary[:, label_num],
        )
        row[f"f1_{label_num}"] = f1_score(
            gt_binary[:, label_num],
            pred_binary[:, label_num],
            zero_division=0,
        )
        try:
            row[f"auc_{label_num}"] = roc_auc_score(
                gt_binary[:, label_num],
                probs[:, label_num],
            )
        except ValueError:
            row[f"auc_{label_num}"] = np.nan

    row["acc_all"] = report["accuracy"]
    row["f1_all"] = report["macro avg"]["f1-score"]
    try:
        row["auc_all"] = roc_auc_score(
            gt_binary,
            probs,
            average="macro",
        )
    except ValueError:
        row["auc_all"] = np.nan
    return row


def save_all_res_from_infer_csvs(save_dir:str, model_name:str, seeds:list) -> str:
    all_rows = []
    for rs in seeds:
        infer_csv_path = os.path.join(save_dir, f"seed_{rs}", f"{model_name}.csv")
        infer_df = pd.read_csv(infer_csv_path)
        all_rows.append(build_all_res_row_from_infer_df(infer_df))

    all_res_df = pd.DataFrame(all_rows)
    ordered_columns = []
    for label_num in LABEL_NUMS:
        ordered_columns.extend([f"acc_{label_num}", f"f1_{label_num}", f"auc_{label_num}"])
    ordered_columns.extend(["acc_all", "f1_all", "auc_all"])
    all_res_df = all_res_df[ordered_columns]

    all_res_path = os.path.join(save_dir, "all_res.csv")
    all_res_df.to_csv(all_res_path, index=False)
    return all_res_path


def save_all_res_from_llm_csvs(save_dir:str, result_name:str, repeat_nums:list) -> str:
    all_rows = []
    for repeat in repeat_nums:
        llm_csv_path = os.path.join(save_dir, f"repeat_{repeat}", f"{result_name}.csv")
        llm_df = pd.read_csv(llm_csv_path)
        all_rows.append(build_all_res_row_from_llm_df(llm_df))

    all_res_df = pd.DataFrame(all_rows)
    ordered_columns = []
    for label_num in LABEL_NUMS:
        ordered_columns.extend([f"acc_{label_num}", f"f1_{label_num}", f"auc_{label_num}"])
    ordered_columns.extend(["acc_all", "f1_all", "auc_all"])
    all_res_df = all_res_df[ordered_columns]

    all_res_path = os.path.join(save_dir, "all_res.csv")
    all_res_df.to_csv(all_res_path, index=False)
    return all_res_path


def eval_bert(model_name:str,device:str,dataset_split_method:str):
    '''
    model_name:sobert|codebert|robert
    device:'cuda:0'
    dataset_split_method:random|time
    '''
    assert model_name in ["sobert","codebert","robert"], "model_name 传参错误"
    save_dir = os.path.join(exp_data_dir,f"{model_name}_res")
    os.makedirs(save_dir,exist_ok=True)
    # save_file_name = "res.joblib"
    # save_path = os.path.join(save_dir,save_file_name)
    # all_res = {}
    seeds = list(range(42,42+15))
    for rs in seeds:
        print(f"随机数种子:{rs}")
        test_df = build_test_df(dataset_split_method, rs)
        trained_model_dir = os.path.join(exp_data_dir,"trained_models",model_name,f"ft_model_{rs}")
        gt_labels, p_labels, probs = infer_trained_model(trained_model_dir, test_df, device=device)
        predict_df = build_prediction_df(test_df, gt_labels, p_labels, probs)
        # res = build_report(gt_labels, p_labels)
        # all_res[rs] = res

        predict_save_dir = os.path.join(save_dir, f"seed_{rs}")
        os.makedirs(predict_save_dir, exist_ok=True)
        predict_save_path = os.path.join(predict_save_dir, f"{model_name}.csv")
        predict_df.to_csv(predict_save_path, index=False)
        print(f"{model_name} seed {rs} 测试集推理结果保存在:{predict_save_path}")

    all_res_path = save_all_res_from_infer_csvs(save_dir, model_name, seeds)
    print(f"{model_name} 15次推理指标CSV保存在:{all_res_path}")

    # joblib.dump(all_res,save_path)
    # print(f"{model_name}实验指标保存在:{save_path}")



def eval_tfidf_and_word2vec(method_name):
    assert method_name in ["tfidf","word2vec"], "method_name 传参错误"
    save_dir = os.path.join(exp_data_dir,f"{method_name}_res")
    os.makedirs(save_dir,exist_ok=True)
    # save_file_name = "res.joblib"
    # save_path = os.path.join(save_dir,save_file_name)
    all_res_csv_path = os.path.join(save_dir, "all_res.csv")
    clf_names = ["LR","DT","RF","SVM","KNN"]
    metric_columns = []
    for label_num in LABEL_NUMS:
        metric_columns.extend([f"acc_{label_num}", f"f1_{label_num}", f"auc_{label_num}"])
    metric_columns.extend(["acc_all", "f1_all", "auc_all"])

    # all_res = {}
    all_res_rows = []
    for rs in range(42,42+15):
        print(f"随机数种子:{rs}")
        # all_res[rs] = {}
        all_res_row = {}
        predict_dir = os.path.join(exp_data_dir,f"trained_{method_name}",f"seed_{rs}")
        for clf_name in clf_names:
            print(f"分类器名称:{clf_name}")
            clf_df = pd.read_csv(os.path.join(predict_dir,f"{clf_name}.csv"))
            # gt_labels = list(clf_df["True"])
            # pred_col = "pred" if "pred" in clf_df.columns else "Pred"
            # p_labels = list(clf_df[pred_col])
            # cls_res = build_report(gt_labels, p_labels)
            # all_res[rs][clf_name] = cls_res
            cls_all_res_row = build_all_res_row_from_infer_df(clf_df)
            for metric_col in metric_columns:
                all_res_row[f"{clf_name}_{metric_col}"] = cls_all_res_row[metric_col]
        all_res_rows.append(all_res_row)

    # joblib.dump(all_res,save_path)
    # print(f"{method_name}实验指标保存在:{save_path}")

    all_res_df = pd.DataFrame(all_res_rows)
    ordered_columns = [
        f"{clf_name}_{metric_col}"
        for clf_name in clf_names
        for metric_col in metric_columns
    ]
    all_res_df = all_res_df[ordered_columns]
    all_res_df.to_csv(all_res_csv_path, index=False)
    print(f"{method_name}实验指标CSV保存在:{all_res_csv_path}")

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
    result_name = f"{llm_name}_prob"
    save_dir = os.path.join(exp_data_dir,f"{result_name}_res")
    os.makedirs(save_dir,exist_ok=True)
    repeat_nums = list(range(1,16))
    all_res_path = save_all_res_from_llm_csvs(save_dir, result_name, repeat_nums)
    print(f"{llm_name} 15次LLM指标CSV保存在:{all_res_path}")


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
    # bert系列
    # device = "cuda:0"
    # bertname = "robert" # sobert|codebert|robert
    # dataset_split_method = "time" # random|time
    # eval_bert(bertname, device, dataset_split_method)

    # 传统系列
    # eval_tfidf_and_word2vec("word2vec") # tfidf|word2vec

    # 大模型系列
    # eval_llm("claude") # chatgpt|claude

    # eval_xwj()
    # eval_xwj_from_all_res()
    pass
if __name__ == "__main__":
    exp_data_dir = "/data/mml/DL_bug_classification/"
    main()
