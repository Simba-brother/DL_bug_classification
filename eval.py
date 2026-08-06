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
    # probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1) # 预测class idx
            # probs = torch.softmax(outputs.logits, dim=-1) # logits -> probs

            p_labels.extend(preds.cpu().numpy()) 
            gt_labels.extend(labels.cpu().numpy())
            # probs.extend(probs.max(dim=-1).values.cpu().numpy())
    # 统计指标
    res = classification_report(gt_labels, p_labels,output_dict=True)
    # 返回统计指标
    return res

def eval_model(model_name:str,device:str):
    '''
    model_name:sobert|codebert|robert
    device:'cuda:0'
    '''
    assert model_name in ["sobert","codebert","robert"], "model_name 传参错误"
    save_dir = os.path.join(exp_data_dir,f"{model_name}_res")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    test_df = pd.read_csv(test_csv_path)
    all_res = {}
    for rs in range(42,42+15):
        print(f"随机数种子:{rs}")
        trained_model_dir = os.path.join(exp_data_dir,"trained_models",model_name,f"ft_model_{rs}")
        res = testing(trained_model_dir,test_df,rs=42, device=device)
        all_res[rs] = res
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
        for cls_name in ["LR","DT","RF","SVM","KNN"]:
            print(f"分类器名称:{cls_name}")
            cls_df = pd.read_csv(os.path.join(predict_dir,f"{cls_name}.csv"))
            gt_labels = list(cls_df["True"])
            p_labels = list(cls_df["Pred"])
            cls_res = classification_report(gt_labels, p_labels,output_dict=True)
            all_res[rs][cls_name] = cls_res
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
        labelname = llm_labelname2labelname[llm_labelname]
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
    test_df = pd.read_csv(test_csv_path)
    for row_id,row in test_df.iterrows():
        labelname_to_labelnum[row["Label"]] = row["LabelNum"]

    for repeat in range(1,16):
        print(f"重复id:{repeat}")
        all_res[repeat] = {}
        llm_df = pd.read_csv(os.path.join(exp_data_dir,f"{llm_name}_res",f"repeat_{repeat}",f"{llm_name}.csv"))
        gt_labelnames = list(llm_df["Label"])
        llm_labelnames = list(llm_df["Answer"])
        gt_labels = convert_labelname2labelnum(gt_labelnames,labelname_to_labelnum)
        p_labels = convert_labelname2labelnum(convert_llmlabelname2labelname(llm_labelnames),labelname_to_labelnum)
        assert len(gt_labels) == len(p_labels), "label转换出错了"
        llm_res = classification_report(gt_labels,p_labels,output_dict=True)
        all_res[repeat] = llm_res
    joblib.dump(all_res,save_path)
    print(f"{llm_name}实验指标保存在:{save_path}")


 

def main():
    # device = "cuda:0"
    # eval_model("robert", device) # sobert|codebert|robert
    # eval_tfidf_and_word2vec("word2vec") # tfidf|word2vec
    eval_llm("claude") # chatgpt|claude
if __name__ == "__main__":
    exp_data_dir = "/data/mml/DL_bug_classification"
    test_csv_path = "reconstruct_dataset/test_dataset.csv"
    main()