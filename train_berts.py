from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch
from torch.optim import AdamW
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
import os
from collections import defaultdict
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

def evaluate(model, val_loader, device):
    model.eval()
    losses = []
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            loss = outputs.loss
            losses.append(loss.item())

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='macro')
    return accuracy, np.mean(losses), f1


def build_dataset(dataset_split_method:str,split_seed:int):
    if dataset_split_method == "time":
        trainval_df = pd.read_csv("reconstruct_dataset/trainval_dataset.csv")
        test_df = pd.read_csv("reconstruct_dataset/test_dataset.csv")
        num_labels = trainval_df["LabelNum"].nunique() # 应该是6(5个DL bug,1个no DL bug)
        # 从trainval中划分出，train和val
        # val_size = int(0.1 * trainval_df.shape[0])
        val_size = test_df.shape[0] # val和test数据量保持一致
        X_train, X_val, y_train, y_val = train_test_split(list(trainval_df['Text']), 
                                                        list(trainval_df['LabelNum']), 
                                                        test_size=val_size, 
                                                        stratify=trainval_df['LabelNum'], 
                                                        random_state=int(split_seed))
        # test_df中构建出X_test,y_test
        X_test, y_test = list(test_df["Text"]), list(test_df["LabelNum"])
        return X_train,y_train,X_val,y_val,X_test,y_test,num_labels
    elif dataset_split_method == "time_tvt":
        print("train|val|test严格按照时间切分")
        train_df = pd.read_csv("reconstruct_dataset/time_tvt/train_dataset.csv")
        val_df = pd.read_csv("reconstruct_dataset/time_tvt/val_dataset.csv")
        test_df = pd.read_csv("reconstruct_dataset/time_tvt/test_dataset.csv")
        num_labels = train_df["LabelNum"].nunique() # 应该是6(5个DL bug,1个no DL bug)
        print(f"训练集中分类数:{num_labels}")
        X_train, y_train = list(train_df["Text"]), list(train_df["LabelNum"])
        X_val, y_val = list(val_df["Text"]), list(val_df["LabelNum"])
        X_test, y_test = list(test_df["Text"]), list(test_df["LabelNum"])
        return X_train,y_train,X_val,y_val,X_test,y_test,num_labels

    elif dataset_split_method == 'random':
        if NOCODE is True:
            df = pd.read_csv("dataset_nocode.csv")
        else:
            df = pd.read_csv("dataset.csv")
        num_labels = df["LabelNum"].nunique() # 应该是6(5个DL bug,1个no DL bug)
        test_size = int(df.shape[0] * 0.15) # test/all = 15%
        val_size = int(df.shape[0] * 0.15) # val/all = 15%
        # 划分出75个测试集，剩下的都是训练集
        X_train, X_test, y_train, y_test = train_test_split(list(df['Text']), list(df['LabelNum']), test_size=test_size, stratify=df['LabelNum'], random_state=int(split_seed))
        # 训练集中再划分出val
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=int(split_seed))
        return X_train,y_train,X_val,y_val,X_test,y_test,num_labels
    else:
        raise Exception("dataset_split_method 参数传递错误")



def build_experiment_configs(experiment_setting:str):
    if experiment_setting == "seed_15":
        return [
            {
                "exp_id": str(split_seed),
                "split_seed": split_seed,
                "repeat_id": None,
            }
            for split_seed in range(42, 42 + 15)
        ]

    if experiment_setting == "seed_5_repeat_3":
        return [
            {
                "exp_id": f"{split_seed}_{repeat_id}",
                "split_seed": split_seed,
                "repeat_id": repeat_id,
            }
            for split_seed in [42, 43, 44, 45, 46]
            for repeat_id in [1, 2, 3]
        ]

    raise ValueError("experiment_setting 只能是 seed_15 或 seed_5_repeat_3")


def train(model_path,save_dir,exp_id,split_seed,device,dataset_split_method):
    '''
    device:"cuda:1"
    dataset_split_method:"random"|"time"
    '''
    model_save_dir = os.path.join(save_dir, f"ft_model_{exp_id}")
    # 数据集
    X_train,y_train,X_val,y_val,X_test,y_test,num_labels = build_dataset(
        dataset_split_method=dataset_split_method,
        split_seed=split_seed,
    )
    print(f"训练集大小:{len(X_train)},验证集大小:{len(X_val)},测试集大小:{len(X_test)}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # 预训练分类模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    # 训练集加载器
    train_loader = DataLoader(TextDataset(X_train, y_train, tokenizer),batch_size=32, shuffle=True)
    # 验证集加载器
    val_loader = DataLoader(TextDataset(X_val, y_val, tokenizer), batch_size=32)
    # 测试集加载器
    test_loader = DataLoader(TextDataset(X_test, y_test, tokenizer), batch_size=32)


    # 模型参数优化器
    optimizer = AdamW(model.parameters(), lr=4e-6)

    # 模型放到gpu上
    model.to(device)
    scaler = torch.amp.GradScaler("cuda") # AMP,加速训练

    num_epochs = 30 # 总共训练30轮次
    best_loss = float('inf') # 无限大
    best_info = {
            "epoch":None,
            "TrainAcc":0.0,
            "TrainLoss":0.0,
            "ValAcc":0.0,
            "ValLoss":0.0,
            "TestAcc":0.0,
            "TestLoss":0.0
    }
    s_time=time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train() # The model enters training mode.
        i=0 # batch num 统计
        losses = [] # 记录每个batch loss
        for batch in train_loader:
            # print(f"{i}/{int(len(trainset)/32)+1}")
            # sys.stdout.write(f'\r{i}/{int(len(X_train)/32)}')  # 使用 \r 回到行首
            # sys.stdout.flush()
            i+=1
            # 优化器清零
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # 前向传播并得到batch loss
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss # batch loss
            losses.append(loss.item())
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward() # loss反向传播
            scaler.step(optimizer) # 优化器做参数优化
            scaler.update() # 参数更新
        '''
          val_res[0]  # 验证集准确率 accuracy
          val_res[1]  # 验证集平均损失 loss
          val_res[2]  # 验证集宏平均 F1
        '''
        # 当前轮次epoch训练完了，在train/val/test上进行评估
        res = evaluate(model, train_loader, device) # trainset eval res
        val_res = evaluate(model, val_loader, device) # valset eval res
        test_res = evaluate(model, test_loader, device) # testset eval res

        
        # 只保存当前 split_seed + repeat_id 下 val_loss 最小的模型
        if val_res[1] < best_loss:
            best_loss = val_res[1]
            best_info["epoch"] = epoch+1
            best_info["TrainAcc"] = res[0]
            best_info["TrainLoss"] = np.mean(losses)
            best_info["ValAcc"] = val_res[0]
            best_info["ValLoss"] = val_res[1]
            best_info["TestAcc"] = test_res[0]
            best_info["TestLoss"] = test_res[1]
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir) # save tokenizer
        
        epoch_end_time = time.time()
        elapsed_time = int(epoch_end_time - epoch_start_time)
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"epoch:{epoch + 1}训练耗时：{hours:02d}小时 {minutes:02d}分钟 {seconds:02d}秒")
        print(f"Epoch {epoch + 1}/{num_epochs} - Train_Acc: {res[0]} - Train_Loss: {np.mean(losses)} - Val_acc: {val_res[0]} - Val_loss: {val_res[1]} - Test_acc: {test_res[0]} - Test_loss: {test_res[1]}")
        
    e_time=time.time()
    elapsed_time = int(e_time - s_time)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Best Info: Epoch {best_info['epoch']}/{num_epochs} - Acc: {best_info['TrainAcc']} - Loss: {best_info['TrainLoss']} - Val_acc: {best_info['ValAcc']} - Val_loss: {best_info['ValLoss']} - test_acc: {best_info['TestAcc']} - test_loss: {best_info['TestLoss']}")
    print(f"总训练耗时：{hours:02d}小时 {minutes:02d}分钟 {seconds:02d}秒")
    print(f'训练模型保存在:{model_save_dir}')
    return best_info

def main():
    device = "cuda:5"
    experiment_setting = "seed_5_repeat_3" # seed_15|seed_5_repeat_3
    experiment_configs = build_experiment_configs(experiment_setting)
    repeat_num = len(experiment_configs) # 总重复实验次数
    print(f"实验重复次数:{repeat_num}")
    dataset_split_method = "random" # random|time|time_tvt(不用)
    model_name = "codebert" # sobert|codebert|robert
    model_path = None
    if model_name == "sobert":
        model_path= "./model"
    elif model_name == "codebert":
        model_path = "./codebert-base"
    elif model_name == "robert":
        model_path = "./roberta-base"
    else:
        raise Exception("model path 参数错误")
    save_dir = os.path.join(exp_data_dir,f"trained_models",model_name)
    os.makedirs(save_dir,exist_ok=True)
    for experiment_config in experiment_configs:
        exp_id = experiment_config["exp_id"]
        split_seed = experiment_config["split_seed"]
        repeat_id = experiment_config["repeat_id"]
        print(
            f"实验设置:{experiment_setting},"
            f"数据集切分随机种子:{split_seed},"
            f"重复id:{repeat_id},实验id:{exp_id}"
        )
        train(
            model_path,
            save_dir,
            exp_id,
            split_seed,
            device,
            dataset_split_method,
        )

if __name__ == "__main__":
    exp_data_dir = "/data/mml/DL_bug_classification"
    os.makedirs(exp_data_dir,exist_ok=True)
    NOCODE = False
    if NOCODE is False:
        exp_data_dir = os.path.join(exp_data_dir,"exp")
    else:
        exp_data_dir = os.path.join(exp_data_dir,"exp_nocode")
    pid = os.getpid()
    print(f"PID:{pid}")
    main()
