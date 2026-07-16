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

def train(rs=42, device='cuda:0'):
    # 语言模型目录
    model_path= "./model"
    s_time=time.time()
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # 预训练分类模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5) # 应该是6(5个DL bug,1个no DL bug)
    # 数据集
    df = pd.read_csv("./dataset.csv")

    # 划分出75个测试集，剩下的都是训练集
    X_train, X_test, y_train, y_test = train_test_split(list(df['Text']), list(df['LabelNum']), test_size=75, stratify=df['LabelNum'], random_state=int(rs))
    # 训练集中再划分出val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=75, stratify=y_train, random_state=int(rs))

    # print(Counter(df['LabelNum']), len(df))
    # print(len(X_train), len(X_val), len(X_test))

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
    scaler = torch.cuda.amp.GradScaler() # AMP,加速训练

    num_epochs = 30 
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train() # The model enters training mode.
        i=0
        losses = []
        for batch in train_loader:
            # print(f"{i}/{int(len(trainset)/32)+1}")
            # sys.stdout.write(f'\r{i}/{int(len(X_train)/32)}')  # 使用 \r 回到行首
            # sys.stdout.flush()
            i+=1
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss # batch loss
            losses.append(loss.item())
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        '''
          val_res[0]  # 验证集准确率 accuracy
          val_res[1]  # 验证集平均损失 loss
          val_res[2]  # 验证集宏平均 F1
        '''
        res = evaluate(model, train_loader, device) # trainset eval res
        val_res = evaluate(model, val_loader, device) # valset eval res
        test_res = evaluate(model, test_loader, device) # testset eval res

        # 保存val_loss最小的模型
        if val_res[1] < best_loss:
            best_loss = val_res[1]
            model.save_pretrained(f"ft_model_{rs}") # save model
            tokenizer.save_pretrained(f"ft_model_{rs}") # save tokenizer

        print(f"Epoch {epoch + 1}/{num_epochs} - Acc: {res[0]} - Loss: {np.mean(losses)} - Val_acc: {val_res[0]} - Val_loss: {val_res[1]} - test_acc: {test_res[0]} - test_loss: {test_res[1]}")

    e_time=time.time()
    print(e_time-s_time)


def testing(rs=42, device='cuda:0'):
    '''
    测试函数
    '''
    df = pd.read_csv("./dataset.csv")
    # 加载回来tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"/home/xwj/exp33/ft_model_{rs}/", use_fast=True)
    # 加载回model
    model = AutoModelForSequenceClassification.from_pretrained(f"/home/xwj/exp33/ft_model_{rs}/") 
    model.to(device)

    X_train, X_test, y_train, y_test = train_test_split(list(df['Text']), list(df['LabelNum']), test_size=75, stratify=df['LabelNum'], random_state=int(rs))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=75, stratify=y_train, random_state=int(rs))
    
    train_loader = DataLoader(TextDataset(X_train, y_train, tokenizer),batch_size=32, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val, tokenizer), batch_size=32)
    test_loader = DataLoader(TextDataset(X_test, y_test, tokenizer), batch_size=32)

    model.eval()
    losses = []
    val_preds = [] # 预测 class idx list
    val_labels = [] # 真值class idx list
    preds_prob = [] # 预测 prob
    with torch.no_grad():
        for batch in test_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1) # 预测class idx
            probs = torch.softmax(outputs.logits, dim=-1) # logits -> probs

            loss = outputs.loss
            losses.append(loss.item())

            val_preds.extend(preds.cpu().numpy()) 
            val_labels.extend(labels.cpu().numpy())
            preds_prob.extend(probs.max(dim=-1).values.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='macro')
    print(classification_report(val_labels, val_preds))
    print(accuracy, np.mean(losses), f1)
    # print(res_dict)

    # for i in range(len(val_labels)):
    #     print(f"{df['Id'][i]}\t{val_labels[i]}\t{val_preds[i]}\t{preds_prob[i]}")

    res = classification_report(val_labels, val_preds,output_dict=True)
    print(f"{res['accuracy']}\t{res['macro avg']['f1-score']}\t \
            {res['0']['precision']}\t{res['0']['recall']}\t{res['0']['f1-score']}\t  \
            {res['1']['precision']}\t{res['1']['recall']}\t{res['1']['f1-score']}\t  \
            {res['2']['precision']}\t{res['2']['recall']}\t{res['2']['f1-score']}\t  \
            {res['3']['precision']}\t{res['3']['recall']}\t{res['3']['f1-score']}\t  \
            {res['4']['precision']}\t{res['4']['recall']}\t{res['4']['f1-score']}")
    return accuracy,f1,res['0']['f1-score'],res['1']['f1-score'],res['2']['f1-score'],res['3']['f1-score'],res['4']['f1-score']

if __name__ == "__main__":
    train()
    # testing()
