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
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            str(text),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
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
    model_name = "/home/xwj/exp33/model/"
    s_time=time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    df = pd.read_csv("/data2/xwj/dataset.csv")

    X_train, X_test, y_train, y_test = train_test_split(list(df['Text']), list(df['LabelNum']), test_size=75, stratify=df['LabelNum'], random_state=int(rs))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=75, stratify=y_train, random_state=int(rs))

    # print(Counter(df['LabelNum']), len(df))
    # print(len(X_train), len(X_val), len(X_test))

    train_loader = DataLoader(TextDataset(X_train, y_train, tokenizer),batch_size=32, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val, tokenizer), batch_size=32)
    test_loader = DataLoader(TextDataset(X_test, y_test, tokenizer), batch_size=32)

    optimizer = AdamW(model.parameters(), lr=4e-6)

    model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 30
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
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
                loss = outputs.loss
            losses.append(loss.item())
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        res = evaluate(model, train_loader, device)
        val_res = evaluate(model, val_loader, device)
        test_res = evaluate(model, test_loader, device)

        # 保存val_loss最小的模型
        if val_res[1] < best_loss:
            best_loss = val_res[1]
            model.save_pretrained(f"ft_model_{rs}")
            tokenizer.save_pretrained(f"ft_model_{rs}")

        print(f"Epoch {epoch + 1}/{num_epochs} - Acc: {res[0]} - Loss: {np.mean(losses)} - Val_acc: {val_res[0]} - Val_loss: {val_res[1]} - test_acc: {test_res[0]} - test_loss: {test_res[1]}")

    e_time=time.time()
    print(e_time-s_time)


def testing(rs=42, device='cuda:0'):
    df = pd.read_csv("/data2/xwj/dataset.csv")
    
    tokenizer = AutoTokenizer.from_pretrained(f"/home/xwj/exp33/ft_model_{rs}/", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(f"/home/xwj/exp33/ft_model_{rs}/") 
    model.to(device)

    X_train, X_test, y_train, y_test = train_test_split(list(df['Text']), list(df['LabelNum']), test_size=75, stratify=df['LabelNum'], random_state=int(rs))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=75, stratify=y_train, random_state=int(rs))
    
    train_loader = DataLoader(TextDataset(X_train, y_train, tokenizer),batch_size=32, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val, tokenizer), batch_size=32)
    test_loader = DataLoader(TextDataset(X_test, y_test, tokenizer), batch_size=32)

    model.eval()
    losses = []
    val_preds = []
    val_labels = []
    preds_prob = []
    with torch.no_grad():
        for batch in test_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(outputs.logits, dim=-1)

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
    print(f"{res['accuracy']}\t{res['macro avg']['f1-score']}")
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
