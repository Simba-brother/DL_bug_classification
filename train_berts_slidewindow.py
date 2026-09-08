from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os


class TextDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer: AutoTokenizer,
        max_length=512,
        window_stride=256,
        max_windows_per_sample=4,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_stride = window_stride # 窗口的跨步
        self.max_windows_per_sample = max_windows_per_sample # 每条样本最多保留的窗口数量
        special_tokens_count = self.tokenizer.num_special_tokens_to_add(pair=False) # [CLS]|[SEP]2个special token
        self.content_max_len = self.max_length - special_tokens_count # 文本最大长度
        if self.content_max_len <= 0:
            raise ValueError("max_length is too small for tokenizer special tokens")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if self.max_windows_per_sample <= 0:
            raise ValueError("max_windows_per_sample must be positive")

    def __len__(self):
        return len(self.texts) # 多少条文本数据集

    def _build_token_windows(self, token_ids):
        # 窗格其实是一个二维数组
        if len(token_ids) <= self.content_max_len:
            # 就一个窗口就可以了
            return [token_ids]
        # 准备放置窗口们
        windows = []
        start = 0
        while start < len(token_ids):
            end = start + self.content_max_len
            windows.append(token_ids[start:end]) # 划出一个窗口
            if end >= len(token_ids): # 最后一个窗口了跳出
                break
            start += self.window_stride # 窗口步长
        # 从收集的窗口中随机选择等距离窗口
        if len(windows) > self.max_windows_per_sample:
            selected_window_indices = np.linspace(
                0,
                len(windows) - 1,
                num=self.max_windows_per_sample,
                dtype=int,
            )
            windows = [windows[window_idx] for window_idx in selected_window_indices]
        return windows

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 使用tokenizer对文本进行划分token（不加special token）,返回token_ids
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        # 获得滑动窗口划分结果
        token_windows = self._build_token_windows(token_ids)

        input_ids_list = []
        attention_mask_list = []
        # 遍历各个窗口，其实每个窗口就相当于一个text
        for window_token_ids in token_windows:
            inputs = self.tokenizer.prepare_for_model(
                window_token_ids, # windowed token ids
                add_special_tokens=True, # 前后加上 special token
                max_length=self.max_length, # 512
                padding="max_length", # 不够，则填充到512
                truncation=True, # 大于512，则直接截断到512
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids_list.append(inputs["input_ids"].squeeze(0))
            attention_mask_list.append(inputs["attention_mask"].squeeze(0))

        return {
            "input_ids": torch.stack(input_ids_list, dim=0), # 新增最外侧一个dim堆叠起来
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def slide_window_collate_fn(batch):
    batch_size = len(batch)
    max_windows = max(item["input_ids"].shape[0] for item in batch) # 本批次中最大窗口数量
    seq_len = batch[0]["input_ids"].shape[1] # seq len

    '''
    它等价于：
    torch.zeros(
        (batch_size, max_windows, seq_len),
        dtype=batch[0]["input_ids"].dtype,
        device=batch[0]["input_ids"].device
    )
    '''
    input_ids = batch[0]["input_ids"].new_zeros((batch_size, max_windows, seq_len))
    attention_mask = batch[0]["attention_mask"].new_zeros((batch_size, max_windows, seq_len))
    window_mask = torch.zeros((batch_size, max_windows), dtype=torch.bool)
    labels = torch.stack([item["labels"] for item in batch], dim=0)

    for row_idx, item in enumerate(batch):
        num_windows = item["input_ids"].shape[0] # 本条text的滑动窗口的数量
        input_ids[row_idx, :num_windows] = item["input_ids"]
        attention_mask[row_idx, :num_windows] = item["attention_mask"]
        window_mask[row_idx, :num_windows] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "window_mask": window_mask,
        "labels": labels,
    }


def mean_window_logits(model, input_ids, attention_mask, window_mask):
    batch_size, max_windows, seq_len = input_ids.shape

    flat_window_mask = window_mask.reshape(-1)
    # 该批次数据中真窗口
    flat_input_ids = input_ids.reshape(batch_size * max_windows, seq_len)[flat_window_mask]
    flat_attention_mask = attention_mask.reshape(batch_size * max_windows, seq_len)[flat_window_mask]

    outputs = model(flat_input_ids, attention_mask=flat_attention_mask)
    valid_logits = outputs.logits

    flat_batch_idx = (
        torch.arange(batch_size, device=input_ids.device)
        .unsqueeze(1)
        .expand(batch_size, max_windows)
        .reshape(-1)
    )
    valid_batch_idx = flat_batch_idx[flat_window_mask]

    summed_logits = valid_logits.new_zeros((batch_size, valid_logits.shape[-1]))
    summed_logits.index_add_(0, valid_batch_idx, valid_logits)

    window_counts = window_mask.sum(dim=1).clamp(min=1).to(valid_logits.dtype).unsqueeze(-1)
    return summed_logits / window_counts


def evaluate(model, val_loader, device):
    model.eval()
    losses = []
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            window_mask = batch["window_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = mean_window_logits(model, input_ids, attention_mask, window_mask)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=-1)

            losses.append(loss.item())
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average="macro")
    return accuracy, np.mean(losses), f1


def build_dataset(dataset_split_method: str, split_seed: int):
    if dataset_split_method == "time":
        trainval_df = pd.read_csv("reconstruct_dataset/trainval_dataset.csv")
        test_df = pd.read_csv("reconstruct_dataset/test_dataset.csv")
        num_labels = trainval_df["LabelNum"].nunique()
        val_size = test_df.shape[0]
        X_train, X_val, y_train, y_val = train_test_split(
            list(trainval_df["Text"]),
            list(trainval_df["LabelNum"]),
            test_size=val_size,
            stratify=trainval_df["LabelNum"],
            random_state=int(split_seed),
        )
        X_test, y_test = list(test_df["Text"]), list(test_df["LabelNum"])
        return X_train, y_train, X_val, y_val, X_test, y_test, num_labels

    elif dataset_split_method == "time_tvt":
        print("train|val|test split strictly by time")
        train_df = pd.read_csv("reconstruct_dataset/time_tvt/train_dataset.csv")
        val_df = pd.read_csv("reconstruct_dataset/time_tvt/val_dataset.csv")
        test_df = pd.read_csv("reconstruct_dataset/time_tvt/test_dataset.csv")
        num_labels = train_df["LabelNum"].nunique()
        print(f"num labels in training set:{num_labels}")
        X_train, y_train = list(train_df["Text"]), list(train_df["LabelNum"])
        X_val, y_val = list(val_df["Text"]), list(val_df["LabelNum"])
        X_test, y_test = list(test_df["Text"]), list(test_df["LabelNum"])
        return X_train, y_train, X_val, y_val, X_test, y_test, num_labels

    elif dataset_split_method == "random":
        if NOCODE is True:
            df = pd.read_csv("dataset_nocode.csv")
        else:
            df = pd.read_csv("dataset.csv")
        num_labels = df["LabelNum"].nunique()
        test_size = int(df.shape[0] * 0.15)
        val_size = int(df.shape[0] * 0.15)
        X_train, X_test, y_train, y_test = train_test_split(
            list(df["Text"]),
            list(df["LabelNum"]),
            test_size=test_size,
            stratify=df["LabelNum"],
            random_state=int(split_seed),
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=int(split_seed),
        )
        return X_train, y_train, X_val, y_val, X_test, y_test, num_labels

    else:
        raise Exception("dataset_split_method parameter error")


def build_experiment_configs(experiment_setting: str):
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

    raise ValueError("experiment_setting must be seed_15 or seed_5_repeat_3")


def train(model_path, save_dir, exp_id, split_seed, device, dataset_split_method):
    """
    device:"cuda:1"
    dataset_split_method:"random"|"time"
    """
    model_save_dir = os.path.join(save_dir, f"ft_model_{exp_id}")

    X_train, y_train, X_val, y_val, X_test, y_test, num_labels = build_dataset(
        dataset_split_method=dataset_split_method,
        split_seed=split_seed,
    )
    print(f"train size:{len(X_train)}, val size:{len(X_val)}, test size:{len(X_test)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    train_dataset = TextDataset(X_train, y_train, tokenizer)
    val_dataset = TextDataset(X_val, y_val, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)

    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=slide_window_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=slide_window_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=slide_window_collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=4e-6)

    model.to(device)
    scaler = torch.amp.GradScaler("cuda")

    num_epochs = 30
    best_loss = float("inf")
    best_info = {
        "epoch": None,
        "TrainAcc": 0.0,
        "TrainLoss": 0.0,
        "ValAcc": 0.0,
        "ValLoss": 0.0,
        "TestAcc": 0.0,
        "TestLoss": 0.0,
    }
    s_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        i = 0
        losses = []
        for batch in train_loader:
            i += 1
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            window_mask = batch["window_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                logits = mean_window_logits(model, input_ids, attention_mask, window_mask)
                loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        res = evaluate(model, train_loader, device)
        val_res = evaluate(model, val_loader, device)
        test_res = evaluate(model, test_loader, device)

        if val_res[1] < best_loss:
            best_loss = val_res[1]
            best_info["epoch"] = epoch + 1
            best_info["TrainAcc"] = res[0]
            best_info["TrainLoss"] = np.mean(losses)
            best_info["ValAcc"] = val_res[0]
            best_info["ValLoss"] = val_res[1]
            best_info["TestAcc"] = test_res[0]
            best_info["TestLoss"] = test_res[1]
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)

        epoch_end_time = time.time()
        elapsed_time = int(epoch_end_time - epoch_start_time)
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"epoch:{epoch + 1} time:{hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train_Acc: {res[0]} - Train_Loss: {np.mean(losses)} - "
            f"Val_acc: {val_res[0]} - Val_loss: {val_res[1]} - "
            f"Test_acc: {test_res[0]} - Test_loss: {test_res[1]}"
        )

    e_time = time.time()
    elapsed_time = int(e_time - s_time)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"Best Info: Epoch {best_info['epoch']}/{num_epochs} - "
        f"Acc: {best_info['TrainAcc']} - Loss: {best_info['TrainLoss']} - "
        f"Val_acc: {best_info['ValAcc']} - Val_loss: {best_info['ValLoss']} - "
        f"test_acc: {best_info['TestAcc']} - test_loss: {best_info['TestLoss']}"
    )
    print(f"total training time:{hours:02d}h {minutes:02d}m {seconds:02d}s")
    print(f"trained model saved in:{model_save_dir}")
    return best_info


def main():
    device = "cuda:4"
    experiment_setting = "seed_5_repeat_3"
    experiment_configs = build_experiment_configs(experiment_setting)
    repeat_num = len(experiment_configs)
    print(f"repeat num:{repeat_num}")
    dataset_split_method = "random"
    model_name = "sobert"
    model_path = None
    if model_name == "sobert":
        model_path = "./model"
    elif model_name == "codebert":
        model_path = "./codebert-base"
    elif model_name == "robert":
        model_path = "./roberta-base"
    else:
        raise Exception("model path parameter error")
    save_dir = os.path.join(exp_data_dir, "trained_models", model_name)
    os.makedirs(save_dir, exist_ok=True)
    for experiment_config in experiment_configs:
        exp_id = experiment_config["exp_id"]
        split_seed = experiment_config["split_seed"]
        repeat_id = experiment_config["repeat_id"]
        print(
            f"experiment setting:{experiment_setting},"
            f"split seed:{split_seed},"
            f"repeat id:{repeat_id}, exp id:{exp_id}"
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
    os.makedirs(exp_data_dir, exist_ok=True)
    NOCODE = False
    if NOCODE is False:
        exp_data_dir = os.path.join(exp_data_dir, "exp_slidewindow")
    else:
        exp_data_dir = os.path.join(exp_data_dir, "exp_nocode_slidewindow")
    pid = os.getpid()
    print(f"PID:{pid}")
    main()
