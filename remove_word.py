from __future__ import annotations

import nltk
import pandas as pd
import re
from pathlib import Path
import os
import math
from collections import Counter
from itertools import combinations


def tokenize_words(text:str, stop_words:set[str]) -> list[str]:
    tokens = nltk.word_tokenize(text)

    return [
        token.lower()
        for token in tokens
        if token.isalpha() and token.lower() not in stop_words
    ]

def collect_words(text_list:list[str]) -> set[str]:
    all_text = " ".join(text_list)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = tokenize_words(all_text, stop_words)
    return set(words)


def remove_words(dl_bug_df:pd.DataFrame, words:set[str]) -> dict[str, pd.DataFrame]:
    remove_word2df = {}

    for remove_word in words:
        pattern = re.compile(rf"\b{re.escape(remove_word)}\b", re.IGNORECASE)
        removed_df = dl_bug_df.copy()
        removed_df["Text"] = (
            removed_df["Text"]
            .fillna("")
            .astype(str)
            .apply(lambda text: re.sub(r"\s+", " ", pattern.sub("", text)).strip())
        )
        remove_word2df[remove_word] = removed_df
    assert len(remove_word2df.keys()) == len(words), "删除词遍历出错"
    return remove_word2df


def safe_filename_word(word:str) -> str:
    safe_word = re.sub(r"[^A-Za-z0-9._-]+", "_", word).strip("._-")
    return safe_word or "empty"


def save_remove_word_dfs(remove_word2df:dict[str, pd.DataFrame], output_dir:Path|str) -> None:
    for remove_word, df in remove_word2df.items():
        csv_path = os.path.join(output_dir,f"remove_{safe_filename_word(remove_word)}.csv")
        df.to_csv(csv_path, index=False)


def get_remove_word_datasets():
    test_df = pd.read_csv(test_dataset_csv_path)
    dl_bug_df = test_df[test_df["Label"] != "Others"].reset_index(drop=True) # 如果不写 drop=True，旧索引会变成一列叫 index 的新列
    text_list = list(dl_bug_df["Text"])
    words = collect_words(text_list)
    remove_word2df = remove_words(dl_bug_df,words)
    save_dir = os.path.join(exp_root_dir,"remove_word")
    os.makedirs(save_dir,exist_ok=True)
    save_remove_word_dfs(remove_word2df, save_dir)
    print(f"word删除数据集保存在:{save_dir}")

def collect_combinewords(text_list:list[str], min_support:float=0.1) -> list[tuple[str, ...]]:
    stop_words = set(nltk.corpus.stopwords.words("english"))

    # 每条 Text 作为一个 transaction，同一条文本中的重复词只计一次。
    transactions = [
        set(tokenize_words("" if pd.isna(text) else str(text), stop_words))
        for text in text_list
    ]
    if not transactions:
        return []

    # support=0.1 表示 itemset 至少出现在 10% 的 transaction 中。
    min_count = max(1, math.ceil(len(transactions) * min_support))
    item_counter = Counter()
    for transaction in transactions:
        item_counter.update(transaction)

    # 先找出频繁 1-itemsets，后续用 Apriori 性质逐层扩展。
    current_frequent_itemsets = {
        frozenset([word])
        for word, count in item_counter.items()
        if count >= min_count
    }
    combinewords_list = []
    k = 2

    while current_frequent_itemsets:
        candidates = set()
        frequent_itemsets = list(current_frequent_itemsets)

        # 由频繁 (k-1)-itemsets 组合生成 k-itemset 候选。
        for idx, itemset in enumerate(frequent_itemsets):
            for other_itemset in frequent_itemsets[idx + 1:]:
                candidate = itemset | other_itemset
                if len(candidate) != k:
                    continue
                # Apriori 剪枝：候选的所有 (k-1) 子集都必须是频繁项集。
                if all(
                    frozenset(subset) in current_frequent_itemsets
                    for subset in combinations(candidate, k - 1)
                ):
                    candidates.add(candidate)

        if not candidates:
            break

        # 统计每个候选 itemset 出现在多少条 Text 中。
        candidate_counter = Counter()
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counter[candidate] += 1

        # 保留 support 达到阈值的候选 itemsets。
        current_frequent_itemsets = {
            candidate
            for candidate, count in candidate_counter.items()
            if count >= min_count
        }
        combinewords_list.extend(
            tuple(sorted(itemset))
            for itemset in current_frequent_itemsets
            if len(itemset) >= 2
        )
        k += 1

    return sorted(combinewords_list, key=lambda itemset: (len(itemset), itemset))

def get_remove_combineword_dataset():
    test_df = pd.read_csv(test_dataset_csv_path)
    dl_bug_df = test_df[test_df["Label"] != "Others"].reset_index(drop=True) # 如果不写 drop=True，旧索引会变成一列叫 index 的新列
    text_list = list(dl_bug_df["Text"])
    collect_combinewords(text_list)


if __name__ == "__main__":
    exp_root_dir = "/data/mml/DL_bug_classification"
    dataset_csv_path = "./dataset.csv"
    test_dataset_csv_path = "./reconstruct_dataset/test_dataset.csv"
    # trainval_dataset_csv_path = "./reconstruct_dataset/trainval_dataset.csv"

    main()
