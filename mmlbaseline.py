import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss

def tokenize_texts(texts):
    """分词并删除非字母词和英文停用词。"""
    stop_words = set(stopwords.words('english'))
    return [
        [
            word
            for word in word_tokenize(text.lower())
            if word.isalpha() and word not in stop_words
        ]
        for text in texts
    ]


def word2vec_embedding(tokenized_texts, model):
    """使用已经在训练集上拟合的 Word2Vec 模型生成句向量。"""
    sentences_vector = []
    for sentence in tokenized_texts:
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(model.vector_size)
        sentences_vector.append(sentence_vector)
    return sentences_vector


def tfidf_embedding(tokenized_texts, vectorizer, fit=False):
    """拟合或复用同一个 TF-IDF 向量器生成文本向量。"""
    sentences = [' '.join(tokens) for tokens in tokenized_texts]
    if fit:
        return vectorizer.fit_transform(sentences)
    return vectorizer.transform(sentences)


def build_dataset_split(dataset_split_method, split_seed):
    """
    按照 mmltrain.py 的 dataset_split_method 构建 train/val/test。
    dataset_split_method: random|time
    """
    if dataset_split_method == "time":
        trainval_df = pd.read_csv("reconstruct_dataset/trainval_dataset.csv")
        test_df = pd.read_csv("reconstruct_dataset/test_dataset.csv")

        val_size = int(0.1 * trainval_df.shape[0])
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=val_size,
            stratify=trainval_df["LabelNum"],
            random_state=int(split_seed),
        )
        X_train, y_train = list(train_df["Text"]), list(train_df["LabelNum"])
        X_val, y_val = list(val_df["Text"]), list(val_df["LabelNum"])
        X_test, y_test = list(test_df["Text"]), list(test_df["LabelNum"])
        test_ids = list(test_df["Id"])
        return X_train, y_train, X_val, y_val, X_test, y_test, test_ids

    if dataset_split_method == "random":
        df = pd.read_csv("dataset.csv")
        test_size = int(df.shape[0] * 0.15)
        val_size = int(df.shape[0] * 0.15)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["LabelNum"],
            random_state=int(split_seed),
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            stratify=train_df["LabelNum"],
            random_state=int(split_seed),
        )
        X_train, y_train = list(train_df["Text"]), list(train_df["LabelNum"])
        X_val, y_val = list(val_df["Text"]), list(val_df["LabelNum"])
        X_test, y_test = list(test_df["Text"]), list(test_df["LabelNum"])
        test_ids = list(test_df["Id"])
        return X_train, y_train, X_val, y_val, X_test, y_test, test_ids

    raise ValueError("dataset_split_method 只能是 random 或 time")


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


def train_pred(train_vector, val_vector, test_vector, y_train, y_val, y_test, test_ids):
    clf_random_states = [42, 43, 44]
    knn_n_neighbors = [5,6,7]
    clf_candidates = {
        'LR': [
            ('max_iter=1000', LogisticRegression(max_iter=1000))
        ],
        'DT': [
            (f'random_state={seed}', DecisionTreeClassifier(random_state=seed))
            for seed in clf_random_states
        ],
        'RF': [
            (
                f'random_state={seed}',
                RandomForestClassifier(n_estimators=100, random_state=seed),
            )
            for seed in clf_random_states
        ],
        'SVM': [
            (
                f'random_state={seed}',
                SVC(kernel='linear', random_state=seed, probability=True),
            )
            for seed in clf_random_states
        ],
        'KNN': [
            (f'n_neighbors={n}', KNeighborsClassifier(n_neighbors=n))
            for n in knn_n_neighbors
        ],
    }

    result_dfs = {}
    for clf_name, candidates in clf_candidates.items():
        print(f'分类器名称:{clf_name}')
        best_clf = None
        best_loss = float('inf')
        for config, clf in candidates:
            clf.fit(train_vector, y_train)
            val_probs = clf.predict_proba(val_vector)
            val_loss = log_loss(y_val, val_probs, labels=clf.classes_)
            print(f'{config}, 验证集loss:{val_loss}')

            if val_loss < best_loss:
                best_loss = val_loss
                best_clf = clf

        y_pred = best_clf.predict(test_vector)
        probs = best_clf.predict_proba(test_vector)
        probs_by_label = np.zeros((len(y_test), 6))
        for prob_col_idx, label_num in enumerate(best_clf.classes_):
            probs_by_label[:, int(label_num)] = probs[:, prob_col_idx]

        result_df = pd.DataFrame({
            'Id': test_ids,
            'True': y_test,
            'pred': y_pred
        })
        for label_num in range(6):
            result_df[f'prob_{label_num}'] = probs_by_label[:, label_num]
        result_dfs[clf_name] = result_df

    return result_dfs

def baseline_method(split_seed, baseline_name, dataset_split_method, exp_id=None):
    '''
    split_seed:训练集/验证集/测试集切分随机数
    baseline_name:word2vec|tfidf
    dataset_split_method:random|time
    exp_id:实验保存id。不传时默认等于 split_seed，兼容 seed_15 旧目录。
    '''
    if exp_id is None:
        exp_id = str(split_seed)

    X_train, y_train, X_val, y_val, X_test, y_test, test_ids = build_dataset_split(
        dataset_split_method,
        split_seed,
    )

    print(f"训练集大小:{len(X_train)},验证集大小:{len(X_val)}, 测试集大小:{len(X_test)}")
    train_tokens = tokenize_texts(X_train)
    val_tokens = tokenize_texts(X_val)
    test_tokens = tokenize_texts(X_test)

    if baseline_name == "word2vec":
        # Word2Vec 只在训练集上拟合，三组数据共享同一词向量空间。
        word2vec_model = Word2Vec(
            train_tokens,
            vector_size=100,
            window=5,
            min_count=2,
            workers=1,
            seed=split_seed,
        )
        train_vector = word2vec_embedding(train_tokens, word2vec_model)
        val_vector = word2vec_embedding(val_tokens, word2vec_model)
        test_vector = word2vec_embedding(test_tokens, word2vec_model)
    elif baseline_name == 'tfidf':
        # TF-IDF 只在训练集上拟合，验证集和测试集复用训练集词表。
        vectorizer = TfidfVectorizer()
        train_vector = tfidf_embedding(train_tokens, vectorizer, fit=True)
        val_vector = tfidf_embedding(val_tokens, vectorizer)
        test_vector = tfidf_embedding(test_tokens, vectorizer)
    else:
        raise Exception("baseline name参数错误")

    result_dfs = train_pred(
        train_vector,
        val_vector,
        test_vector,
        y_train,
        y_val,
        y_test,
        test_ids
    )

    # 保存
    save_dir = os.path.join(exp_data_dir,f"trained_{baseline_name}",f"seed_{exp_id}")
    os.makedirs(save_dir,exist_ok=True)
    for clf_name,df in result_dfs.items():
        save_file_name = clf_name+".csv"
        save_file_path = os.path.join(save_dir,save_file_name)
        df.to_csv(save_file_path,index=False)
    print(f"结果保存在:{save_dir}")


def main():
    s_time=time.time()
    method_name = "word2vec" # tfidf|word2vec
    dataset_split_method = "time" # random|time
    experiment_setting = "seed_5_repeat_3" # seed_15|seed_5_repeat_3
    experiment_configs = build_experiment_configs(experiment_setting)
    print(f"基线名称:{method_name}")
    print(f"数据集切分方式:{dataset_split_method}")
    print(f"实验设置:{experiment_setting}")
    repeat_num = len(experiment_configs) # 重复实验次数
    for experiment_config in experiment_configs:
        exp_id = experiment_config["exp_id"]
        split_seed = experiment_config["split_seed"]
        repeat_id = experiment_config["repeat_id"]
        print(
            f"数据集切分随机种子:{split_seed},"
            f"重复id:{repeat_id},实验id:{exp_id}"
        )
        baseline_method(
            split_seed,
            method_name,
            dataset_split_method,
            exp_id=exp_id,
        )
    e_time=time.time()
    elapsed_time = int(e_time - s_time)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"总耗时：{hours:02d}小时 {minutes:02d}分钟 {seconds:02d}秒")
    print(f"当前时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    pid = os.getpid()
    print(f"pid:{pid}")
    exp_data_dir = "/data/mml/DL_bug_classification/time5_3"
    main()
