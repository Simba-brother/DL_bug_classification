import sys
import time
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

def word2vec_embedding(texts)->list:
    # 获得停用词set
    stop_words = set(stopwords.words('english'))
    sentences = []
    for i in texts:
        # token化+删除停用词
        new_str = [word for word in word_tokenize(i.lower()) if word.isalpha() and word not in stop_words]
        sentences.append(new_str)
    # 获得w2v的模型
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    # 存储每个句子的向量（句向量）。
    sentences_vector = []
    # 遍历每个句子
    for sentence in sentences:
        # 获得该句子每个词的vectior
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if word_vectors:
            # 句子向量其实就是词向量的均值
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(model.vector_size)  # 如果没有有效词，返回零向量
        sentences_vector.append(sentence_vector)
    return sentences_vector

def tfidf_embedding(texts):
    stop_words = set(stopwords.words('english'))
    sentences = []
    for i in texts:
        new_str = " ".join([word for word in word_tokenize(i.lower()) if word.isalpha() and word not in stop_words])
        sentences.append(new_str)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    return tfidf_matrix.toarray()



def train_pred(train_vector, val_vector, test_vector, y_train, y_val, y_test):
    clf_random_states = [42, 43, 44]
    knn_n_neighbors = [5,6,7]
    clf_candidates = {
        'LR': [
            (f'random_state={seed}', LogisticRegression(random_state=seed))
            for seed in clf_random_states
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
        result_dfs[clf_name] = pd.DataFrame({
            'True': y_test,
            'Pred': y_pred,
            'Probs': probs.tolist(),
        })

    return result_dfs

def baseline_method(trainval_df,test_df,rs,baseline_name):
    '''
    rs:训练集验证集切分随机数
    baseline_name:word2vec|tfidf
    '''
    
    # 从trainval中划分出，train和val
    val_size = int(0.1 * trainval_df.shape[0])
    X_train, X_val, y_train, y_val = train_test_split(list(trainval_df['Text']), list(trainval_df['LabelNum']), 
                                                      test_size=val_size, stratify=trainval_df['LabelNum'], random_state=rs)
    # test_df中构建出X_test,y_test
    X_test, y_test = list(test_df["Text"]), list(test_df["LabelNum"])

    print(f"训练集大小:{len(X_train)},测试集大小:{len(X_test)}")
    if baseline_name == "word2vec":
        # 都是句子level的vector
        train_vector = word2vec_embedding(X_train)
        val_vector = word2vec_embedding(X_val)
        test_vector = word2vec_embedding(X_test)
    elif baseline_name == 'tfidf':
        # 都是句子level的vector
        train_vector = tfidf_embedding(X_train)
        val_vector = tfidf_embedding(X_val)
        test_vector = tfidf_embedding(X_test)
    else:
        raise Exception("baseline name参数错误")

    result_dfs = train_pred(
        train_vector,
        val_vector,
        test_vector,
        y_train,
        y_val,
        y_test
    )

    # 保存
    save_dir = os.path.join(exp_data_dir,f"trained_{baseline_name}",f"seed_{rs}")
    os.makedirs(save_dir,exist_ok=True)
    for cls_name,df in result_dfs.items():
        save_file_name = cls_name+".csv"
        save_file_path = os.path.join(save_dir,save_file_name)
        df.to_csv(save_file_path,index=False)
    print(f"结果保存在:{save_dir}")


def main():
    s_time=time.time()
    method_name = "tfidf" # word2vec|tfidf
    print(f"基线名称:{method_name}")
    repeat_num = 15 # 重复实验次数
    for rs in range(42,42+repeat_num):
        print(f"随机种子:{rs}")
        baseline_method(trainval_df,test_df,rs,method_name)
    e_time=time.time()
    elapsed_time = int(e_time - s_time)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"总耗时：{hours:02d}小时 {minutes:02d}分钟 {seconds:02d}秒")


if __name__ == "__main__":
    exp_data_dir = "/data/mml/DL_bug_classification"
    # 数据集(测试集划分出来)
    trainval_df = pd.read_csv("reconstruct_dataset/trainval_dataset.csv")
    test_df = pd.read_csv("reconstruct_dataset/test_dataset.csv")
    # 类别数
    num_labels = trainval_df["LabelNum"].nunique() # 应该是6(5个DL bug,1个no DL bug)
    main()
