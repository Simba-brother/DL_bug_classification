import sys
import time
import pandas as pd
import numpy as np
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
from sklearn.metrics import accuracy_score, classification_report

def word2vec_embedding(texts):
    stop_words = set(stopwords.words('english'))
    sentences = []
    for i in texts:
        new_str = [word for word in word_tokenize(i.lower()) if word.isalpha() and word not in stop_words]
        sentences.append(new_str)
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    sentences_vector = []
    for sentence in sentences:
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if word_vectors:
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

def train_pred(X, y, m, data_rs=42, clf_rs=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=90, random_state=int(data_rs), stratify=y)
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=90, random_state=int(data_rs), stratify=y_train)


    clfs = {'LR':LogisticRegression(random_state=int(clf_rs)), 
            'DT':DecisionTreeClassifier(random_state=int(clf_rs)), 
            'RF':RandomForestClassifier(n_estimators=100, random_state=int(clf_rs)),
            'SVM':SVC(kernel='linear', random_state=int(clf_rs), probability=True),
            'KNN':KNeighborsClassifier(n_neighbors=5)}
    df = pd.DataFrame()
    df['True'] = y_test
    time_all = []
    for clf_name, clf in clfs.items():
        s = time.time()
        clf.fit(X_train, y_train)
        time_all.append(time.time()-s)
        s = time.time()
        y_pred = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        time_all.append(time.time()-s)
        df[clf_name] = y_pred
        for i in range(6):
            df[f'{clf_name}_prob_{i}'] = probs[:,i]

    df.to_csv(f"results/{m}/{data_rs}_{clf_rs}.csv",index=False)
    return time_all

if __name__ == "__main__":
    m = sys.argv[1]
    df = pd.read_csv("dataset/dataset.csv")

    time_all_list = []
    for data_rs in [42,43,44,45,46]:
        for clf_rs in [42,43,44]:
            if m == "word2vec":
                s = time.time()
                word2vec_vector = word2vec_embedding(df['Text'])
                embedding_time = time.time()-s
                time_all = train_pred(word2vec_vector, df['LabelNum'], m, data_rs, clf_rs)
                time_all.append(embedding_time)
                time_all_list.append(time_all)
            elif m == "tfidf":
                s = time.time()
                tfidf_vector = tfidf_embedding(df['Text'])
                embedding_time = time.time()-s
                time_all = train_pred(tfidf_vector, df['LabelNum'], m, data_rs, clf_rs)
                time_all.append(embedding_time)
                time_all_list.append(time_all)

    columns = []
    for i in ['LR','DT','RF','SVM','KNN']:
        columns.append(f'{m}_{i}_train')
        columns.append(f'{m}_{i}_test')
    columns.append('embedding')
    data = pd.DataFrame(time_all_list, columns=columns)
    data.to_csv(f"results/{m}/time.csv",index=False)
