import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from nltk.tokenize import word_tokenize
import gensim.downloader
from tqdm import tqdm
import pickle
import ast
import nltk
nltk.download('punkt_tab')

def load_data():
    random_state = 42
    df = pd.read_csv("data/final.csv")
    y = df["delta"]
    xtrain, xtest, ytrain, ytest = train_test_split(df, y, test_size=0.2, random_state=random_state)
    df_train = xtrain
    df_test = xtest
    return df_train, df_test

def get_stats(df):
    counts = df["delta"].value_counts().to_dict()
    return counts

class StockGAP():
    def __init__(self, feature_type):
        self.feature_type = feature_type
        if feature_type == 'glove':
            self.glove = gensim.downloader.load('glove-wiki-gigaword-200')
            self.feature_type = feature_type
    
    def save_model(self, filename):
        if self.clf is not None:
            pickle.dump(self.clf, open(filename, 'wb'))
    
    def load_model(self, filename):
        self.clf = pickle.load(open(filename, 'rb'))
    
    def fit_tfidf(self, train):
        X = None
        self.vectorizer = TfidfVectorizer(ngram_range =(1, 1), max_df=0.8, min_df=2)
        matrix = self.vectorizer.fit_transform(train["headlines"])
        X = matrix
        return X
    
    def get_glove_feature(self, df):
        features = []
        tokenized = df["headlines"].tolist()
        for row in tokenized:
            row = ast.literal_eval(row)
            accum = np.zeros(self.glove.vector_size)
            num_vecs = 0
            for token in row:
                try:
                    accum += self.glove[token]
                    num_vecs += 1
                except:
                    pass
            features.append(accum / num_vecs)
        features = np.array(features)
        return features
    
    def cross_validation(self, X, y):
        best_hyperparameter = {}
        hidden_opts = [100, 50]
        lr_opts = [0.0001, 0.01]
        alpha_opts = [0.0001, 0.01]

        combs = itertools.product(hidden_opts, lr_opts, alpha_opts)
        for c in combs:
            (hidden, lr, alpha) = c
            classifier = MLPClassifier(solver="adam", hidden_layer_sizes=(hidden, hidden),
                activation="relu", learning_rate_init=lr, alpha=alpha, tol=1e-3)
            cv_score = cross_val_score(classifier, X, y, cv=5, scoring="f1_macro")
            mean_score = np.mean(cv_score)
            if best_hyperparameter == {} or mean_score > best_hyperparameter["mean_f1_macro"]:
                best_hyperparameter = {"hidden_layer": [hidden, hidden],
                    "learning_rate": lr, "alpha": alpha, "mean_f1_macro": mean_score, "f1_macro": cv_score}
        return best_hyperparameter
    
    def fit(self, X, y, hyperparameter):
        self.clf = MLPClassifier(solver="adam", hidden_layer_sizes=tuple(hyperparameter["hidden_layer"]), activation="relu",
            learning_rate_init=hyperparameter["learning_rate"], alpha=hyperparameter["alpha"], tol=1e-3)
        self.clf.fit(X, y)
    
    def test_performance(self, test):
        accuracy, f1 = 0, 0
        X = None
        if self.feature_type == "tfidf":
            X = self.vectorizer.transform(test["headlines"])
        else:
            X = get_glove_feature(test)
        y = test["delta"]
        prediction = self.clf.predict(X)
        accuracy = accuracy_score(y, prediction)
        f1 = f1_score(y, prediction, average="macro")
        return accuracy, f1