#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 2019/12/19
    function: 逻辑回归-Python实现
    versions: 1.0
    @author: Zero
    @copyright: Apache License, Version 2.0
"""
from math import exp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# label parse
def parseRecord(df_record):
    df_group = df_record.groupby(by="label")
    label_list = list(df_group.groups.keys())
    for idx, names in enumerate(label_list):
        df_record = df_record.replace(names, idx)
        print(f"{names}-->{idx}")
    return df_record


# data
def create_data(file_path, names):
    df = pd.read_csv(file_path, names=names)
    datas = parseRecord(df)
    return datas.iloc[:100, :-1], datas.iloc[:100, -1]


class LogisticReressionClassifier:
    def __init__(self, max_iter=100, learning_rate=0.0001):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x, w):
        return 1 / (1 + exp(-np.dot(x, w)))

    def __error(self, x, y, w):
        return y - self.sigmoid(x, w)

    def data_matrix(self, X):
        data_mat = []
        for d in X.values:
            data_mat.append([*d, 1.0])
        return data_mat

    def fit(self, X, y):
        data_mat = self.data_matrix(X)  # m*n
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(data_mat)):
                error = self.__error(np.array(data_mat[i]), y.iloc[i], self.weights)
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
            print(f"iter_:{iter_}---->error:{error}\n")
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)


if __name__ == "__main__":
    file_path = "../datas/iris.data"
    names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    X, y = create_data(file_path, names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lr_clf = LogisticReressionClassifier()

    lr_clf.fit(X_train, y_train)
    score = lr_clf.score(X_test, y_test)
    print(f"score:{score}")
