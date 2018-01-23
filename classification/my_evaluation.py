"""
Name: my_evaluation.py
Date: 2017.07.23
Usage: kadai1.py や kadai2.py で必要な関数を呼び出して使用
Description:
    線形分類器の誤分類の個数を求める関数と分類後の最小マージンを求める関数
"""

import numpy as np

def error_counter(w, features, labels):
    """
    分類誤差を計算する.
    """
    error_sum = 0
    for i in range(len(labels)):
        score = np.dot(w, features[i])
        print(i, score)
        if(labels[i]==0 and score<0):
            error_sum += 1
        elif(labels[i]==1 and score>=0):
            error_sum += 1
    return error_sum

def compute_margin(w, x):
    """
    パラメータとデータからマージンを計算する.
    """
    dif = np.linalg.norm(w)
    if(dif != 0):
        return np.absolute(np.dot(w,x)) / np.linalg.norm(w)
    else:
        print("分母が0になりました.")
        return

def search_min_margin(w, X):
    """
    データ点で最小となるマージンを計算する.
    """
    min_marg = compute_margin(w,X[0])
    for i in range(len(X)):
        temp = compute_margin(w, X[i])
        if(min_marg > temp):
            min_marg = temp
    return min_marg
