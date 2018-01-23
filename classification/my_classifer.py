"""
Name: my_classifer.py
Date: 2017.07.23
Usage: kadai1.pyとkadai2.pyで呼び出して使用
Description:
    kadai1.pyとkadai2.py用の関数群.
"""

import numpy as np

def func(x, a, b):
    """
    識別関数
    """
    return a * x + b

def make_dat(size, mu, cov):
    """
    データ生成
    """
    return np.random.multivariate_normal(mu, cov, size)

def sigmoid(f):
    """
    シグモイド関数
    """
    return 1 / (1 + np.exp(-f))

def fisher_cls(cls1, cls2, mean1, mean2):
    """
    フィッシャー線形判別の学習
    """
    s_w = np.zeros((len(cls1[0]),len(cls1[0])))
    for i in range(len(cls1)):
        x_i = np.matrix(cls1[i]).reshape(len(cls1[i]),1)
        mean1 = np.matrix(mean1).reshape(len(cls1[i]),1)
        s_w += (x_i - mean1) * np.transpose(x_i - mean1)
    for i in range(len(cls2)):
        x_i = np.matrix(cls2[i]).reshape(len(cls2[i]),1)
        mean2 = np.matrix(mean2).reshape(len(cls2[i]),1)
        s_w += (x_i - mean2) * np.transpose(x_i - mean2)
    s_w_inv = np.linalg.inv(s_w)
    w = s_w_inv * (mean1 - mean2)
    return w

def perceptron_cls(x, t):
    """
    単純パーセプトロン+誤り訂正学習
    """
    #パラメータを初期化
    w = np.ones(len(x[0,:]))
    eta = 0.01
    count = 0
    num_correct = 0
    #全てのデータを正確に分類するまでループ
    while(num_correct < len(x)):
        num_correct = 0
        for i in range(len(x)):
            #正しく分類できたとき
            if(np.dot(w, x[i,:]) * t[i] > 0):
                num_correct += 1
            #できなければ学習率を更新
            else:
                w += eta * x[i,:] * t[i]
        count += 1
    print("単純パーセプトロンのパラメータ学習のためにwを"+str(count)+"回更新しました.")
    return w

def irls_cls(phi, t, cond):
    """
    IRLSによるロジスティック回帰
    """
    count = 0
    w = np.ones(len(phi[0,:]))
    #w[0] = 1.0
    sub = cond
    while(sub>=cond):
        count += 1
        #Rとyを計算
        R = np.zeros((len(phi),len(phi)))
        y = []
        for i in range(len(phi)):
            f = np.dot(w, phi[i,:])
            y_i = sigmoid(f)
            R[i, i] = y_i * (1 - y_i)
            y.append(y_i)
        #ヘッセ行列を計算
        phi_T = phi.transpose()
        Hesse = np.dot(phi_T, np.dot(R, phi))
        #パラメータを更新
        w_new = w - np.dot(np.linalg.inv(Hesse), np.dot(phi_T, (y-t)))
        #収束しているか確認
        if(np.linalg.norm(w) == 0):
            print("なんかやべーやつおる!",count)
        sub = np.linalg.norm(w_new - w) / np.linalg.norm(w)
        w = w_new
        #収束していればループを抜ける.
    print("ロジスティック回帰のパラメータ学習のためにwを"+str(count)+"回更新しました.")
    return w
