"""
Name: kadai2.py
Date: 2017.07.23
Usage: python3 kadai2.py
Description:
    irisデータの3クラスのうち2クラスを選び,該当するデータを分類する識別関数を学習する.
    手法はフィッシャー線形判別,単純パーセプトロン+誤り訂正学習,IRLSによるロジスティック回帰.
"""

import numpy as np
from pylab import *
from sklearn import datasets
import my_classifer as my_cls
import my_evaluation as my_eval

if __name__ == '__main__':
    #irisデータを読み込む
    iris = datasets.load_iris()
    features = iris.data[:100]
    cls1 = iris.data[:50]
    cls2 = iris.data[50:100]
    labels = iris.target[:100]

    #ここからフィッシャー線形判別
    #各クラスの平均を求める
    mean1 = np.mean(cls1, axis=0)
    mean2 = np.mean(cls2, axis=0)

    #フィッシャー線形判別の学習
    w_fisher = my_cls.fisher_cls(cls1, cls2, mean1, mean2)
    w_fisher = np.array(w_fisher).reshape(1,len(w_fisher))[0]
    print("フィッシャー線形判別のパラメータ")
    print(w_fisher)
    #フィッシャー線形判別の精度を確認
    print("フィッシャー線形判別の分類精度")
    error_fisher = my_eval.error_counter(w_fisher, features, labels)
    print("正解率は",(len(labels)-error_fisher)/len(labels))
    #誤分類していない場合はマージンを計算
    if(error_fisher == 0):
        print("フィッシャー線形判別でのマージン")
        margin_fisher = my_eval.search_min_margin(w_fisher, features)
        print(margin_fisher)

    #ここから単純パーセプトロン
    #クラス1を+1,クラス2を-1とする教師データを作成
    t = labels.copy()
    t[t==1] = -1
    t[t==0] = 1

    #データに1を加える.
    #print(type(features))
    X_percep = np.hstack((np.ones((len(features),1)), features))

    #単純パーセプトロンを学習
    w_percep = my_cls.perceptron_cls(X_percep, t)
    print("単純パーセプトロンのパラメータ")
    print(w_percep)

    #単純パーセプトロンの精度を確認
    print("単純パーセプトロンの分類精度")
    error_percep = my_eval.error_counter(w_percep, X_percep, labels)
    print("正解率は",(len(labels)-error_percep)/len(labels))
    if(error_percep == 0):
        print("単純パーセプトロンでのマージン")
        margin_percep = my_eval.search_min_margin(w_percep, X_percep)
        print(margin_percep)

    #ここからIRLSでロジスティック回帰
    #クラス1を+1,クラス2を0として教師データを再設定
    t[t==-1] = 0

    #基底関数phiを設定
    phi = X_percep

    #IRLSでロジスティック回帰を学習
    cond = 0.01
    w_irls = my_cls.irls_cls(phi, t, cond)
    print("ロジスティック回帰のパラメータ")
    print(w_irls)

    #ロジスティック回帰の精度を確認
    print("ロジスティック回帰の分類精度")
    error_irls = my_eval.error_counter(w_irls, phi, labels)
    print("正解率は",(len(labels)-error_irls)/len(labels))
    if(error_irls == 0):
        print("ロジスティック回帰でのマージン")
        margin_irls = my_eval.search_min_margin(w_irls, phi)
        print(margin_irls)
