"""
Name: kadai1.py
Date: 2017.07.23
Usage: kadai1.py
Description:
    2クラスの2次元データを線形分離する.データは人工的に発生させる.
    モデルはフィッシャー線形判別,単純パーセプトロン+誤り訂正学習,IRLSによる識別モデルである.
"""

import numpy as np
from pylab import *
from my_classifer import *
import my_evaluation as my_eval

if __name__ == '__main__':
    #訓練データを作成
    cov = np.linalg.inv([[30,10],[10,15]])
    cls1 = []
    cls2 = []
    cls1.extend(make_dat(80, [0,0], cov))
    cls1.extend(make_dat(20, [-2,-2], cov))
    cls2.extend(make_dat(100, [1,1], cov))

    #クラス1を0,クラス2を1とする正解データのラベルを作成
    labels = [0 for i in range(len(cls1))]
    for i in range(len(cls2)):
        labels.append(1)
    labels = np.array(labels)

    #訓練データを描画
    x, y = np.transpose(np.array(cls1))
    plot(x, y, 'r.', label='class1')
    x, y = np.transpose(np.array(cls2))
    plot(x, y, 'b.', label='class2')

    #各クラスの平均を求める
    mean1 = np.mean(cls1, axis=0)
    mean2 = np.mean(cls2, axis=0)

    #フィッシャー線形判別の学習
    w_fisher = fisher_cls(cls1, cls2, mean1, mean2)
    w_fisher = np.array(w_fisher).reshape(1,len(w_fisher))[0]

    #フィッシャー線形判別の識別境界を描画
    a = -(w_fisher[0]/w_fisher[1])
    mean = (mean1 + mean2) / 2
    b = -a * mean[0] + mean[1]
    x_fisher = np.linspace(-3, 3, 1000)
    y_fisher = [func(x_n, a, b) for x_n in x_fisher]
    plot(x_fisher, y_fisher, 'g-', label='Fisher')

    #ここから単純パーセプトロン
    #クラス1を+1,クラス2を-1とする教師データを作成
    t = labels.copy()
    t[t==1] = -1.0
    t[t==0] = 1.0

    #データを[1, x軸の値, y軸の値]とする
    X_percep = np.hstack((ones((len(cls1)+len(cls2),1)), np.vstack((cls1, cls2))))

    #単純パーセプトロンの学習
    w_percep = perceptron_cls(X_percep, t)

    #パーセプトロンの識別境界を描画
    a = -w_percep[1]/w_percep[2]
    b = - w_percep[0]/w_percep[2]
    x_percep = np.linspace(-3, 3, 1000)
    y_percep = [func(x_i, a, b) for x_i in x_percep]
    plot(x_percep, y_percep, 'c-', label='Perceptron')

    #ここからIRLSによるロジスティック回帰
    #クラス1を1.0,クラス2を0.0として教師データを再設定
    t[t==-1.0] = 0.0

    #基底関数phiを設定
    phi = X_percep

    #収束条件
    cond = 0.01

    #IRLSでパラメータを推定
    w_irls = irls_cls(phi, t, cond)

    #ロジスティック回帰の識別境界を描画
    a = -w_irls[1]/w_irls[2]
    b = - w_irls[0]/w_irls[2]
    x_irls = np.linspace(-3, 3, 1000)
    y_irls = [func(x_i, a, b) for x_i in x_irls]
    plot(x_irls, y_irls, 'm-', label='IRLS')

    legend()
    xlim(-3, 3)
    ylim(-3, 3)
    savefig("classifer_output.png")
    show()
