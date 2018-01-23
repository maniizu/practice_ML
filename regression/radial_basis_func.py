"""
Name: report3.py
Date: 2017.06.18
Usage: python3 radial_basis_func.py
Description:
    定義された適当な関数を動径基底関数により近似する.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def true_func(x):
    """
    定義された適当な関数
    """
    f = -100 * x * (x - 1) * (x - 0.5) ** 2
    return f

def basis_func(x, m):
    """
    動径基底関数を生成して返す.
    """
    if m == 0:
        m = 1
    u = np.arange(m) / m
    phi = np.exp(-1 * (x - u) ** 2 / 3.0)
    return phi

def desigh_mat(X, phi, m):
    """
    計画行列を計算する.
    phiは動径基底関数.
    """
    large_phi = []
    for x in X:
        large_phi.append(phi(x, m))
    return np.array(large_phi)

def pred_w(large_phi, T):
    """
    パラメータwを推定する.
    """
    large_phi_T = large_phi.T
    w = np.linalg.inv(large_phi_T.dot(large_phi)).dot(large_phi_T).dot(T)
    return w

def pred_func(w, x):
    """
    近似関数の生成
    """
    func = w.dot(basis_func(x, len(w)))
    return func

if __name__ == '__main__':
    np.random.seed(1)
    for n in [10, 100, 1000]:
        X = np.random.rand(n)
        T = true_func(X) + np.random.normal(0, 0.1, size=len(X))
        phi = basis_func
        for m in [1, 4, 9]:
            print((n,m))
            w = pred_w(desigh_mat(X, phi, m), T)
            #print(w)
            #関数をプロットする
            Y = []
            for x in np.arange(0, 1, 0.01):
                Y.append(pred_func(w,x))
            plt.plot(np.arange(0, 1, 0.01), Y, color='red', label='pred_func')
            plt.plot(np.arange(0, 1, 0.01), true_func(np.arange(0, 1, 0.01)), color='green', label='true_func')
            plt.scatter(X, T, marker='+')
            plt.legend()
            plt.xlabel("xsamples")
            plt.ylabel("values")
            plt.savefig("RBF(" + str(n) + "," + str(m) + ")")
            plt.show()

            error = np.sum((true_func(np.arange(0, 1, 0.01)) - Y) ** 2) / 2
            print(error)
