"""
Name : maximum_likelihood.py
Date : 2017.5.30
Usage : python3 maximumlikelihood.py plot_params
    param : 0の場合は1回だけ推定を行う
            1の場合は10回推定を行い平均時間を求める.
Description :
    ロジスティック回帰のパラメータ(a,b)を勾配上昇法と
    ニュートンラフソン法で推定する.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MaximumLikelihood:
    def __init__(self,N):
        #データサイズを設定
        self.N = N
        #パラメータの真値の設定
        self.true_a = 0.8
        self.true_b = -0.3
        self.true_sigma = 0.1
        #パラメータの初期値を設定
        self.est_a = 0
        self.est_b = 0
        #推定値の履歴を保存
        self.log_a = [self.est_a]
        self.log_b = [self.est_b]
        #収束条件の設定
        self.cond = 0.0000001
        #学習率の設定
        self.eta = 0.0001
        self.eta_newton = 1.0
        #説明変数Xの生成(X~N(0,1^2))
        self.X = np.random.randn(self.N,)
        self.Y = np.zeros(self.N)
        self.eps = pow(self.true_sigma, 2) * np.random.randn(self.N,)

    def logistic_reg(self):
        """
        ロジスティック回帰でYを生成
        """
        self.Y = (1.0 /
        (1.0 + np.exp(-1.0 * self.true_a * self.X
        - self.true_b))) + self.eps

    def get_grad(self):
        """
        勾配ベクトルを算出
        """
        grad = np.zeros(2)
        s = 1.0 / (1.0 + np.exp(-self.est_a * self.X - self.est_b))
        grad[0] = np.sum(self.X * s * (1.0 - s) * (self.Y - s)
        / pow(self.true_sigma, 2))
        grad[1] = np.sum(s * (1.0 - s) * (self.Y - s)
        / pow(self.true_sigma, 2))
        return grad

    def get_hesse(self):
        """
        ヘッセ行列を算出
        """
        hesse = np.zeros((2, 2))
        s = 1.0 / (1.0 + np.exp(-self.est_a * self.X
        - self.est_b))
        temp = s * (1.0 - s) * (self.Y - 2.0 * s * (self.Y + 1.0) + 3.0 * pow(s, 2))
        hesse[0][0] = np.sum(self.X * self.X * temp / pow(self.true_sigma, 2))
        hesse[0][1] = np.sum(self.X * temp / pow(self.true_sigma, 2))
        hesse[1][0] = hesse[0][1]
        hesse[1][1] = np.sum(temp / pow(self.true_sigma, 2))
        return hesse

    def steepest(self):
        """
        勾配上昇法を実施
        """
        counter = 0
        while True:
            counter += 1
            grad = self.get_grad()
            delta = self.eta * grad
            self.est_a += delta[0]
            self.est_b += delta[1]
            if sys.argv[1] == '0':
                self.log_a.append(self.est_a)
                self.log_b.append(self.est_b)
            flag = True
            for i in range(len(delta)):
                if np.abs(delta[i]) > self.cond:
                    flag = False
            if flag:
                break
        print((self.true_a-self.est_a, self.true_b-self.est_b)
        , counter)
        #1回だけ推定する場合は推定値の変化の様子をプロット
        if sys.argv[1] == '0':
            self.plot_params()
            plt.savefig("steepest_size"+str(self.N))
            plt.show()

    def newton(self):
        """
        ニュートンラフソン法を実施
        """
        counter = 0
        while True:
            counter += 1
            grad = self.get_grad()
            hesse = self.get_hesse()
            if np.linalg.det(hesse) != 0:
                inv_hesse = np.linalg.inv(hesse)
            flag = True
            delta = inv_hesse.dot(grad)
            for i in range(len(delta)):
                if(np.abs(delta[i] * self.eta_newton)
                > self.cond):
                    flag = False
            self.est_a -= delta[0] * self.eta_newton
            self.est_b -= delta[1] * self.eta_newton
            if sys.argv[1] == '0':
                self.log_a.append(self.est_a)
                self.log_b.append(self.est_b)
            if flag:
                break
        print((self.true_a-self.est_a, self.true_b-self.est_b)
        , counter)
        #1回だけ推定する場合は推定値の変化の様子をプロット
        if sys.argv[1] == '0':
            self.plot_params()
            plt.savefig("newton_size"+str(self.N))
            plt.show()

    def plot_params(self):
        """
        描画用の関数
        """
        p1, = plt.plot(np.arange(len(self.log_a)), self.log_a
        , marker='o')
        p2, = plt.plot(np.arange(len(self.log_b)), self.log_b
        , marker='o')
        plt.legend([p1, p2], ["est_a", "est_b"])
        plt.xlabel("times")
        plt.ylabel("value")

if __name__ == '__main__':
    #引数が0のときは推定を1回だけ行う
    if sys.argv[1] == '0':
        print('='*10 + 'ここから勾配上昇法' + '='*10)
        for i in [20, 50, 100]:
            print('='*20)
            ML = MaximumLikelihood(i)
            ML.logistic_reg()
            ML.steepest()
        print('='*10 + 'ここからニュートンラフソン法' + '='*10)
        for i in [20, 50, 100]:
            print('='*20)
            ML = MaximumLikelihood(i)
            ML.logistic_reg()
            ML.newton()

    #行数が1のときは10回推定を行い,推定平均時間を求める
    if sys.argv[1] == '1':
        print('='*10 + 'ここから勾配上昇法を各データサイズについて10回繰り返す' + '='*10)
        for i in [20, 50, 100]:
            start = time.clock()
            for j in range(10):
                print('='*20)
                ML = MaximumLikelihood(i)
                ML.logistic_reg()
                ML.steepest()
            end = time.clock()
            print("データサイズ{0}の推定平均時間は{1}"
            .format(i, (end-start)/10))
        print('='*10 + 'ここからニュートンラフソン法を各データサイズについて10回繰り返す' + '='*10)
        for i in [20, 50, 100]:
            start = time.clock()
            for j in range(10):
                print('='*20)
                ML = MaximumLikelihood(i)
                ML.logistic_reg()
                ML.newton()
            end = time.clock()
            print("データサイズ{0}の推定平均時間は{1}"
            .format(i, (end-start)/10))
