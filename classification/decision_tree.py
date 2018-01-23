"""
Name: decision_tree.py
Date: 2017.8.13
Usage: python3 decision_tree.py
Description:
    情報利得により分類を行う決定木.
    連続値を属性として持つデータを2クラスに分類する.
    データから生成された決定木を正解率で評価する.
"""

import numpy as np
import pandas as pd

def make_dat(size, mu, cov):
    """
    平均mu,共分散covの正規分布に従うsize個のデータを生成
    """
    return np.random.multivariate_normal(mu, cov, size)

def accuracy(true_cls, pred_cls):
    """
    正解率の計算
    """
    return sum((true_cls == pred_cls)==True) / len(true_cls)

class Node:
    """
    ノードクラス
    """
    def __init__(self):
        """
        Nodeの情報を初期化する
        """
        #子ノードのリスト
        self.left = None
        self.right = None
        #分割する属性の番号
        self.attribute = None
        #分割する属性の取りうる値
        self.threshold = None
        #クラス番号
        self.label = None

    def build_tree(self, data, target):
        """
        木の構築を行う.
        data: ノードに与えられたデータ
        target: データのクラス
        """

        print("="*30)

        #全てのデータが同じクラスのとき
        if len(np.unique(target)) <= 1:
            #クラスをラベル付けして終了
            self.label = target[0]
            print("all data have a same class")
            return

        #このノードのクラスを設定
        class_cnt = {i: len(target[target==i]) for i in np.unique(target)}
        self.label= max(class_cnt.items(), key=lambda x:x[1])[0]
        #attribute_listが空集合(データがない)のとき
        print("the number of atrribute is",len(data.columns))
        if len(data.columns) == 0:
            return

        #最も良い分割を初期化
        best_gain_ratio = -np.inf
        best_attr = None
        best_threshould = None

        #利得率最大となる属性を選択
        for i in range(len(data.ix[0,:])):
            if len(np.unique(data.ix[:,i])) != 1:
                sub_gain_ratio = self.gain_ratio_func(i, data, target)
                if sub_gain_ratio > best_gain_ratio:
                    best_gain_ratio = sub_gain_ratio
                    best_attr = i
        self.attribute = data.columns[best_attr]

        print("gain ratio is",best_gain_ratio)
        print("devision attribute is",self.attribute)

        #分割のしきい値を求める
        self.threshold = (max(data.ix[:,best_attr])+min(data.ix[:,best_attr]))/2

        #それぞれの子ノードについて再帰
        data_left = data.ix[data.ix[:,best_attr]<=self.threshold]
        data_left = data_left.drop(self.attribute, axis=1)
        data_left.index = range(len(data_left))
        tar_left = target.ix[data.ix[:,best_attr]<=self.threshold]
        tar_left.index = range(len(tar_left))
        self.left = Node()
        self.left.build_tree(data_left, tar_left)
        data_right = data.ix[data.ix[:,best_attr]>self.threshold]
        data_right = data_right.drop(self.attribute, axis=1)
        data_right.index = range(len(data_right))
        tar_right = target.ix[data.ix[:,best_attr]>self.threshold]
        tar_right.index = range(len(tar_right))
        self.right = Node()
        self.right.build_tree(data_right, tar_right)

    def info_func(self, target):
        """
        データDの情報量
        """
        num_yes = len(target.ix[target==1])
        num_no = len(target.ix[target==0])
        if num_yes==0 or num_no==0:
            return 0
        yes = (-num_yes / len(target)) * np.log2(num_yes/len(target))
        no = (-num_no / len(target)) * np.log2(num_no/len(target))
        return yes + no

    def sub_info_func(self, attr, data, target):
        """
        属性xによるデータDの情報量
        """
        sub_info = 0
        for i in set(data.ix[:,attr]):
            coef = len(data.ix[data.ix[:,attr]==i]) / len(data)
            sub_info += coef * self.info_func(target.ix[data.ix[:,attr]==i])
        return sub_info

    def gain_func(self, attr, data, target):
        """
        属性xによる情報利得
        """
        return self.info_func(target) - self.sub_info_func(attr, data, target)

    def splitinfo_func(self, attr, data):
        """
        属性xの分割によるDの情報量
        """
        split_info = 0
        for i in set(data.ix[:,attr]):
            coef = len(data.ix[data.ix[:,attr]==i]) / len(data)
            split_info += -coef * np.log2(coef)
        return split_info

    def gain_ratio_func(self, attr, data, target):
        """
        属性xによる利得率
        """
        split_info = self.splitinfo_func(attr, data)
        return self.gain_func(attr,data,target)/split_info

    def predict(self, a_dat):
        """
        ある1つのデータa_datはどのクラスに分類されるか予測する
        """
        if self.attribute != None:
            if a_dat[self.attribute] <= self.threshold:
                return self.left.predict(a_dat)
            else:
                return self.right.predict(a_dat)
        else:
            return self.label

class DecisionTree:
    """
    決定木クラス.
    """
    def __init__(self):
        """
        初期化
        """
        self.root = None

    def fit(self, data, target):
        """
        フィッティング
        """
        self.root = Node()
        self.root.build_tree(data, target)
        pass

    def predict(self, data):
        """
        あるデータセットdataの各データをクラスに分類する
        """
        pred_cls = []
        for i in range(len(data)):
            pred_cls.append(self.root.predict(data.ix[i,:]))
        return np.array(pred_cls)

if __name__ == '__main__':
    #データ生成
    cov = np.random.randint(1, 5, (5, 5))
    mu1 = [1, 1, 1, 1, 1]
    mu2 = [-1, -1, -1, -1, -1]
    cls1 = make_dat(50, mu1, cov)
    cls2 = make_dat(50, mu2, cov)
    X = pd.DataFrame(np.vstack((cls1, cls2)),columns=list('ABCDE'))
    Y = [1 for i in range(len(cls1))]
    for i in range(len(cls2)):
        Y.append(0)
    Y = pd.Series(Y)

    #決定木を学習する
    tree = DecisionTree()
    tree.fit(X, Y)
    #クラスに分類する
    pred_cls = tree.predict(X)
    #正解率と誤り率
    acc = accuracy(Y, pred_cls)
    print("="*40, end='\n\n')
    print("the accuracy is", acc)
    print("the error rate is", 1-acc)
