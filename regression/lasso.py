"""
Name: lasso.py
Date: 2017.12.21
Usage: python3 kadai3.py を端末上で入力.
Description:
    Boston Housing データを用いてデータ解析を行う.
    手法はLassoで,最小二乗誤差と回帰係数で評価する.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    #データを読み込む
    boston = load_boston()
    X = boston.data
    y = pd.DataFrame(boston.target, columns=['MEDV'])

    #データを標準化
    sc = StandardScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X), \
        columns=boston.feature_names)

    #学習データとテストデータに分ける
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.1, random_state=123)

    #lassoを学習
    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train, y_train)

    #lassoで予測
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)

    #結果を出力
    print('MSE train: {0:.3f}, MSE test: {0:.3f}'.format(\
        mean_squared_error(y_train, y_train_pred),\
            mean_squared_error(y_test, y_test_pred)))
    print('name:\tcoef')
    for i in range(len(X.columns)):
        print('{0}:\t{1:.3f}'.format(X.columns[i],\
            lasso.coef_[i]))
