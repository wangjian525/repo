# !pip install surprise
from surprise import SVD,SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
import time

from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive/Colab Notebooks/L6 MF/')
time1 = time.time()
# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)
train_set = data.build_full_trainset()

# 使用funkSVD
def funk():
    algo = SVD(biased=False)

    # 定义K折交叉验证迭代器，k=3
    kf = KFold(n_splits=3)
    for trainset, testset in kf.split(data):
        # 训练并预测
        algo.fit(trainset)
        predictions = algo.test(testset)
        # 计算RMSE
        accuracy.rmse(predictions, verbose=True) # verbose 输出当前跌代，默认False

    uid = str(196)
    iid = str(302)
    # 输出uid对iid的预测结果
    pred = algo.predict(uid, iid, r_ui=4, verbose=True)

    time2 = time.time()
    print(time2-time1)

def bais():
    algo = SVD(biased=Ture)

    # 定义K折交叉验证迭代器，k=3
    kf = KFold(n_splits=3)
    for trainset, testset in kf.split(data):
        # 训练并预测
        algo.fit(trainset)
        predictions = algo.test(testset)
        # 计算RMSE
        accuracy.rmse(predictions, verbose=True) # verbose 输出当前跌代，默认False

    uid = str(196)
    iid = str(302)
    # 输出uid对iid的预测结果
    pred = algo.predict(uid, iid, r_ui=4, verbose=True)

    time2 = time.time()
    print(time2-time1)

def SVD_pp():
    algo = SVDpp()

    # 定义K折交叉验证迭代器，k=3
    kf = KFold(n_splits=3)
    for trainset, testset in kf.split(data):
        # 训练并预测
        algo.fit(trainset)
        predictions = algo.test(testset)
        # 计算RMSE
        accuracy.rmse(predictions, verbose=True) # verbose 输出当前跌代，默认False

    uid = str(196)
    iid = str(302)
    # 输出uid对iid的预测结果
    pred = algo.predict(uid, iid, r_ui=4, verbose=True)

    time2 = time.time()
    print(time2-time1)
funk()
bais()
SVD_pp()
