from surprise import KNNWithMeans
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import KFold

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./knn_cf/ratings.csv', reader=reader)
# trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})

# 定义k折交叉验证，k=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练与预测
    algo.fit(trainset)
    pred = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(pred, verbose=True)
    accuracy.mse(pred, verbose=True)

uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)
print(pred)


