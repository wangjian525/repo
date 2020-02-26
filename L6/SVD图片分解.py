from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# 取前k个特征，对图像进行还原
def get_image_feature(s, k):
    # 对于S, 只保留前k个特征值
    s_temp = np.zeros(s.shape[0])  # 返回给定行状和类型的用0填充的数组  s_temp.shape:(1080,)
    s_temp[0: k] = s[0: k]
    # 一维数组s_temp*多维数组np.identity，则np.identity每行元素分别对应乘s_temp的每行元素
    s = s_temp * np.identity(s.shape[0])  # np.identity(n),输入 n*n 的单位方阵  s.shape:1080*1080
    # 用新的s_temp 以及p, q重构A
    # s为构造的以特征值为主对角线的矩阵
    temp = np.dot(p, s)
    temp = np.dot(temp, q)
    plt.imshow(temp, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()


# 加载256色图片
image = Image.open('./256.bmp')
A = np.array(image)
# 显示原图像
plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
# 对图像矩阵A进行奇异值分解，得到p, s, q  p:m*m   q: n*n   s: (1080,）
p, s, q = svd(A, full_matrices=False)
# 取前k个特征，对图像进行还原
get_image_feature(s, 5)
get_image_feature(s, 50)
get_image_feature(s, 500)
