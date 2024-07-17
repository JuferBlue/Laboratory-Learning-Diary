"""
@author:JuferBlue
@file:六.数组的函数.py
@date:2024/7/17 18:51
@description:
"""
import numpy as np

# 矩阵乘积
arr1 = np.arange(5)
arr2 = np.arange(5)
print(np.dot(arr1, arr2))

# 数学函数
arr_v = np.array([-1, 0, 1])
# 绝对值
abs_v = np.abs(arr_v)
print(abs_v)

# 三角函数：.sin()  .cos()  .tan()

# 指数函数：.exp()

# 对数函数：.log()  .log10()

# 聚合函数
# max min
arr3 = np.array([1, 2, 3, 4, 5])
print(arr3.max())

# sum prod
print(np.sum(arr3, axis=0))  # 维度一求和
print(np.sum(arr3, axis=1))  # 维度二求和

# 均值函数mean() 标准差函数std()
print(arr3.mean())
print(arr3.std())
