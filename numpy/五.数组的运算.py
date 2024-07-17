"""
@author:JuferBlue
@file:五.数组的运算.py
@date:2024/7/17 16:52
@description:
"""

import numpy as np

# 数组与系数之间的运算
# 加减乘除   幂  取整取余
a = np.array([1, 2, 3])
print(a + 1)


# 数组与数组之间的运算
# 加法 减法

# 乘法-对应位置上面的数相乘
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a * b) # [4, 10, 18]
# 除法和幂方都是对应位置相乘


# 广播 - 不同形状数组之间的运算
# 如果是向量与矩阵之间做运算，向量自动升级为行矩阵；
# 如果是向量与矩阵之间做运算，向量自动升级为行矩阵；

# 向量广播
a = np.array([1, 1, 1])
b = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
print(a + b)

# 列矩阵广播
arr1 = np.arange(3).reshape(3, 1)
print(arr1)
arr2 = np.ones((3, 5))
print(arr2)
print(arr1 * arr2)
