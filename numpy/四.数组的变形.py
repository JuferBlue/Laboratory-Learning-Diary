"""
@author:JuferBlue
@file:四.数组的变形.py
@date:2024/7/17 16:26
@description:
"""

import numpy as np


# 向量的转置
arr1 = np.arange(1, 10) # 直接创建出来的是向量,需要转为矩阵才能转
arr1 = arr1.reshape(1,-1)
# .T 表示转置
arr1_T = arr1.T
print(arr1_T)

#矩阵的转置：直接.T


# 数组翻转
# 向量翻转
arr2 = np.arange(1, 10)
print(arr2)
arr2 = np.flipud(arr2)
print(arr2)
# 矩阵翻转
arr3 = np.arange(1,21).reshape(4,5)
print(arr3)
# 矩阵左右翻转
arr3_lr = np.fliplr(arr3)
print(arr3_lr)
# 矩阵上下翻转
arr3_ud = np.flipud(arr3)
print(arr3_ud)


# 数组的重塑
# .reshape()方法

# 数组的拼接
arr4 = np.array([1,2,3,4])
arr5 = np.array([5,6,7,8])
arr6 = np.concatenate((arr4,arr5))
print(arr6)
#矩阵拼接
arr7 = np.array([[1,2,3,4],[5,6,7,8]])
arr8 = np.array([[9,10,11,12],[13,14,15,16]])
#按第一个维度拼接,默认参数axis=0
arr9 = np.concatenate((arr7,arr8))
print(arr9)
#按第二个维度拼接
arr10 = np.concatenate((arr7,arr8),axis=1)
print(arr10)

# 数组的分裂
arr11 = np.arange(10,100,10)
arrs1,arrs2,arrs3 = np.split(arr11,[2,8]) # 表示在索引[2]和索引[8]的位置截断
print(arrs1)
print(arrs2)
print(arrs3)
# 矩阵的分裂
arr12 = np.arange(1,9,).reshape(2,4)
print(arr12)
# 按第一个维度分裂-行
arrs4,arrs5 = np.split(arr12,[1])
print(arrs4)
print(arrs5)
# 按第二个维度分裂-列
arrs6,arrs7,arrs8 = np.split(arr12,[1,3],axis=1)
print(arrs6)
print(arrs7)
print(arrs8)

