"""
@author:JuferBlue
@file:三.数组的索引.py
@date:2024/7/17 16:11
@description:
"""

import numpy as np

# 访问一维
arr1 = np.arange(1,10)
print(arr1) # [1 2 3 4 5 6 7 8 9]
print(arr1[3]) # 正向访问
print(arr1[-3]) # 负向访问
# 修改数组元素
arr1[3] = 100
print(arr1)

# 访问矩阵
arr2 = np.arange(1,10).reshape(3,3)
print(arr2)
#访问元素
print(arr2[0,2])
#修改元素
arr2[0,2] = 100
print(arr2)


# 花式索引
arr3 = np.arange(0,90,10)
print(arr3)
print(arr3[[2,5,8]])
# 矩阵的花式索引
arr4 = np.arange(0,17).reshape(4,4)
print(arr4)
print(arr4[[0,1,2],[0,1,2]]) # [0,0]和[1,1]和[2,2]



# 访问数组的切片:同列表切片类似
