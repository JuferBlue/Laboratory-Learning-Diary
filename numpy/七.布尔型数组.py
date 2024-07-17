"""
@author:JuferBlue
@file:七.布尔型数组.py
@date:2024/7/17 19:27
@description:
"""

import numpy as np

arr = np.arange(1, 7).reshape(2, 3)
print(arr)

# 数组与数字作比较
print(arr >= 4)

# 数组与数组比较
arr1 = np.array([1, 4, 5, 8])
arr2 = np.array([1, 5, 6, 8])
print(arr1 >= arr2)

# 布尔数组中True的数量
# np.sum()统计布尔型数组中True的个数
arr3 = np.random.normal(0, 1, 10000)
num = np.sum(np.abs(arr3) < 1)
print(num)
# np.any():只要布尔型数组中有一个及其以上的True 就返回True
arr4 = np.arange(1, 10)
arr5 = np.arange(9, 0, -1)
print(arr4)
print(arr5)
print(np.any(arr4 == arr5))
# np.all():当布尔型数组里全是 True 时，才返回 True

# 满足条件的元素所在位置
# np.where()
arr = np.random.normal(500, 70, 10000)
print(np.where(arr > 650))
