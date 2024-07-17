"""
@author:JuferBlue
@file:二.数组的创建.py
@date:2024/7/17 16:00
@description:
"""
import numpy as np

# .array():已知具体数值创建
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

# .arange():创建等差数列
arr2 = np.arange(1, 10, 2)
print(arr2)

# .zeros():创建全零数组
# .ones():创建全一数组
arr3 = np.zeros(5)
arr4 = np.ones(5)
print(arr3)
print(arr4)

# .random.random():0-1均匀分布的浮点型随机数组
arr5 = np.random.random(5)
print(arr5)

# .random.randint():创建指定范围整数数组
arr6 = np.random.randint(1, 10, 5) # 第三个参数为形状
print(arr6)

# .random.normal()：服从正态分布的随机数组
arr7 = np.random.normal(0, 1, 5)
print(arr7)
