"""
@author:JuferBlue
@file:一.数组基础.py
@date:2024/7/17 15:25
@description:
"""

import numpy as np

# 创建整数组
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1) # [1 2 3 4 5]

# 创建浮点数组
# 数组里面有一个是浮点数则整个数组都是浮点数
arr2 = np.array([1.2, 2, 3, 4, 5])
print(arr2) # [1. 2. 3. 4. 5.]


# 同化定理
# 往整数型数组里插入浮点数，该浮点数会自动被截断为整数
# 往浮点型数组里插入整数，该整数会自动升级为浮点数；
arr3 = np.array([1, 2, 3, 4, 5])
arr3[0] = 100.99
print(arr3) # [100 2 3 4 5]

# 共同改变定理
# 数组中数据类型的改变会导致全体共同改变
# 规范改变：.astype()
arr4 = np.array([1, 2, 3, 4, 5])
print(arr4) # [1 2 3 4 5]
arr5 = arr4.astype(np.float64)
print(arr5) # [1. 2. 3. 4. 5.]
# 运算时改变
arr6 = np.array([1, 2, 3, 4, 5])
print(arr6+1.0) # [2. 3. 4. 5. 6.]

# 一维数组和二维数组
# .ones(): 创建一个全为1的数组
arr7 = np.ones(3)
print(arr7) # [1. 1. 1.]
arr8 = np.ones((1, 3)) #(行,列)
print(arr8) # [[1. 1. 1.]]
arr9 = np.ones((2,3,4)) #(维度,行,列)
print(arr9) # [[[1. 1. 1.]]]
# .shape: 查看数组的形状
print(arr7.shape) # (3,)
print(arr8.shape) # (1, 3)
print(arr9.shape) # (1, 1, 3)

#不同数字组之间的转换
arr10 = np.arange(1,10)
print(arr10) # [1 2 3 4 5 6 7 8 9]
# 转换成二维数组:.reshape()
arr11 = arr10.reshape((3,-1)) # 1表示行，-1表示列数自己计算。计算出小数会报错
print(arr11)





