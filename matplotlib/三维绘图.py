"""
@author:JuferBlue
@file:三维绘图.py
@date:2024/7/18 10:52
@description:
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义函数 f(x, y) = x^2 + y^2
def f(x, y):
    return x ** 2 + y ** 2


# 创建网格数据
x = np.linspace(-5, 5, 100)
print(x)
y = np.linspace(-5, 5, 100)
print(y)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 点 (1, 1) 处的偏导数
x0, y0 = 1, 1
z0 = f(x0, y0)
df_dx = 2 * x0
df_dy = 2 * y0


# 切平面方程
def tangent_plane(x, y, x0, y0, z0, df_dx, df_dy):
    return z0 + df_dx * (x - x0) + df_dy * (y - y0)


# 生成切平面数据
T = tangent_plane(X, Y, x0, y0, z0, df_dx, df_dy)

# 绘图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制函数表面
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# 绘制切平面
ax.plot_surface(X, Y, T, color='red', alpha=0.5)

# 绘制点 (1, 1) 处的点
ax.scatter(x0, y0, z0, color='black', s=100, label='Point (1, 1)')

# 设置标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Function Surface and Tangent Plane')

plt.legend()
plt.show()
