"""
@author:JuferBlue
@file:绘制简单图像.py
@date:2024/7/17 22:15
@description:
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label='sin')
# supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.plot(x, y2, linestyle='--', label='cos')  # 设置线型
plt.xlabel('x')  # 添加x轴标签
plt.ylabel('y')  # 添加y轴标签
plt.title('sin & cos')  # 添加标题
plt.legend()  # 显示图例
plt.show()
