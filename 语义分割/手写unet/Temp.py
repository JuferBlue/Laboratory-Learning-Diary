"""
@author:JuferBlue
@file:Temp.py
@date:2024/9/23 8:10
@description:
"""
import matplotlib.pyplot as plt
import numpy as np

# 假设 pred_mask 是你的预测结果，维度为 (H, W)
# 例如，这里随机生成一个 256x256 大小的预测输出，类别值为 0, 1, 2, 3。
pred_mask = np.random.randint(0, 4, (256, 256))

# 伪彩色显示
plt.imshow(pred_mask, cmap='jet')  # 使用伪彩色显示类别
plt.title('Predicted Segmentation Mask')
plt.colorbar()  # 添加颜色条，显示类别映射
plt.show()

# 如果你想用灰度显示，可以使用：
plt.imshow(pred_mask, cmap='gray')  # 使用灰度显示类别
plt.title('Predicted Segmentation Mask (Gray)')
plt.colorbar()  # 显示灰度图的颜色条
plt.show()
