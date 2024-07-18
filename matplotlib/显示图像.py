"""
@author:JuferBlue
@file:显示图像.py
@date:2024/7/17 22:31
@description:
"""
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./lena.jpeg')
plt.imshow(img)
plt.show()
