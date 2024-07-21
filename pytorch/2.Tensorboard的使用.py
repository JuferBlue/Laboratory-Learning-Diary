"""
@author:JuferBlue
@file:2.Tensorboard的使用.py
@date:2024/7/18 20:40
@description:
"""

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 首先需要new一个SummaryWriter对象：
writer = SummaryWriter(log_dir="logs")

# 添加内容
# 1-添加折线图：add_scalar 标题 y x
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

# 2-添加图片：add_image
image_path = './img.png'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)
writer.add_image('test_img', img_array, 1, dataformats='HWC')  # 必须传入numpy或者tensor数据类型

writer.close()
