"""
@author:JuferBlue
@file:utils.py
@date:2024/9/22 7:56
@description:
"""
from PIL import Image
from torchvision import transforms
import numpy as np
def keep_image_size_open(path,size=(256,256)):
    from PIL import Image  # 导入PIL库中的Image模块
    # 打开指定路径的图像文件
    img = Image.open(path)
    # 获取图像的最大尺寸（宽度和高度中的较大值）
    temp = max(img.size)
    # 创建一个新的RGB模式的正方形空白图像，背景为黑色
    mask = Image.new(img.mode, (temp, temp), (0, 0, 0))
    # 将原始图像粘贴到新图像的左上角
    mask.paste(img, (0, 0))
    # 将新图像调整为目标尺寸
    mask = mask.resize(size)
    return mask


palette = [
    (0, 0, 0),       # 类别 0 - 背景
    (128, 0, 0),     # 类别 1
    (0, 128, 0),     # 类别 2
    (128, 128, 0),   # 类别 3
    (0, 0, 128),     # 类别 4
    (128, 0, 128),   # 类别 5
    (0, 128, 128),   # 类别 6
    (128, 128, 128), # 类别 7
    (64, 0, 0),      # 类别 8
    (192, 0, 0),     # 类别 9
    (64, 128, 0),    # 类别 10
    (192, 128, 0),   # 类别 11
    (64, 0, 128),    # 类别 12
    (192, 0, 128),   # 类别 13
    (64, 128, 128),  # 类别 14
    (192, 128, 128), # 类别 15
    (0, 64, 0),      # 类别 16
    (128, 64, 0),    # 类别 17
    (0, 192, 0),     # 类别 18
    (128, 192, 0),   # 类别 19
    (0, 64, 128)     # 类别 20
]


# 将单通道类别图像转换为彩色图像
def apply_color_map(output_img):
    # 将 output_img 从 tensor 转为 numpy
    output_img = output_img.cpu().numpy().astype(np.uint8)

    # 创建一个 RGB 图像
    color_img = np.zeros((output_img.shape[0], output_img.shape[1], 3), dtype=np.uint8)

    # 根据类别值，应用调色板中的颜色
    for class_id, color in enumerate(palette):
        color_img[output_img == class_id] = color

    return color_img

if __name__ == '__main__':
    mask = keep_image_size_open('img.png')
    # 使用 ToTensor 将 PIL 图像转换为 Tensor
    mask_tensor = transforms.ToTensor()(mask)  # 这里要使用 () 来应用函数
    print(mask_tensor.shape)