"""
@author:JuferBlue
@file:utils.py
@date:2024/9/22 7:56
@description:
"""
from PIL import Image


def keep_image_size_open(path,size=(256,256)):
    from PIL import Image  # 导入PIL库中的Image模块
    # 打开指定路径的图像文件
    img = Image.open(path)
    # 获取图像的最大尺寸（宽度和高度中的较大值）
    temp = max(img.size)
    # 创建一个新的RGB模式的正方形空白图像，背景为黑色
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    # 将原始图像粘贴到新图像的左上角
    mask.paste(img, (0, 0))
    # 将新图像调整为目标尺寸
    mask = mask.resize(size)
    return mask