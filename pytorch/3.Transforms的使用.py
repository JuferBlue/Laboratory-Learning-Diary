"""
@author:JuferBlue
@file:3.Transforms的使用.py
@date:2024/7/18 20:45
@description:
"""

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "./img.png"
img = Image.open(img_path)

writer = SummaryWriter("logs")
# 将图片转为张量
# 会将图片的像素值从0-255转为0-1
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("toTensor", img_tensor)

# normalize图片
# 统一数据分布：使不同图像或不同通道的数据分布接近正态分布，这有助于模型的学习和收敛。
# 加速训练：通过减少数据中的尺度差异，归一化可以帮助优化算法更快地找到全局最小值。
# 避免梯度消失或爆炸：通过将数据缩放到相似的尺度，可以避免因数据尺度过大或过小而导致的梯度消失或爆炸问题。
# 提高模型泛化能力：归一化后的数据更接近于模型期望的输入范围，这有助于模型在未见过的数据上表现得更好。
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)
# # ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

# resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
print(img_resize.shape)
writer.add_image("Resize", img_resize, 0)
#
# Compose resize 等比缩放
# compose 执行流水作业
trans_resize_2 = transforms.Resize(512)
# 先resize在toTensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)
#
# randomcrop随机裁剪
trans_random = transforms.RandomCrop(50)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
