"""
@author:JuferBlue
@file:7.现有模型的使用和修改.py
@date:2024/7/20 19:34
@description:
"""
import torchvision

import torch
import torchvision.models as models
from torch import nn

# 加载预训练的VGG16模型
vgg16_true = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# 或者如果你不想要预训练权重
vgg16_false = models.vgg16(weights=None)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(root="./CIFAR10_data", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

# 如何利用现有的网络进行修改
# 添加一个线性层
# vgg16_true.add_module("add_linear",nn.Linear(1000,10))
# print(vgg16_true)

# 在classifier中加一层
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)
print(vgg16_false)

# 修改指定层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
