"""
@author:JuferBlue
@file:featureMap.py
@date:2024/9/6 15:25
@description:
"""
# 导入相关的包
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os

# 利用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    ToTensor()
])
# 准备数据
train_data = torchvision.datasets.MNIST(root="../数据集/mnist", train=True, transform=transform,
                                        download=True)
test_data = torchvision.datasets.MNIST(root="../数据集/mnist", train=False, transform=transform,
                                       download=True)

# 查看数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集长度为{train_data_size}")
print(f"测试数据集长度为{test_data_size}")

# 创建数据加载器
train_dataloader = DataLoader(train_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
# 实例化网络
lenet = LeNet()
lenet = lenet.to(device)

# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
lenet.apply(init_weights)