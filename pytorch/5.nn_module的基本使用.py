"""
@author:JuferBlue
@file:5.nn_module的基本使用.py
@date:2024/7/19 8:53
@description:
"""

import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    # 定义了数据流经网络的前向传播逻辑
    def forward(self, input):
        output = input + 1;
        return output


net = Net()
x = torch.tensor(1.0)
# x传入到forward的input参数
output = net(x)
print(output)
