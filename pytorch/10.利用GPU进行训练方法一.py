"""
@author:JuferBlue
@file:10.利用GPU进行训练方法一.py
@date:2024/7/20 23:57
@description:
"""
# 需要利用cuda的数据
# 网络模型
# 数据（输入，标注）
# 损失函数

# 使用方法1：.cuda()


import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from my_model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./CIFAR10_data", download=True, train=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="./CIFAR10_data", download=True, train=False,
                                         transform=torchvision.transforms.ToTensor())

# 查看两个数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 创建数据加载器
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 实例化网络
net = myNet()
if torch.cuda.is_available():
    net = net.cuda()

# 损失函数：交叉熵
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 定义优化器：随机梯度下降法
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0

# 设置训练网络的循环次数
epoch = 10

# 添加tensor_board
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("------第{}轮训练开始------".format(i + 1))
    # 训练步骤开始
    net.train()
    for data in train_dataloader:
        # 每次一批64张
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新调整参数

        total_train_step += 1

        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存每一轮的模型
    torch.save(net, "myNet_{}.pth".format(i))
    print("模型已保存")

writer.close()
