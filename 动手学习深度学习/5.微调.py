"""
@author:JuferBlue
@file:5.微调.py
@date:2024/8/26 16:32
@description: 微调ResNet18以适应CIFAR-10数据集
"""
# 导入相关的包
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# 指定训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对图片的预处理
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.RandomCrop((224, 224)),
    torchvision.transforms.ToTensor(),
    normalize
])

# 导入数据集
train_data = torchvision.datasets.CIFAR10(root='../数据集/cifar10', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='../数据集/cifar10', train=False, transform=transform, download=True)

# 保存数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 创建数据加载器
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

# 加载预训练模型
resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
# 修改最后一层以适应CIFAR-10的类别数
num_ftrs = resnet18.fc.in_features
resnet18.fc = torch.nn.Linear(num_ftrs, 10)
# 将模型移动到GPU上
resnet18 = resnet18.to(device)


# 定义优化器，为最后一层设置不同的学习率
# 优化器
learning_rate = 0.0005
optimizer = torch.optim.SGD(resnet18.parameters(), lr=learning_rate)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 设置训练网络的一些参数
epoch = 5  # 设置训练的轮数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数

# 开始训练
for i in range(epoch):
    print("------第{}轮训练开始------".format(i + 1))
    resnet18.train()  # 确保模型处于训练模式
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = resnet18(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))

    # 测试过程
    resnet18.eval()  # 确保模型处于评估模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 关闭梯度计算
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = resnet18(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum().item()
            total_accuracy += accuracy

    # 输出一个轮次的信息
    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率:{total_accuracy / test_data_size}")

