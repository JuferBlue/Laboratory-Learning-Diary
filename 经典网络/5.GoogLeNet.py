# 导入相关的包
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import torch.nn.functional as F

# 指定训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 对图片的预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
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
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 定义Inception模块
class Inception(nn.Module):
    # c1,c2,c3,c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__()
        # 路径1：1x1卷积
        self.path1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 路径2：1x1卷积+3x3卷积
        self.path2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.path2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 路径3：1x1卷积+5x5卷积
        self.path3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.path3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 路径4：3x3最大池化+1x1卷积
        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        path1 = F.relu(self.path1(x))  # 添加F.relu进行激活
        path2 = F.relu(self.path2_2(F.relu(self.path2_1(x))))  # 嵌套路径需要嵌套激活
        path3 = F.relu(self.path3_2(F.relu(self.path3_1(x))))
        path4 = F.relu(self.path4_2(self.path4_1(x)))  # 注意：path4_1是池化层，通常不跟激活函数
        return torch.cat([path1, path2, path3, path4], dim=1)


# 实现每一个stage
b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)
googlenet = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
googlenet.to(device)

# 观察每一层输出的形状
x = torch.randn(size=(1, 3, 224, 224), device=device)
for blk in googlenet:
    x = blk(x)
    print(blk.__class__.__name__, 'output shape: \t\t', x.shape)


# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.1
optimizer = torch.optim.SGD(googlenet.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

# 设置训练网络的循环次数
epoch = 30

# 开始训练
for i in range(epoch):
    # 训练过程
    print("------第{}轮训练开始------".format(i + 1))
    googlenet.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = googlenet(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))

    # 测试过程
    googlenet.eval()
    with torch.no_grad():
        total_test_loss = 0
        total_accuracy = 0
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = googlenet(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()
            total_test_step += 1

    # 输出一个轮次的信息
    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率:{total_accuracy / test_data_size}")
    total_test_step += 1

    # 创建文件夹用来保存模型
    dir_path = os.path.join(".", "模型保存", "5-GoogLeNet")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 保存每一轮的模型
    torch.save(googlenet.state_dict(), f"./模型保存/5-GoogLeNet/googlenet{i}.pth")
    print("模型保存成功")
