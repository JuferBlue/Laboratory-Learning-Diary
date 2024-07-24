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
train_dataloader = DataLoader(train_data, batch_size=256)
test_dataloader = DataLoader(test_data, batch_size=256)


# 定义一个残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y+=X
        return F.relu(Y)




# 定义残差网络
# 开始部分和googlenet一样
b1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
resnet = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(512, 10))
resnet.to(device)


# 观察每一层输出的形状
x = torch.randn(size=(1, 3, 224, 224), device=device)
for blk in resnet:
    x = blk(x)
    print(blk.__class__.__name__, 'output shape: \t\t', x.shape)



# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
resnet.apply(init_weights)


# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.05
optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate)

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
    resnet.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = resnet(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))

    # 测试过程
    resnet.eval()
    with torch.no_grad():
        total_test_loss = 0
        total_accuracy = 0
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = resnet(imgs)
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
    dir_path = os.path.join(".", "模型保存", "6-ResNet")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 保存每一轮的模型
    torch.save(resnet.state_dict(), f"./模型保存/6-ResNet/resnet{i}.pth")
    print("模型保存成功")
