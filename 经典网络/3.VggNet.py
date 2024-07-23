# 导入相关的包
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os

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

# 定义vgg块
# num_convs：卷积层的个数
# in_channels：输入通道数
# out_channels：输出通道数
def vgg_block(num_convs, in_channels, out_channels):
    layers = [] # 创建一个列表，用于存放vgg块中的卷积层和池化层
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# vgg块的参数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# vgg模型
def vgg(conv_arch):
    conv_blocks = []  # 创建一个列表，用于存放vgg块
    in_channels = 3  # 输入通道数
    # 遍历conv_arch，创建vgg块
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blocks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )


vgg11 = vgg(conv_arch)
vgg11 = vgg11.to(device)

# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
vgg11.apply(init_weights)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.05
optimizer = torch.optim.SGD(vgg11.parameters(), lr=learning_rate)

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
    vgg11.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = vgg11(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))

    # 测试过程
    vgg11.eval()
    with torch.no_grad():
        total_test_loss = 0
        total_accuracy = 0
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = vgg11(imgs)
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
    dir_path = os.path.join(".", "模型保存", "3-VggNet")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 保存每一轮的模型
    torch.save(vgg11.state_dict(), f"./模型保存/3-VggNet/vggnet{i}.pth")
    print("模型保存成功")
