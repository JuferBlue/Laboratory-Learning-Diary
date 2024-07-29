# 导入相关的包
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

# 对图片的数据进行预处理
transform = torchvision.transforms.Compose([
    ToTensor(),
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    # torchvision.transforms.RandomResizedCrop(
    #     size=(224,224),
    #     scale=(0.1, 1.0),
    #     ratio=(0.5, 2)
    # ),
    # torchvision.transforms.ColorJitter(
    #     brightness=0.5, # 亮度  0.5表示在原图像的亮度上改变50%
    #     contrast=0.5, # 对比度
    #     saturation=0.5, # 饱和度
    #     hue=0.5 # 色调
    # ),
    torchvision.transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # 每个通道的均值
        std=[0.2023, 0.1994, 0.2010]   # 每个通道的标准差
    )
])
# 自定义模型
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
# b1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
#                    nn.BatchNorm2d(64),
#                    nn.ReLU(),
#                    nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *resnet_block(64, 64, 2, first_block=True),
            *resnet_block(64, 128, 2),
            *resnet_block(128, 256, 2),
            *resnet_block(256, 512, 2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.model(x)

# 数据集选择
# eg:dataset = "CIFAR10"
dataset_name = "CIFAR10"
# 训练集的批次大小
# eg:batch_size = 64
batch_size = 128
# 学习率的大小
# eg:learning_rate = 0.01
learning_rate = 0.05
# 训练的轮数
# eg:epoch = 10
epoch = 15
# 模型名称
# eg:model_name = "myNet"
model_name = "myResNet18"
# 模型
# eg:net_model = LeNet
net_model = ResNet18


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def train_my_model(dataset_name, transform, batch_size, net_model, learning_rate, epoch, model_name):
    # 指定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 动态获取数据集类
    dataset_class = getattr(torchvision.datasets, dataset_name)
    # 训练集
    train_data = dataset_class(root='../数据集/{}'.format(dataset_name.lower()), train=True, transform=transform,
                               download=True)
    # 测试集
    test_data = dataset_class(root='../数据集/{}'.format(dataset_name.lower()), train=False, transform=transform,
                              download=True)

    # 保存数据集的长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集的样本数为：{}".format(train_data_size))
    print("测试集的样本数为：{}".format(test_data_size))

    # 创建数据加载器
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # 实例化网络模型
    net = net_model()
    net.to(device)

    # 初始化模型参数

    net.apply(init_weights)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # 优化器
    learn_rate = learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)

    # 设置训练网络的一些记录数
    # 记录已训练的总次数
    total_train_step = 0
    # 记录已训练的总样本数
    total_train_data = 0
    # 记录测试的次数
    total_test_step = 0
    # 记录训练时间
    total_train_time = 0
    # 收集每轮训练 的平均loss用于绘图
    train_loss_map = []
    # 收集每轮训练的正确率用于绘图
    train_accuracy_map = []
    # 收集每轮的测试的loss用于绘图
    test_loss_map = []
    # 收集每轮测试的正确率用于绘图
    test_accuracy_map = []

    # 设置训练网络的循环次数
    train_epoch = epoch

    # 开始训练
    for i in range(train_epoch):
        print(f"----------------------第{i + 1}轮训练开始----------------------")
        # 开始计时
        start_time = time.time()
        # 训练步骤开始
        net.train()
        # 记录一轮训练的次数
        epoch_train_step = 0
        # 记录一轮训练的总误差
        epoch_train_loss = 0
        # 记录一轮训练正确的样本数
        epoch_train_right_data = 0
        for data in train_dataloader:
            # 计算损失值
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            outputs = net(images)
            loss = loss_fn(outputs, targets)
            # 反向传播更新梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新记录值
            total_train_step += 1  # 总训练次数加1
            epoch_train_step += 1  # 单轮训练次数加1
            total_train_data += batch_size  # 总训练样本数加上批数
            epoch_train_loss += loss.item()  # 单轮训练总误差更新
            accuracy = (outputs.argmax(1) == targets).sum().item()  # 一个批次训练正确的样本数
            epoch_train_right_data += accuracy  # 一轮训练正确的样本数更新
            if total_train_step % 100 == 0:
                print("训练次数：{}|训练样本数：{}|单次训练误差(loss):{}".format(total_train_step, total_train_data,
                                                                               loss.item()))
        # 计算一轮训练的平均loss
        epoch_train_average_loss = epoch_train_loss / epoch_train_step
        train_loss_map.append(epoch_train_average_loss)  # 将数据加入到列表中用于绘图
        # 计算一轮训练的正确率
        epoch_train_accuracy = epoch_train_right_data / train_data_size
        train_accuracy_map.append(epoch_train_accuracy)

        # 测试步骤开始
        net.eval()
        with torch.no_grad():
            # 一轮测试的总误差
            epoch_test_loss = 0
            # 一轮测试的次数
            epoch_test_step = 0
            # 一轮测试的正确次数
            epoch_test_right_data = 0
            for data in test_dataloader:
                # 计算损失值
                images, targets = data
                images = images.to(device)
                targets = targets.to(device)
                outputs = net(images)
                loss = loss_fn(outputs, targets)
                # 更新记录值
                epoch_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum().item()
                epoch_test_right_data += accuracy
                epoch_test_step += 1
        total_test_step += 1
        # 结束计时
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_train_time += elapsed_time
        print(f"本轮训练时间:{elapsed_time:.4f}s")
        print(f"累积训练时间:{total_train_time:.4f}s")

        # 计算一轮测试的平均loss
        epoch_test_average_loss = epoch_test_loss / epoch_test_step
        test_loss_map.append(epoch_test_average_loss)  # 将数据加入到列表中用于绘图
        # 计算一轮测试的正确率
        epoch_test_accuracy = epoch_test_right_data / test_data_size
        test_accuracy_map.append(epoch_test_accuracy)

        # 输出一个轮次的信息
        # 训练信息
        print(f"------训练数据------")
        print(f"本轮训练集上总误差(train_total_loss):{epoch_train_loss}")
        print(f"本轮训练集上的平均误差(train_average_loss):{epoch_train_average_loss}")
        print(f"本轮训练集上的正确率(train_accuracy):{epoch_train_right_data / train_data_size}")
        # 测试信息
        print(f"------测试数据------")
        print(f"本轮测试集上总误差(test_total_loss):{epoch_test_loss}")
        print(f"本轮测试集上的平均误差(test_average_loss):{epoch_test_average_loss}")
        print(f"本轮测试集上的正确率(test_accuracy):{epoch_test_right_data / test_data_size}")

        # 创建文件夹用来保存模型
        dir_path = os.path.join(".", "模型保存", "{}".format(model_name))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 保存每一轮模型
        torch.save(net.state_dict(), "./模型保存/" + "{}/".format(model_name) + "net{}.pth".format(i))
        print("模型保存成功")

    # 绘图
    # draw_train_loss_map(train_loss_map) # 绘制训练loss曲线
    # draw_test_loss_map(test_loss_map) # 绘制测试loss曲线
    # draw_train_accuracy_map(train_accuracy_map) # 绘制训练正确率曲线
    # draw_test_accuracy_map(test_accuracy_map) # 绘制测试正确率曲线
    draw_loss_and_accuracy_curves(train_loss_map, test_loss_map, train_accuracy_map, test_accuracy_map)


# 集合图像
def draw_loss_and_accuracy_curves(train_loss_map, test_loss_map, train_accuracy_map, test_accuracy_map):
    epochs = range(1, len(train_loss_map) + 1)
    plt.figure(figsize=(10, 8))
    # 绘制训练损失曲线
    plt.plot(epochs, train_loss_map, 'o-', color='blue', label='Train Loss')
    # 绘制测试损失曲线
    plt.plot(epochs, test_loss_map, 'o-', color='green', label='Test Loss')
    # 绘制训练准确率曲线（使用虚线）
    plt.plot(epochs, train_accuracy_map, 's--', color='red', label='Train Accuracy')
    # 绘制测试准确率曲线（使用虚线）
    plt.plot(epochs, test_accuracy_map, 's--', color='orange', label='Test Accuracy')
    plt.xlabel('Epoch')  # X轴标签
    plt.ylabel('Value')  # Y轴标签
    plt.title('Training and Testing Loss and Accuracy')  # 图标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    # 设置X轴的刻度为整数
    plt.xticks(epochs, [f'{i}' for i in epochs], rotation=45)
    plt.show()


# 训练模型
train_my_model(dataset_name, transform, batch_size, net_model, learning_rate, epoch, model_name)