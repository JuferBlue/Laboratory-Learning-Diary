import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import time

# 利用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    ToTensor(),
    torchvision.transforms.Resize(28),
])

# 准备数据
train_data = torchvision.datasets.CIFAR10(root="../数据集/cifar10", train=True, transform=transform,
                                        download=True)
test_data = torchvision.datasets.CIFAR10(root="../数据集/cifar10", train=False, transform=transform,
                                       download=True)

# 查看数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集长度为{train_data_size}")
print(f"测试数据集长度为{test_data_size}")

# 创建数据加载器
train_dataloader = DataLoader(train_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)


# 创建网络模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),
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

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
learning_rate = 0.1
optimizer = torch.optim.SGD(lenet.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练时间
total_train_time = 0

# 设置训练网络的循环次数
epoch = 100

for i in range(epoch):
    print(f"--------第{i + 1}轮训练开始--------")
    # 开始计时
    start_time = time.time()
    # 训练步骤开始
    lenet.train()
    for data in train_dataloader:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        outputs = lenet(images)
        loss = loss_fn(outputs, targets)

        # 更新梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

        if total_train_step % 100 == 0:
            print("训练次数{}, loss: {:.4f}".format(total_train_step, loss.item()))


    # 测试步骤开始
    lenet.eval()
    total_accuracy = 0
    total_test_loss=0
    # 测试时不需要更新梯度
    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            outputs = lenet(images)
            loss = loss_fn(outputs, targets)
            total_test_loss+=loss.item()
            accuracy = (outputs.argmax(1) == targets).sum().item()
            total_accuracy += accuracy

    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_train_time += elapsed_time
    print(f"本轮训练时间:{elapsed_time:.4f}s")
    print(f"累积训练时间:{total_train_time:.4f}s")

    # 输出一个轮次的信息
    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率:{total_accuracy/test_data_size}")
    total_test_step += 1


    # 创建文件夹用来保存模型
    dir_path = os.path.join(".", "模型保存", "1-LeNet")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 保存每一轮的模型
    torch.save(lenet.state_dict(), f"./模型保存/1-LeNet/lenet{i}.pth")
    print("模型保存成功")






