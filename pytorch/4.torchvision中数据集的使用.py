"""
@author:JuferBlue
@file:4.torchvision中数据集的使用.py
@date:2024/7/19 8:14
@description:
"""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# root: 存放数据集的地址
# train: True: 训练集, False: 测试集
# download: 是否下载数据集
# transform: 数据预处理
test_data = torchvision.datasets.CIFAR10(root="./CIFAR10_data", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# data_set: 数据集
# num_workers: 多线程
# batch_size: 批量打包数据
# shuffle: 是否打乱数据
# drop_last: 是否删除最后一个数据
test_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=64, shuffle=True, drop_last=False)

# img: 图片数据 类型:torch.Tensor
# target: 标签
img, target = test_data[0]
print(img.shape)
print(target)
print(test_data.classes[target])

writer = SummaryWriter(log_dir="logs")
# 每一轮的加载所有的数据
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1
writer.close()
