"""
@author:JuferBlue
@file:1.读取数据.py
@date:2024/7/18 19:21
@description:如何载入自己的数据集
"""
from torch.utils.data import Dataset
from PIL import Image
import os


# 自定义数据集需要继承Dataset类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 数据集路径
        self.label_dir = label_dir  # 标签-文件夹名字
        self.path = os.path.join(self.root_dir, self.label_dir)  # 数据集路径+标签-文件夹名字
        self.img_path = os.listdir(self.path)  # 文件夹下所有图片名字

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 根据索引获取图片名字
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 路径+图片名字=图片路径
        img = Image.open(img_item_path)  # 打开图片转换为PIL.Image对象
        label = self.label_dir  # 获取标签
        return img, label  # 根据返回图片和标签

    def __len__(self):
        return len(self.img_path)  # 返回数据集长度


root_dir = "./hymenoptera_data/train"  # 训练集路径
ants_label = "ants"  # 蚂蚁数据集文件夹名
bees_label = "bees"  # 蜜蜂数据集文件夹名
ants_dataset = MyData(root_dir, ants_label)
bees_dataset = MyData(root_dir, bees_label)

train_dataset = ants_dataset + bees_dataset  # 合并所有的训练集

# 测试
img, label = train_dataset[123]
print(label)
img.show()
