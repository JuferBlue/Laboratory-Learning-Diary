"""
@author:JuferBlue
@file:get_data.py
@date:2024/9/21 18:37
@description:
"""
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class SteelDefectDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mode='train'):
        self.img_dir = img_dir  # 图片路径
        self.mask_dir = mask_dir  # 标签路径
        self.mode = mode

        if mode == 'train':
            self.train_file = "NEU-Seg/ImageSets/Segmentation/train.txt"
            self.img_arr = self.read_split_file(self.train_file)
            self.mask_arr = self.img_arr
        elif mode == 'valid':
            self.val_file = "NEU-Seg/ImageSets/Segmentation/val.txt"
            self.img_arr = self.read_split_file(self.val_file)
            self.mask_arr = self.img_arr
        elif mode == 'test':
            self.test_file = "NEU-Seg/ImageSets/Segmentation/test.txt"
            self.img_arr = self.read_split_file(self.test_file)

        self.real_len = len(self.img_arr)

        print('Finished reading the {} set of Steel Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __len__(self):
        return self.real_len

    def __getitem__(self, idx):
        # 获取文件名
        single_image_name = self.img_arr[idx]

        # 构造图像和标签的路径
        img_path = os.path.join(self.img_dir, single_image_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, single_image_name + '.png')

        # 加载图像
        image = Image.open(img_path).convert('RGB')  # 假设图像是灰度图，如果是RGB图像，使用 'RGB'
        mask = Image.open(mask_path)  # 假设标签是单通道的灰度图
        # 将 RGB 图像转换为灰度图 (单通道)
        image = transforms.functional.rgb_to_grayscale(image)

        # 图像的转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 可根据需求修改尺寸
            transforms.ToTensor()           # 将图像转换为张量并归一化到 [0, 1] 范围
        ])
        image = transform(image)  # 图像转为张量

        # 标签图像的处理
        mask = mask.resize((224, 224))  # 将mask也调整为相同的尺寸
        mask = torch.tensor(np.array(mask), dtype=torch.long)  # 将标签转换为long类型张量

        if self.mode == 'test':
            return image  # 如果是测试集，只返回图像
        else:
            return image, mask  # 训练和验证时返回图像和标签

    def read_split_file(self, file_path):
        """读取 .txt 文件中的图像文件名列表"""
        with open(file_path, "r") as file:
            file_names = file.read().splitlines()
        return file_names



#






