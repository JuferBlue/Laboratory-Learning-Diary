"""
@author:JuferBlue
@file:Data.py
@date:2024/9/28 23:07
@description:
"""
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class NEU(Dataset):
    def __init__(self,model='train'):
        self.model = model
        self.path = './data/NEU-Seg'

        self.img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.label_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long()),  # 将标签转换为LongTensor
        ])

        #根据模式选择对应的文件列表
        self.txt_path = os.path.join(self.path, 'ImageSets','Segmentation')
        if self.model=='train':
            img_name_path = os.path.join(self.txt_path,'train.txt')
        elif self.model=='val':
            img_name_path = os.path.join(self.txt_path,'test.txt')
        elif self.model=='test':
            img_name_path = os.path.join(self.txt_path,'val.txt')
        elif self.model=='trainval':
            img_name_path = os.path.join(self.txt_path,'trainval.txt')

        # 打开并读取文件列表
        with open(img_name_path, 'r') as file:
            self.img_name_list = file.readlines()

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        img_name = self.img_name_list[index].strip()
        img_path = os.path.join(self.path,'JPEGImages',img_name+'.jpg')
        label_path = os.path.join(self.path,'SegmentationClass',img_name+'.png')

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path)


        image = self.img_transform(image)
        label = self.label_transform(label).squeeze(0)

        return image, label


if __name__ == '__main__':
    data = NEU(model='trainval')
    print(data.__len__())
    # 第一个0是getitem里的index，第二个0，1是元组数据索引
    print(data[0][0].shape)
    print(data[0][1].shape)

    img = data[187][0]
    #打印每个像素点的值

    annotation = data[187][1]
    print(annotation[100:120, 100:120])

    annotation.unsqueeze_(0)
    img = img.permute(1, 2, 0).numpy()
    annotation = annotation.permute(1, 2, 0).numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(annotation)
    plt.show()

    # print(annotation[80:100, 100:120])