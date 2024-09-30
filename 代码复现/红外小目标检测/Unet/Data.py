"""
@author:JuferBlue
@file:Data.py
@date:2024/9/26 10:37
@description:
"""
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

class IRSTD(Dataset):
    def __init__(self,train=True,transform=None):
        self.train = train
        self.path = './data/IRSTD-1k'
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        #根据模式选择对应的文件列表
        if self.train:
            img_name_path = os.path.join(os.path.join(self.path,'trainval.txt'))
        else:
            img_name_path = os.path.join(os.path.join(self.path,'test.txt'))

        # 打开并读取文件列表
        with open(img_name_path, 'r') as file:
            img_name_list = file.readlines()

        #初始化测试图像和标签图像名列表
        self.image_list = []
        self.label_list = []

        for line in img_name_list:
            image_path = os.path.join(self.path,'IRSTD1k_Img',line.strip()+'.png')
            label_path = os.path.join(self.path,'IRSTD1k_Label',line.strip()+'.png')
            self.image_list.append(image_path)
            self.label_list.append(label_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label_path = self.label_list[index]

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')


        image = self.transform(image)
        label = self.transform(label)

        # 将标签从0和255转为0和1
        label = (label > 0).float()

        return image, label


if __name__ == '__main__':
    data = IRSTD(train=False)
    print(data.__len__())
    # 第一个0是getitem里的index，第二个0，1是元组数据索引
    print(data[0][0].shape)
    print(data[0][1].shape)

    img = data[0][0]
    #打印每个像素点的值

    annotation = data[0][1]
    img = img.permute(1, 2, 0).numpy()
    annotation = annotation.permute(1, 2, 0).numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(annotation, cmap='gray')
    plt.show()

    # print(annotation[80:100, 100:120])
