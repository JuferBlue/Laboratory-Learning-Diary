"""
@author:JuferBlue
@file:data.py
@date:2024/9/22 7:50
@description:
"""
import os

import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np



class MyDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.name = os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path,'SegmentationClass',segment_name)
        image_path = os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg'))

        # 定义预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图片转换为Tensor，并归一化至[0, 1]
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long()),  # 将标签转换为LongTensor
        ])


        segment_image = keep_image_size_open(segment_path)
        image_path = keep_image_size_open(image_path)

        segment_image = target_transform(segment_image).squeeze(0)
        image_path = transform(image_path)

        # annotation_array = np.array(segment_image,dtype=np.int64)
        # annotation_tensor = torch.tensor(annotation_array)

        return image_path, segment_image


if __name__ == '__main__':
    data = MyDataset('VOC/VOCdevkit/VOC2007')
    # 第一个0是getitem里的index，第二个0，1是元组数据索引
    print(data[0][0].shape)
    print(data[0][1].shape)
    img = data[0][0]
    annotation = data[0][1]
    img = img.permute(1,2,0).numpy()
    annotation = annotation.numpy()

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(annotation,cmap='gray')
    plt.show()

    print(annotation[80:100, 100:120])


