"""
@author:JuferBlue
@file:SIRST_AUG_DATA.py
@date:2024/9/28 21:29
@description:
"""
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

class SIRST_AUG(Dataset):
    def __init__(self,train=True,transform=None):
        self.train = train
        self.path = './data/sirst_aug'
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        #根据模式选择对应的文件夹
        if self.train:
            img_name_path = os.path.join(self.path,'trainval')
        else:
            img_name_path = os.path.join(self.path,'test')

        #获取ima_name_path下所有的文件名存入列表
        self.images_path = os.path.join(img_name_path, 'images')
        self.masks_path = os.path.join(img_name_path, 'masks')


        self.name_list = os.listdir(self.images_path)



    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        img_name = self.name_list[index]
        img_path = os.path.join(self.images_path, img_name)
        label_path = os.path.join(self.masks_path, img_name)

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')


        image = self.transform(image)
        label = self.transform(label)

        # 将标签从0和255转为0和1
        label = (label > 0).float()

        return image, label


if __name__ == '__main__':
    data = SIRST_AUG(train=True)
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