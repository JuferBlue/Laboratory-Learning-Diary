"""
@author:JuferBlue
@file:test.py
@date:2024/9/22 9:33
@description:
"""
import os.path

import torch

from UNet import *
from utils import keep_image_size_open
from data import *
from torchvision.utils import save_image
from torchvision import transforms
from utils import *

net = UNet(3,21).cuda()

weight_path = 'params/unet.pth'

if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('load weight success')
else:
    print('load weight failed')

_input = input('please input image path:')
img = keep_image_size_open(_input)

transform = transforms.Compose([
            transforms.ToTensor(),  # 将图片转换为Tensor，并归一化至[0, 1]
        ])

img_data = transform(img).cuda()

img_data = torch.unsqueeze(img_data,dim=0) # 升维，没有批次维度

out = net(img_data)
out = out.squeeze(0)

output_img = torch.argmax(out, dim=0)
color_mapped_img = apply_color_map(output_img)
# 将 NumPy 图像转换为 PIL 图像并保存
pil_img = Image.fromarray(color_mapped_img)
pil_img.save('result/output.png')
