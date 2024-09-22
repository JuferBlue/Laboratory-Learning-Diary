"""
@author:JuferBlue
@file:test.py
@date:2024/9/22 9:33
@description:
"""
import os.path

import torch

from net import *
from utils import keep_image_size_open
from data import *
from torchvision.utils import save_image

net = UNet().cuda()

weight_path = 'params/unet.pth'

if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('load weight success')
else:
    print('load weight failed')

_input = input('please input image path:')
img = keep_image_size_open(_input)
img_data = transforms(img).cuda()

img_data = torch.unsqueeze(img_data,dim=0) # 升维，没有批次维度

out = net(img_data)
save_image(out, 'result/result.jpg')
print(out)