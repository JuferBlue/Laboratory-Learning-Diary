"""
@author:JuferBlue
@file:8.模型的保存和再使用.py
@date:2024/7/20 19:58
@description:
"""

import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)

# 保存方式1
# 保存了结构也保存了参数
torch.save(vgg16, "vgg16_method1.pth")
# 保存方式1-加载模式1
vgg16_method1 = torch.load("vgg16_method1.pth")

# 保存方式2(官方推荐)
# 只保存参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# 保存方式2-加载模式2
vgg16_method2 = torch.load("vgg16_method2.pth")

# 陷阱
# class MyModule(torch.nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 20, 5)
#     def forward(self,x):
#         x = self.conv1(x)
#         return x
#
# myModule = MyModule()
# torch.save(myModule, "myModule.pth")

# 自己写的这个网络保存后再导入 仍旧需要有原来的模型才能导入，不然会报错

myModule = torch.load("myModule.pth")
print(myModule)
