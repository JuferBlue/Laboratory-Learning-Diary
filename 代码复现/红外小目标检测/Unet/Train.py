"""
@author:JuferBlue
@file:Train.py
@date:2024/9/26 12:08
@description:
"""


import torch
from torch.utils.data import DataLoader;
from SIRST_AUG_DATA import *
from UNet_X16 import *
from UNet_X16_improved import *
from Utils import *
from Loss import *

batch_size = 4
learning_rate = 0.05
epochs = 100
weight_path = 'params/unet.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = SIRST_AUG(train=True)
dataset_test = SIRST_AUG(train=False)


train_length = len(dataset_train)
test_length = len(dataset_test)

print("训练集大小："+str(train_length))
print("测试集大小："+str(test_length))

train_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

unet = UNet_X16_improved(1,1).to(device)

if os.path.exists(weight_path):
    unet.load_state_dict(torch.load(weight_path))
    print('加载模型成功')
else:
    print('未找到模型')

loss_fn = SoftIoULoss()
loss_fn.to(device)
optimizer = torch.optim.SGD(unet.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0001)
# Poly learning rate function
def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power=0.9):
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

max_iter = len(train_data_loader)*epochs




for epoch in range(epochs):
    #训练部分
    print(f'--------------------epoch:{epoch+1}--------------------')
    unet.train()
    epoch_train_loss = 0
    epoch_train_iou = 0
    for i , (img, label) in enumerate(train_data_loader):

        iter = epoch * len(train_data_loader) + i  # 当前迭代数
        poly_lr_scheduler(optimizer, learning_rate, iter, max_iter, power=0.9)

        img = img.to(device)
        label = label.to(device)
        # print("标签形状："+str(label.shape))

        output = unet(img)
        # print("输出形状："+str(output.shape))
        loss = loss_fn(output, label)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        iou = compute_iou(output, label)
        epoch_train_iou += iou

        #保存图片



        if (i+1) % 50 == 0:
            print(f'epoch:{epoch+1}\t\timg_num:{(i+1)*batch_size}\t\tmiou:{epoch_train_iou/((i+1)*batch_size)}\t\ttrain_loss===>>>>{loss.item()}')
    print(f"第{epoch+1}轮训练结束：")
    print(f"当前epoch训练的miou为：{epoch_train_iou/((i+1)*batch_size)}")
    print(f"当前epoch测试的loss为：{epoch_train_loss}")

    #测试部分
    unet.eval()
    epoch_test_loss=0
    epoch_test_iou = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.to(device)
            label = label.to(device)
            output = unet(img)
            loss = loss_fn(output, label)
            epoch_test_loss += loss.item()
            iou = compute_iou(output, label)
            epoch_test_iou += iou
        print(f"第{epoch+1}轮测试结束：")
        print(f"当前epoch测试的miou为：{epoch_test_iou/test_length}")
        print(f"当前epoch测试的loss为：{epoch_test_loss}")

    # 保存模型
    torch.save(unet.state_dict(), weight_path)
    print('模型已保存')
