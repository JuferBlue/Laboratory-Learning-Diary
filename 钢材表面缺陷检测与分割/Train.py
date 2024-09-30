"""
@author:JuferBlue
@file:Train.py
@date:2024/9/29 8:41
@description:
"""
import torch
from Data import *
from torch.utils.data import DataLoader
from UNet import *
from torch import nn,optim

batch_size = 8
learning_rate = 1e-4
epochs = 100
weight_path = 'params/unet.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = NEU(model='trainval')
dataset_test = NEU(model='test')

train_length = len(dataset_train)
test_length = len(dataset_test)

print("训练集大小：",train_length)
print("测试集大小：",test_length)

train_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

unet = UNet(1,4).to(device)

if os.path.exists(weight_path):
    unet.load_state_dict(torch.load(weight_path))
    print('加载模型成功')
else:
    print('未找到模型')

optimizer = optim.Adam(unet.parameters(),lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

for epoch in range(epochs):
    # 训练部分
    print(f'--------------------epoch:{epoch + 1}--------------------')
    unet.train()
    for i, (img, label) in enumerate(train_data_loader):
        img = img.to(device)
        label = label.to(device)
        # print(label.shape)


        output = unet(img)
        # print(output.shape)

        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_img = torch.argmax(output, dim=1)
        # print(output_img.shape)
        if i % 10 == 0:
            print(loss.item())
    # 保存模型
    torch.save(unet.state_dict(), weight_path)
    print('模型已保存')