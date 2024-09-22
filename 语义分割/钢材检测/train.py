"""
@author:JuferBlue
@file:train.py
@date:2024/9/21 17:16
@description:
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from get_data import SteelDefectDataset
from unet import UNet
# 假设 SteelDefectDataset 和 UNet 已经被定义
# from get_data import SteelDefectDataset
# from unet_model import UNet

# 超参数
num_epochs = 20
batch_size = 8
learning_rate = 0.001

# 创建训练集和验证集的数据集
train_dataset = SteelDefectDataset(img_dir="NEU-Seg/JPEGImages", mask_dir="NEU-Seg/SegmentationClass", mode='train')
val_dataset = SteelDefectDataset(img_dir="NEU-Seg/JPEGImages", mask_dir="NEU-Seg/SegmentationClass", mode='valid')

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, num_classes=2).to(device)  # 假设二分类任务
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适合分类任务
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播和优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 验证阶段
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    with torch.no_grad():  # 禁止梯度计算
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f"Validation Loss after Epoch {epoch+1}: {val_loss/len(val_loader):.4f}")

print("Training complete.")
