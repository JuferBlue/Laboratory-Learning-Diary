"""
@author:JuferBlue
@file:Loss.py
@date:2024/9/28 22:03
@description:
"""
import torch
import torch.nn as nn
class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: 模型的预测 (N, C, H, W) -> 二值化或经过sigmoid的输出
        # targets: 标签 (N, C, H, W) -> 二值化的ground truth
        intersection = (preds * targets).sum(dim=(2, 3))  # 交集
        total = (preds + targets).sum(dim=(2, 3))         # 并集
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)  # SoftIoU公式
        return 1 - iou.mean()