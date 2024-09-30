"""
@author:JuferBlue
@file:Utils.py
@date:2024/9/26 16:00
@description:
"""


def compute_iou(output, label, threshold=0.5):
    # 将输出二值化
    output = (output > threshold).float()

    batch_size = output.size(0)  # 获取 batch_size
    iou_sum = 0

    for i in range(batch_size):
        # 计算交集
        intersection = (output[i] * label[i]).sum()

        # 计算并集
        union = output[i].sum() + label[i].sum() - intersection

        # 避免除以零
        if union == 0:
            iou = 1.0  # 如果没有前景目标，则将 IoU 设为 1
        else:
            iou = intersection / union

        iou_sum += iou

    # 返回批次的总iou
    return iou_sum

