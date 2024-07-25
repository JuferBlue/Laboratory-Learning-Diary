"""
@author:JuferBlue
@file:Test.py
@date:2024/7/17 14:33
@description:
"""

import matplotlib.pyplot as plt


def draw_loss_and_accuracy_curves(train_loss_map, test_loss_map, train_accuracy_map, test_accuracy_map):
    epochs = range(1, len(train_loss_map) + 1)

    plt.figure(figsize=(10, 8))

    # 绘制训练损失曲线
    plt.plot(epochs, train_loss_map, 'o-', color='blue', label='Train Loss')

    # 绘制测试损失曲线
    plt.plot(epochs, test_loss_map, 'o-', color='green', label='Test Loss')

    # 绘制训练准确率曲线（使用虚线）
    plt.plot(epochs, train_accuracy_map, 's--', color='red', label='Train Accuracy')

    # 绘制测试准确率曲线（使用虚线）
    plt.plot(epochs, test_accuracy_map, 's--', color='orange', label='Test Accuracy')

    plt.xlabel('Epoch')  # X轴标签
    plt.ylabel('Value')  # Y轴标签
    plt.title('Training and Testing Loss and Accuracy')  # 图标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    # 设置X轴的刻度为整数
    plt.xticks(epochs, [f'{i}' for i in epochs], rotation=45)
    plt.show()


# 示例数据
train_loss_map = [0.8, 0.7, 0.6, 0.5, 0.4]
test_loss_map = [0.9, 0.8, 0.7, 0.6, 0.5]
train_accuracy_map = [0.1, 0.2, 0.3, 0.4, 0.5]
test_accuracy_map = [0.15, 0.25, 0.35, 0.45, 0.55]

# 调用函数绘制曲线
draw_loss_and_accuracy_curves(train_loss_map, test_loss_map, train_accuracy_map, test_accuracy_map)


