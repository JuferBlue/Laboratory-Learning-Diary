from PIL import Image
import numpy as np

# # 打开二值图像文件
# img_path = "img_1.png"
# img = Image.open(img_path)
#
# # 将图像转换为numpy数组
# img_array = np.array(img)
#
# # 输出图像的像素点
# print(img_array[40:50, 20:30])  # 仅输出前10行和10列的像素值以节省显示空间
#
#
# img2 = Image.open(img_path).convert('L')
# img_array = np.array(img2)
#
# # 输出图像的像素点
# print(img_array[40:50, 20:30])  # 仅输出前10行和10列的像素值以节省显示空间
#
# from PIL import Image

# 打开图片
img_path = "img_1.png"  # 替换为你的图片路径
img = Image.open(img_path)

# 查看图像的模式
print(f"The image mode is: {img.mode}")
