from PIL import Image
import numpy as np
from torchvision import transforms

from PIL import Image
import numpy as np
from collections import Counter

from PIL import Image
import numpy as np
from collections import Counter


def count_pixels(image_path):
    # 加载图像并转换为灰度图像
    image = Image.open(image_path)  # 'L'表示灰度图
    image_array = np.array(image)

    # 使用 Counter 计算每个像素值的个数
    pixel_counts = Counter(image_array.flatten())

    # 过滤掉出现次数为 0 的像素值
    pixel_counts = {pixel_value: count for pixel_value, count in pixel_counts.items() if count > 0}

    return pixel_counts


if __name__ == "__main__":
    image_path = 'img_1.png'  # 替换为你的图片路径
    pixel_counts = count_pixels(image_path)

    # 打印结果
    for pixel_value, count in pixel_counts.items():
        print(f"像素值: {pixel_value}, 个数: {count}")

