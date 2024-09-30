"""
@author:JuferBlue
@file:Infer.py
@date:2024/9/29 22:49
@description:
"""
from PIL import Image
from torchvision import transforms
from UNet import *
import os
import numpy as np

unet = UNet(1,4)

model_path = './params/unet.pth'
unet.load_state_dict(torch.load(model_path))


path = './data/NEU-Seg/ImageSets/Segmentation/test.txt'
names = []
with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        name = line.strip()
        names.append(name)

img_path = './data/NEU-Seg/JPEGImages'
save_path = './baseline_predictions'

transform = transforms.Compose([
    transforms.ToTensor(),
])

unet.eval()
with torch.no_grad():
    for i in range(len(names)):
        img_name = names[i] + '.jpg'
        image = Image.open(img_path + '/' + img_name).convert('L')
        image = transform(image)
        image = image.unsqueeze(0)
        output = unet(image)
        output = torch.argmax(output, dim=1)
        output = output.squeeze(0)
        output = output.numpy()

        np.save(os.path.join(save_path, 'prediction_' + names[i] + '.npy'), output)








