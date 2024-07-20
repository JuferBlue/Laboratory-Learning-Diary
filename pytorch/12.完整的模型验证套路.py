"""
@author:JuferBlue
@file:12.完整的模型验证套路.py
@date:2024/7/21 0:15
@description:
"""
import torch
from PIL import Image
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# img_path = "./dog.png"
# img_path = "./plane.png"
img_path = "./horse.png"
img = Image.open(img_path)
# img.show()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
img = transform(img)

print(img.shape)


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("./myNet_9.pth")
model.to(device)
print(model)
img = torch.reshape(img, (1, 3, 32, 32))
img = img.to(device)
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
