"""
@author:JuferBlue
@file:train.py
@date:2024/9/22 9:37
@description:
"""
from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = ('VOC/VOCdevkit/VOC2007')
save_path = 'train_image'


if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path),batch_size=2,shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('load weight success')
    else:
        print('no weight')


    opt = optim.Adam(net.parameters(),lr=0.001)
    loss_fn = nn.BCELoss()
    loss_fn.to(device)

    epoch = 1

    while True:
        for i,(image,segment_image) in enumerate(data_loader):
            image,segment_image = image.to(device),segment_image.to(device)
            out_image = net(image)
            loss = loss_fn(out_image,segment_image)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%5==0:
                print(f'{epoch}--{i}--train_loss===>>>>{loss.item()}')
            if i%50==0:
                torch.save(net.state_dict(),weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image,_segment_image,_out_image],dim=0)
            save_image(img,f'{save_path}/{i}.png')
        epoch += 1