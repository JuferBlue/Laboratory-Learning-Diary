"""
@author:JuferBlue
@file:Train.py
@date:2024/9/22 9:37
@description:
"""
from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from torchvision.utils import save_image
from UNet import *
from utils import *
# from net import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = ('VOC/VOCdevkit/VOC2007')
save_path = 'train_image'


if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path),batch_size=4,shuffle=True)
    net = UNet(3,21).to(device)


    # def weights_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.xavier_normal_(m.weight)
    # net.apply(weights_init)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('load weight success')
    else:
        print('no weight')


    opt = optim.Adam(net.parameters(),lr=0.0001)
    # loss_fn = nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    loss_fn.to(device)

    epoch = 1

    while True:
        net.train()
        for i,(image,segment_image) in enumerate(data_loader):
            image,segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            # print(image.shape, segment_image.shape,out_image.shape)
            loss = loss_fn(out_image,segment_image)

            opt.zero_grad()

            loss.backward()

            opt.step()

            if i%5==0:
                print(f'{epoch}--{i}--train_loss===>>>>{loss.item()}')
            if i%50==0:
                torch.save(net.state_dict(),weight_path)

            #
            print(out_image.shape)
            out_img = out_image[0]
            image = image[0]
            segment_image = segment_image[0]
            print(out_img.shape)
            output_img = torch.argmax(out_img, dim=0)


            print(output_img.shape)
            color_mapped_img = apply_color_map(output_img)

            # 将 NumPy 图像转换为 PIL 图像并保存
            pil_img = Image.fromarray(color_mapped_img)
            # pil_img.save(f'{save_path}/output{i}.png')

            # 将原始 image 转换为 NumPy 数组并转换为 PIL Image
            image_np = image.permute(1, 2, 0).cpu().numpy()  # 转为 (H, W, C) 格式
            image_np = (image_np * 255).astype(np.uint8)  # 假设 image 范围在 [0, 1]，先转为 255 范围
            image_pil = Image.fromarray(image_np)

            blended_img = Image.blend(image_pil, pil_img, alpha=0.5)
            blended_img.save(f'{save_path}/output{i}.png')

        epoch += 1