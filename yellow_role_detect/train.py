import torch

from net import Net_v1,Net_v2
from torch import nn
from torch.utils.data import DataLoader
from dataset import dataset
import os
import numpy as np
from PIL import Image,ImageDraw
from utils import iou

save_path = "model/net_v2.pt"
DEVICE = "cuda"
if __name__ == '__main__':
    data_set = dataset("data")

    net = Net_v2().to(DEVICE)
    #加载预训练权重
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        print("已加载预训练权重！")
    loss_func = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters())
    Train = False
    while True:
        if Train:
            train_loader = DataLoader(dataset=data_set,batch_size=90,shuffle=True)
            for i,(x,y) in enumerate(train_loader):
                # x = x.reshape(-1,300*300*3).to(DEVICE)
                x = x.permute(0,3,1,2).to(DEVICE)
                y = y.to(DEVICE)

                out = net(x)
                loss = loss_func(out,y)

                opt.zero_grad()
                loss.backward()
                opt.step()
                if i%10 ==0:
                    torch.save(net.state_dict(), save_path)
                    print(loss.item())

        else:
            test_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=True)
            for i, (x, y) in enumerate(test_loader):
                # x = x.reshape(-1, 300 * 300 * 3).to(DEVICE)
                x = x.permute(0, 3, 1, 2).to(DEVICE)
                y = y.to(DEVICE)

                out = net(x)
                loss = loss_func(out, y)
                print(loss.item())

                # x = x.reshape(-1,300,300,3).cpu()
                x = x.permute(0, 2,3,1).cpu()
                out = out.detach().cpu().numpy()*300
                y = y.detach().cpu().numpy() * 300
                print("iou=",iou(out[0],y[0]))
                img_data = np.array((x[0]+0.5)*255,dtype=np.uint8)
                img = Image.fromarray(img_data,"RGB")
                draw = ImageDraw.Draw(img)
                draw.rectangle(np.array(y[0]), outline="red",width=2)
                draw.rectangle(np.array(out[0]), outline="yellow",width=2)
                img.show()

