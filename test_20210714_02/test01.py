import torch.optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda"

train_data = datasets.CIFAR10("D:\data\CIFAR10_data",train=True,transform=transforms.ToTensor(),download=True)
test_data = datasets.CIFAR10("D:\data\CIFAR10_data",train=False,transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(train_data,batch_size=512,shuffle=True)
test_laoder = DataLoader(test_data,batch_size=100,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(32*32*3,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )
    def forward(self,x):
        return self.fc_layer(x)

if __name__ == '__main__':
    summaryWriter = SummaryWriter("logs")
    net = Net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    # loss_func = nn.MSELoss()
    # nn.BCELoss()
    # nn.BCEWithLogitsLoss()
    # loss_func = nn.NLLLoss()
    loss_func = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(10000):
        sum_loss = 0
        sum_acc = 0
        for i,(img,label) in enumerate(train_loader):
            img,label = img.to(DEVICE),label.to(DEVICE)

            img = img.reshape(-1,32*32*3)

            out = net(img)

            # label = one_hot(label,10)
            loss = loss_func(out,label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            #acc = torch.mean(torch.eq(torch.argmax(out,dim=1),torch.argmax(label,dim=1)).float())
            acc = torch.mean(torch.eq(torch.argmax(out, dim=1), label).float())
            sum_acc = sum_acc + acc
            sum_loss = sum_loss + loss
            if i%10 ==0  and i!=0:
                _loss = sum_loss/10
                _acc = sum_acc/10
                summaryWriter.add_scalar("acc",_acc,step)
                summaryWriter.add_scalar("loss",_loss,step)
                print("loss:",_loss.item())
                print("acc:",_acc.item())
                sum_loss = 0
                sum_acc = 0
                step +=1
