import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
from early_stopping import EarlyStopping
import sys
import torch.nn.init as init


class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def train(self, data):
        n_epoch = self.params['n_epoch']

        # 将输入的训练集按8：2的方式划分为训练数据和验证数据
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_ds, val_ds = torch.utils.data.random_split(data, (train_size, test_size))

        dim = data.X.shape[1:]

        # 加载网络和优化器
        self.clf = self.net(dim=dim, pretrained=self.params['pretrained'], num_classes=self.params['num_class']).to(
            self.device)
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        # 加载训练和验证数据
        train_loader = DataLoader(train_ds, shuffle=True, **self.params['loader_tr_args'])
        val_loader = DataLoader(val_ds, shuffle=True, **self.params['loader_va_args'])

        # to track the validation loss as the model trains
        valid_losses = []
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=5, verbose=True, save_path="./")

        for epoch in tqdm(range(1, n_epoch + 1), ncols=100, file=sys.stdout):
            # 训练
            self.clf.train()
            for batch_idx, (x, y, idxs) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
            # 每一轮进行一次验证
            self.clf.eval()
            for batch_idx, (x, y, idxs) in enumerate(val_loader):
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                valid_losses.append(loss.item())
            # early_stop
            valid_loss = np.average(valid_losses)
            early_stopping(valid_loss, self.clf)
            # stop train
            if early_stopping.early_stop:
                tqdm.write("Early stopping at {} epoch".format(epoch - 1))
                break

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        labels = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
                labels[idxs] = y.cpu()
        return probs, labels

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    def get_model(self):
        return self.clf

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings

    def get_grad_embeddings(self, data):
        self.clf.eval()
        embDim = self.clf.get_embedding_dim()
        nLab = self.params['num_class']
        embeddings = np.zeros([len(data), embDim * nLab])

        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embeddings[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                    1 - batchProbs[j][c]) * -1.0
                        else:
                            embeddings[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                    -1 * batchProbs[j][c]) * -1.0

        return embeddings


class CIFAR10_Net(nn.Module):
    def __init__(self, dim=28 * 28, pretrained=False, num_classes=10):
        super().__init__()
        # resnet18 = models.resnet18(pretrained=pretrained)
        resnet18 = models.efficientnet_b7(pretrained=pretrained)
        features_tmp = nn.Sequential(*list(resnet18.children())[:-1])
        # features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.features = nn.Sequential(*list(features_tmp))
        self.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                        nn.Linear(2560, num_classes))
        # print(resnet18)

        self.dim = resnet18.classifier[1].in_features
        # print(self.dim)
        # exit()

    def forward(self, x):
        feature = self.features(x)
        x = feature.view(feature.size(0), -1)
        output = self.classifier(x)
        return output, x

    def get_embedding_dim(self):
        return self.dim


class MNIST_Net(nn.Module):
    def __init__(self, dim=28 * 28, pretrained=False, num_classes=10):
        super().__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier = nn.Linear(resnet18.fc.in_features, num_classes)
        self.dim = resnet18.fc.in_features

    def forward(self, x):
        x = self.conv(x)
        feature = self.features(x)
        x = feature.view(feature.size(0), -1)
        output = self.classifier(x)
        return output, x

    def get_embedding_dim(self):
        return self.dim


def get_net(name, args_task, device):
    if name == 'CIFAR10':
        return Net(CIFAR10_Net, args_task, device)
    elif name == 'MNIST':
        return Net(MNIST_Net, args_task, device)
    else:
        raise NotImplementedError
