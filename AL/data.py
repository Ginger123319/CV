import numpy as np
import torch
import random
import os
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset


class MNIST_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy(), mode='L')
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, args_task):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.args_task = args_task

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs][idx]

    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return self.handler(X, Y, self.args_task['transform_train'])

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],
                                          self.args_task['transform_train'])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs],
                                            self.args_task['transform_train'])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.args_task['transform_train'])

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.args_task['transform'])

    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]

    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test

    # add new
    def cal_acc(self, preds, labels):
        return 1.0 * (labels == preds).sum().item() / len(preds)


def get_CIFAR10(handler, args_task):
    data_train = datasets.CIFAR10(r'D:\Python\source\CIFAR10_DATA', train=True, download=False)
    data_test = datasets.CIFAR10(r'D:\Python\source\CIFAR10_DATA', train=False, download=False)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data,
                torch.LongTensor(data_test.targets), handler, args_task)


def get_MNIST(handler, args_task):
    raw_train = datasets.MNIST(r'D:\Python\source\MNIST_DATA', train=True, download=False)
    raw_test = datasets.MNIST(r'D:\Python\source\MNIST_DATA', train=False, download=False)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)
