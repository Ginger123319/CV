import torch
import cfg
from utils import Utils
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DrugData(Dataset):
    def __init__(self, root):
        super().__init__()
        self.dataset = []
        self.utils = Utils()
        # print(word_vec.shape)
        self.new_dict = np.load(cfg.word2vec_path, allow_pickle=True).item()
        with open(root, encoding="utf-8") as f:
            f.readline()
            for line in f.readlines():
                line = line.split()
                tag = self.utils.get_name_list().index(line[1])
                self.dataset.append([line[0], tag])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        elem = self.dataset[index][0]
        # 内存爆炸问题：在init中添加了下列代码，一次性将所有词的词向量都查询出来，查询次数过多，内存爆炸
        # 在getitem中进行查询，查询次数可以控制，与batch_size一致，可以减少查询次数，减少内存消耗
        # 思维提升：用的时候再去查询加载有助于性能优化，热加载
        word_vec = torch.zeros(self.utils.get_max_len()[0], 300)
        for i, s in enumerate(elem):
            try:
                vec = self.new_dict[s]
            except:
                continue
            word_vec[i] = torch.Tensor(vec)
        elem = word_vec
        tag = self.dataset[index][1]
        return elem, tag


class DrugTagData(Dataset):
    def __init__(self):
        super().__init__()
        self.utils = Utils()
        self.dataset = self.utils.get_name_list()
        # print(word_vec.shape)
        self.new_dict = np.load(cfg.word2vec_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        elem = self.dataset[index]
        # 内存爆炸问题：在init中添加了下列代码，一次性将所有词的词向量都查询出来，查询次数过多，内存爆炸
        # 在getitem中进行查询，查询次数可以控制，与batch_size一致，可以减少查询次数，减少内存消耗
        # 思维提升：用的时候再去查询加载有助于性能优化，热加载
        word_vec = torch.zeros(self.utils.get_max_len()[0], 300)
        for i, s in enumerate(elem):
            try:
                vec = self.new_dict[s]
            except:
                continue
            word_vec[i] = torch.Tensor(vec)
        elem = word_vec
        return elem


if __name__ == '__main__':
    drug_data = DrugData(cfg.test_path)
    # print(drug_data.__len__())
    loader = DataLoader(drug_data, batch_size=100, shuffle=True)
    for i, (inputs, label) in enumerate(loader):
        print(i)
        print(inputs.shape)
        print(label)
    # drug_tag_data = DrugTagData()
    # # print(drug_data.__len__())
    # loader = DataLoader(drug_tag_data, batch_size=100, shuffle=True)
    # for i, inputs in enumerate(loader):
    #     print(i)
    #     print(inputs.shape)

