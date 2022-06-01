import torch
from word2vec import get_vector
import cfg
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DrugData(Dataset):
    def __init__(self, root):
        super().__init__()
        self.dataset = []
        # print(word_vec.shape)
        new_dict = np.load(cfg.word2vec_path, allow_pickle=True).item()
        with open(root, encoding="utf-8") as f:
            f.readline()
            cls_set = set()
            input_max_len = 0
            output_max_len = 0
            for line in f.readlines()[0:5]:
                # print(line.split())
                line = line.split()
                # print(line[0])
                word_vec = torch.zeros(57, 300)
                for i, s in enumerate(line[0]):
                    print("i:{}".format(i))
                    # print(get_vector(s))
                    try:
                        vec = new_dict[s]
                    except:
                        vec = torch.zeros(300, )
                    word_vec[i] = torch.Tensor(vec)
                input_vec = word_vec
                word_vec = torch.zeros(25, 300)
                for i, s in enumerate(line[1]):
                    # print(get_vector(s))
                    print("j:{}".format(i))
                    try:
                        vec = new_dict[s]
                    except:
                        vec = torch.zeros(300, )
                    word_vec[i] = torch.Tensor(vec)
                tag_vec = word_vec
                self.dataset.append((input_vec, tag_vec))
                # print(self.dataset[0][0].shape, self.dataset[0][1].shape)
                # exit()
                # cls_set.add(line[1])
                # 计算输入和标签中的字符最长长度
                # if len(line[0]) > input_max_len:
                #     input_max_len = len(line[0])
                # if len(line[1]) > output_max_len:
                #     output_max_len = len(line[1])
            # print(len(cls_set))
            # print(input_max_len)
            # print(output_max_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        elem = self.dataset[index][0]
        tag = self.dataset[index][1]
        return elem, tag


if __name__ == '__main__':
    drug_data = DrugData(cfg.save_path)
    print(drug_data.__len__())
    loader = DataLoader(drug_data, batch_size=5, shuffle=True)
    for i, (inputs, label) in enumerate(loader):
        print(i)
        print(inputs.shape)
        print(label.shape)
