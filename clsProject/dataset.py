import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import sample_add, split_data

train_file = r"..\..\source\enzyme\train\train_file.txt"
test_file = r"..\..\source\enzyme\test\test_file.txt"
s_sequence = "0GAVLIPFYWSTCMNQDEKRHX"
cls_num = len(s_sequence)


def string_2num_list(s):
    s_dic = {}
    s_list = []
    for i in range(cls_num):
        s_dic[s_sequence[i]] = i
    for i in range(len(s)):
        s_list.append(s_dic[s[i]])
    # print(s_list)
    # print(s_dic)
    # print(cls_num)
    return s_list


def my_one_hot(n_list):
    result = []
    # print(base)
    for i in n_list:
        if i == 0:
            base = [0 for x in range(cls_num)]
            result.append(base)
        else:
            base = [0 for x in range(cls_num)]
            base[i] = 1
            # print(base)
            result.append(base)
        # base[i] = 0
    # print(result)
    return torch.Tensor(result)


class MyData(Dataset):
    def __init__(self, is_train=True):
        file_path = train_file if is_train else test_file
        self.dataset = []
        max_len, min_len = split_data()
        self.list0 = []
        self.list1 = []
        with open(file_path) as t:
            # data = t.readlines()
            # print(data)
            for line in t:
                # print(line.split('.')[1])
                # 一行末尾的换行符一定要去除，不一定会显示
                flag = line.split('.')[1].strip('\n')
                # print(flag)
                if flag == "0":
                    self.list0.append(line.split('.')[0])
                else:
                    self.list1.append(line.split('.')[0])

        if is_train:
            r_list0, r_list1, l0, l1 = sample_add(self.list0, self.list1, max_len, min_len)
            # 先用增样后的数据训练，因此往dataset中添加的是增样后的数据r_list0, r_list1
            # 当需要用真实数据训练时，将l0, l1添加到dataset中
            # r_list0 = l0
            # r_list1 = l1
            for i in r_list0:
                tag = 0
                num_list = string_2num_list(i)
                r_array = my_one_hot(num_list)
                # print(r_array.shape)
                # print(r_array[300])
                self.dataset.append((r_array.unsqueeze(dim=0), tag))
                # exit()
            for i in r_list1:
                tag = 1
                num_list = string_2num_list(i)
                r_array = my_one_hot(num_list)
                self.dataset.append((r_array.unsqueeze(dim=0), tag))
            # print(self.dataset[-1][1])
            # exit()
        else:
            # 测试集
            test0, test1 = sample_add(self.list0, self.list1, max_len, min_len, False)
            for i in test0:
                tag = 0
                num_list = string_2num_list(i)
                r_array = my_one_hot(num_list)
                # print(r_array.shape)
                # print(r_array[300])
                self.dataset.append((r_array.unsqueeze(dim=0), tag))
                # exit()
            for i in test1:
                tag = 1
                num_list = string_2num_list(i)
                r_array = my_one_hot(num_list)
                self.dataset.append((r_array.unsqueeze(dim=0), tag))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        d = self.dataset[index]
        # print(type(d))
        return d[0], d[1]


if __name__ == '__main__':
    # num_list = string_2num_list("0LPTSNPAQELEARQLGRTTRDDLINGNSASCADVIFIYARGSTETGN")
    # print(my_one_hot(num_list).shape)
    data = MyData(is_train=True)
    # print(data[10][0].shape)
    data_loader = DataLoader(data, batch_size=30, shuffle=True)
    for i, (x, y) in enumerate(data_loader):
        # print(i)
        print(x.shape)
        print(y)
        break
