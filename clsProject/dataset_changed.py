import torch
from torch.utils.data import Dataset, DataLoader

train_file = r"..\..\source\enzyme\train\train_file.txt"
train_0_file = r"..\..\source\enzyme\train\train_0_file.txt"
train_1_file = r"..\..\source\enzyme\train\train_1_file.txt"
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
    return s_list


def my_one_hot(n_list):
    result = []
    for i in n_list:
        if i == 0:
            base = [0 for x in range(cls_num)]
            result.append(base)
        else:
            base = [0 for x in range(cls_num)]
            base[i] = 1
            result.append(base)
    return torch.Tensor(result)


class MyData(Dataset):
    def __init__(self, is_train=True, is_add=True):
        super().__init__()
        self.dataset = []
        if is_train:
            file = train_0_file if is_add else train_file
        else:
            file = test_file
        with open(file) as f:
            for elem in f:
                elem = elem.split('.')
                tag = int(elem[1].strip('\n'))
                elem = elem[0].ljust(632, "0")
                elem = my_one_hot(string_2num_list(elem))
                self.dataset.append((elem.unsqueeze(dim=0), tag))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        d = self.dataset[index]
        return d[0], d[1]


if __name__ == '__main__':
    data = MyData(is_train=False, is_add=False)
    print(data[10][0].shape)
    data_loader = DataLoader(data, batch_size=30, shuffle=True)
    for i, (x, y) in enumerate(data_loader):
        # x = x.permute(0, 3, 1, 2)
        print(x.shape)
        print(y)
        break
