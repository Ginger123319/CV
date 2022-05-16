import os
import torch
from sklearn import preprocessing

import cfg
from torch.utils.data import Dataset


class StockData(Dataset):
    def __init__(self, path, is_train=True):
        super().__init__()
        self._dataset = []
        _sub_filename = "train.txt" if is_train else "test.txt"
        filename = os.path.join(path, _sub_filename)
        with open(filename) as f:
            for line in f:
                row_list = line.strip("\n").split(":")
                # eval将列表两边的引号去除，还原为列表类型
                _elem = eval(row_list[1])
                _tag = int(row_list[0])
                self._dataset.append([_tag, _elem])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        _tag = self._dataset[index][0]
        _elem = self._dataset[index][1]
        _tag = torch.tensor(_tag, dtype=torch.float32)
        z_score = preprocessing.StandardScaler()
        _elem = z_score.fit_transform(_elem)
        _elem = torch.Tensor(_elem)
        # print(z_score)
        # exit()
        # # 归一化处理
        # for i in range(len(_elem)):
        #     # 成交量归一化
        #     _elem[i][-2] /= 1e+8
        #     # 股价归一化
        #     flag = torch.ceil(torch.max(_elem[i][0:4]))
        #     if flag != 0:
        #         _elem[i][0:4] /= flag
        #     _elem[i][-1] /= 100
        return _tag, _elem


if __name__ == '__main__':
    test_data = StockData(cfg.data_dir)
    print(test_data[0][1].shape)
    print(test_data[50003][1].dtype)
    print(test_data[50003][0].dtype)
