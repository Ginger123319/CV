import os

import torch
import numpy as np
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import cfg


class CodeData(Dataset):
    def __init__(self, path):
        super().__init__()

        self._path = path
        self._trans = transforms.Compose([transforms.ToTensor()])
        self._dataset = []
        for name in os.listdir(path):
            _label_str = name.split(".")[0]
            _label_list = [int(x) for x in _label_str]
            _label = torch.tensor(_label_list)
            _img_path = os.path.join(path, name)
            self._dataset.append((_img_path, _label))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        _img_path = self._dataset[index][0]
        _img_data = Image.open(_img_path)
        _label = self._dataset[index][1]
        _label = one_hot(_label, 10)
        return self._trans(_img_data), _label.float()


if __name__ == '__main__':
    test_data = CodeData(cfg.validate_dir)
    print(test_data[0][1].dtype)
