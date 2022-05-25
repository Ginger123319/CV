import torch
import data_pre
from torch.utils.data import Dataset, DataLoader


class CBowData(Dataset):
    def __init__(self, filename):
        self._dataset = []
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            word_str = "".join(lines)
        word_list, word_len = data_pre.split_word(word_str)
        sen_list = data_pre.split_sentence(lines).split()
        index_list = data_pre.get_index(sen_list, word_list)
        # print(index_list)
        for index in index_list:
            for elem in index:
                tag = elem
                self._dataset.extend(data_pre.add_padding(index, word_len))
        # print(self._dataset)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return torch.tensor(self._dataset[index][0]), torch.tensor(self._dataset[index][1])


if __name__ == '__main__':
    bow = CBowData("word.txt")
    # print(len(bow))
    # print(bow[0])
    dataloader = DataLoader(bow, batch_size=5, shuffle=True)
    for i, (data, tag) in enumerate(dataloader):
        print(data.shape)
        print(tag.shape)
