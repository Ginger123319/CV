import cfg
import numpy as np


# word2vec = {}
# with open(cfg.embeddings_filepath, encoding="utf-8") as f:
#     # print(f)
#     f.readline()
#     for line in f.readlines():
#         # print(type(line))
#         line = line.split()
#         # print(line[1:])
#         key = line[0]
#         value = np.array(line[1:], dtype=np.float32)
#         # print(line)
#         word2vec[key] = value
# np.save(cfg.word2vec_path, word2vec)
# print("dict save success！")
# item()函数取出保存的字典

def get_vector(string):
    new_dict = np.load(cfg.word2vec_path, allow_pickle=True).item()
    result = new_dict[string]
    return result


if __name__ == '__main__':
    pass
