import jieba


# 拆词
def split_word(string):
    ret = jieba.lcut(string)
    ret_set = set(ret)
    if "\n" in ret_set:
        ret_set.remove("\n")
    # print(ret_set)
    ret_list = list(ret_set)
    # print(len(ret_list))
    return ret_list, len(ret_list)


# 拆句,在一句话结尾标识符，如句号感叹号等字符后面加换行符
def split_sentence(str_list):
    sen_str = ""
    for line in str_list:
        # print(line)
        for end in ["。", "！"]:
            if end in line:
                line = line.replace(end, end + "\n")
        sen_str += line
    return sen_str


def get_index(sen_list_p, word_list_p):
    # 将单个句子的分词在所有句子的分词列表中的索引取出来
    a = []
    for i in sen_list_p:
        word = jieba.lcut(i)
        b = []
        for j in word:
            num = word_list_p.index(j)
            b.append(num)
        a.append(b)
    return a


# 增加padding，将输入长度补齐到4，当第一个第二个和最后两个作为标签时候，需要补齐
def add_padding(index_list, word_len):
    dataset = []
    for i, elem in enumerate(index_list):
        inputs = []
        tag = elem
        # print(elem)
        # print(elem)
        if i - 2 < 0:
            inputs.extend(index_list[0:i] + index_list[i + 1:i + 3])
        else:
            inputs.extend(index_list[i - 2:i] + index_list[i + 1:i + 3])
        # print(inputs)
        while len(inputs) < 4:
            if i - 1 <= 0:
                inputs.insert(0, word_len)
            if i + 2 >= len(index_list):
                inputs.insert(word_len - 1, word_len + 1)

        dataset.append((inputs, tag))
    return dataset


if __name__ == '__main__':
    # 分句
    with open("word.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        word_str = "".join(lines)
    word_list, length = split_word(word_str)
    sen_list = split_sentence(lines).split()
    print(get_index(sen_list, word_list))
