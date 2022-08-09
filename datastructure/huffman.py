# 如何将数据建立成一个huffman树
# 如何将建立好的Huffman树进行编码
class Node(object):
    def __init__(self, name=None, value=None):
        self._name = name
        self._value = value
        self._left = None
        self._right = None


class HuffmanTree(object):
    # 根据Huffman树的思想：以节点为基础，反向建立Huffman树
    def __init__(self, char_weights):
        self.Leav = [Node(part[0], part[1]) for part in char_weights]
        # 根据输入的字符及其频数生成节点
        while len(self.Leav) != 1:
            # 当reverse不为true时（默认为false)，sort就是从小到大输出
            # 因为要排序的是结构体里面的某个值，所以要用参数key
            # lambda是一个隐函数，是固定写法
            self.Leav.sort(key=lambda node: node._value, reverse=True)
            c = Node(value=(self.Leav[-1]._value + self.Leav[-2]._value))
            # print(c._value)
            # 数组pop默认删除最后一个元素，参数为-1可加可不加，并返回该值
            c._left = self.Leav.pop(-1)
            c._right = self.Leav.pop(-1)
            # 在数组最后添加新对象
            self.Leav.append(c)
        # 最后一个节点作为根节点
        self.root = self.Leav[0]
        # 此时的新节点他没有名字，就连root根节点也没有名字，因为它只是中间节点不是叶子节点
        # 根据测试集合，正好生成五个新节点，总共十一个元素，最长的编码序列为10，所以初始化为10的Buffer存储0\1
        self.Buffer = list(range(10))

    # 用递归的思想生成编码
    # 刚开始从根节点开始编码，因为它没有名字，所以无法对其进行编码，但是可以递归到它的左节点。
    # 如果其左节点还是没有名字，就会一直进入self_pre(node._left, length + 1)，直到本次执行的节点有名字。
    def pre(self, tree, length):
        node = tree
        if not node:
            return
        elif node._name:
            print(node._name + '  encoding:', end=''),
            for i in range(length):
                print(self.Buffer[i], end='')
            print('\n')
            return
        self.Buffer[length] = 0
        self.pre(node._left, length + 1)
        self.Buffer[length] = 1
        self.pre(node._right, length + 1)

    # 生成哈夫曼编码
    def get_code(self):
        self.pre(self.root, 0)


if __name__ == '__main__':
    # 输入的是字符及其频数
    char_weights = [('a', 6), ('b', 4), ('c', 10), ('d', 8), ('f', 12), ('g', 2)]
    tree = HuffmanTree(char_weights)
    # print(tree)
    tree.get_code()
