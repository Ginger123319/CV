import shutil
import torch
import os
import cfg
from net import CBowNet
from torch import optim
from matplotlib import pyplot as plt
import numpy as np

DEVICE = "cuda"


# 采用数据集的方式实现手写数字识别的训练和测试


class Test:
    def __init__(self, root):
        # 创建网络对象
        self.net = CBowNet().to(DEVICE)
        self.net.eval()
        # 优化器
        self.opt = optim.Adam(self.net.parameters())
        # 加载参数
        if os.path.exists(cfg.param_path) and os.path.exists(cfg.opt_path):
            try:
                self.net.load_state_dict(torch.load(cfg.param_path))
                self.opt.load_state_dict(torch.load(cfg.opt_path))
                print("Loaded!")
            except RuntimeError:
                os.remove(cfg.param_path)
                os.remove(cfg.opt_path)
                print("参数异常，重新开始训练")
        else:
            print("No Param!")

    def __call__(self, *args, **kwargs):
        return self.net.get_emb()


if __name__ == '__main__':
    # 删除log文件
    if os.path.exists(r"./log"):
        shutil.rmtree(r"./log")
        print("log is deleted！")

    train = Test(cfg.train_dir)
    emb = train()
    print(emb.shape)
    # 词向量画图
    x1 = emb[:, 0].cpu().detach().numpy()
    y1 = emb[:, 1].cpu().detach().numpy()
    z1 = emb[:, 2].cpu().detach().numpy()
    fig = plt.figure()
    ax = plt.gca(projection="3d")
    ax.scatter(x1, y1, z1)
    # 读取wordlist
    with open(cfg.word_list_dir, "r", encoding="utf-8") as f:
        word_list = f.read().split("\n")
        print(len(word_list))
    num = np.array(word_list)
    print(num)
    for i in range(len(word_list)):
        ax.text(x1[i], y1[i], z1[i], num[i])
        print(num[i])
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = 'SimHei'
    # 设置正常显示符号
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()
