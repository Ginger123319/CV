import sys
import os
import re
import random
import math
import datetime

import arguments
from parameters import *
from utils import *
import argparse
import numpy as np
import warnings
import torch
from data import CIFAR10_Handler, get_CIFAR10, MNIST_Handler, get_MNIST
from nets import get_net
from utils import get_strategy
from pprint import pprint
from temp import get_theta

torch.set_printoptions(profile='full')

# parameters
args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
NUM_ROUND = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# recording
sys.stdout = Logger(
    os.path.abspath('') + '/logfile/' + DATA_NAME + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(
        NUM_INIT_LB) + '_' + str(args_input.quota) + '_normal_log.txt')
warnings.filterwarnings('ignore')


# 获取阈值，筛选图片，训练循环过程中使用的方法
def main_process(strategy, dataset):
    # 获取阈值
    prediction, labels = strategy.predict_prob(dataset.get_test_data())
    # print(prediction[:30])
    theta = get_theta(prediction, labels)
    # 获取训练集中未标注的部分进行预测
    unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data()
    print(len(unlabeled_idxs))  # 49000
    unlabeled_prediction, labels = strategy.predict_prob(unlabeled_data)
    probs, pred_labels = unlabeled_prediction.max(1)
    # 将大于阈值的图片设为已标注
    autolabel_mask = probs > theta
    # 打印大于阈值的图片张数
    print(autolabel_mask.sum())  # 5418取出对应图片的索引
    autolabel_index = unlabeled_idxs[autolabel_mask]
    # 计算大于阈值自动标注图片的标注精度
    auto_preds = pred_labels[autolabel_mask]
    auto_labels = dataset.Y_train[autolabel_index]
    print(dataset.cal_acc(auto_preds, auto_labels))  # 0.79还行
    # 并且从label_idxs中删除(暂时采用删除的方式)
    print(len(autolabel_index))  # 5418
    print(len(dataset.labeled_idxs))  # 50000
    # 删除索引并且删除对应的图片
    dataset.labeled_idxs = np.delete(dataset.labeled_idxs, autolabel_index)
    dataset.X_train = np.delete(dataset.X_train, autolabel_index, axis=0)
    dataset.Y_train = np.delete(dataset.Y_train, autolabel_index)
    print(len(dataset.X_train))
    print(len(dataset.labeled_idxs))  # 44582
    # 计算还剩下多少未标注的图片
    # 获取训练集中未标注的部分进行预测
    dataset.n_pool = dataset.n_pool - len(autolabel_index)
    unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data()
    print(len(unlabeled_idxs))
    # 返回自动标注后剩余的图片数量
    return len(unlabeled_idxs)


if __name__ == '__main__':
    # start experiment

    iteration = args_input.iteration

    all_acc = []
    acq_time = []

    # repeate # iteration trials
    while iteration > 0:

        iteration = iteration - 1

        # data, network, strategy
        args_task = args_pool[DATA_NAME]
        dataset = get_CIFAR10(CIFAR10_Handler, args_task)  # load dataset
        net = get_net('CIFAR10', args_task, device)  # load network
        strategy = get_strategy(args_input.ALstrategy, dataset, net, args_input, args_task)  # load strategy

        start = datetime.datetime.now()

        # generate initial labeled pool
        dataset.initialize_labels(args_input.initseed)

        # record acc performance
        acc = np.zeros(NUM_ROUND + 1)

        # only for special cases that need additional data
        new_X = torch.empty(0)
        new_Y = torch.empty(0)

        # print info
        print(DATA_NAME)
        print('RANDOM SEED {}'.format(SEED))
        print(type(strategy).__name__)

        # round 0 accuracy
        if args_input.ALstrategy == 'WAAL':
            strategy.train(model_name=args_input.ALstrategy)
        else:
            strategy.train()
        preds = strategy.predict(dataset.get_test_data())
        acc[0] = dataset.cal_test_acc(preds)
        print('Round 0\ntesting accuracy {}'.format(acc[0]))
        print('\n')

        rest_img_num = main_process(strategy, dataset)
        query_pool = rest_img_num
        NUM_QUERY = 1000
        NUM_ROUND = int(query_pool / NUM_QUERY)
        for i in range(1, NUM_ROUND + 1):
            q_idxs = strategy.query(NUM_QUERY)
            # update
            strategy.update(q_idxs)
            # train
            strategy.train()
            # round rd accuracy
            preds = strategy.predict(dataset.get_test_data())
            print(dataset.cal_test_acc(preds))
            query_pool = main_process(strategy, dataset)
            # 查询难例(一次查询1000张)
            if query_pool < 40000:
                break
        # save model
        timestamp = re.sub('\.[0-9]*', '_', str(datetime.datetime.now())).replace(" ", "_").replace("-",
                                                                                                    "").replace(
            ":",
            "")
        model_path = './modelpara/' + timestamp + DATA_NAME + '_' + STRATEGY_NAME + '_' + str(
            NUM_QUERY) + '_' + str(
            NUM_INIT_LB) + '_' + str(args_input.quota) + '.params'
        end = datetime.datetime.now()
        acq_time.append(round(float((end - start).seconds), 3))
        torch.save(strategy.get_model().state_dict(), model_path)
