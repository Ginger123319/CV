from torchvision import transforms
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, KCenterGreedyPCA, BALDDropout, \
    AdversarialBIM, AdversarialDeepFool, VarRatio, MeanSTD, BadgeSampling, CEALSampling, \
    LossPredictionLoss, VAAL, WAAL
from parameters import *
from torchvision import transforms
import sys
import os
import numpy as np
import math
import torch


def get_mean_stddev(datax):
    return round(np.mean(datax), 4), round(np.std(datax), 4)


def get_aubc(quota, bsize, resseq):
    # it is equal to use np.trapz for calculation
    ressum = 0.0
    if quota % bsize == 0:
        for i in range(len(resseq) - 1):
            ressum = ressum + (resseq[i + 1] + resseq[i]) * bsize / 2

    else:
        for i in range(len(resseq) - 2):
            ressum = ressum + (resseq[i + 1] + resseq[i]) * bsize / 2
        k = quota % bsize
        ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
    ressum = round(ressum / quota, 3)

    return ressum


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_strategy(STRATEGY_NAME, dataset, net, args_input, args_task):
    if STRATEGY_NAME == 'RandomSampling':
        return RandomSampling(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'LeastConfidence':
        return LeastConfidence(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'MarginSampling':
        return MarginSampling(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'EntropySampling':
        return EntropySampling(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'LeastConfidenceDropout':
        return LeastConfidenceDropout(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'MarginSamplingDropout':
        return MarginSamplingDropout(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'EntropySamplingDropout':
        return EntropySamplingDropout(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'KMeansSampling':
        return KMeansSampling(dataset, net, args_input, args_task)
    # elif STRATEGY_NAME == 'KMeansSamplingGPU':
    #     return KMeansSamplingGPU(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'KCenterGreedy':
        return KCenterGreedy(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'KCenterGreedyPCA':
        return KCenterGreedyPCA(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'BALDDropout':
        return BALDDropout(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'VarRatio':
        return VarRatio(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'MeanSTD':
        return MeanSTD(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'BadgeSampling':
        return BadgeSampling(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'LossPredictionLoss':
        return LossPredictionLoss(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'AdversarialBIM':
        return AdversarialBIM(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'AdversarialDeepFool':
        return AdversarialDeepFool(dataset, net, args_input, args_task)
    elif 'CEALSampling' in STRATEGY_NAME:
        return CEALSampling(dataset, net, args_input, args_task)
    elif STRATEGY_NAME == 'WAAL':
        return WAAL(dataset, net, args_input, args_task)
    else:
        raise NotImplementedError
