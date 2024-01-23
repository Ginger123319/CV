# -*- encoding: utf-8 -*-
import os, abc, six
from os import path as P
import pandas as pd


def load_iris():
    from sklearn.datasets import load_iris
    X, y = load_iris(True)
    df = pd.DataFrame(data=X, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    def replace2str(label):
        if label == 0:
            return 'Iris-setosa'
        elif label == 1:
            return 'Iris-versicolor'
        else:
            return 'Iris-virginica'

    df['Species'] = [replace2str(l) for l in y]

    return df
