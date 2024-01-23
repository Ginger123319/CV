# coding: utf-8
import mxnet as mx
import numpy as np
import cv2
from collections import namedtuple
import glob
import time


def predict_imgs(img_BGR):
    ctx = mx.gpu()
    sym, arg_params, aux_params = mx.model.load_checkpoint('./model/resnet-18-fire', 20)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 128, 128))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    Batch = namedtuple('Batch', ['data'])

    start = time.time()
    # img_BGR = cv2.imread(_img)
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (120, 256))
    img = img - 127.5
    input_image = img[:, :, :, np.newaxis]
    input_image = input_image.transpose([3, 2, 0, 1])
    mod.forward(Batch([mx.nd.array(input_image, ctx)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    max_id = np.argmax(prob)
    # cv2.imshow('img', img_BGR)
    # cv2.waitKey(1)
    print('Result: It costs %f ms.' % ((time.time() - start) * 1000.0))

    return max_id

# if __name__ == '__main__':
#     predict_imgs()
