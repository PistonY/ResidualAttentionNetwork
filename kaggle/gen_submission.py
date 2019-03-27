# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/27/19
import os
import pandas as pd
import numpy as np
from mxnet import image, nd
from mxnet.gluon import data as gdata
from kaggle.utils import get_ems
from tqdm import tqdm


map_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
test_ds = gdata.vision.ImageFolderDataset('/home/yxv/DataSets/test', flag=1)


def trans_test(data):
    im = data.astype(np.float32) / 255.
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)

    im = nd.transpose(im, (2, 0, 1))
    return im


test_data = gdata.DataLoader(test_ds.transform_first(trans_test),
                             256, shuffle=False, last_batch='keep')


def get_fs(ctx, net, rate):
    assert len(net) == len(rate)
    preds = []
    for X, _ in tqdm(test_data):
        y_hat = get_ems(net, X.as_in_context(ctx), rate)
        preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    preds = [map_list[i] for i in preds]
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    # df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
    df.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    import mxnet as mx
    from gluoncv.model_zoo.residual_attentionnet import cifar_residualattentionnet452
    from model.residual_attention_network import ResidualAttentionModel_32input

    ctx = mx.gpu()
    net1 = cifar_residualattentionnet452()
    net1.load_parameters('../cifar_param/cifar_att452/test_epoch215_0.97570.param', ctx)
    net1.hybridize(static_alloc=True, static_shape=True)

    net2 = ResidualAttentionModel_32input(additional_stage=True)
    net2.load_parameters('../cifar_param/cifar_att92/test_epoch215_0.97140.param', ctx)
    net2.hybridize(static_alloc=True, static_shape=True)

    net = [net1, net2]
    rate = [0.7, 0.3]
    get_fs(ctx, net, rate)
