# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/27/19
import mxnet as mx
from kaggle.utils import get_ems
from tqdm import tqdm


def test(net, ctx, test_loader, rate):
    metric = mx.metric.Accuracy()
    test_loss = mx.metric.Loss()
    for batch in tqdm(test_loader):
        trans = batch[0].as_in_context(ctx)
        labels = batch[1].as_in_context(ctx)
        output = get_ems(net, trans, rate)
        metric.update(labels, output)

    _, test_acc = metric.get()
    _, test_loss = test_loss.get()
    test_str = ("Test Loss: %f, Test acc %f." % (test_loss, test_acc))
    print(test_str)


if __name__ == '__main__':
    from gluoncv.model_zoo.residual_attentionnet import cifar_residualattentionnet452
    from model.residual_attention_network import ResidualAttentionModel_32input
    from train_cifar import val_data

    ctx = mx.gpu()
    net1 = cifar_residualattentionnet452()
    net1.load_parameters('../cifar_param/cifar_att452/test_epoch215_0.97570.param', ctx)
    net1.hybridize(static_alloc=True, static_shape=True)

    net2 = ResidualAttentionModel_32input(additional_stage=True)
    net2.load_parameters('../cifar_param/cifar_att92/test_epoch215_0.97140.param', ctx)
    net2.hybridize(static_alloc=True, static_shape=True)

    net = [net1, net2]
    rate = [0.7, 0.3]
    test(net, ctx, val_data, rate)
