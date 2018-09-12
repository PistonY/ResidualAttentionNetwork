#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_cifar.py
# @Author: Piston Yang
# @Date  : 18-9-6


from model.residual_attention_network import ResidualAttentionModel_92_32input_update
import mxnet as mx
from mxnet import gluon, image, nd, autograd
from mxnet.gluon import loss as gloss
import numpy as np
import datetime
import os
from lib.piston_util import format_time, inf_train_gen

os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'

batch_size = 64


def transformer(data, label):
    im = data.asnumpy()
    im = np.pad(im, pad_width=((4, 4), (4, 4), (0, 0)), mode='constant')
    im = nd.array(im) / 255.
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), rand_crop=True, rand_mirror=True,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, label


def trans_test(data, label):
    im = data.astype(np.float32) / 255.
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)

    im = nd.transpose(im, (2, 0, 1))
    return im, label


train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=False, transform=trans_test),
    batch_size=batch_size)


def test(test_net, ctx, test_loader, epoch):
    print("Start testing iter %d." % epoch)
    Loss = gloss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()
    test_loss = mx.metric.Loss()
    for batch in test_loader:
        trans = batch[0].as_in_context(ctx)
        labels = batch[1].as_in_context(ctx)
        output = test_net(trans)
        loss = Loss(output, labels)
        test_loss.update(0, loss)
        metric.update(labels, output)

    _, test_acc = metric.get()
    _, test_loss = test_loss.get()
    test_net.save_parameters('cifar_param/test_iter%d_%.5f.param' % (epoch, test_acc))
    test_str = ("\033[35mtest_Loss: %f, test acc %f\033[0m" % (test_loss, test_acc))
    print(test_str)


def train(train_net, iterations, lr, wd, ctx, lr_period: tuple, lr_decay, train_loader, test_loader, cat_interval):
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(train_net.collect_params(),
                            'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    # trainer = gluon.Trainer(train_net.collect_params(),
    #                         'rmsprop', {'learning_rate': lr, 'wd': wd})

    train_gen = inf_train_gen(train_loader)
    Loss = gloss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()
    train_loss = mx.metric.Loss()
    prev_time = datetime.datetime.now()

    metric.reset()
    train_loss.reset()
    for iteration in range(int(iterations)):
        batch = next(train_gen)
        trans = batch[0].as_in_context(ctx)
        labels = batch[1].as_in_context(ctx)
        with autograd.record():
            output = train_net(trans)
            loss = Loss(output, labels)
        loss.backward()
        trainer.step(batch_size)
        train_loss.update(0, loss)
        metric.update(labels, output)
        if iteration % cat_interval == cat_interval - 1:
            cur_time = datetime.datetime.now()
            time_str = format_time(prev_time, cur_time)
            _, train_acc = metric.get()
            _, epoch_loss = train_loss.get()
            epoch_str = ("Iter %d. Loss: %.5f, Train acc %f."
                         % (iteration, epoch_loss, train_acc))
            prev_time = cur_time
            print("\033[32m" + epoch_str + time_str + 'lr ' + str(trainer.learning_rate) + "\033[0m")
            test(train_net, ctx, test_loader, iteration)
        if iteration in lr_period:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)


if __name__ == '__main__':
    ctx = mx.gpu(3)

    net = ResidualAttentionModel_92_32input_update()
    net.hybridize()
    net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
    iterations = 240e3
    train(train_net=net, iterations=iterations, lr=0.1, wd=1e-4, ctx=ctx, lr_period=(72e3, 144e3, 216e3), lr_decay=0.1,
          train_loader=train_data, test_loader=val_data, cat_interval=1e3)
