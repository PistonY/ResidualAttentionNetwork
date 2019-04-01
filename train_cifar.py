#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_cifar.py
# @Author: Piston Yang
# @Date  : 18-9-6


from model.residual_attention_network import ResidualAttentionModel_32input
import mxnet as mx
from mxnet import gluon, image, nd, autograd
from mxnet.gluon import loss as gloss
import numpy as np
import time
import os
from lib.piston_util import cutout
from gluoncv.utils.lr_scheduler import LRSequential, LRScheduler
import logging

os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'

batch_size = 128
train_samples = int(5e4)
dtype = 'float32'
assert dtype in ('float32', 'float16')
random_eraser = cutout()


def transformer(data, label):
    im = data.asnumpy()
    im = np.pad(im, pad_width=((4, 4), (4, 4), (0, 0)), mode='constant')
    im = random_eraser(im)
    im = nd.array(im) / 255.
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), rand_crop=True, rand_mirror=True,
                                    mean=mx.nd.array([0.4914, 0.4824, 0.4467]),
                                    std=mx.nd.array([0.2471, 0.2435, 0.2616]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, label


def trans_test(data, label):
    im = data.astype(np.float32) / 255.
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                                    mean=mx.nd.array([0.4914, 0.4824, 0.4467]),
                                    std=mx.nd.array([0.2471, 0.2435, 0.2616]))
    for aug in auglist:
        im = aug(im)

    im = nd.transpose(im, (2, 0, 1))
    return im, label


train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, num_workers=12, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=False, transform=trans_test),
    batch_size=batch_size, last_batch='keep')


def label_transform(label, classes):
    ind = label.astype('int')
    res = nd.zeros((ind.shape[0], classes), ctx=label.context)
    res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1
    return res


def test(test_net, ctx, test_loader, iteration, logger):
    Loss = gloss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()
    test_loss = mx.metric.Loss()
    for batch in test_loader:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [test_net(X.astype(dtype, copy=False)) for X in data]
        losses = [Loss(yhat, y.astype(dtype, copy=False)) for yhat, y in zip(outputs, label)]
        test_loss.update(0, losses)
        metric.update(label, outputs)

    _, test_acc = metric.get()
    _, test_loss = test_loss.get()
    test_net.save_parameters('cifar_param/test_epoch%d_%.5f.param' % (iteration, test_acc))
    test_str = ("Test Loss: %f, Test acc %f." % (test_loss, test_acc))
    logger.info(test_str)


def train(train_net, epochs, lr, wd, ctx, warmup_epochs, train_loader, test_loader, use_mixup, logger):
    num_batches = train_samples // batch_size
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=lr,
                    nepochs=warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler('cosine', base_lr=lr, target_lr=0,
                    nepochs=epochs - warmup_epochs,
                    iters_per_epoch=num_batches)
    ])
    opt_params = {'learning_rate': lr, 'momentum': 0.9, 'wd': wd, 'lr_scheduler': lr_scheduler}
    if dtype != 'float32':
        opt_params['multi_precision'] = True
    trainer = gluon.Trainer(train_net.collect_params(), 'nag', opt_params)

    Loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    metric = mx.metric.RMSE()
    train_loss = mx.metric.Loss()
    alpha = 1
    classes = 10

    print("Start training with mixup.")
    for epoch in range(epochs):
        metric.reset()
        train_loss.reset()
        st_time = time.time()
        for i, batch in enumerate(train_loader):
            lam = np.random.beta(alpha, alpha)
            if epoch >= (epochs - 20) or not use_mixup:
                lam = 1

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            trans = [lam * X + (1 - lam) * X[::-1] for X in data]
            labels = []
            for Y in label:
                y1 = label_transform(Y, classes)
                y2 = label_transform(Y[::-1], classes)
                labels.append(lam * y1 + (1 - lam) * y2)

            with autograd.record():
                outputs = [train_net(X.astype(dtype, copy=False)) for X in trans]
                losses = [Loss(yhat, y.astype(dtype, copy=False)) for yhat, y in zip(outputs, labels)]
            for l in losses:
                l.backward()
            trainer.step(batch_size)
            train_loss.update(0, losses)
            metric.update(labels, outputs)

        cur_time = time.time() - st_time
        eps_samples = int(train_samples // cur_time)
        _, train_acc = metric.get()
        _, epoch_loss = train_loss.get()
        epoch_str = ("Epoch %d. Loss: %.5f, Train RMSE %.5f. %d samples/s. lr %.5f"
                     % (epoch, epoch_loss, train_acc, eps_samples, trainer.learning_rate))
        logger.info(epoch_str)
        test(train_net, ctx, test_loader, epoch, logger)


if __name__ == '__main__':
    from gluoncv.model_zoo.residual_attentionnet import cifar_residualattentionnet452

    ctx = [mx.gpu(i) for i in range(4)]
    assert batch_size // len(ctx) == 0, "Pre batch on each GPU should be same."
    mix_up = True
    no_wd = True

    # net = ResidualAttentionModel_32input(additional_stage=True)
    net = cifar_residualattentionnet452()
    net.hybridize(static_alloc=True, static_shape=True)
    net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
    if dtype != 'float32':
        net.cast('float16')
    if no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = 'Attention92_cifar10_train.log'
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    epochs = 220
    lr = 0.1 * (batch_size // 64)
    wd = 1e-4

    train(train_net=net, epochs=epochs, lr=lr, wd=wd, ctx=ctx, warmup_epochs=5,
          train_loader=train_data, test_loader=val_data, use_mixup=mix_up, logger=logger)
