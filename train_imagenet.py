# @File  : train_imagenet.py
# @Author: X.Yang
# @Contact : pistonyang@gmail.com
# @Date  : 18-9-27

from model.residual_attention_network import ResidualAttentionModel
from mxnet.gluon.data.vision import ImageFolderDataset
import mxnet as mx
from mxnet import gluon, image, nd, autograd
from mxnet.gluon import loss as gloss, utils as gutils
import datetime
from lib.piston_util import format_time, inf_train_gen
import logging


def transformer(data, label):
    jitter_param = 0.4
    lighting_param = 0.1
    im = data
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224),
                                    rand_crop=True,
                                    rand_resize=True,
                                    rand_mirror=True,
                                    brightness=jitter_param,
                                    saturation=jitter_param,
                                    contrast=jitter_param,
                                    pca_noise=lighting_param,
                                    mean=True,
                                    std=True)

    for aug in auglist:
        im = aug(im)

    im = nd.transpose(im, (2, 0, 1))
    return im, label


def trans_test(data, label):
    im = data
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    mean=True,
                                    std=True)
    for aug in auglist:
        im = aug(im)

    im = nd.transpose(im, (2, 0, 1))
    return im, label


def test(test_net, ctx, test_loader, iteration, logger):
    # print("Start testing iter %d." % iteration)
    Loss = gloss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()
    metric_top5 = mx.metric.TopKAccuracy(5)
    test_loss = mx.metric.Loss()
    for batch in test_loader:
        trans = gutils.split_and_load(batch[0], ctx)
        labels = gutils.split_and_load(batch[1], ctx)
        outputs = [test_net(tran) for tran in trans]
        losses = [Loss(output, label) for output, label in zip(outputs, labels)]
        test_loss.update(0, losses)
        metric.update(labels, outputs)
        metric_top5.update(labels, outputs)
    _, test_top1_acc = metric.get()
    _, test_top5_acc = metric_top5.get()
    _, test_loss = test_loss.get()

    if test_top1_acc >= 0.7:
        test_net.save_parameters('imagenet_param/test_iter%d_%.5f.param' % (iteration, test_top1_acc))
    test_str = ("test_Loss: %f, test top1-acc %f, test top5-acc %f." % (test_loss, test_top1_acc, test_top5_acc))
    logger.info(test_str)


def train(train_net, iterations, trainer, ctx, lr_period: tuple, lr_decay, train_loader, test_loader, cat_interval):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = 'Attention56_train.log'
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    net.collect_params().reset_ctx(ctx)
    train_gen = inf_train_gen(train_loader)
    Loss = gloss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()
    metric_top5 = mx.metric.TopKAccuracy(5)
    train_loss = mx.metric.Loss()
    prev_time = datetime.datetime.now()

    metric.reset()
    train_loss.reset()

    for iteration in range(int(iterations)):
        batch = next(train_gen)
        trans = gutils.split_and_load(batch.data[0], ctx)
        labels = gutils.split_and_load(batch.label[0], ctx)

        with autograd.record():
            outputs = [train_net(tran) for tran in trans]
            losses = [Loss(output, label) for output, label in zip(outputs, labels)]

        for loss in losses:
            loss.backward()

        trainer.step(batch_size)
        train_loss.update(0, losses)
        metric.update(labels, outputs)
        metric_top5.update(labels, outputs)
        if iteration % cat_interval == cat_interval - 1:
            cur_time = datetime.datetime.now()
            time_str = format_time(prev_time, cur_time)
            _, top1_acc = metric.get()
            _, top5_acc = metric_top5.get()
            _, epoch_loss = train_loss.get()
            metric.reset()
            metric_top5.reset()
            train_loss.reset()
            epoch_str = ("Iter %d. Loss: %.5f, Train top1-acc %f, Train top5-acc %f."
                         % (iteration, epoch_loss, top1_acc, top5_acc))
            prev_time = cur_time
            logger.info(epoch_str + time_str + 'lr ' + str(trainer.learning_rate))
            test(train_net, ctx, test_loader, iteration, logger)
        if iteration in lr_period:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)


if __name__ == '__main__':
    batch_size = 64
    iterations = 530e3
    wd = 1e-4
    lr = 0.1
    lr_period = tuple([iterations * i for i in (0.3, 0.6, 0.9)])
    lr_decay = 0.1
    cat_interval = 10e3
    num_workers = 12
    num_gpus = 2
    ctx = [mx.gpu(i) for i in range(num_gpus)]

    net = ResidualAttentionModel()
    net.hybridize(static_alloc=True)
    net.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(),
                            'nag', {'learning_rate': lr,
                                    'momentum': 0.9,
                                    'wd': wd})

    train_data = gluon.data.DataLoader(
        ImageFolderDataset('/system1/Dataset/ImageNet/ILSVRC2012_img_train',
                           transform=transformer),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='discard')

    val_data = gluon.data.DataLoader(
        ImageFolderDataset('/system1/Dataset/ImageNet/test',
                           transform=trans_test),
        batch_size=batch_size, num_workers=num_workers)

    train(train_net=net, iterations=iterations, trainer=trainer, ctx=ctx, lr_period=lr_period,
          lr_decay=lr_decay, train_loader=train_data, test_loader=val_data, cat_interval=cat_interval)

