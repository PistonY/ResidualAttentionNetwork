#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : piston_util.py
# @Author: Piston Yang
# @Date  : 18-9-7
import numpy as np


def format_time(prev_time, cur_time) -> str:
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d." % (h, m, s)
    return time_str


def inf_train_gen(loader):
    while True:
        for batch in loader:
            yield batch


def rescale(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min().asscalar()
    if x_max is None:
        x_max = x.max().asscalar()
    return (x - x_min) / (x_max - x_min)


def cutout(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    # copied by:https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


if __name__ == '__main__':
    from gluoncv.model_zoo import cifar_residualattentionnet92
    from mxnet import nd

    net = cifar_residualattentionnet92()
    net.initialize()
    net.summary(nd.random.randn(1, 3, 32, 32))
