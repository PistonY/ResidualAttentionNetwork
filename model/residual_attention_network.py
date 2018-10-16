#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : residual_attention_network.py
# @Author: Piston Yang
# @Date  : 18-9-6

from __future__ import absolute_import
from mxnet.gluon import nn
from model.attention_module import *


class ResidualAttentionModel_448input(nn.HybridBlock):

    def __init__(self, classes=1000, additional_stage=False, **kwargs):
        """
         Input size is 448
        :param classes: Output classes
        :param additional_stage: If False means Attention56, True means Attention92
        :param kwargs:
        """
        self.additional_stage = additional_stage
        super(ResidualAttentionModel_448input, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            # 112 x 112
            self.residual_block0 = ResidualBlock(128, in_channels=64)
            self.attention_module0 = AttentionModule_stage0(128)
            self.residual_block1 = ResidualBlock(256, in_channels=128, stride=2)
            # 56 x 56
            self.attention_module1 = AttentionModule_stage1(256)
            self.residual_block2 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module2 = AttentionModule_stage2(512)
            if additional_stage:
                self.attention_module2_2 = AttentionModule_stage2(512)
            self.residual_block3 = ResidualBlock(1024, in_channels=512, stride=2)
            self.attention_module3 = AttentionModule_stage3(1024)
            if additional_stage:
                self.attention_module3_2 = AttentionModule_stage3(1024)
                self.attention_module3_3 = AttentionModule_stage3(1024)
            self.residual_block4 = ResidualBlock(2048, in_channels=1024, stride=2)
            self.residual_block5 = ResidualBlock(2048)
            self.residual_block6 = ResidualBlock(2048)
            self.mpool2 = nn.HybridSequential()
            with self.mpool2.name_scope():
                self.mpool2.add(nn.BatchNorm())
                self.mpool2.add(nn.Activation('relu'))
                self.mpool2.add(nn.AvgPool2D(pool_size=7, strides=1))
            self.fc = nn.Conv2D(classes, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.mpool1(x)
        x = self.residual_block0(x)
        x = self.attention_module0(x)

        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        if self.additional_stage:
            x = self.attention_module2_2(x)
        x = self.residual_block3(x)

        x = self.attention_module3(x)
        if self.additional_stage:
            x = self.attention_module3_2(x)
            x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)

        return x


class ResidualAttentionModel(nn.HybridBlock):

    def __init__(self, classes=1000, additional_stage=False, **kwargs):
        super(ResidualAttentionModel, self).__init__(**kwargs)
        """
        input size 224
        :param classes: Output classes 
        :param additional_stage: If False means Attention56, True means Attention92.
        :param kwargs: 
        """
        self.additional_stage = additional_stage
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.residual_block1 = ResidualBlock(256, in_channels=64)
            self.attention_module1 = AttentionModule_stage1(256)
            self.residual_block2 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module2 = AttentionModule_stage2(512)
            if additional_stage:
                self.attention_module2_2 = AttentionModule_stage2(512)
            self.residual_block3 = ResidualBlock(1024, in_channels=512, stride=2)
            self.attention_module3 = AttentionModule_stage3(1024)
            if additional_stage:
                self.attention_module3_2 = AttentionModule_stage3(1024)
                self.attention_module3_3 = AttentionModule_stage3(1024)
            self.residual_block4 = ResidualBlock(2048, in_channels=1024, stride=2)
            self.residual_block5 = ResidualBlock(2048)
            self.residual_block6 = ResidualBlock(2048)
            self.mpool2 = nn.HybridSequential()
            with self.mpool2.name_scope():
                self.mpool2.add(nn.BatchNorm())
                self.mpool2.add(nn.Activation('relu'))
                self.mpool2.add(nn.AvgPool2D(pool_size=7, strides=1))
            self.fc = nn.Conv2D(classes, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.mpool1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        if self.additional_stage:
            x = self.attention_module2_2(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        if self.additional_stage:
            x = self.attention_module3_2(x)
            x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)

        return x


class ResidualAttentionModel_32input(nn.HybridBlock):
    def __init__(self, classes=10, additional_stage=False, **kwargs):
        super(ResidualAttentionModel_32input, self).__init__(**kwargs)
        """
        Input size 32
        :param classes: Output classes, default 10
        :param additional_stage: If False means Attention56, True means Attention92. default False
        :param kwargs:
        """
        self.additional_stage = additional_stage
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            # 32 x 32
            # self.mpool1 = nn.MaxPool2D(pool_size=2, strides=2, padding=0)

            self.residual_block1 = ResidualBlock(128, in_channels=32)
            self.attention_module1 = AttentionModule_stage2(128, size1=32, size2=16)
            self.residual_block2 = ResidualBlock(256, in_channels=128, stride=2)
            self.attention_module2 = AttentionModule_stage3(256, size1=16)
            if additional_stage:
                self.attention_module2_2 = AttentionModule_stage3(256, size1=16)
            self.residual_block3 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module3 = AttentionModule_stage4(512)
            if additional_stage:
                self.attention_module3_2 = AttentionModule_stage4(512)
                self.attention_module3_3 = AttentionModule_stage4(512)
            self.residual_block4 = ResidualBlock(1024, in_channels=512)
            self.residual_block5 = ResidualBlock(1024)
            self.residual_block6 = ResidualBlock(1024)
            self.mpool2 = nn.HybridSequential()
            with self.mpool2.name_scope():
                self.mpool2.add(nn.BatchNorm())
                self.mpool2.add(nn.Activation('relu'))
                self.mpool2.add(nn.AvgPool2D(pool_size=8, strides=1))
            self.fc = nn.Conv2D(classes, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        if self.additional_stage:
            x = self.attention_module2_2(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        if self.additional_stage:
            x = self.attention_module3_2(x)
            x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)

        return x


if __name__ == '__main__':
    cifar_net = ResidualAttentionModel_32input(additional_stage=True)
    cifar_net.initialize()
    print(cifar_net(nd.random.normal(shape=(3, 3, 32, 32))))
