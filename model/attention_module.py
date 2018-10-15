#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : attention_module.py
# @Author: Piston Yang
# @Date  : 18-9-

from mxnet.gluon import nn
from mxnet import nd
from .basic_layer import ResidualBlock


class UpsamplingBilinear2d(nn.HybridBlock):
    def __init__(self, size, **kwargs):
        super(UpsamplingBilinear2d, self).__init__(**kwargs)
        self.size = size

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.contrib.BilinearResize2D(x, self.size, self.size)


class AttentionModule_stage0(nn.HybridBlock):
    def __init__(self, channels, size1=112, size2=56, size3=28, size4=14, **kwargs):
        """
        Input size is 112 x 112
        :param channels:
        :param type:
        :param size1:
        :param size2:
        :param size3:
        :param size4:
        :param kwargs:
        """
        super(AttentionModule_stage0, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax1_blocks = ResidualBlock(channels)
            self.skip1_connection_residual_block = ResidualBlock(channels)

            self.mpool2 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax2_blocks = ResidualBlock(channels)
            self.skip2_connection_residual_block = ResidualBlock(channels)

            self.mpool3 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax3_blocks = ResidualBlock(channels)
            self.skip3_connection_residual_block = ResidualBlock(channels)

            self.mpool4 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax4_blocks = nn.HybridSequential()
            with self.softmax4_blocks.name_scope():
                self.softmax4_blocks.add(ResidualBlock(channels))
                self.softmax4_blocks.add(ResidualBlock(channels))

            self.interpolation4 = UpsamplingBilinear2d(size=size4)
            self.softmax5_blocks = ResidualBlock(channels)

            self.interpolation3 = UpsamplingBilinear2d(size=size3)
            self.softmax6_blocks = ResidualBlock(channels)

            self.interpolation2 = UpsamplingBilinear2d(size=size2)
            self.softmax7_blocks = ResidualBlock(channels)

            self.interpolation1 = UpsamplingBilinear2d(size=size1)

            self.softmax8_blocks = nn.HybridSequential()
            with self.softmax8_blocks.name_scope():
                self.softmax8_blocks.add(nn.BatchNorm())
                self.softmax8_blocks.add(nn.Activation('relu'))
                self.softmax8_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax8_blocks.add(nn.BatchNorm())
                self.softmax8_blocks.add(nn.Activation('relu'))
                self.softmax8_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax8_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        # 56 x 56

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        # 28 x 28

        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)
        # 14 x 14

        out_mpool4 = self.mpool4(out_softmax3)
        out_softmax4 = self.softmax4_blocks(out_mpool4)
        # 7 x 7

        out_interp4 = F.elemwise_add(self.interpolation4(out_softmax4), out_softmax3)
        out = F.elemwise_add(out_interp4, out_skip3_connection)

        out_softmax5 = self.softmax5_blocks(out)
        out_interp3 = F.elemwise_add(self.interpolation3(out_softmax5), out_softmax2)
        out = F.elemwise_add(out_interp3, out_skip2_connection)

        out_softmax6 = self.softmax5_blocks(out)
        out_interp2 = F.elemwise_add(self.interpolation2(out_softmax6), out_softmax1)
        out = F.elemwise_add(out_interp2, out_skip1_connection)

        out_softmax7 = self.softmax7_blocks(out)
        out_interp1 = F.elemwise_add(self.interpolation1(out_softmax7), out_trunk)

        out_softmax8 = self.softmax8_blocks(out_interp1)
        out = F.elemwise_add(F.ones_like(out_softmax8), out_softmax8)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage1(nn.HybridBlock):
    def __init__(self, channels, size1=56, size2=28, size3=14, **kwargs):
        """
        Input size is 56 x 56
        :param channels:
        :param type:
        :param size1:
        :param size2:
        :param size3:
        """
        super(AttentionModule_stage1, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax1_blocks = ResidualBlock(channels)
            self.skip1_connection_residual_block = ResidualBlock(channels)

            self.mpool2 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax2_blocks = ResidualBlock(channels)
            self.skip2_connection_residual_block = ResidualBlock(channels)

            self.mpool3 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.softmax3_blocks = nn.HybridSequential()
            with self.softmax3_blocks.name_scope():
                self.softmax3_blocks.add(ResidualBlock(channels))
                self.softmax3_blocks.add(ResidualBlock(channels))

            self.interpolation3 = UpsamplingBilinear2d(size=size3)
            self.softmax4_blocks = ResidualBlock(channels)

            self.interpolation2 = UpsamplingBilinear2d(size=size2)
            self.softmax5_blocks = ResidualBlock(channels)

            self.interpolation1 = UpsamplingBilinear2d(size=size1)

            self.softmax6_blocks = nn.HybridSequential()
            with self.softmax6_blocks.name_scope():
                self.softmax6_blocks.add(nn.BatchNorm())
                self.softmax6_blocks.add(nn.Activation('relu'))
                self.softmax6_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax6_blocks.add(nn.BatchNorm())
                self.softmax6_blocks.add(nn.Activation('relu'))
                self.softmax6_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax6_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)

        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)

        out_interp3 = F.elemwise_add(self.interpolation3(out_softmax3), out_softmax2)
        out = F.elemwise_add(out_interp3, out_skip2_connection)

        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = F.elemwise_add(self.interpolation2(out_softmax4), out_softmax1)
        out = F.elemwise_add(out_interp2, out_skip1_connection)

        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = F.elemwise_add(self.interpolation1(out_softmax5), out_trunk)

        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = F.elemwise_add(F.ones_like(out_softmax6), out_softmax6)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage2(nn.HybridBlock):
    def __init__(self, channels, size1=28, size2=14, **kwargs):
        """
        Input size is 28 x 28
        :param channels:
        :param type:
        :param size1:
        :param size2:
        """
        super(AttentionModule_stage2, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax1_blocks = ResidualBlock(channels)
            self.skip1_connection_residual_block = ResidualBlock(channels)

            self.mpool2 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.softmax2_blocks = nn.HybridSequential()
            with self.softmax2_blocks.name_scope():
                self.softmax2_blocks.add(ResidualBlock(channels))
                self.softmax2_blocks.add(ResidualBlock(channels))

            self.interpolation2 = UpsamplingBilinear2d(size=size2)
            self.softmax3_blocks = ResidualBlock(channels)

            self.interpolation1 = UpsamplingBilinear2d(size=size1)

            self.softmax4_blocks = nn.HybridSequential()
            with self.softmax4_blocks.name_scope():
                self.softmax4_blocks.add(nn.BatchNorm())
                self.softmax4_blocks.add(nn.Activation('relu'))
                self.softmax4_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax4_blocks.add(nn.BatchNorm())
                self.softmax4_blocks.add(nn.Activation('relu'))
                self.softmax4_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax4_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_interp2 = F.elemwise_add(self.interpolation2(out_softmax2), out_softmax1)
        out = F.elemwise_add(out_interp2, out_skip1_connection)

        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = F.elemwise_add(self.interpolation1(out_softmax3), out_trunk)

        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = F.elemwise_add(F.ones_like(out_softmax4), out_softmax4)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage3(nn.HybridBlock):
    def __init__(self, channels, size1=14, **kwargs):
        """
        Input size is 14 x 14
        :param channels:
        :param type:
        :param size1:
        :param size2:
        :param size3:
        :param size4:
        :param kwargs:
        """
        super(AttentionModule_stage3, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.softmax1_blocks = nn.HybridSequential()
            with self.softmax1_blocks.name_scope():
                self.softmax1_blocks.add(ResidualBlock(channels))
                self.softmax1_blocks.add(ResidualBlock(channels))

            self.interpolation1 = UpsamplingBilinear2d(size=size1)

            self.softmax2_blocks = nn.HybridSequential()
            with self.softmax2_blocks.name_scope():
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_interp1 = F.elemwise_add(self.interpolation1(out_softmax1), out_trunk)

        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = F.elemwise_add(F.ones_like(out_softmax2), out_softmax2)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage4(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        """
        Input size is 14 x 14
        :param channels:
        :param type:
        :param size1:
        :param size2:
        :param size3:
        :param size4:
        :param kwargs:
        """
        super(AttentionModule_stage4, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.softmax1_blocks = nn.HybridSequential()
            with self.softmax1_blocks.name_scope():
                self.softmax1_blocks.add(ResidualBlock(channels))
                self.softmax1_blocks.add(ResidualBlock(channels))

            self.softmax2_blocks = nn.HybridSequential()
            with self.softmax2_blocks.name_scope():
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_softmax1 = self.softmax1_blocks(x)

        out_softmax2 = self.softmax2_blocks(out_softmax1)
        out = F.elemwise_add(F.ones_like(out_softmax2), out_softmax2)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)

        return out_last
