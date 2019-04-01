# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 4/1/19

import unittest
from mxnet import nd
from model.residual_attention_network import ResidualAttentionModel_32input, ResidualAttentionModel


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cifar_data = nd.random.normal(shape=(1, 3, 32, 32))
        self.cifar_att56 = ResidualAttentionModel_32input()
        self.cifar_att56.initialize()

        self.cifar_att92 = ResidualAttentionModel_32input(additional_stage=True)
        self.cifar_att92.initialize()

        self.imgnet_data = nd.random.normal(shape=(1, 3, 224, 224))
        self.att56 = ResidualAttentionModel()
        self.att56.initialize()

        self.att92 = ResidualAttentionModel(additional_stage=True)
        self.att92.initialize()

    def test_model(self):
        self.assertEqual((1, 10), self.cifar_att56(self.cifar_data).shape)
        self.assertEqual((1, 10), self.cifar_att92(self.cifar_data).shape)
        self.assertEqual((1, 1000), self.att56(self.imgnet_data).shape)
        self.assertEqual((1, 1000), self.att92(self.imgnet_data).shape)


if __name__ == '__main__':
    unittest.main()
