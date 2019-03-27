# -*- coding: utf-8 -*-
# Author: X.Yang
# Concat: pistonyang@gmail.com
# Date  : 3/27/19

__all__ = ['get_ems']


def get_ems(net: list, trans, rate: list):
    assert len(net) == len(rate)
    output = []
    for n, r in zip(net, rate):
        output.append(n(trans) * r)
    return sum(output)
