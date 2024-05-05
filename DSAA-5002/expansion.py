#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/11/19 10:43
# @Author: ShaineSecret
# @File: expansion.py
import numpy as np

def dk_function(X, knots):
    '''
    final function for prepocessing
    :param X: data
    :param knots: knots for expansion
    :return: new X
    '''
    X_dk = X
    k_j = len(knots)
    for h in range(k_j-3):
        # print('++++++++++++++++++++++++++++')
        # print('1', basis_fuction(X, knots, h))
        # print('============================')
        # print('2', basis_fuction(X, knots, k_j-2))
        # print('----------------------------')
        X_dk = np.column_stack([X_dk, basis_fuction(X, knots, h) - basis_fuction(X, knots, k_j-2)])
    return X_dk

def basis_fuction(X, knots, h):
    a = np.power(np.abs(X-knots[h]), 3)
    b = np.power(np.abs(X-knots[-1]), 3)
    # print('aaaaaaaaaaaaaaaaaaaaaaa')
    # print(a)
    # print('aaaaaaaaaaaaaaaaaaaaaaa')
    # print('bbbbbbbbbbbbbbbbbbbbb')
    # print(b)
    # print('bbbbbbbbbbbbbbbbbbbbb')
    return ((a - b) / (knots[-1] - knots[h]))

if __name__ == '__main__':
    X = np.random.randint(0,10,(4,3))
    print(dk_function(X, [0, 0.2, 0.4, 0.6, 0.8, 1]))