#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import numpy as np
import pyailib as pl

X = np.random.randn(1, 3, 4, 2)
print(X.shape)
V = pl.contrast(X, caxis=None, mode='way1', reduction='mean')
print(V)

X = np.random.randn(1, 3, 4, 2)
print(X.shape)
V = pl.contrast(X, caxis=-1, mode='way1', reduction='mean')
print(V)

X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
print(X.shape)
V = pl.contrast(X, caxis=None, mode='way1', reduction='mean')
print(V)
