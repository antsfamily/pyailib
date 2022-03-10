#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import numpy as np
import pyailib as pl


X = np.random.randn(1, 3, 4, 2)
V = pl.frobenius(X, caxis=-1, p=2, reduction='mean')
print(V)

X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
V = pl.frobenius(X, caxis=None, p=2, reduction='mean')
print(V)
