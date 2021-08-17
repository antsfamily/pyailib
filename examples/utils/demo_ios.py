#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import pyailib as pl

a = np.random.randn(3, 4)
b = 10
c = [1, 2, 3]
d = {'1': 1, '2': a}
s = 'Hello, the future!'
t = (0, 1)

pl.saveh5('./data.h5', {'a': a, 'b': b, 'c': c, 'd': d, 's': s})
data = pl.loadh5('./data.h5', keys=['a', 's'])
print(data.keys())

print("==========")
# saveh5('./data.h5', {'t': t}, 'w')
pl.saveh5('./data.h5', {'t': t}, 'a')
pl.saveh5('./data.h5', {'t': (2, 3, 4)}, 'a')
data = pl.loadh5('./data.h5')

for k, v in data.items():
    print(k, v)

x = pl.loadyaml('../../data/files/demo.yaml', 'trainid')
print(x, type(x))
x = pl.loadjson('../../data/files/demo.json', 'trainid')
print(x, type(x))

x = pl.loadyaml('../../data/files/demo.yaml', 'lr')
print(x, type(x))
x = pl.loadjson('../../data/files/demo.json', 'lr')
print(x, type(x))
