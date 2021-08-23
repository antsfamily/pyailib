#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-07-06 09:32:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

import pyailib as pl
import matplotlib.pyplot as plt

bboxes = [[100, 100, 200, 200], [300, 300, 400, 500]]
labels = ['dog', 'cat']
scores = [0.987, None]
edgecolors = [pl.DISTINCT_COLORS['DistinctNormRGB20'][0], None]
edgecolors = pl.DISTINCT_COLORS['DistinctNormRGB20'][0:2]
linewidths = 1

fontdict = {'family': 'Times New Roman',
            'style': 'italic',
            'weight': 'normal',
            'color': 'darkred',
            'size': 12,
            }

x = pl.imread('../../data/images/LenaRGB512.tif')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(x)

pl.plot_bbox(bboxes, labels=labels, scores=scores, edgecolors=edgecolors, linewidths=linewidths, fontdict=fontdict, textpos='TopLeft', ax=ax)
plt.axis('off')
plt.savefig('./bbbox.png', bbox_inches='tight', pad_inches=0)
plt.show()
