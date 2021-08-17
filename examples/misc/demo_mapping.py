#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import pyailib as pl
import matplotlib.pyplot as plt


# datafile = '/mnt/d/DataSets/sar/ERS/mat/E2_81988_STD_F327/E2_81988_STD_F327/ERS2_SAR_SLC=E2_81988_STD_F327(sl=1el=8192).mat'

# sardata = pl.loadmat(datafile)['sardata']

# print(sardata[0][0][1].shape)
# SI = sardata[0][0][1]
# SI = th.from_numpy(SI)

datafile = '/mnt/e/ws/github/antsfamily/torchsar/torchsar/examples/imaging/data/ALPSRP020160970SI.tiff'

SI = pl.imread(datafile)
print(SI.shape, SI.dtype)


SI1 = pl.mapping(SI, method='1Sigma')
SI2 = pl.mapping(SI, method='5Sigma')
SI3 = pl.mapping(SI, method='10Sigma')

plt.figure()
plt.imshow(SI, cmap='gray')
plt.figure()
plt.imshow(SI1, cmap='gray')
plt.figure()
plt.imshow(SI2, cmap='gray')
plt.figure()
plt.imshow(SI3, cmap='gray')

plt.show()
