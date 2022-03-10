#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


import pyailib as pl

files = pl.listxfile(listdir='../../data/images/', exts='.png', recursive=False, filelist=[])
print(files)

filepath = pl.pathjoin('a', 'b', 'c', '.d')
print(filepath)

filepath = '../../data/files/log.log'

v = pl.readnum(filepath, pmain='Train', psub='loss: ', vfn=float, nshots=10)
print(v)
