#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
from pyailib.utils.const import EPS


def entropy(X, caxis=None, mode='shannon', reduction='mean'):
    r"""compute the entropy of the inputs

    .. math::
        {\rm ENT} = -\sum_{n=0}^N p_i{\rm log}_2 p_n

    where $N$ is the number of pixels, $p_n=\frac{|X_n|^2}{\sum_{n=0}^N|X_n|^2}$.

    Parameters
    ----------
    X : numpy array
        The complex or real inputs, for complex inputs, both complex and real representations are surpported.
    caxis : int or None
        If :attr:`X` is complex-valued, :attr:`caxis` is ignored. If :attr:`X` is real-valued and :attr:`caxis` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`caxis` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    mode : str, optional
        The entropy mode: ``'shannon'`` or ``'natural'`` (the default is 'shannon')
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : scalar or numpy array
        The entropy of the inputs.
    """

    if mode in ['Shannon', 'shannon', 'SHANNON']:
        logfunc = np.log2
    if mode in ['Natural', 'natural', 'NATURAL']:
        logfunc = np.log

    if np.iscomplex(X).any():
        X = (X * X.conj()).real
    else:
        if type(caxis) is int:
            if X.shape[caxis] != 2:
                raise ValueError('The complex input is represented in real-valued formation, but you specifies wrong axis!')
            X = np.power(X, 2).sum(axis=caxis)
        if caxis is None:
            X = np.power(X, 2)

    if X.dtype not in [np.float32, np.float64]:
        X = X.to(np.float32)

    if X.ndim == 2:
        axis = (0, 1)
    if X.ndim == 3:
        axis = (1, 2)
    if X.ndim == 4:
        axis = (1, 2, 3)
    if X.ndim == 5:
        axis = (1, 2, 3, 4)

    P = np.sum(X, axis, keepdims=True)
    p = X / (P + EPS)
    S = -np.sum(p * logfunc(p + EPS), axis)
    if reduction == 'mean':
        S = np.mean(S)
    if reduction == 'sum':
        S = np.sum(S)

    return S


if __name__ == '__main__':

    X = np.random.randn(1, 3, 4, 2)
    S = entropy(X, caxis=-1, mode='shannon')
    print(S)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    S = entropy(X, caxis=None, mode='shannon')
    print(S)
