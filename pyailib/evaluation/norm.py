#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np


def frobenius(X, caxis=None, p=2, reduction='mean'):
    r"""Compute frobenius norm

    .. math::
       \|\bm X\|_p^p = (\sum{x^p})^{1/p}

    Parameters
    ----------
    X : numpy array
        The complex or real inputs, for complex inputs, both complex and real representations are surpported.
    caxis : int or None
        If :attr:`X` is complex-valued, :attr:`caxis` is ignored. If :attr:`X` is real-valued and :attr:`caxis` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`caxis` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    p : int, optional
        Description
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : scalar or numpy array
        The norm of the inputs.

    """

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

    D = X.ndim
    dim = tuple(range(1, D))
    X = np.power(np.mean(np.power(X, p), axis=dim), 1. / p)

    if reduction == 'mean':
        F = np.mean(X)
    if reduction == 'sum':
        F = np.sum(X)

    return F


if __name__ == '__main__':

    X = np.random.randn(1, 3, 4, 2)
    V = frobenius(X, caxis=-1, p=2, reduction='mean')
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = frobenius(X, caxis=None, p=2, reduction='mean')
    print(V)
