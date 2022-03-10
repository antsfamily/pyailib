#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
from pyailib.utils.const import EPS


def contrast(X, caxis=None, mode='way1', reduction='mean'):
    r"""Compute contrast of an complex image

    ``'way1'`` is defined as follows, see [1]:

    .. math::
       C = \frac{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}{{\rm E}(|I|^2)}


    ``'way2'`` is defined as follows, see [2]:

    .. math::
        C = \frac{{\rm E}(|I|^2)}{\left({\rm E}(|I|)\right)^2}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"

    Parameters
    ----------
    X : numpy ndarray
        The image array.
    caxis : int or None
        If :attr:`X` is complex-valued, :attr:`caxis` is ignored. If :attr:`X` is real-valued and :attr:`caxis` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`caxis` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    mode : str, optional
        ``'way1'`` or ``'way2'``
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    scalar
        The contrast value of input.

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

    D = X.ndim
    axis = tuple(range(1, D))

    if X.dtype not in [np.float32, np.float64]:
        X = X.to(np.float32)

    if mode in ['way1', 'WAY1']:
        Xmean = X.mean(axis=axis, keepdims=True)
        C = np.sqrt(np.power(X - Xmean, 2).mean(axis=axis, keepdims=True)) / (Xmean + EPS)
    if mode in ['way2', 'WAY2']:
        C = X.mean(axis=axis, keepdims=True) / (np.pow((np.sqrt(X).mean(axis=axis, keepdims=True)), 2) + EPS)

    if reduction == 'mean':
        C = np.mean(C)
    if reduction == 'sum':
        C = np.sum(C)
    return C


if __name__ == '__main__':

    X = np.random.randn(1, 3, 4, 2)
    print(X.shape)
    V = contrast(X, caxis=None, mode='way1', reduction='mean')
    print(V)

    X = np.random.randn(1, 3, 4, 2)
    print(X.shape)
    V = contrast(X, caxis=-1, mode='way1', reduction='mean')
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = contrast(X, caxis=None, mode='way1', reduction='mean')
    print(V)
