#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-05 16:36:03
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

import numpy as np
from .arrayops import sl


def nextpow2(x):
    r"""get the next higher power of 2.

    Given an number :math:`x`, returns the first p such that :math:`2^p >=|x|`. 

    Args:
        x (int or float): an number.

    Returns:
        int: Next higher power of 2.

    Examples:

        ::

            print(prevpow2(-5), nextpow2(-5))
            print(prevpow2(5), nextpow2(5))
            print(prevpow2(0.3), nextpow2(0.3))
            print(prevpow2(7.3), nextpow2(7.3))
            print(prevpow2(-3.5), nextpow2(-3.5))

            # output
            2 3
            2 3
            -2 -1
            2 3
            1 2

    """

    return int(np.ceil(np.log2(np.abs(x) + 1e-32)))


def prevpow2(x):
    r"""get the previous lower power of 2.

    Given an number :math:`x`, returns the first p such that :math:`2^p <=|x|`. 

    Args:
        x (int or float): an number.

    Returns:
        int: Next higher power of 2.

    Examples:

        ::

            print(prevpow2(-5), nextpow2(-5))
            print(prevpow2(5), nextpow2(5))
            print(prevpow2(0.3), nextpow2(0.3))
            print(prevpow2(7.3), nextpow2(7.3))
            print(prevpow2(-3.5), nextpow2(-3.5))

            # output
            2 3
            2 3
            -2 -1
            2 3
            1 2

    """
    
    return int(np.floor(np.log2(np.abs(x) + 1e-32)))


def ebeo(a, b, op='+'):
    r"""element by element operation

    Element by element operation.

    Parameters
    ----------
    a : list, tuple or ndarray
        The first list/tuple/ndarray.
    b : list, tuple or ndarray
        The second list/tuple/ndarray.
    op : str, optional
        Supported operations are:
        - ``'+'`` or ``'add'`` for addition (default)
        - ``'-'`` or ``'sub'`` for substraction
        - ``'*'`` or ``'mul'`` for multiplication
        - ``'/'`` or ``'div'`` for division
        - ``'**'`` or ``pow`` for power
        - ``'<'``, or ``'lt'`` for less than
        - ``'<='``, or ``'le'`` for less than or equal to
        - ``'>'``, or ``'gt'`` for greater than
        - ``'>='``, or ``'ge'`` for greater than or equal to
        - ``'&'`` for bitwise and
        - ``'|'`` for bitwise or
        - ``'^'`` for bitwise xor
        - function for custom operation.

    Raises
    ------
    TypeError
        If the specified operator not in the above list, raise a TypeError.
    """
    if op in ['+', 'add']:
        return [i + j for i, j in zip(a, b)]
    if op in ['-', 'sub']:
        return [i - j for i, j in zip(a, b)]
    if op in ['*', 'mul']:
        return [i * j for i, j in zip(a, b)]
    if op in ['/', 'div']:
        return [i / j for i, j in zip(a, b)]
    if op in ['**', '^', 'pow']:
        return [i ** j for i, j in zip(a, b)]
    if isinstance(op, str):
        raise TypeError("Not supported operation: " + op + "!")
    else:
        return [op(i, j) for i, j in zip(a, b)]


def real2cplx(X, axis=-1):
    r"""convert real-valued array to complex-valued array

    Convert real-valued array (the size of :attr:`axis` -th dimension is 2) to complex-valued array

    Args:
        X (numpy array): real-valued array.
        axis (int, optional): the complex axis. Defaults to -1.

    Returns:
        numpy array: complex-valued array

    Examples:

        ::

            import numpy as np

            np.random.seed(2020)

            Xreal = np.random.randint(0, 30, (3, 2, 4))
            Xcplx = real2cplx(Xreal, axis=1)
            Yreal = cplx2real(Xcplx, axis=0)

            print(Xreal, Xreal.shape, 'Xreal')
            print(Xcplx, Xcplx.shape, 'Xcplx')
            print(Yreal, Yreal.shape, 'Yreal')
            print(np.sum(Yreal[0] - Xcplx.real), np.sum(Yreal[1] - Xcplx.imag), 'Error')

            # output
            [[[ 0  8  3 22]
            [ 3 27 29  3]]

            [[ 7 24 29 16]
            [ 0 24 10  9]]

            [[19 11 23 18]
            [ 3  6  5 16]]] (3, 2, 4) Xreal

            [[[ 0. +3.j  8.+27.j  3.+29.j 22. +3.j]]

            [[ 7. +0.j 24.+24.j 29.+10.j 16. +9.j]]

            [[19. +3.j 11. +6.j 23. +5.j 18.+16.j]]] (3, 1, 4) Xcplx

            [[[[ 0.  8.  3. 22.]]

            [[ 7. 24. 29. 16.]]

            [[19. 11. 23. 18.]]]


            [[[ 3. 27. 29.  3.]]

            [[ 0. 24. 10.  9.]]

            [[ 3.  6.  5. 16.]]]] (2, 3, 1, 4) Yreal

            0.0 0.0, Error
    """

    idxreal = sl(np.ndim(X), axis=axis, idx=[[0]])
    idximag = sl(np.ndim(X), axis=axis, idx=[[1]])

    return X[idxreal] + 1j * X[idximag]


def cplx2real(X, axis=-1):
    r"""convert complex-valued array to real-valued array

    Args:
        X (numpy array): complex-valued array
        axis (int, optional): complex axis for real-valued array. Defaults to -1.

    Returns:
        numpy array: real-valued array

    Examples:

        ::

            import numpy as np

            np.random.seed(2020)

            Xreal = np.random.randint(0, 30, (3, 2, 4))
            Xcplx = real2cplx(Xreal, axis=1)
            Yreal = cplx2real(Xcplx, axis=0)

            print(Xreal, Xreal.shape, 'Xreal')
            print(Xcplx, Xcplx.shape, 'Xcplx')
            print(Yreal, Yreal.shape, 'Yreal')
            print(np.sum(Yreal[0] - Xcplx.real), np.sum(Yreal[1] - Xcplx.imag), 'Error')

            # output
            [[[ 0  8  3 22]
            [ 3 27 29  3]]

            [[ 7 24 29 16]
            [ 0 24 10  9]]

            [[19 11 23 18]
            [ 3  6  5 16]]] (3, 2, 4) Xreal

            [[[ 0. +3.j  8.+27.j  3.+29.j 22. +3.j]]

            [[ 7. +0.j 24.+24.j 29.+10.j 16. +9.j]]

            [[19. +3.j 11. +6.j 23. +5.j 18.+16.j]]] (3, 1, 4) Xcplx

            [[[[ 0.  8.  3. 22.]]

            [[ 7. 24. 29. 16.]]

            [[19. 11. 23. 18.]]]


            [[[ 3. 27. 29.  3.]]

            [[ 0. 24. 10.  9.]]

            [[ 3.  6.  5. 16.]]]] (2, 3, 1, 4) Yreal

            0.0 0.0, Error
    """

    return np.stack((X.real, X.imag), axis=axis)


if __name__ == '__main__':

    import numpy as np

    np.random.seed(2020)

    Xreal = np.random.randint(0, 30, (3, 2, 4))
    Xcplx = real2cplx(Xreal, axis=1)
    Yreal = cplx2real(Xcplx, axis=0)

    print(Xreal, Xreal.shape, 'Xreal')
    print(Xcplx, Xcplx.shape, 'Xcplx')
    print(Yreal, Yreal.shape, 'Yreal')
    print(np.sum(Yreal[0] - Xcplx.real), np.sum(Yreal[1] - Xcplx.imag), 'Error')

    print(prevpow2(-5), nextpow2(-5))
    print(prevpow2(5), nextpow2(5))
    print(prevpow2(0.3), nextpow2(0.3))
    print(prevpow2(7.3), nextpow2(7.3))
    print(prevpow2(-3.5), nextpow2(-3.5))


