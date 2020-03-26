#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging

import numpy

def from_positive_part(may_negative):
    """ 
    >>> from_positive_part(numpy.array([1, -2, 3]))
    array([ 1.,  0.,  3.])
    """
    return (may_negative + abs(may_negative))/2.0

def from_negative_part(may_negative):
    """
    >>> from_negative_part(numpy.array([1, -2, 3]))
    array([ 0., 2.,  0.])
    """
    return (may_negative - abs(may_negative))/(-2.0)

def guess_by_svd(A, k):
    # A should be a non-negative matrix
    nz = numpy.nonzero(A < 0)
    if nz[0].any():
        raise ValueError('non-zero elements in A',
                         [ ((r, c), v) for v, r, c in zip(A[nz], *nz)])
    (m, n) = A.shape
    if min(m, n, k) != k:
        raise ValueError('rank too large', k)

    W = numpy.zeros( (m, k), dtype = float)
    H = numpy.zeros( (k, n), dtype = float)

    # 1st SVD
    U, s, V = numpy.linalg.svd(A, full_matrices = False, compute_uv = True)
    #logging.error(['shape of U', U.shape])
    #logging.error(['shape of V', V.shape])
    #logging.info(["SVD S", s])
    U = U.astype(numpy.float64)
    s = s.astype(numpy.float64)
    V = V.astype(numpy.float64)

    # every component must have the same sign
    # (Perron-Frobenius theorem)
    i = 0
    if numpy.average(U[:, i])>= 0:
        W[:, 0] = U[:, i]
        H[0, :] = s[0] * V[i, :]
    else:
        W[:, 0] = -U[:, i]
        H[0, :] = -s[0] * V[i, :]

    for i in range(1, k):
        uu, vv = U[:, i], V[i, :]
        uup = from_positive_part(uu)
        uun = from_negative_part(uu)
        vvp = from_positive_part(vv)
        vvn = from_negative_part(vv)

        n_uup = numpy.linalg.norm(uup)
        n_vvp = numpy.linalg.norm(vvp)
        n_uun = numpy.linalg.norm(uun)
        n_vvn = numpy.linalg.norm(vvn)

        termp = n_uup * n_vvp
        termn = n_uun * n_vvn

        if termp >= termn: # choose which side to use 
            # use positive half
            W[:, i] = uup / n_uup 
            H[i, :] = s[i] * termp * vvp / n_vvp
        else:
            # use negative half
            W[:, i] = uun / n_uun 
            H[i, :] = s[i] * termn * vvn / n_vvn

    return W, H, s

def test_k():
    """
    >>> guess_by_svd(numpy.array([[1,2,3], [2,1,2]]), 20)
    Traceback (most recent call last):
    ...
    ValueError: ('rank too large', 20)
    """

def test_negative():
    """
    >>> guess_by_svd(numpy.array([[1,2,3], [2,-1,-2]]), 2)
    Traceback (most recent call last):
    ...
    ValueError: ('non-zero elements in A', [((1, 1), -1), ((1, 2), -2)])
    """

def test_a():
    """
    >>> A = numpy.array([[1,0,1,0], [0,1,0,1], [2, 0, 2, 0]])
    >>> W, H = guess_by_svd(A, 2)
    >>> reconstructed = numpy.dot(W, H)
    >>> numpy.allclose(A, reconstructed, atol = .00001)
    True
    """

if __name__ == "__main__":
    import doctest
    doctest.testmod()
