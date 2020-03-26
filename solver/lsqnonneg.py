#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
A Python implementation of NNLS algorithm

References:
[1]  Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems, Prentice-Hall, Chapter 23, p. 161, 1974.

Contributed by Klaus Schuch (schuch@igi.tugraz.at)
based on MATLAB's lsqnonneg function

from diffusion-mri,
Copyright (c) 2007-2010 <Bing Jian and Baba C. Vemuri>
"""

import numpy

def lsqnonneg(C, d, x0=None, tol=None, itmax_factor=3):
    '''Linear least squares with nonnegativity constraints

    (x, resnorm, residual) = lsqnonneg(C,d) returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real
    '''

    eps = 2.22e-16    # from matlab
    def norm1(x):
        return abs(x).sum().max()

    def msize(x, dim):
        s = x.shape
        if dim >= len(s):
            return 1
        else:
            return s[dim]

    if tol is None:
        tol = 10*eps*norm1(C)*(max(C.shape)+1)

    C = numpy.asarray(C)

    (m,n) = C.shape
    P = numpy.zeros(n)
    Z = numpy.arange(1, n+1)

    if x0 is None:
        x=P
    else:
        if any(x0 < 0):
            x=P
        else:
            x=x0

    ZZ=Z

    resid = d - numpy.dot(C, x)
    w = numpy.dot(C.T, resid)

    outeriter=0
    it=0
    itmax=itmax_factor*n
    exitflag=1

    # outer loop to put variables into set to hold positive coefficients
    while numpy.any(Z) and numpy.any(w[ZZ-1] > tol):
        outeriter += 1

        t = w[ZZ-1].argmax()
        t = ZZ[t]

        P[t-1]=t
        Z[t-1]=0

        PP = numpy.where(P <> 0)[0]+1
        ZZ = numpy.where(Z <> 0)[0]+1

        CP = numpy.zeros(C.shape)

        CP[:, PP-1] = C[:, PP-1]
        CP[:, ZZ-1] = numpy.zeros((m, msize(ZZ, 1)))

        z=numpy.dot(numpy.linalg.pinv(CP), d)

        z[ZZ-1] = numpy.zeros((msize(ZZ,1), msize(ZZ,0)))

        # inner loop to remove elements from the positve set which no longer belong
        while numpy.any(z[PP-1] <= tol):
            it += 1

            if it > itmax:
                max_error = z[PP-1].max()
                raise Exception('Exiting: Iteration count (=%d) exceeded\n Try raising the tolerance tol. (max_error=%d)' % (it, max_error))

            QQ = numpy.where((z <= tol) & (P <> 0))[0]
            alpha = min(x[QQ]/(x[QQ] - z[QQ]))
            x = x + alpha*(z-x)

            ij = numpy.where((abs(x) < tol) & (P <> 0))[0]+1
            Z[ij-1] = ij
            P[ij-1] = numpy.zeros(max(ij.shape))
            PP = numpy.where(P <> 0)[0]+1
            ZZ = numpy.where(Z <> 0)[0]+1

            CP[:, PP-1] = C[:, PP-1]
            CP[:, ZZ-1] = numpy.zeros((m, msize(ZZ, 1)))

            z=numpy.dot(numpy.linalg.pinv(CP), d)
            z[ZZ-1] = numpy.zeros((msize(ZZ,1), msize(ZZ,0)))

        x = z
        resid = d - numpy.dot(C, x)
        w = numpy.dot(C.T, resid)

    return (x, sum(resid * resid), resid)

if __name__=='__main__':
    C = numpy.array([[1.0, 1.0],
                     [2.0, 1.0],
                     [5.0, 1.0],
                     [6.0, 1.0],
                     [10.0, 1.0]])

    d = numpy.array([3, 5, 11, 13, 21])

    [x, resnorm, residual] = lsqnonneg(C, d)

    print [x, resnorm, residual]
