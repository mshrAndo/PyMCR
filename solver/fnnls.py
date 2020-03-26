#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg.linalg as la

def find(data):
    return np.nonzero(data)[0]
def norm(data, d):
    return max(abs(data).sum(axis=0))

def fnnls(XtX, Xty, tolerance = None):
    
    # initialize variables

    m = XtX.shape[0]
    n = XtX.shape[1]
    
    if tolerance == None:
        eps = 2.2204e-16
        tolerance = 10 * eps * norm(XtX, 1) * max(m, n);

    P = np.zeros(n, np.int16)
    P[:] = -1
    Z = np.arange(0,n)

    z = np.zeros(m, np.float)
    x = np.array(P)
    ZZ = np.array(Z)

    w = Xty - np.dot(XtX, x)

    # set up iteration criterion
    iter = 0
    itmax = 30 * n

    # outer loop to put variables into set to hold positive coefficients
    while np.any(Z) and np.any(w[ZZ] > tolerance):
        wt = w[ZZ].max()
        t = find(w[ZZ] == wt)
        t = t[-1:][0]
        t = ZZ[t]
        P[t] = t
        Z[t] = -1
        PP = find(P != -1)

        ZZ = find(Z != -1)
        if len(PP) == 1:
            XtyPP = Xty[PP]
            XtXPP = XtX[PP, PP]
            z[PP] = XtyPP / XtXPP
        else:
            XtyPP = np.array(Xty[PP])
            XtXPP = np.array(XtX[PP, np.array(PP)[:, np.newaxis]])
            z[PP] = np.dot(XtyPP, la.inv(XtXPP))
        z[ZZ] = 0

        # inner loop to remove elements from the positive set which no longer belong
        while np.any(z[PP] <= tolerance) and (iter < itmax) :
            iter += 1
            iztol = find(z <= tolerance)
            ip = find(P[iztol] != -1)
            QQ = iztol[ip]

            if len(QQ) == 1:
                alpha = x[QQ] / (x[QQ] - z[QQ])
            else :
                x_xz = x[QQ] / (x[QQ] - z[QQ])
                alpha = x_xz.min()

            x += alpha * (z - x)
            iabs = find(abs(x) < tolerance)
            ip = find(P[iabs] != -1)
            ij = iabs[ip]

            Z[ij] = np.array(ij)
            P[ij] = -1
            PP = find(P != -1)
            if len(PP) == 0:
                break
            ZZ = find(Z != -1)

            if len(PP) == 1:
                XtyPP = Xty[PP]
                XtXPP = XtX[PP, PP]
                z[PP] = XtyPP / XtXPP
            else:
                XtyPP = np.array(Xty[PP])
                XtXPP = np.array(XtX[PP, np.array(PP)[:, np.newaxis]])
                z[PP] = np.dot(XtyPP, la.inv(XtXPP))
            z[ZZ] = 0

        x = np.array(z)
        w = Xty - np.dot(XtX, x)
    return x, w

if __name__ == '__main__' :
    X = np.array( [[2, 1],
                   [1, 2]])

    XtX = np.dot(X.T, X)

    for y in ([0,0],
              [0,3],
              [3,0],
              [3,3],
              ):
        Xty = np.dot(X.T, np.array(y))
    
        print y, fnnls(XtX, Xty)
