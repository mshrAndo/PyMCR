#!/usr/bin/python
# -*- coding: utf-8 -*-
"""optimize by alternate least square"""
import logging
import math

import numpy
import scipy.optimize

import fnnls

def norm(vector):
    """just a Frobenius norm for now 
    >>> norm([1,1,1,1])
    2.0
    """
    return numpy.linalg.norm(vector)


import scipy.optimize._nnls as _nnls_F
def _scipy_optimize_nnls(A, b):
    # non-error checking version
    m,n = A.shape

    w   = numpy.zeros((n,), dtype=numpy.double)
    zz  = numpy.zeros((m,), dtype=numpy.double)
    index = numpy.zeros((n,), dtype=int)

    x,rnorm,mode = _nnls_F.nnls(A,m,n,b,w,zz,index)	#fortran func. mode: SUCCESS-FAILURE FLAG, x & rnorm are same as the return from scipy.optimize.nnls
    if mode != 1:
        raise RuntimeError("too many iterations")

    return x, rnorm		#Tuple (x, rnorm), x:np.ndarray, rnorm:float

def _nnls(matrix, vector):
    """
    >>> _nnls(numpy.array([ [1, 0], [0, 1] ]), numpy.array([2, 3]))
    (array([ 2.,  3.]), 0.0)
    >>> _nnls(numpy.array([ [2, 1], [0, 2] ]), numpy.array([3, 2]))
    (array([ 1.,  1.]), 0.0)
    >>> (x, r) = _nnls(numpy.array([ [2, 1], [5, 2] ]), numpy.array([1, 3]))
    >>> r > 0
    True
    """
    #return scipy.optimize.nnls(matrix, vector)
    return _scipy_optimize_nnls(matrix, vector)

def _nnls_matrix(matrix, vector_list):
    """
    >>> _nnls_matrix(numpy.array([ [1, 0], [0, 1] ]), numpy.array([[1, 2], [3, 4]]))
    (array([[ 1.,  2.],
           [ 3.,  4.]]), [0.0, 0.0])
    """
    solutions = [_nnls(matrix, vector) for vector in vector_list]
    return (numpy.vstack(s[0] for s in solutions),	#s: Tuple (x, rnorm)
            [s[1] for s in solutions])

def _fnnls_matrix(matrix, vector_list):
    """
    >>> _fnnls_matrix(numpy.array([ [1, 0], [0, 1] ]), numpy.array([[1, 2], [3, 4]]))[0]
    array([[ 1.,  2.],
           [ 3.,  4.]])
    """
    XtX = numpy.dot(matrix.T, matrix)
    solutions = [fnnls.fnnls(XtX, numpy.dot(matrix.T, vector))
                 for vector in vector_list]
    return (numpy.vstack(s[0] for s in solutions),
            [s[1] for s in solutions])

def least_square_proj(A, bs):
    """
    >>> least_square_proj(numpy.array([ [1, 0], [0, 1] ]), numpy.array([[1, 2, 3], [3, 4, 5]]))
    array([[ 1.,  2.,  3.],
           [ 3.,  4.,  5.]])
    """
    X, residues, rank, s = numpy.linalg.lstsq(A.T, bs)
    # project to positive side
    ng = numpy.nonzero(X <= 0)
    X[ng] = 0
    return X


#import ctypes
#import numpy

#gotoblas = numpy.ctypeslib.load_library("libgoto2", '.')
#nnls = numpy.ctypeslib.load_library("libnnls", '.')
#nnls.NNLS.argtypes = [numpy.ctypeslib.ndpointer(dtype = numpy.double),
#                      numpy.ctypeslib.ndpointer(dtype = numpy.double),
#                      numpy.ctypeslib.ndpointer(dtype = numpy.double),
#                      ctypes.c_int,
#                      ctypes.c_int,
#                      ctypes.c_int,
#                      ]

#nnls.NNLS.restype = ctypes.c_int


def NNLS_gotoblas(A, b):
    M, N = A.shape
    S = b.shape[1]
    x = numpy.zeros(shape = (N, S), order = 'F', dtype = numpy.double)
    r = nnls.NNLS(numpy.asarray(A, order = 'F', dtype = numpy.double),
                  numpy.asarray(b, order = 'F', dtype = numpy.double),
                  x, S, M, N)
    logging.error(['ret', r])
    return x

def nn_least_square(A, bs):
    """
    >>> nn_least_square(numpy.array([ [1, 0], [0, 1] ]), numpy.array([[1, 2, 3], [3, 4, 5]]))
    array([[ 1.,  2.,  3.],
           [ 3.,  4.,  5.]])

    >>> nn_least_square(numpy.array([ [1, 1], [3, 1] ]), numpy.array([[1, 2, 3], [3, 4, 5]]))
    array([[ 1.  ,  1.25,  1.5 ],
           [ 0.  ,  0.25,  0.5 ]])
    """
    #X, residues = _fnnls_matrix(A, bs.T)

    X, residues = _nnls_matrix(A, bs.T)

    #X2 =  NNLS_gotoblas(A, bs)
    #raise ValueError(X.T, X2) 
    return X.T

def als(A, W, H, normalize_W = True, method = 'NNLS',
        coef_L1W = 0, coef_L2W = 0,
        coef_L1H = 0, coef_L2H = 0,
        omit_norm1 = False,
        fix_W = 0,
        Hw = None):	#added
    """alternate least square, without non-negative restriction"""
    if A is None:
        return (None, None)

    if not ((W is None) or (H is None)):
        pre = numpy.dot(W, H)
    else:
        pre = None

    if method == "LS":
        optimize = least_square_proj
    elif method == "NNLS":
        optimize = nn_least_square
    else:
        raise ValueError(method)

    if not W is None: # update H for W; solve __ Wt * W * H = Wt * A
        if normalize_W:
            # try other norms?
            for c in range(W.shape[1]):
                n = norm(W[:, c])
                if n > 0:
                    W[:, c] /= n

        mat_A = numpy.dot(W.T, W)
        mat_b = numpy.dot(W.T, A)

        if coef_L2H > 0.0:
            logging.debug(['L2 norm for H', coef_L2H, mat_A.shape])
            mat_A += (numpy.identity( mat_A.shape[0] ) * coef_L2H)

        if coef_L1H > 0.0:
            logging.debug(['L1 norm for H', coef_L1H])
            mat_A += (numpy.ones( mat_A.shape ) * coef_L1H)

        H = optimize(mat_A, mat_b)	#guessing H with least squares; optimize = least_square_proj or nn_least_square
        
        if not Hw is None:	#added
        	H = H * Hw
        
    if not H is None:
        if type(fix_W) == numpy.ndarray:
            #fix_W = numpy.array([True,False,True,True]) ... as mask array.
            #True: Fix, False: to be calculated
            knowns = numpy.dot(W[:,fix_W], H[fix_W])
            rest_A = (A - knowns)
            shrinked_H = H[fix_W==False]
            mat_A = numpy.dot(shrinked_H, shrinked_H.T)
            mat_b = numpy.dot(shrinked_H, rest_A.T)
            
            if coef_L2W > 0.0:
                logging.debug(['L2 norm for W', coef_L2W, mat_A.shape])
                mat_A += (numpy.identity( mat_A.shape[0] ) * coef_L2W)
                
            if coef_L1W > 0.0:
                logging.debug(['L1 norm for W', coef_L1W])
                mat_A += (numpy.ones( mat_A.shape ) * coef_L1W)
                #for i in range(fix_W):
                #    mat_b -= numpy.outer(numpy.ones(mat_A.shape[0]) * coef_L1W, W[:, i])
                
            X = optimize(mat_A, mat_b)
            W[:, fix_W==False] = X.T            
            
        elif fix_W == 0:
            # solve H * Ht * Wt = H * At
            mat_A = numpy.dot(H, H.T)
            mat_b = numpy.dot(H, A.T)

            if coef_L2W > 0.0:
                logging.debug(['L2 norm for W', coef_L2W, mat_A.shape])
                mat_A += (numpy.identity( mat_A.shape[0] ) * coef_L2W)

            if coef_L1W > 0.0:
                logging.debug(['L1 norm for W', coef_L1W])
                mat_A += (numpy.ones( mat_A.shape ) * coef_L1W)

            X = optimize(mat_A, mat_b)
            W = X.T            

        elif fix_W < (W.shape)[1]:            
            knowns = numpy.outer(W[:, (fix_W -1)], H[(fix_W -1), :])
            rest_A = (A - knowns)
            shrinked_H = H[fix_W:, :]
            mat_A = numpy.dot(shrinked_H, shrinked_H.T)
            mat_b = numpy.dot(shrinked_H, rest_A.T)
            
            if coef_L2W > 0.0:
                logging.debug(['L2 norm for W', coef_L2W, mat_A.shape])
                mat_A += (numpy.identity( mat_A.shape[0] ) * coef_L2W)

            if coef_L1W > 0.0:
                logging.debug(['L1 norm for W', coef_L1W])
                mat_A += (numpy.ones( mat_A.shape ) * coef_L1W)
                for i in range(fix_W):
                    mat_b -= numpy.outer(numpy.ones(mat_A.shape[0]) * coef_L1W,
                                         W[:, i])
            X = optimize(mat_A, mat_b)
            W[:, fix_W:] = X.T            

    # calc err
    err_mat = A - numpy.dot(W, H)
    norm1 = norm(err_mat)
    norm2 = numpy.sqrt(numpy.sum(err_mat**2, 1))

    logging.debug(["ALS norm", norm1])

    return W, H, (norm1, norm2)

if __name__ == "__main__":
    import doctest
    doctest.testmod()