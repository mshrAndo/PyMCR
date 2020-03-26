#!/usr/bin/python
# -*- coding: utf-8 -*-

import ctypes
import numpy

gotoblas = numpy.ctypeslib.load_library("libgoto2", '.')
nnls = numpy.ctypeslib.load_library("libnnls", '.')
nnls.NNLS.argtypes = [numpy.ctypeslib.ndpointer(dtype = numpy.double),
                      numpy.ctypeslib.ndpointer(dtype = numpy.double),
                      numpy.ctypeslib.ndpointer(dtype = numpy.double),
                      ctypes.c_int,
                      ctypes.c_int,
                      ctypes.c_int,
                      ]

nnls.NNLS.restype = ctypes.c_int


def NNLS(A, bs):
    M, N = A.shape
    S = b.shape[1]
    x = numpy.zeros(shape = (N, S), dtype = numpy.double)
    nnls.NNLS(A, b, x, S, M, N)
    return x