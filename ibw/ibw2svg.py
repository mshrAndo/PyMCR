#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import cairo
import logging

import numpy
import ibw

def load_file(path):
    if not path:
        return None
    wave = ibw.IgorWave()
    wave.load_file(open(path, "rb"))
    if not (wave and wave.validate()) :
        return None

    A = numpy.frombuffer(wave.blob, dtype = numpy.float32)
    rows, cols = wave.wave_header['nDim'][:2]
    A = numpy.reshape(A, (rows, cols), 'F') # to 2D array
    logging.info(['loaded ibw:', path, A.shape, A.max(), A.min()])
    return A



A = load_file(sys.argv[1])
top = numpy.amax(A)
bottom = numpy.amin(A)

print top, bottom
W, H = (400, 200)
s = cairo.SVGSurface(open('graph.svg', 'w'), W, H)


c = cairo.Context(s)

w, h = A.shape
mat = cairo.Matrix(W/w, 0,
                   0, H/(bottom -top),
                   0, H)

colors = [(1, 0, 0, .3),
          (0, 1, 0, .3),
          (0, 0, 1, .3),
          ]
c.set_line_width(1)
for row in range(1, w):
    c.set_source_rgba(*(colors[0]))
    c.save()
    c.set_matrix(mat)
    c.move_to(0, A[row][0])
    for col in range(h):
        c.line_to(col, A[row][col])
    c.restore()
    c.stroke()
    if row > 100:
        break
c.stroke()
c.show_page()