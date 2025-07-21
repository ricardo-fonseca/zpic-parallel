# distutils: language = c++
# cython: language_level=3

cimport em2d.laser.defs as laser

cdef class PlaneWave:
    cdef laser.PlaneWave * obj

cdef class Gaussian:
    cdef laser.Gaussian * obj
