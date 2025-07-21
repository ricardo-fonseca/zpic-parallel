# distutils: language = c++
# cython: language_level=3

cimport em2d.density.defs as density

cdef class None:
    cdef density.None * obj

cdef class Uniform:
    cdef density.Uniform * obj

cdef class Step:
    cdef density.Step * obj

cdef class Slab:
    cdef density.Slab * obj
