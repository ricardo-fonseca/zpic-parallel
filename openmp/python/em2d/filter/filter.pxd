# distutils: language = c++
# cython: language_level=3

cimport em2d.filter.defs as filter

cdef class None:
    cdef filter.None * obj

cdef class Binomial:
    cdef filter.Binomial * obj

cdef class Compensated:
    cdef filter.Compensated * obj
