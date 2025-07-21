# distutils: language = c++
# cython: language_level=3

cimport em2d.udist.defs as udist

cdef class None:
    cdef udist.None * obj

cdef class Cold:
    cdef udist.Cold * obj

cdef class Thermal:
    cdef udist.Thermal * obj

cdef class ThermalCorr:
    cdef udist.ThermalCorr * obj
