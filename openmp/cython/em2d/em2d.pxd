# distutils: language = c++
# cython: language_level=3

###############################################################################
# Standard C/C++ 
#
from libc.stdint cimport uint32_t, uint64_t
from libcpp.string cimport string
from libcpp.vector cimport vector


###############################################################################
# Utilities
#
cdef extern from "../../em2d/utils.h":
    pass

###############################################################################
# vector types
#
from em2d.vec_types cimport *

###############################################################################
# bnd class
#
from em2d.bnd cimport *

###############################################################################
# grid class
#

from em2d.grid cimport *

###############################################################################
# vec3grid class
#

from em2d.vec3grid cimport *

cdef class Vec3Grid:
    cdef vec3grid.vec3grid[ float3 ] * obj
    cdef bint is_view

    cdef associate( Vec3Grid self, vec3grid.vec3grid[ float3 ] * src )

###############################################################################
# Current class
#

cimport em2d.current

cdef class Current:
    cdef em2d.current.Current * obj
    cdef bint is_view
    cdef Vec3Grid J

    cdef associate( self, em2d.current.Current * src )


###############################################################################
# EMF class
#

cimport em2d.emf

cdef class EMF:
    cdef em2d.emf.EMF * obj    
    cdef bint is_view
    cdef Vec3Grid E
    cdef Vec3Grid B

    cdef associate( EMF self, em2d.emf.EMF * src )

###############################################################################
# Particles
#

cimport em2d.particles


###############################################################################
# Simulation
#

cimport em2d.simulation

cdef class Simulation:
    cdef em2d.simulation.Simulation * obj
    cdef EMF emf
    cdef Current current
    cdef bint mov_window