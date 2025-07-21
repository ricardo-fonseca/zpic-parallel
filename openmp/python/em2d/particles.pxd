###############################################################################
# Particles
#

from libc.stdint cimport uint32_t
from em2d.vec_types cimport *

cdef extern from "../../em2d/particles.h":
    cdef cppclass Particles:
        uint2 ntiles
        uint2 nx
        int * np
        int * offset

        int2 *ix
        float2 *x
        float3 *u

        uint32_t max_part

        int2 periodic
        uint2 dims

        Particles( uint2 ntiles, uint2 nx, uint32_t max_part )