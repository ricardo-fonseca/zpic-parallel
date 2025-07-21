###############################################################################
# vector types
#

cdef extern from "../../em2d/vec_types.h":
    cdef struct uint2:
        unsigned int x
        unsigned int y

    cdef struct int2:
        int x
        int y

    cdef struct float2:
        float x
        float y

    cdef struct float3:
        float x
        float y
        float z
