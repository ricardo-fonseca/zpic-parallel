from em2d.vec_types cimport *

cdef extern from "../../em2d/zpic.h" namespace "zpic":
    void sys_info()
    float courant( float2 dx )
    float courant( uint2 gnx, float2 box )
    float courant( uint2 ntiles, uint2 nx, float2 box )
