###############################################################################
# Current class
#

from em2d.vec3grid cimport *
cimport em2d.filter.defs as filter

cdef extern from "../../em2d/current.h":
    cdef cppclass Current:
        vec3grid[float3] * J
        float2 box

        Current( uint2 ntiles, uint2 nx, float2 box, double dt )

        int get_iter()
        double get_dt()

        void advance()
        void zero()
        void save( int )

        void set_filter( filter.Digital & new_filter )