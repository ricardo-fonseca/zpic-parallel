cdef extern from "../../em2d/emf.h" namespace "emf":
    cdef enum field:
        e, b

###############################################################################
# EMF class
#

from em2d.vec3grid cimport *
cimport em2d.current as current

cdef extern from "../../em2d/emf.h":
    cdef cppclass EMF:
        vec3grid[float3] * E
        vec3grid[float3] * B
        float2 box

        EMF( uint2, uint2, float2, double )

        int get_iter()
        double get_dt()
        int set_moving_window()
        
        void advance()
        void advance( current.Current & current )
        void save( int, int )

