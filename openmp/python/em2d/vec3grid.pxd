###############################################################################
# vec3grid class
#
from libcpp.string cimport string
from em2d.vec_types cimport *
from em2d.bnd cimport *

cdef extern from "../../em2d/vec3grid.h":
    cdef cppclass vec3grid[V]:
        V * d_buffer
        uint2 ntiles
        uint2 nx
        bnd_uint gc
        int2 periodic
        uint2 dims
        uint2 ext_nx
        unsigned int offset
        unsigned int tile_vol
        string name

        vec3grid( uint2, uint2, bnd_uint ) except +
        vec3grid( uint2, uint2 ) except +

        size_t buffer_size()
        int zero()
        void set( V )
        void add( vec3grid & )

        # This is a workaround it should be gather( int, S )
        unsigned int gather( int, void * )

        void copy_to_gc()
        void add_from_gc()
        void x_shift_left( unsigned int )

        void kernel3_x( float, float, float )
        void kernel3_y( float, float, float )

        void save( int, string )

ctypedef vec3grid[float3] vec3grid_float
