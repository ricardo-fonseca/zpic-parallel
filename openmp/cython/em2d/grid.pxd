###############################################################################
# grid class
#
from libcpp.string cimport string
from em2d.vec_types cimport *
from em2d.bnd cimport *

cdef extern from "../../em2d/grid.h":
    cdef cppclass grid[T]:
        T * d_buffer
        uint2 ntiles
        uint2 nx
        bnd_uint gc
        int2 periodic
        uint2 dims
        uint2 ext_nx
        unsigned int offset
        unsigned int tile_vol
        string name

        grid( uint2, uint2, bnd_uint ) except +
        grid( uint2, uint2 ) except +

        size_t buffer_size()
        int zero()
        void set( T )
        void add( grid & )
        unsigned int gather( T * )

        void copy_to_gc()
        void add_from_gc()
        void x_shift_left( unsigned int )

        void kernel3_x( T, T, T )
        void kernel3_y( T, T, T )

        void save( string file )

        uint2 get_ntiles()
        uint2 get_dims()

ctypedef grid[float] grid_float