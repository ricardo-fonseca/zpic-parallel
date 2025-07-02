# distutils: language = c++
# cython: language_level=3

###############################################################################
# Standard C/C++ 
#
from libc.stdint cimport uint32_t, uint64_t
from libcpp.string cimport string

###############################################################################
# Utilities
#
cdef extern from "../em2d/utils.h":
    pass

###############################################################################
# vector types
#

cdef extern from "../em2d/vec_types.h":
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

###############################################################################
# bnd class
#

cdef cppclass __bnd_pair[T]:
    T lower
    T upper


cdef extern from "../em2d/bnd.h":
    cdef cppclass bnd[T]:
        __bnd_pair[T] x
        __bnd_pair[T] y
        bnd() except +

ctypedef bnd[unsigned int] bnd_uint

###############################################################################
# grid class
#

cdef extern from "../em2d/grid.h":
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


###############################################################################
# vec3grid class
#
cdef extern from "../em2d/vec3grid.h":
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


###############################################################################
# EMF class
#

cdef extern from "../em2d/emf.h":
    cdef cppclass EMF:
        vec3grid[float3] * E
        vec3grid[float3] * B
        float2 box

        EMF( uint2, uint2, float2, double )

        int get_iter()
        int set_moving_window()
        
        void advance()
        # void advance( Current )
        void save( int, int )


###############################################################################
# Laser pulse classes
#

cdef extern from "../em2d/laser.h" namespace "Laser":
    cdef cppclass Pulse:
        float start
        float fwhm
        float rise
        float flat
        float fall
        float a0
        float omega0
        float polarization

        float cos_pol
        float sin_pol

        unsigned int filter

        int validate()
        
        Pulse()
        int add( EMF & )

    cdef cppclass PlaneWave(Pulse):
        PlaneWave()
        int add( EMF & )

    cdef cppclass Gaussian(Pulse):
        float W0
        float focus
        float axis

        Gaussian()
        int add_gaussian "add"( EMF & )
