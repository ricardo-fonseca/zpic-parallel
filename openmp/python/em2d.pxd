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
# Current class
#

cdef extern from "../em2d/current.h":
    cdef cppclass Current:
        vec3grid[float3] * J
        float2 box

        Current( uint2 ntiles, uint2 nx, float2 box, double dt )

        int get_iter()
        double get_dt()

        void advance()
        void zero()
        void save( int )


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
        double get_dt()
        int set_moving_window()
        
        void advance()
        void advance( Current )
        void save( int, int )


###############################################################################
# Laser pulse classes
#

cimport laser

###############################################################################
# Particles
#

cdef extern from "../em2d/particles.h":
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


###############################################################################
# 
# UDistribution

cimport udist


###############################################################################
# 
# Density

cimport density

###############################################################################
# Species
#

cdef extern from "../em2d/species.h":
    cdef cppclass Species:
        string name
        float m_q
        int push_type

        Species( string name, float m_q, uint2 ppc )

        void initialize( float2 box, uint2 ntiles, uint2 nx, double dt, int id )
        void set_udist( udist.Type & new_udist )
        void set_density( density.Profile & new_density )

        void advance( EMF & emf, Current & current )

        void deposit_charge( grid[float] & charge )

        void save()
        void save_charge()
        void save_phasespace( int quant, float2 range, int size )
        void save_phasespace( int quant0, float2 range0, int size0,
                              int quant1, float2 range1, int size1 )
        
        void gather( int quant, float * data )
        uint64_t np_total() 
        int get_iter()
        float get_dt()



###############################################################################
# Simulation
#

cdef extern from "../em2d/simulation.h":
    cdef cppclass Simulation:
        uint2 ntiles
        uint2 nx
        float2 box
        double dt

        EMF emf
        Current current
        vector[Species] species

        Simulation( uint2 ntiles, uint2 nx, float2 box, double dt )
        void set_moving_window() 
        void add_species( Species & sp )

        Species * get_species( string name )

        void advance()
        void advance_mov_window()

        unsigned int get_iter()
        double get_t()

        void energy_info()

