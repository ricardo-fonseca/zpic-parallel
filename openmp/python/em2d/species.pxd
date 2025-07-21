###############################################################################
# Species
#

from libc.stdint cimport uint64_t
from libcpp.string cimport string

from em2d.vec_types cimport *
from em2d.grid cimport *

cimport em2d.udist.defs as udist
cimport em2d.density.defs as density
cimport em2d.emf as emf
cimport em2d.current as current

cdef extern from "../../em2d/species.h":
    cdef cppclass Species:
        string name
        float m_q
        int push_type

        Species( string name, float m_q, uint2 ppc )

        void initialize( float2 box, uint2 ntiles, uint2 nx, double dt, int id )
        void set_udist( udist.Type & new_udist )
        void set_density( density.Profile & new_density )

        void advance( emf.EMF & emf, current.Current & current )

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

