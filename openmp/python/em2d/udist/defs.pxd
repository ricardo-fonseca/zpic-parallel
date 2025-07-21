###############################################################################
# 
# UDistribution

from em2d.vec_types cimport *
from em2d.particles cimport Particles

cdef extern from "../../../em2d/udist.h" namespace "UDistribution":
    cdef cppclass Type:
        pass

    cdef cppclass None(Type):
        None()
        void set( Particles & part, unsigned seed )

    cdef cppclass Cold(Type):
        float3 ufl
        Cold( float3 ufl )
        void set( Particles & part, unsigned seed )


    cdef cppclass Thermal(Type):
        float3 uth
        float3 ufl
        Thermal( float3 uth, float3 ufl )
        void set( Particles & part, unsigned seed )


    cdef cppclass ThermalCorr(Type):
        float3 uth
        float3 ufl
        int npmin
        ThermalCorr( float3 uth, float3 ufl, int npmin )
        void set( Particles & part, unsigned seed )

