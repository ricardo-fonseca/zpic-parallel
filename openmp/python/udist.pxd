###############################################################################
# 
# UDistribution

cdef extern from "../em2d/vec_types.h":
    cdef struct float3:
        float x
        float y
        float z

cdef extern from "../em2d/udist.h" namespace "UDistribution":
    cdef cppclass Type:
        pass

    cdef cppclass None(Type):
        pass

    cdef cppclass Cold(Type):
        float3 ufl
        Cold( float3 ufl )

    cdef cppclass Thermal(Type):
        float3 uth
        float3 ufl
        Thermal( float3 uth, float3 ufl )

    cdef cppclass ThermalCorr(Type):
        float3 uth
        float3 ufl
        int npmin
        ThermalCorr( float3 uth, float3 ufl, int npmin )