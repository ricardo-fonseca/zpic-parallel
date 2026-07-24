###############################################################################
# 
# Density

cdef extern from "../../../em2d/density.h" namespace "Density":
    cdef cppclass Profile:
        float n0
        Profile( float n0 )

    cdef cppclass None( Profile ):
        None( )

    cdef cppclass Uniform( Profile ):
        Uniform( float n0 )

    cdef cppclass Step( Profile ):
        float pos
        int dir
        Step( int dir, float n0, float pos )

    cdef cppclass Slab( Profile ):
        float begin
        float end
        int dir
        Slab( int, float, float, float )