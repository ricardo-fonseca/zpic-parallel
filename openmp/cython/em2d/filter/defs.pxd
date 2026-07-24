###############################################################################
# 
# Filter

cdef extern from "../../../em2d/filter.h" namespace "Filter":
    cdef cppclass Digital:
        pass

    cdef cppclass None( Digital ):
        pass

    cdef cppclass Binomial( Digital ):      
        Binomial( int dir, unsigned order )

    cdef cppclass Compensated( Binomial ):
        Compensated( int dir, unsigned order )
