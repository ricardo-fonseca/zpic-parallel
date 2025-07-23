# distutils: language = c++
# cython: language_level=3

###############################################################################
# 
# Filter

cimport em2d.coord as coord 
cimport em2d.filter.defs as filter

cdef class None:
    """None()

    Class representing an all pass digital filter
    """
    # cdef filter.None * obj

    def __cinit__(self ):
        self.obj = new filter.None( )

    def __dealloc__(self):
        del self.obj

cdef class Binomial:
    """Binomial( dir, level = 1 )
    
    Class representing binomial digital filter

    Parameters
    ----------
    dir : string
        Filter direction, must be one of 'x' or 'y'
    level : int
        Filter level (number of times the kernel will be applied ), defaults
        to 1 (minimal value). 
    """
    # cdef filter.Binomial * obj

    def __cinit__(self, dir, unsigned level = 1 ):
        cdef coord.cart dir_ = {'x':coord.cart.x, 'y':coord.cart.y }[dir]
        self.obj = new filter.Binomial( dir_, level )

    def __dealloc__(self):
        del self.obj

cdef class Compensated:
    """Compensated( dir, level = 1 )
    
    Class representing compensated binomial digital filter. The filter will
    apply a binomial filter with the specified level, followed by a compensator
    kernel.

    Parameters
    ----------
    dir : string
        Filter direction, must be one of 'x' or 'y'
    level : int
        Filter level (number of times the kernel will be applied ), defaults
        to 1 (minimal value). 
    """
    # cdef filter.Compensated * obj

    def __cinit__(self, dir, unsigned level = 1 ):
        cdef coord.cart dir_ = {'x':coord.cart.x, 'y':coord.cart.y }[dir]
        self.obj = new filter.Compensated( dir_, level )

    def __dealloc__(self):
        del self.obj
