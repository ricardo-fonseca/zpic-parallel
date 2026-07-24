# distutils: language = c++
# cython: language_level=3

###############################################################################
# 
# Density

cimport em2d.coord as coord 
cimport em2d.density.defs as profile

cdef class None:
    """None()

    Class representing a 0 density profile
    """
    # cdef profile.None * obj

    def __cinit__(self ):
        self.obj = new profile.None( )

    def __dealloc__(self):
        del self.obj

cdef class Uniform:
    """Uniform( n0 )

    Class representing a uniform density profile

    Parameters
    ----------
    n0 : float
        Density value
    """
    # cdef profile.Uniform * obj

    def __cinit__(self, float n0 ):
        self.obj = new profile.Uniform( n0 )

    def __dealloc__(self):
        del self.obj
    
    @property
    def n0( self ):
        """Density value"""
        return self.obj.n0

cdef class Step:
    """Step( dir, n0, pos )

    Class representing a step density profile

    Parameters
    ----------
    dir : string
        Step direction, must be one of 'x' or 'y'
    n0 : float
        Density value
    pos : float
        Step position in simulation units
    """
    # cdef profile.Step * obj

    def __cinit__(self, dir, float n0, float pos ):
        cdef coord.cart dir_ = {'x':coord.cart.x, 'y':coord.cart.y }[dir]
        self.obj = new profile.Step( dir_, n0, pos )

    def __dealloc__(self):
        del self.obj
    
    @property
    def n0( self ):
        """Density value"""
        return self.obj.n0

    @property
    def pos( self ):
        """Edge position"""
        return self.obj.pos

    @property
    def dir( self ):
        """Step direction"""
        return {0:'x',1:'y'}[ self.obj.dir ]

cdef class Slab:
    """Slab( dir, n0, begin, end )

    Class representing a slab density profile


    Parameters
    ----------
    dir : string
        Step direction, must be one of 'x' or 'y'
    n0 : float
        Density value
    begin : float
        Position of the beggining of the slab in simulation units
    end : float
        Position of the end of the slab in simulation units
    """
    # cdef profile.Slab * obj

    def __cinit__(self, dir, float n0, float begin, float end ):
        cdef coord.cart dir_ = {'x':coord.cart.x, 'y':coord.cart.y }[dir]
        self.obj = new profile.Slab( dir_, n0, begin, end )

    def __dealloc__(self):
        del self.obj
    
    @property
    def n0( self ):
        """Density value"""
        return self.obj.n0

    @property
    def dir( self ):
        """Step direction"""
        return {0:'x',1:'y'}[ self.obj.dir ]

    @property
    def begin( self ):
        """Slab begin position"""
        return self.obj.begin

    @property
    def end( self ):
        """Slab end position"""
        return self.obj.end