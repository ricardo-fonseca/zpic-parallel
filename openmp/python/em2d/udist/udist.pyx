# distutils: language = c++
# cython: language_level=3

###############################################################################
# Udist
#

from em2d.vec_types cimport *
cimport em2d.udist.defs as udist

cdef class None:
    """None()

    Class representing a frozen (0 fluid, 0 temperature) momentum distribution
    """
    # cdef udist.None * obj

    def __cinit__(self):
        self.obj = new udist.None( )

    def __dealloc__(self):
        del self.obj

cdef class Cold:
    """Cold( ufl )

    Class representing a cold (0 temperature) momentum distribution

    Parameters
    ----------
    ufl : { float, float, float }
        Fluid momentum
    """
    # cdef udist.Cold * obj

    def __cinit__( self, list ufl ):

        cdef float3 _ufl
        _ufl.x = ufl[0]
        _ufl.y = ufl[1]
        _ufl.z = ufl[2]

        self.obj = new udist.Cold( _ufl )

    def __dealloc__(self):
        del self.obj

cdef class Thermal:
    """Thermal( uth, ufl )

    Class representing a thermal momentum distribution

    Parameters
    ----------
    uth : { float, float, float }
        Temperature distribution
    ufl : { float, float, float }
        Fluid momentum
    """

    # cdef udist.Thermal * obj

    def __cinit__( self, list uth, list ufl ):

        cdef float3 _uth
        _uth.x = uth[0]
        _uth.y = uth[1]
        _uth.z = uth[2]

        cdef float3 _ufl
        _ufl.x = ufl[0]
        _ufl.y = ufl[1]
        _ufl.z = ufl[2]

        self.obj = new udist.Thermal( _uth, _ufl )

    def __dealloc__(self):
        del self.obj

cdef class ThermalCorr:
    """Thermal( uth, ufl, npmin = 2 )

    Class representing a thermal momentum distribution. Momentum is corrected
    to minimize fluid fluctuations.

    Parameters
    ----------
    uth : { float, float, float }
        Temperature distribution
    ufl : { float, float, float }
        Fluid momentum
    npmin : int
        Minimum number of particles in a cell to apply momentum correction
    """

    # cdef udist.ThermalCorr * obj

    def __cinit__( self, list uth, list ufl, int npmin = 2 ):

        cdef float3 _uth
        _uth.x = uth[0]
        _uth.y = uth[1]
        _uth.z = uth[2]

        cdef float3 _ufl
        _ufl.x = ufl[0]
        _ufl.y = ufl[1]
        _ufl.z = ufl[2]

        self.obj = new udist.ThermalCorr( _uth, _ufl, npmin )

    def __dealloc__(self):
        del self.obj
