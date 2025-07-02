# distutils: language = c++
# cython: language_level=3

import numpy as np

###############################################################################
# Grid
#

from em2d cimport grid as cppGrid

cdef class Grid:
    cdef cppGrid[ float ] * obj

    def __cinit__(self, list ntiles, list nx, list gc):

        cdef uint2 lntiles
        lntiles.x = ntiles[0]
        lntiles.x = ntiles[1]

        cdef uint2 lnx
        lnx.x = nx[0]
        lnx.x = nx[1]

        cdef bnd_uint lgc
        lgc.x.lower = gc[0,0]
        lgc.x.upper = gc[0,1]
        lgc.y.lower = gc[1,0]
        lgc.y.upper = gc[1,1]

        self.obj = new cppGrid[ float ]( lntiles, lnx, lgc )

    def __cinit__(self, list ntiles, list nx ):

        cdef uint2 lntiles
        lntiles.x = ntiles[0]
        lntiles.x = ntiles[1]

        cdef uint2 lnx
        lnx.x = nx[0]
        lnx.x = nx[1]

        self.obj = new cppGrid[ float ]( lntiles, lnx )

    def __dealloc__(self):
        del self.obj
    
    def buffer_size(self):
        return self.obj.buffer_size()
    
    def zero(self):
        return self.obj.zero()
    
    def set(self, float val):
        self.obj.set( val )
    
    def add(self, Grid rhs):
        return self.obj.add( rhs.obj[0] )
    
    def gather( self ):
        dst = np.zeros( shape = [ self.obj.dims.x, self.obj.dims.y ], dtype = np.float32 )
        cdef float [:,:] buffer = dst
        self.obj.gather( & buffer[ 0, 0 ] )
        return dst
    
    def copy_to_gc( self ):
        self.obj.copy_to_gc()

    def add_from_gc( self ):
        self.obj.add_from_gc()

    def x_shift_left( self, unsigned int shift ):
        self.obj.x_shift_left( shift )
    
    def kernel_x( self, float a, float b, float c ):
        self.obj.kernel3_x( a, b, c )

    def kernel_y( self, float a, float b, float c ):
        self.obj.kernel3_y( a, b, c )

    def save( self, str filename ):
        self.obj.save( filename.encode('utf-8') )


###############################################################################
# Vec3Grid
#

from em2d cimport vec3grid as cppVec3Grid
cimport fcomp

cdef class Vec3Grid:
    cdef cppVec3Grid[ float3 ] * obj

    def __cinit__(self, list ntiles, list nx, int[:,:] gc):

        cdef uint2 _ntiles
        _ntiles.x = ntiles[0]
        _ntiles.y = ntiles[1]

        cdef uint2 _nx
        _nx.x = nx[0]
        _nx.y = nx[1]

        cdef bnd_uint _gc
        _gc.x.lower = gc[0,0]
        _gc.x.upper = gc[0,1]
        _gc.y.lower = gc[1,0]
        _gc.y.upper = gc[1,1]

        self.obj = new cppVec3Grid[ float3 ]( _ntiles, _nx, _gc )

    def __cinit__(self, list ntiles, list nx ):

        cdef uint2 _ntiles
        _ntiles.x = ntiles[0]
        _ntiles.y = ntiles[1]

        cdef uint2 _nx
        _nx.x = nx[0]
        _nx.y = nx[1]

        self.obj = new cppVec3Grid[ float3 ]( _ntiles, _nx )

    def __dealloc__(self):
        del self.obj
    
    def buffer_size(self):
        return self.obj.buffer_size()
    
    def zero(self):
        return self.obj.zero()
    
    def set(self, float3 val):
        self.obj.set( val )
    
    def add(self, Vec3Grid rhs):
        return self.obj.add( rhs.obj[0] )
    
    def gather( self, fc ):
        dst = np.zeros( shape = [ self.obj.dims.x, self.obj.dims.y ], dtype = np.float32 )
        cdef float [:,:] buffer = dst
        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.y}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.gather( _fc, & buffer[ 0, 0 ] )
        
        return dst
    
    def copy_to_gc( self ):
        self.obj.copy_to_gc()

    def add_from_gc( self ):
        self.obj.add_from_gc()

    def x_shift_left( self, unsigned int shift ):
        self.obj.x_shift_left( shift )
    
    def kernel_x( self, float a, float b, float c ):
        self.obj.kernel3_x( a, b, c )

    def kernel_y( self, float a, float b, float c ):
        self.obj.kernel3_y( a, b, c )

    def save( self, fc, str filename ):
        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.y}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.save( _fc, filename.encode('utf-8') )

###############################################################################
# EMF
#

from em2d cimport EMF as cppEMF
cimport emf

cdef class pyEMF:
    cdef cppEMF * obj

    def __cinit__(self, list ntiles, list nx, list box, double dt):

        cdef uint2 _ntiles
        _ntiles.x = ntiles[0]
        _ntiles.y = ntiles[1]

        cdef uint2 _nx
        _nx.x = nx[0]
        _nx.y = nx[1]

        cdef float2 _box
        _box.x = box[0]
        _box.y = box[1]

        self.obj = new cppEMF( _ntiles, _nx, _box, dt )

    def __dealloc__(self):
        del self.obj

    def save( self, fld, fc ):
        tmp = {'e':emf.field.e, 'b':emf.field.b }
        cdef emf.field _field = tmp[fld]

        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.z}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.save( _field, _fc )


###############################################################################
# Laser
#

from em2d cimport Gaussian as cppGaussian
from em2d cimport Pulse as cppPulse

cdef class pyGaussian:
    cdef cppGaussian * obj

    def __cinit__( self, *, float start = 0.0, float fwhm = 0.0,
				   float rise = 0.0, float flat = 0.0, float fall = 0.0,
				   float a0 = 0.0, float omega0 = 0.0, float polarization = 0.0,
                   float cos_pol = 0.0, float sin_pol = 0.0,
                   float W0 = 0, float focus = 0, float axis = 0 ):

        self.obj = new cppGaussian()

        self.obj.start = start
        self.obj.fwhm = fwhm
        self.obj.rise = rise
        self.obj.flat = flat
        self.obj.fall = fall
        self.obj.a0 = a0
        self.obj.omega0 = omega0
        self.obj.polarization = polarization
        self.obj.cos_pol = cos_pol
        self.obj.sin_pol = sin_pol

        self.obj.W0    = W0
        self.obj.focus = focus
        self.obj.axis  = axis

    def __dealloc__(self):
        del self.obj

    def add( self, pyEMF emf ):
        self.obj.add( emf.obj[0] )

