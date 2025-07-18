# distutils: language = c++
# cython: language_level=3

import numpy as np
import visxd

###############################################################################
# Grid
#

from em2d cimport grid as cppGrid

cdef class Grid:
    cdef cppGrid[ float ] * obj
    cdef bint is_view

    def __cinit__(self, list ntiles = None, list nx = None, list gc = None ):

        cdef uint2 lntiles
        cdef uint2 lnx
        cdef bnd_uint lgc


        if ( ntiles is None ):
            self.obj = NULL
            self.is_view = True
        else:
            lntiles.x = ntiles[0]
            lntiles.x = ntiles[1]

            lnx.x = nx[0]
            lnx.x = nx[1]

            if ( gc is None ):
                self.obj = new cppGrid[ float ]( lntiles, lnx )
            else:
                lgc.x.lower = gc[0,0]
                lgc.x.upper = gc[0,1]
                lgc.y.lower = gc[1,0]
                lgc.y.upper = gc[1,1]
                self.obj = new cppGrid[ float ]( lntiles, lnx, lgc )

            self.is_view = False

    cdef associate( self, cppGrid[ float ] * src ):
        if ( not self.is_view ):
            raise Exception( "Grid object is not of view type, cannot associate existing cppGrid object")
        self.obj = src  

    def __dealloc__(self):
        if ( not self.is_view ):
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
    cdef bint is_view

    def __cinit__(self, list ntiles = None, list nx = None, list gc = None ):

        cdef uint2 _ntiles
        cdef uint2 _nx
        cdef bnd_uint _gc

        if ( ntiles is None ):
            self.obj = NULL
            self.is_view = True
        else:        
            _ntiles.x = ntiles[0]
            _ntiles.y = ntiles[1]

            _nx.x = nx[0]
            _nx.y = nx[1]

            if ( gc is None ):
                self.obj = new cppVec3Grid[ float3 ]( _ntiles, _nx )
            else:
                _gc.x.lower = gc[0,0]
                _gc.x.upper = gc[0,1]
                _gc.y.lower = gc[1,0]
                _gc.y.upper = gc[1,1]

                self.obj = new cppVec3Grid[ float3 ]( _ntiles, _nx, _gc )

            self.is_view = False

    cdef associate( self, cppVec3Grid[ float3 ] * src ):
        if ( not self.is_view ):
            raise Exception( "Vec3Grid object is not of view type, cannot associate existing cppVec3Grid object")

        self.obj = src

    def __dealloc__(self):
        if ( not self.is_view ):
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
        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.z}
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
# Current
#

from em2d cimport Current as cppCurrent

cdef class pyCurrent:
    cdef cppCurrent * obj
    cdef bint is_view
    cdef Vec3Grid J

    def __cinit__(self, list ntiles = None, list nx = None, list box = None, double dt = 0):

        cdef uint2 _ntiles
        cdef uint2 _nx
        cdef float2 _box

        self.J = Vec3Grid()

        if ( ntiles is not None ):
            _ntiles.x = ntiles[0]
            _ntiles.y = ntiles[1]

            _nx.x = nx[0]
            _nx.y = nx[1]

            _box.x = box[0]
            _box.y = box[1]

            self.obj = new cppCurrent( _ntiles, _nx, _box, dt )
            self.is_view = False

            self.J.associate( self.obj.J )
        else:
            self.obj = NULL
            self.is_view = True              

    cdef associate( self, cppCurrent * src ):
        if ( not self.is_view ):
            raise Exception( "pyCurrent object is not of view type, cannot associate existing cppCurrent object")
        self.obj = src
        self.J.associate( self.obj.J )

    def __dealloc__(self):
        if ( not self.is_view ):
            del self.obj

    @property
    def J( self ):
        return self.J
    
    @property
    def n( self ):
        return self.obj.get_iter()

    def save( self, fc ):
        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.z}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.save( _fc )    

    def plot( self, fc, **kwargs ):

        flabel = 'J_' + fc
        data = self.J.gather( fc )

        frange = [
            [ 0, self.obj.box.x ],
            [ 0, self.obj.box.y ]
        ]
        time = self.obj.get_iter() * self.obj.get_dt()

        visxd.plot2d( data, range = frange, 
            title  = "$\\sf {} $\n$t = {:g} \\;[\\sf {}]$".format( flabel, time, "1 / \\omega_n" ),
            xtitle = "$\\sf {} \\;[{}]$".format( 'x', 'c / \\omega_n' ),
            ytitle = "$\\sf {} \\;[{}]$".format( 'y', 'c / \\omega_n' ),
            vtitle = "$\\sf {} \\;[{}]$".format( flabel, 'm_e c \\omega_n e^{-1}' ),
            **kwargs
        )

    def vplot( self, **kwargs ):

        flabel = 'In\\text{-}plane\\;current'
        xdata = self.J.gather( 'x' )
        ydata = self.J.gather( 'y' )

        frange = [
            [ 0, self.obj.box.x ],
            [ 0, self.obj.box.y ]
        ]
        time = self.obj.get_iter() * self.obj.get_dt()

        visxd.plot2d( np.sqrt( xdata*xdata + ydata*ydata ), range = frange, 
            title  = "$\\sf {} $\n$t = {:g} \\;[\\sf {}]$".format( flabel, time, "1 / \\omega_n" ),
            xtitle = "$\\sf {} \\;[{}]$".format( 'x', 'c / \\omega_n' ),
            ytitle = "$\\sf {} \\;[{}]$".format( 'y', 'c / \\omega_n' ),
            vtitle = "$\\sf {} \\;[{}]$".format( flabel, 'e \\omega_n^2 / c' ),
            **kwargs
        )

###############################################################################
# EMF
#

from em2d cimport EMF as cppEMF
cimport emf

cdef class pyEMF:
    cdef cppEMF * obj
    cdef bint is_view

    cdef Vec3Grid E
    cdef Vec3Grid B

    def __cinit__(self, list ntiles = None, list nx = None, list box = None, double dt = 0):

        cdef uint2 _ntiles
        cdef uint2 _nx
        cdef float2 _box

        self.E = Vec3Grid()
        self.B = Vec3Grid()

        if ( ntiles is not None ):
            _ntiles.x = ntiles[0]
            _ntiles.y = ntiles[1]

            _nx.x = nx[0]
            _nx.y = nx[1]

            _box.x = box[0]
            _box.y = box[1]

            self.obj = new cppEMF( _ntiles, _nx, _box, dt )
            self.is_view = False

            self.E.associate( self.obj.E )
            self.B.associate( self.obj.B )
        else:
            self.obj = NULL
            self.is_view = True

    cdef associate( self, cppEMF * src ):
        if ( not self.is_view ):
            raise Exception( "pyEMF object is not of view type, cannot associate existing cppEMF object")
        self.obj = src  
        self.E.associate( self.obj.E )
        self.B.associate( self.obj.B )

    def __dealloc__(self):
        if ( not self.is_view ):
            del self.obj
    
    @property
    def E( self ):
        return self.E

    @property
    def B( self ):
        return self.B

    @property
    def n( self ):
        return self.obj.get_iter()

    @property
    def box( self ):
        return [ self.obj.box.x, self.obj.box.y ]

    def save( self, fld, fc ):
        tmp = {'e':emf.field.e, 'b':emf.field.b }
        cdef emf.field _field = tmp[fld]

        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.z}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.save( _field, _fc )

    def advance( self ):
        self.obj.advance()
    
    def advance( self, pyCurrent current ):
        self.obj.advance( current.obj[0] )

    def plot( self, fld, fc, **kwargs ):

        if ( fld == 'E' ):
            flabel = 'E_' + fc
            data = self.E.gather( fc )
        elif ( fld == 'B' ):
            flabel = 'B_' + fc
            data = self.B.gather( fc )
        else:
            raise Exception( "Invalid field ")

        frange = [
            [ 0, self.obj.box.x ],
            [ 0, self.obj.box.y ]
        ]
        time = self.obj.get_iter() * self.obj.get_dt()

        visxd.plot2d( data, range = frange, 
            title  = "$\\sf {} $\n$t = {:g} \\;[\\sf {}]$".format( flabel, time, "1 / \\omega_n" ),
            xtitle = "$\\sf {} \\;[{}]$".format( 'x', 'c / \\omega_n' ),
            ytitle = "$\\sf {} \\;[{}]$".format( 'y', 'c / \\omega_n' ),
            vtitle = "$\\sf {} \\;[{}]$".format( flabel, 'm_e c \\omega_n e^{-1}' ),
            **kwargs
        )

    def vplot( self, fld, **kwargs ):

        if ( fld == 'E' ):
            flabel = 'In\\text{-}plane\\;E\\;field'
            xdata = self.E.gather( 'x' )
            ydata = self.E.gather( 'y' )
        elif ( fld == 'B' ):
            flabel = 'In\\text{-}plane\\;B\\;field'
            xdata = self.B.gather( 'x' )
            ydata = self.B.gather( 'y' )
        else:
            raise Exception( "Invalid field ")

        frange = [
            [ 0, self.obj.box.x ],
            [ 0, self.obj.box.y ]
        ]
        time = self.obj.get_iter() * self.obj.get_dt()

        visxd.plot2d( np.sqrt( xdata*xdata + ydata*ydata ), range = frange, 
            title  = "$\\sf {} $\n$t = {:g} \\;[\\sf {}]$".format( flabel, time, "1 / \\omega_n" ),
            xtitle = "$\\sf {} \\;[{}]$".format( 'x', 'c / \\omega_n' ),
            ytitle = "$\\sf {} \\;[{}]$".format( 'y', 'c / \\omega_n' ),
            vtitle = "$\\sf {} \\;[{}]$".format( flabel, 'm_e c \\omega_n e^{-1}' ),
            **kwargs
        )


###############################################################################
# Lasers
#

from em2d cimport PlaneWave as cppPlaneWave
from em2d cimport Gaussian as cppGaussian

cdef class pyPlaneWave:
    cdef cppPlaneWave * obj
    cdef bint is_view

    def __cinit__(self):
        self.obj = NULL
        self.is_view = True  

    cdef associate( self, cppPlaneWave * src ):
        self.obj = src

    def __cinit__( self, *, float start = 0.0, float fwhm = 0.0,
                   float rise = 0.0, float flat = 0.0, float fall = 0.0,
                   float a0 = 0.0, float omega0 = 0.0, float polarization = 0.0,
                   float cos_pol = 0.0, float sin_pol = 0.0 ):

        self.obj = new cppPlaneWave()
        self.is_view = False

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

    def __dealloc__(self):
        if ( not self.is_view ):
            del self.obj

    def add( self, pyEMF emf ):
        # Plane Wave
        self.obj.add( emf.obj[0] )

cdef class pyGaussian:
    cdef cppGaussian * obj
    cdef bint is_view

    def __cinit__(self):
        self.obj = NULL
        self.is_view = True 

    cdef associate( self, cppGaussian * src ):
        self.obj = src

    def __cinit__( self, *, float start = 0.0, float fwhm = 0.0,
                   float rise = 0.0, float flat = 0.0, float fall = 0.0,
                   float a0 = 0.0, float omega0 = 0.0, float polarization = 0.0,
                   float cos_pol = 0.0, float sin_pol = 0.0,
                   float W0 = 0, float focus = 0, float axis = 0 ):

        self.obj = new cppGaussian()
        self.is_view = False

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
        if ( not self.is_view ):
            del self.obj

    def add( self, pyEMF emf ):
        self.obj.add( emf.obj[0] )

###############################################################################
# Particles
#

from em2d cimport Particles as cppParticles
cimport part

###############################################################################
# Udist
#

from em2d cimport None as cppNone
from em2d cimport Cold as cppCold
from em2d cimport Thermal as cppThermal
from em2d cimport ThermalCorr as cppThermalCorr

cdef class pyNone:
    cdef cppNone * obj

    def __cinit__(self ):
        self.obj = new cppNone( )

    def __dealloc__(self):
        del self.obj

cdef class pyCold:
    cdef cppCold * obj

    def __cinit__( self, list ufl ):

        cdef float3 _ufl
        _ufl.x = ufl[0]
        _ufl.y = ufl[1]
        _ufl.z = ufl[2]

        self.obj = new cppCold( _ufl )

    def __dealloc__(self):
        del self.obj

cdef class pyThermal:
    cdef cppThermal * obj

    def __cinit__( self, list uth, list ufl ):

        cdef float3 _uth
        _uth.x = uth[0]
        _uth.y = uth[1]
        _uth.z = uth[2]

        cdef float3 _ufl
        _ufl.x = ufl[0]
        _ufl.y = ufl[1]
        _ufl.z = ufl[2]

        self.obj = new cppThermal( _uth, _ufl )

    def __dealloc__(self):
        del self.obj

cdef class pyThermalCorr:
    cdef cppThermalCorr * obj

    def __cinit__( self, list uth, list ufl, int npmin = 2 ):

        cdef float3 _uth
        _uth.x = uth[0]
        _uth.y = uth[1]
        _uth.z = uth[2]

        cdef float3 _ufl
        _ufl.x = ufl[0]
        _ufl.y = ufl[1]
        _ufl.z = ufl[2]

        self.obj = new cppThermalCorr( _uth, _ufl, npmin )

    def __dealloc__(self):
        del self.obj

###############################################################################
# Species
#

from em2d cimport Species as cppSpecies

cdef class pySpecies:
    cdef cppSpecies * obj
    cdef bint is_view
    
    def __cinit__(self, str name = None, float m_q = 0, list ppc = None, *, udist = None ):

        cdef uint2 _ppc
        if ( ppc is not None ):
            _ppc.x = ppc[0]
            _ppc.y = ppc[1]

            self.obj = new cppSpecies( name.encode('utf-8'), m_q, _ppc )
            self.is_view = False

            if ( udist is not None ):
                self.set_udist( udist )
        else:
            self.obj = NULL
            self.is_view = True

    cdef associate( self, cppSpecies * src ):
        if ( not self.is_view ):
            raise Exception( "pySpecies object is not of view type, cannot associate existing cppSpecies object")
        self.obj = src

    def __dealloc__(self):
        if ( not self.is_view ):
            del self.obj
    
    cdef initialize( self, list box, list ntiles, list nx, double dt, int id ):
        cdef uint2 _ntiles
        _ntiles.x = ntiles[0]
        _ntiles.y = ntiles[1]

        cdef uint2 _nx
        _nx.x = nx[0]
        _nx.y = nx[1]

        cdef float2 _box
        _box.x = box[0]
        _box.y = box[1]

        self.obj.initialize( _box, _ntiles, _nx, dt, id )
    
    @property
    def name( self ):
        return self.obj.name.decode('utf-8')

    def set_udist( self, udist ):

        if ( isinstance( udist, pyNone )):
            self.obj.set_udist( (<pyNone> udist).obj[0] )
        elif ( isinstance( udist, pyCold )):
            self.obj.set_udist( (<pyCold> udist).obj[0] )
        elif ( isinstance( udist, pyThermal )):
            self.obj.set_udist( (<pyThermal> udist).obj[0] )
        elif ( isinstance( udist, pyThermalCorr )):
            self.obj.set_udist( (<pyThermalCorr> udist).obj[0] )
        else:
            raise Exception( "Invalid udist object")

    def save( self ):
        self.obj.save()

    def save_charge( self ):
        self.obj.save_charge()

    def gather( self, q ):
        dst = np.zeros( shape = self.obj.np_total(), dtype = np.float32 )
        cdef float[:] buffer = dst

        cdef part.quant q_ = {
            'x'  : part.quant.x,
            'y'  : part.quant.y,
            'ux' : part.quant.ux,
            'uy' : part.quant.uy,
            'uz' : part.quant.uz
        }[ q ]

        self.obj.gather( q_, & buffer[0] )
        return dst
    
    def plot( self, qx, qy, marker = '.', ms = 0.1, alpha = 0.5, **kwargs ):
        
        qlabels = { 'x':'x', 'y':'y', 'ux':'u_x', 'uy':'u_y', 'uz':'u_z' }
        qunits  = { 'x':'c/\\omega_n', 'y':'c/\\omega_n', 'ux':'c', 'uy':'c', 'uz':'c' }

        time = self.obj.get_iter() * self.obj.get_dt()

        visxd.plot1d( self.gather(qx), self.gather(qy), 
            xtitle = "$\\sf {} \\;[{}]$".format( qlabels[qx], qunits[qx] ),
            ytitle = "$\\sf {} \\;[{}]$".format( qlabels[qy], qunits[qy] ),
            title  = "$\\sf {} - {}/{} $\n$t = {:g} \\;[\\sf {}]$".format( 
                  self.name, qlabels[qy], qlabels[qx], time, "1 / \\omega_n"
                ),
            marker = marker, ms = ms, alpha = alpha,
            **kwargs )

###############################################################################
# Simulation
#

from em2d cimport Simulation as cppSimulation

cdef class pySimulation:
    cdef cppSimulation * obj
    cdef pyEMF emf
    cdef pyCurrent current

    def __cinit__(self, list ntiles, list nx, list box, double dt, *, species = None ):

        cdef uint2 _ntiles
        _ntiles.x = ntiles[0]
        _ntiles.y = ntiles[1]

        cdef uint2 _nx
        _nx.x = nx[0]
        _nx.y = nx[1]

        cdef float2 _box
        _box.x = box[0]
        _box.y = box[1]

        self.obj = new cppSimulation( _ntiles, _nx, _box, dt )

        self.emf = pyEMF()
        self.emf.associate( &self.obj.emf )

        self.current = pyCurrent()
        self.current.associate( &self.obj.current )       

        if ( isinstance( species, pySpecies )):
            self.obj.add_species( (<pySpecies> species).obj[0] )
        elif ( isinstance( species, (list,tuple) ) ):
            for s in species:
                self.obj.add_species( (<pySpecies> s).obj[0] )

    def __dealloc__(self):
        del self.obj
    
    @property
    def t( self ):
        return self.obj.get_t()
    
    @property
    def n( self ):
        return self.obj.get_iter()

    @property
    def ntiles( self ):
        return [ self.obj.ntiles.x, self.obj.ntiles.y ]
    
    @property
    def box( self ):
        return [ self.obj.box.x, self.obj.box.y ]
    
    @property
    def dt( self ):
        return self.obj.dt

    @property
    def emf( self ):
        return self.emf

    @property
    def current( self ):
        return self.current
    
    def add( self, src ):
        if ( isinstance( src, pySpecies )):
            self.obj.add_species( (<pySpecies> src).obj[0] )
        else:
            raise Exception("Invalid src object type")
    
    def advance( self ):
        self.obj.advance()

    def energy_info( self ):
        self.obj.energy_info()

