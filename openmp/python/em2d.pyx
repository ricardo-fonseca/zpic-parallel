# distutils: language = c++
# cython: language_level=3

import numpy as np
import visxd

###############################################################################
# Grid
#

from em2d cimport grid as cppGrid

cdef class Grid:
    """Grid( ntiles = None, nx = None, gc = None)

    Class representing a tiled grid of floats

    Parameters
    ----------
    ntiles : list of integers, optional
        Number of tiles (x,y) in the grid. Defaults to None, meaning the object
        will be a view of an existing cppGrid.
    nx : list of integers, optional
        Tile grid size (x,y)
    gc : list of integers, optional
        Number of guard cells

    See Also
    --------
    Grid.associate()
    """
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
        """associate( cppGrid[ float ] * src )
        
        Associate object with an existing cppGrid object. This requires that the object
        was created with ntiles = None

        Parameters
        ----------
        src : cppGrid[ float ] * src
            Pointer to existing cppGrid object
        """
        if ( not self.is_view ):
            raise Exception( "Grid object is not of view type, cannot associate existing cppGrid object")
        self.obj = src  

    def __dealloc__(self):
        if ( not self.is_view ):
            del self.obj

    def buffer_size(self):
        """buffer_size()

        Returns grid buffer size
        
        Returns
        -------
        buffer_size : int
            Grid buffer size (number of elements)
        """
        return self.obj.buffer_size()
    
    def zero(self):
        """zero()
        
        Sets all grid elements to zero
        """
        self.obj.zero()
    
    def set(self, float val):
        """set( val )

        Set all grid elements to a float value

        Parameters
        ----------
        val : float
            Value to be set on the grid
        """
        self.obj.set( val )
    
    def add(self, Grid rhs):
        """add( rhs )

        Adds another grid object to this one

        Parameters
        ----------
        rhs : Grid
            Grid object to be added
        """
        self.obj.add( rhs.obj[0] )
    
    def gather( self ):
        """gather()

        Gather the values from a tiled grid onto a contiguous grid

        Returns
        -------
        gather : numpy.ndarray
            Contiguous grid with all the grid values
        """
        dst = np.empty( shape = [ self.obj.dims.x, self.obj.dims.y ], dtype = np.float32 )
        cdef float [:,:] buffer = dst
        self.obj.gather( & buffer[ 0, 0 ] )
        return dst
    
    def copy_to_gc( self ):
        """copy_to_gc()

        Copies edge values to neighboring tile guard cells
        """
        self.obj.copy_to_gc()

    def add_from_gc( self ):
        """add_from_gc()

        Adds values from neighboring tile guard cells into edge values
        """
        self.obj.add_from_gc()

    def x_shift_left( self, unsigned int shift ):
        """x_shift_left( shift )

        Shits grid values left by the required amount (shift)

        Parameters
        ----------
        shift : unsigned int
            Number of cells to shift left, must be <= gc.x.upper
        """
        self.obj.x_shift_left( shift )
    
    def kernel_x( self, float a, float b, float c ):
        """kernel_x(a,b,c)

        Performs a convolution with a 3 point kernel along x

        Parameters
        ----------
        a : float
            Left kernel value
        b : float
            Center kernel value
        c : float
            Right kernel value
        """
        self.obj.kernel3_x( a, b, c )

    def kernel_y( self, float a, float b, float c ):
        """kernel_y(a,b,c)

        Performs a convolution with a 3 point kernel along y

        Parameters
        ----------
        a : float
            Left kernel value
        b : float
            Center kernel value
        c : float
            Right kernel value
        """
        self.obj.kernel3_y( a, b, c )

    def save( self, str filename ):
        """save( filename )

        Save grid data to .zdf file (without metadata)

        Parameters
        ----------
        filename : string
            Name of file to write (including path)
        """
        self.obj.save( filename.encode('utf-8') )


###############################################################################
# Vec3Grid
#

from em2d cimport vec3grid as cppVec3Grid
cimport fcomp

cdef class Vec3Grid:
    """Vec3Grid( ntiles = None, nx = None, gc = None)

    Class representing a tiled grid of float3

    Parameters
    ----------
    ntiles : list of integers, optional
        Number of tiles (x,y) in the grid. Defaults to None, meaning the object
        will be a view of an existing cppGrid.
    nx : list of integers, optional
        Tile grid size (x,y)
    gc : list of integers, optional
        Number of guard cells

    See Also
    --------
    Vec3Grid.associate()
    """

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
        """associate( cppVec3Grid[ float ] * src )
        
        Associate object with an existing cppVec3Grid object. This requires that the object
        was created with ntiles = None

        Parameters
        ----------
        src : cppVec3Grid[ float ] * src
            Pointer to existing cppVec3Grid object
        """
        if ( not self.is_view ):
            raise Exception( "Vec3Grid object is not of view type, cannot associate existing cppVec3Grid object")

        self.obj = src

    def __dealloc__(self):
        if ( not self.is_view ):
            del self.obj
    
    def buffer_size(self):
        """buffer_size()

        Returns grid buffer size
        
        Returns
        -------
        buffer_size : int
            Grid buffer size (number of elements)
        """
        return self.obj.buffer_size()
    
    def zero(self):
        """zero()
        
        Sets all grid elements to zero
        """
        return self.obj.zero()
    
    def set(self, float3 val):
        """set( val )

        Set all grid elements to float3 value

        Parameters
        ----------
        val : float3
            Value to be set on the grid
        """
        self.obj.set( val )
    
    def add(self, Vec3Grid rhs):
        """add( rhs )

        Adds another Vec3Grid object to this one

        Parameters
        ----------
        rhs : Grid
            Grid object to be added
        """
        return self.obj.add( rhs.obj[0] )
    
    def gather( self, str fc ):
        """gather( fc )

        Gather the values of the specified field component from a tiled Vec3Grid
        onto a contiguous grid

        Parameters
        ----------
        fc : char
            Field component, must be one of 'x', 'y' or 'z'

        Returns
        -------
        gather : numpy.ndarray
            Contiguous grid with all the grid values of the selected field component
        """
        dst = np.empty( shape = [ self.obj.dims.x, self.obj.dims.y ], dtype = np.float32 )
        cdef float [:,:] buffer = dst
        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.z}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.gather( _fc, & buffer[ 0, 0 ] )
        
        return dst
    
    def copy_to_gc( self ):
        """copy_to_gc()

        Copies edge values to neighboring tile guard cells
        """
        self.obj.copy_to_gc()

    def add_from_gc( self ):
        """add_from_gc()

        Adds values from neighboring tile guard cells into edge values
        """
        self.obj.add_from_gc()

    def x_shift_left( self, unsigned int shift ):
        """x_shift_left( shift )

        Shits grid values left by the required amount (shift)

        Parameters
        ----------
        shift : unsigned int
            Number of cells to shift left, must be <= gc.x.upper
        """
        self.obj.x_shift_left( shift )
    
    def kernel_x( self, float a, float b, float c ):
        """kernel_x(a,b,c)

        Performs a convolution with a 3 point kernel along x. The same kernel
        values are applied to all field components.

        Parameters
        ----------
        a : float
            Left kernel value
        b : float
            Center kernel value
        c : float
            Right kernel value
        """
        self.obj.kernel3_x( a, b, c )

    def kernel_y( self, float a, float b, float c ):
        """kernel_y(a,b,c)

        Performs a convolution with a 3 point kernel along y. The same kernel
        values are applied to all field components.

        Parameters
        ----------
        a : float
            Left kernel value
        b : float
            Center kernel value
        c : float
            Right kernel value
        """
        self.obj.kernel3_y( a, b, c )

    def save( self, str fc, str filename ):
        """save( fc, filename )

        Save grid data of selected field component to .zdf file (without metadata)

        Parameters
        ----------
        fc : string
            Field component, must be one of 'x', 'y' or 'z'

        filename : string
            Name of file to write (including path)
        """
        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.y}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.save( _fc, filename.encode('utf-8') )

###############################################################################
# Current
#

from em2d cimport Current as cppCurrent

cdef class pyCurrent:
    """pyCurrent( ntiles = None, nx = None, box = None, dt = 0)

    Class representing current density

    Parameters
    ----------
    ntiles : list of integers, optional
        Number of tiles (x,y) in the grid. Defaults to None, meaning the object
        will be a view of an existing cppCurrent object.
    nx : list of integers, optional
        Tile grid size (x,y)
    box : list of double, optional
        Simulation box size (x,y) in physical units
    dt : double, optional
        Simulation time-step

    See Also
    --------
    pyCurrent.associate()
    """

    cdef cppCurrent * obj
    """Pointer to corresponding cppCurrent object"""
    cdef bint is_view
    """True if object is a view of an existing cppCurrent object"""
    cdef Vec3Grid J
    """Vec3Grid holding current density values"""

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
        """associate( cppCurrent * src )
        
        Associate object with an existing cppCurrent object. This requires that the object
        was created with ntiles = None

        Parameters
        ----------
        src : cppCurrent * src
            Pointer to existing cppCurrent object
        """
        if ( not self.is_view ):
            raise Exception( "pyCurrent object is not of view type, cannot associate existing cppCurrent object")
        self.obj = src
        self.J.associate( self.obj.J )

    def __dealloc__(self):
        if ( not self.is_view ):
            del self.obj

    @property
    def J( self ):
        """Current Density

        Vec3Grid with current density values
        """
        return self.J
    
    @property
    def n( self ):
        """Iteration number

        Current iteration number
        """
        return self.obj.get_iter()

    def save( self, fc ):
        """save( fc )

        Save selected field component of current density to .zdf file with full metadata

        Parameters
        ----------
        fc : char
            Field component, must be one of 'x', 'y' or 'z'
        """
        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.z}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.save( _fc )    

    def plot( self, fc, **kwargs ):
        """plot( fc, **kwargs )

        Plot selected field component of current density. Plot is done using visxd.plot2d()

        Parameters
        ----------
        fc : string
            Field component, must be one of 'x', 'y' or 'z'
        **kwargs
            Additional keyword arguments to be passed on to visxd.plot2d()
        """

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
        """vplot( **kwargs )

        Plot in-plane (x,y) current density. Plot is done using visxd.plot2d()

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to be passed on to visxd.plot2d()
        """

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
    """pyEMF( ntiles = None, nx = None, box = None, dt = 0)

    Class representing current EM fields

    Parameters
    ----------
    ntiles : list of integers, optional
        Number of tiles (x,y) in the grid. Defaults to None, meaning the object
        will be a view of an existing cppCurrent object.
    nx : list of integers, optional
        Tile grid size (x,y)
    box : list of double, optional
        Simulation box size (x,y) in physical units
    dt : double, optional
        Simulation time-step

    See Also
    --------
    pyEMF.associate()
    """
    cdef cppEMF * obj
    """Pointer to corresponding cppCurrent object"""
    cdef bint is_view
    """True if object is a view of an existing cppCurrent object"""
    cdef Vec3Grid E
    """Vec3Grid holding E field values"""
    cdef Vec3Grid B
    """Vec3Grid holding B field values"""

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
        """associate( cppEMF * src )
        
        Associate object with an existing cppEMF object. This requires that the object
        was created with ntiles = None

        Parameters
        ----------
        src : cppEMF * src
            Pointer to existing cppEMF object
        """
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
        """Electric field

        Vec3Grid with E-field values
        """
        return self.E

    @property
    def B( self ):
        """Magnetic field

        Vec3Grid with B-field values
        """
        return self.B

    @property
    def n( self ):
        """Iteration number

        Current iteration number
        """
        return self.obj.get_iter()

    @property
    def box( self ):
        """Simulation box size

        Simulation box size in simulation units
        """
        return [ self.obj.box.x, self.obj.box.y ]

    def save( self, fld, fc ):
        """save( fld, fc )

        Save selected field component to .zdf file with full metadata

        Parameters
        ----------
        fld : string
            Field type, must be one of 'E' (electric field) or 'B' magnetic field
        fc : string
            Field component, must be one of 'x', 'y' or 'z'
        """
        tmp = {'E':emf.field.e, 'B':emf.field.b }
        cdef emf.field _field = tmp[fld]

        tmp = {'x':fcomp.x, 'y':fcomp.y, 'z':fcomp.z}
        cdef fcomp.cart _fc = tmp[ fc ]
        self.obj.save( _field, _fc )

    def advance( self ):
        """advance()

        Advance EM object 1 iteration, disregarding current density
        """
        self.obj.advance()
    
    def advance( self, pyCurrent current ):
        """advance()
        
        Advance EM object 1 iteration

        Parameters
        ----------
        current : pyCurrent
            Electric current density
        """
        self.obj.advance( current.obj[0] )

    def plot( self, fld, fc, **kwargs ):
        """plot( fld, fc, **kwargs )

        Plot selected field component. Plot is done using visxd.plot2d()

        Parameters
        ----------
        fld : string
            Field type, must be one of 'E' (electric field) or 'B' (magnetic field)
        fc : string
            Field component, must be one of 'x', 'y' or 'z'
        **kwargs
            Additional keyword arguments to be passed on to visxd.plot2d()
        """

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
        """vplot( **kwargs )

        Plot in-plane (x,y) field. Plot is done using visxd.plot2d()

        Parameters
        ----------
        fld : string
            Field type, must be one of 'E' (electric field) or 'B' (magnetic field)
        **kwargs
            Additional keyword arguments to be passed on to visxd.plot2d()
        """

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
    """pyPlaneWave( start = 0, fwhm = 0, rise = 0, flat = 0, fall = 0,
                    a0 = 0, omega0 = 0, polarization = 0,
                    cos_pol = 0, sin_pol = 0 )
    
    Class representing plane wave laser pulse

    Parameters
    ----------
    start : float
        Start position of laser pulse (front)
    fwhm : float
        Laser pulse FWHM, overrides rise / flat / fall parameters
    rise : float
        Laser pulse rise time
    flat : float
        Laser pulse flat time
    fall : float
        Laser pulse fall time
    a0 : float
        Normalized peak vector potential of the pulse
    omega0 : float 
        Laser frequency normalized to simulation frequency
    polarization : float
        Polarization angle in radians. Will be ignored unless cos_pol and
        sin_pol are both 0
    cos_pol : float
        Cosine of the polarization angle
    sin_pol : float
        Sine of the polarization angle
    """
    cdef cppPlaneWave * obj

    def __cinit__( self, *, float start = 0.0, float fwhm = 0.0,
                   float rise = 0.0, float flat = 0.0, float fall = 0.0,
                   float a0 = 0.0, float omega0 = 0.0, float polarization = 0.0,
                   float cos_pol = 0.0, float sin_pol = 0.0 ):

        self.obj = new cppPlaneWave()

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
        del self.obj

    def add( self, pyEMF emf ):
        """add( emf )

        Adds laser pulse to EMF object

        Parameters
        ----------
        emf : pyEMF
            EM fields object
        """
        # Plane Wave
        self.obj.add( emf.obj[0] )

cdef class pyGaussian:
    """pyPlaneWave( start = 0, fwhm = 0, rise = 0, flat = 0, fall = 0,
                    a0 = 0, omega0 = 0, polarization = 0,
                    cos_pol = 0, sin_pol = 0 )
    
    Class representing plane wave laser pulse

    Parameters
    ----------
    start : float
        Start position of laser pulse (front)
    fwhm : float
        Laser pulse FWHM, overrides rise / flat / fall parameters
    rise : float
        Laser pulse rise time
    flat : float
        Laser pulse flat time
    fall : float
        Laser pulse fall time
    a0 : float
        Normalized peak vector potential of the pulse
    omega0 : float 
        Laser frequency normalized to simulation frequency
    polarization : float
        Polarization angle in radians. Will be ignored unless cos_pol and
        sin_pol are both 0
    cos_pol : float
        Cosine of the polarization angle
    sin_pol : float
        Sine of the polarization angle
    W0 : float
        Beam waist
    focus : float
        Focal plane position (x)
    axis : float
        Propagation axis position (y)
    """
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
        """add( emf )

        Adds laser pulse to EMF object

        Parameters
        ----------
        emf : pyEMF
            EM fields object
        """
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
    """pyNone()

    Class representing a frozen (0 fluid, 0 temperature) momentum distribution
    """
    cdef cppNone * obj

    def __cinit__(self ):
        self.obj = new cppNone( )

    def __dealloc__(self):
        del self.obj

cdef class pyCold:
    """pyCold( ufl )

    Class representing a cold (0 temperature) momentum distribution

    Parameters
    ----------
    ufl : { float, float, float }
        Fluid momentum
    """
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
    """pyThermal( uth, ufl )

    Class representing a thermal momentum distribution

    Parameters
    ----------
    uth : { float, float, float }
        Temperature distribution
    ufl : { float, float, float }
        Fluid momentum
    """

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
    """pyThermal( uth, ufl, npmin = 2 )

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
    """pySpecies( name, m_q, ppc, udist = None)

    Class representing a particle species

    Parameters
    ----------
    name : string
        Name of particle species (used for diagnostics)
    m_q : float
        Mass over charge ratio in simulation units (e.g. for electrons this 
        should be -1)
    ppc : { int, int }
        Number of particles per cell. Defaults to None, meaning the object
        will be a view of an existing cppSpecies object.
    udist : object
        Velocity distribution to use, defaults to pyNone (0 velocity)
    """
    cdef cppSpecies * obj
    """Pointer to corresponding cppCurrent object"""
    cdef bint is_view
    """True if object is a view of an existing cppCurrent object"""    

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
        """associate( cppSpecies * src )
        
        Associate object with an existing cppSpecies object. This requires that the object
        was created with ppc = None

        Parameters
        ----------
        src : cppSpecies * src
            Pointer to existing cppSpecies object
        """
        if ( not self.is_view ):
            raise Exception( "pySpecies object is not of view type, cannot associate existing cppSpecies object")
        self.obj = src

    def __dealloc__(self):
        if ( not self.is_view ):
            del self.obj
    
    cdef initialize( self, list box, list ntiles, list nx, double dt, int id ):
        """initialize( box, ntiles, nx, dt, id )

        Initialize data structures and inject initial particle distribution

        Parameters
        ----------
        box : { float, float }
            Simulation box size (x,y)
        ntiles : { int, int }
            Number of tiles (x,y)
        nx : { int, int }
            Tile grid grize (x,y)
        dt : float
            Simulation time step
        id : int
            Species unique identifier. This is also used to initialize the random number generator.
        """
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
        """Species name"""
        return self.obj.name.decode('utf-8')

    def set_udist( self, udist ):
        """set_udist( udist )

        Sets initial generalized velocity (u) distribution

        Parameters
        ----------
        udist : object
            Initial velocity distribution, must be of class pyNone, pyCold, pyThermal or pyThermalCorr
        """

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
        """save()

        Save particle data to .zdf file
        """
        self.obj.save()

    def save_charge( self ):
        """save_charge()

        Deposits species charge density and saves grid to .zdf file
        """
        self.obj.save_charge()

    def gather( self, q ):
        """gather( q )

        Gathers specied quantity into a contiguous grid

        Parameters
        ----------
        q : string
            Quantity to gather, must be one of 'x', 'y', 'ux', 'uy' or 'uz'
        
        Returns
        -------
        gather : numpy.ndarray
            Contiguous NumPy array with the selected quantity 
        """

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
        """plot( qx, qy, marker, ms, alpha, **kwargs )

        Do an x,y scatter plot of every particle using the selected quantities.
        Plot is done using visxd.plot1d()

        Parameters
        ----------
        qx : string
            Quantity to use for x axis. Must must be one of 'x', 'y', 'ux', 'uy' or 'uz'
        qy : string
            Quantity to use for y axis. Must must be one of 'x', 'y', 'ux', 'uy' or 'uz'
        marker : string, default = '.'
            Marker to use for plotting data, defaults to dot ('.')
        ms : float, default = 0.1
            Marker size, defaults to 0.1
        alpha : float
            Marker transparency, defaults to 0.5
        **kwargs
            Additional keyword arguments to be passed on to visxd.plot1d()
        """

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
    """pySimulation( ntiles, nx, box, dt, species = None )

    Class representing an em2d simulation

    Parameters
    ----------
    ntiles : { int, int }
        Number of tiles (x,y) in the simulation grid
    nx : { int, int }
        Size (x,y) of individual tiles
    box : { float, float }
        Simulation box size (x,y) in simulation units
    dt : float
        Simulation time step in simulation units
    species : pySpecies or list of pySpecies
        Species or list of Species to be added to the simulation. Defaults to None,
        meaning no species are to be added at this time. Species may be added later
        using the add() method
    """
    cdef cppSimulation * obj
    """Pointer to corresponding cppSimulation object"""
    cdef pyEMF emf
    """View of the simulation EMF object"""
    cdef pyCurrent current
    """View of the simulation Current object"""

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
        """Simulation time (in simulation units)"""
        return self.obj.get_t()
    
    @property
    def n( self ):
        """Simulation iteration number"""
        return self.obj.get_iter()

    @property
    def ntiles( self ):
        """Simulation number of tiles """
        return ( self.obj.ntiles.x, self.obj.ntiles.y )
    
    @property
    def box( self ):
        """Simulation box size (in simulation units)"""
        return ( self.obj.box.x, self.obj.box.y )
    
    @property
    def dt( self ):
        """Simulation time step (in simulation units)"""
        return self.obj.dt

    @property
    def emf( self ):
        """View of the simulation EM fields"""
        return self.emf

    @property
    def current( self ):
        """View of the simulation current density"""
        return self.current
    
    def add( self, src ):
        """add( src )

        Add object to simulation

        Parameters
        ----------
        src : object
            Object to add to the simulation. Currently only pySpecies objects
            are supported
        """
        if ( isinstance( src, pySpecies )):
            self.obj.add_species( (<pySpecies> src).obj[0] )
        else:
            raise Exception("Invalid src object type")
    
    def advance( self ):
        """advance()

        Advance simulation 1 iteration. This will:
        1. Zero global current
        2. Advance all species in the simulation and deposit current
        3. Update current edge values and guard cells
        4. Advance EM fields using the newly deposited current
        """
        self.obj.advance()

    def energy_info( self ):
        """energe_info()

        Print out energy report
        """
        self.obj.energy_info()

