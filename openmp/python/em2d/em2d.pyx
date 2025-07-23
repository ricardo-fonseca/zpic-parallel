# distutils: language = c++
# cython: language_level=3

import numpy as np
import visxd

###############################################################################
# ZPIC utilities
#
from em2d cimport zpic

class zpic:
    def sys_info():
        """sys_info()
        Prints system information (OpenMP / SIMD support)
        """
        zpic.sys_info()

    def courant( list dx = None, list dims = None, list ntiles = None, list nx = None, list box = None ):
        """courant( dx, dims, ntiles, nx, box )

        Returns the Courant-Friedrichs-Lewy limit for time step

        Parameters
        ----------
        dx : { float, float }
            Simulation cell size (x,y), defaults to None. If specified no other
            parameters are required
        dims : { int, int }
            Simulation grid size (x,y), defaults to None. If dims and dx are None
            user must use both ntiles and nx parameters
        ntiles : { int, int }
            Number of simulation tiles (x,y). Not needed if dims or dx was specified
        nx : { int, int }
            Individual tile grid size. Not needed if dims or dx  was specified
        box : float
            Simulation box size in simulation units. Not needed if dx was specified
        """

        cdef float2 dx_
        cdef uint2 dims_
        cdef uint2 ntiles_
        cdef uint2 nx_
        cdef float2 box_

        if ( dx is not None ):
            dx_.x = dx[0]
            dx_.y = dx[1]
            return zpic.courant( dx_ )
        else:
            box_.x = box[0]
            box_.y = box[1]

            if ( ntiles is None ):
                dims_.x = dims[0]
                dims_.y = dims[1]
                return zpic.courant( dims_, box_ )
            else:
                ntiles_.x = ntiles[0]
                ntiles_.y = ntiles[1]

                nx_.x = nx[0]
                nx_.y = nx[1]
                
                return zpic.courant( ntiles_, nx_, box_ )

###############################################################################
# Grid
#

from em2d cimport grid

cdef class Grid:
    """Grid( ntiles = None, nx = None, gc = None)

    Class representing a tiled grid of floats

    Parameters
    ----------
    ntiles : list of integers, optional
        Number of tiles (x,y) in the grid. Defaults to None, meaning the object
        will be a view of an existing grid.grid.
    nx : list of integers, optional
        Tile grid size (x,y)
    gc : list of integers, optional
        Number of guard cells

    See Also
    --------
    Grid.associate()
    """
    cdef grid.grid[ float ] * obj
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
            lntiles.y = ntiles[1]

            lnx.x = nx[0]
            lnx.y = nx[1]

            if ( gc is None ):
                self.obj = new grid.grid[ float ]( lntiles, lnx )
            else:
                lgc.x.lower = gc[0][0]
                lgc.x.upper = gc[0][1]
                lgc.y.lower = gc[1][0]
                lgc.y.upper = gc[1][1]
                self.obj = new grid.grid[ float ]( lntiles, lnx, lgc )

            self.is_view = False

    cdef associate( self, grid.grid[ float ] * src ):
        """associate( grid.grid[ float ] * src )
        
        Associate object with an existing grid.grid object. This requires that the object
        was created with ntiles = None

        Parameters
        ----------
        src : grid.grid[ float ] * src
            Pointer to existing grid.grid object
        """
        if ( not self.is_view ):
            raise Exception( "Grid object is not of view type, cannot associate existing grid.grid object")
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

        cdef uint2 dims = self.obj.get_dims()
        dst = np.empty( shape = [ dims.y, dims.x ], dtype = np.float32 )
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

from em2d.vec3grid cimport *
from em2d cimport fcomp

cdef class Vec3Grid:
    """Vec3Grid( ntiles = None, nx = None, gc = None)

    Class representing a tiled grid of float3

    Parameters
    ----------
    ntiles : list of integers, optional
        Number of tiles (x,y) in the grid. Defaults to None, meaning the object
        will be a view of an existing grid.grid.
    nx : list of integers, optional
        Tile grid size (x,y)
    gc : list of integers, optional
        Number of guard cells

    See Also
    --------
    Vec3Grid.associate()
    """

    # cdef vec3grid.vec3grid[ float3 ] * obj
    # cdef bint is_view

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
                self.obj = new vec3grid.vec3grid[ float3 ]( _ntiles, _nx )
            else:
                _gc.x.lower = gc[0][0]
                _gc.x.upper = gc[0][1]
                _gc.y.lower = gc[1][0]
                _gc.y.upper = gc[1][1]

                self.obj = new vec3grid.vec3grid[ float3 ]( _ntiles, _nx, _gc )

            self.is_view = False

    cdef associate( self, vec3grid.vec3grid[ float3 ] * src ):
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
        dst = np.empty( shape = [ self.obj.dims.y, self.obj.dims.x ], dtype = np.float32 )
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

from em2d cimport current
cimport em2d.filter.filter

cdef class Current:
    """Current( ntiles = None, nx = None, box = None, dt = 0)

    Class representing current density

    Parameters
    ----------
    ntiles : list of integers, optional
        Number of tiles (x,y) in the grid. Defaults to None, meaning the object
        will be a view of an existing current.Current object.
    nx : list of integers, optional
        Tile grid size (x,y)
    box : list of double, optional
        Simulation box size (x,y) in physical units
    dt : double, optional
        Simulation time-step

    See Also
    --------
    Current.associate()
    """

    # cdef current.Current * obj
    # """Pointer to corresponding current.Current object"""
    # cdef bint is_view
    # """True if object is a view of an existing current.Current object"""
    # cdef Vec3Grid J
    # """Vec3Grid holding current density values"""

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

            self.obj = new current.Current( _ntiles, _nx, _box, dt )
            self.is_view = False

            self.J.associate( self.obj.J )
        else:
            self.obj = NULL
            self.is_view = True              

    cdef associate( self, current.Current * src ):
        """associate( current.Current * src )
        
        Associate object with an existing current.Current object. This requires that the object
        was created with ntiles = None

        Parameters
        ----------
        src : current.Current * src
            Pointer to existing current.Current object
        """
        if ( not self.is_view ):
            raise Exception( "Current object is not of view type, cannot associate existing current.Current object")
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
    
    def set_filter( self, filter ):
        """set_filter( filter )

        Sets current filtering to be applied at each time step

        Parameters
        ----------
        filter : object
            Digital filter to be used, must be one of em2d.filter.None, 
            .Binomial or .Compensated
        """
        if ( isinstance( filter, em2d.filter.filter.None )):
            self.obj.set_filter( (<em2d.filter.filter.None> filter).obj[0] )
        elif ( isinstance( filter, em2d.filter.filter.Binomial )):
            self.obj.set_filter( (<em2d.filter.filter.Binomial> filter).obj[0] )
        elif ( isinstance( filter, em2d.filter.filter.Compensated )):
            self.obj.set_filter( (<em2d.filter.filter.Compensated> filter).obj[0] )
        else:
            raise Exception( "Invalid filter object")

    def save( self, fc ):
        """save( fc )

        Save selected field component of current density to .zdf file with full
        metadata

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

        Plot selected field component of current density. Plot is done using
        visxd.plot2d()

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

from em2d cimport emf

cdef class EMF:
    """EMF( ntiles = None, nx = None, box = None, dt = 0)

    Class representing current EM fields

    Parameters
    ----------
    ntiles : list of integers, optional
        Number of tiles (x,y) in the grid. Defaults to None, meaning the object
        will be a view of an existing current.Current object.
    nx : list of integers, optional
        Tile grid size (x,y)
    box : list of double, optional
        Simulation box size (x,y) in physical units
    dt : double, optional
        Simulation time-step

    See Also
    --------
    EMF.associate()
    """
    # cdef emf.EMF * obj
    # """Pointer to corresponding current.Current object"""
    # cdef bint is_view
    # """True if object is a view of an existing current.Current object"""
    # cdef Vec3Grid E
    # """Vec3Grid holding E field values"""
    # cdef Vec3Grid B
    # """Vec3Grid holding B field values"""

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

            self.obj = new emf.EMF( _ntiles, _nx, _box, dt )
            self.is_view = False

            self.E.associate( self.obj.E )
            self.B.associate( self.obj.B )
        else:
            self.obj = NULL
            self.is_view = True

    cdef associate( self, emf.EMF * src ):
        """associate( emf.EMF * src )
        
        Associate object with an existing emf.EMF object. This requires that the object
        was created with ntiles = None

        Parameters
        ----------
        src : emf.EMF * src
            Pointer to existing emf.EMF object
        """
        if ( not self.is_view ):
            raise Exception( "EMF object is not of view type, cannot associate existing emf.EMF object")
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
    
    def advance( self, Current current ):
        """advance()
        
        Advance EM object 1 iteration

        Parameters
        ----------
        current : Current
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
# Particles
#

from em2d cimport part
from em2d cimport particles



###############################################################################
# Species
#

from em2d cimport species
cimport em2d.udist.udist
cimport em2d.density.density

cdef class Species:
    """Species( name, m_q, ppc, udist = None)

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
        will be a view of an existing species.Species object.
    udist : object
        Velocity distribution to use, defaults to pyNone (0 velocity)
    """
    cdef species.Species * obj
    """Pointer to corresponding current.Current object"""
    cdef bint is_view
    """True if object is a view of an existing species.Species object"""    

    def __cinit__(self, str name = None, float m_q = 0, list ppc = None,
            *, udist = None, density = None ):

        cdef uint2 _ppc
        if ( ppc is not None ):
            _ppc.x = ppc[0]
            _ppc.y = ppc[1]

            self.obj = new species.Species( name.encode('utf-8'), m_q, _ppc )
            self.is_view = False

            if ( udist is not None ):
                self.set_udist( udist )
            
            if ( density is not None ):
                self.set_density( density )
        else:
            self.obj = NULL
            self.is_view = True

    cdef associate( self, species.Species * src ):
        """associate( species.Species * src )
        
        Associate object with an existing species.Species object. This requires that the object
        was created with ppc = None

        Parameters
        ----------
        src : species.Species * src
            Pointer to existing species.Species object
        """
        if ( not self.is_view ):
            raise Exception( "Species object is not of view type, cannot associate existing species.Species object")
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
        """set_udist( dist )

        Sets initial generalized velocity (u) distribution

        Parameters
        ----------
        udist : object
            Initial velocity distribution, must be of class pyNone, pyCold, pyThermal or pyThermalCorr
        """

        if ( isinstance( udist, em2d.udist.udist.None )):
            self.obj.set_udist( (<em2d.udist.udist.None> udist).obj[0] )
        elif ( isinstance( udist, em2d.udist.udist.Cold )):
            self.obj.set_udist( (<em2d.udist.udist.Cold> udist).obj[0] )
        elif ( isinstance( udist, em2d.udist.udist.Thermal )):
            self.obj.set_udist( (<em2d.udist.udist.Thermal> udist).obj[0] )
        elif ( isinstance( udist, em2d.udist.udist.ThermalCorr )):
            self.obj.set_udist( (<em2d.udist.udist.ThermalCorr> udist).obj[0] )
        else:
            raise Exception( "Invalid udist object")
    
    def set_density( self, density ):
        """set_density( density )

        Sets density profile

        Parameters
        ----------
        density : object
            Density profile, must be of class em2d.density.None, .Uniform,
            .Step or .Slab
        """

        if ( isinstance( density, em2d.density.density.None )):
            self.obj.set_density( (<em2d.density.density.None> density).obj[0] )
        elif ( isinstance( density, em2d.density.density.Uniform )):
            self.obj.set_density( (<em2d.density.density.Uniform> density).obj[0] )
        elif ( isinstance( density, em2d.density.density.Step )):
            self.obj.set_density( (<em2d.density.density.Step> density).obj[0] )
        elif ( isinstance( density, em2d.density.density.Slab )):
            self.obj.set_density( (<em2d.density.density.Slab> density).obj[0] )
        else:
            raise Exception( "Invalid density object")

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
    
    def get_charge( self ):
        """get_charge()

        Returns species charge density as a Grid object. Charge density is
        recalculated when the routine is called.

        Returns
        -------
        charge : Grid
            Grid object holding charge density
        """

        ntiles_ = self.obj.get_ntiles()
        nx_     = self.obj.get_nx()

        ntiles = [ ntiles_.x, ntiles_.y ]
        nx     = [ nx_.x, nx_.y ]
        gc     = [ [0,1], [0,1] ]
        charge = Grid( ntiles, nx, gc )

        charge.zero()
        self.obj.deposit_charge( charge.obj[0] )
        charge.add_from_gc()
        
        return charge

    
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

    def plot_charge( self, **kwargs ):
        """plot_charge( fc, **kwargs )

        Plot the charge density for the species. Plot is done using visxd.plot2d()

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to be passed on to visxd.plot2d()
        """

        box = self.obj.get_box()

        frange = [
            [ 0, box.x ],
            [ 0, box.y ]
        ]
        time = self.obj.get_iter() * self.obj.get_dt()

        visxd.plot2d( self.get_charge().gather(), range = frange, 
            title  = "$\\sf {} \\;charge \\;density$\n$t = {:g} \\;[\\sf {}]$".format( self.name, time, "1 / \\omega_n" ),
            xtitle = "$\\sf {} \\;[{}]$".format( 'x', 'c / \\omega_n' ),
            ytitle = "$\\sf {} \\;[{}]$".format( 'y', 'c / \\omega_n' ),
            vtitle = "$\\sf {} - {} \\;[{}]$".format( self.name, '\\rho', 'n_e' ),
            **kwargs
        )

###############################################################################
# Simulation
#

from em2d cimport simulation

cimport em2d.laser.laser

cdef class Simulation:
    """Simulation( ntiles, nx, box, dt, species = None )

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
    species : Species or list of Species
        Species or list of Species to be added to the simulation. Defaults to None,
        meaning no species will be used.
    moving_window : boolean
        Use a moving window for the simulation, defaults to False
    """
    # cdef simulation.Simulation * obj
    # """Pointer to corresponding cppSimulation object"""
    # cdef EMF emf
    # """View of the simulation EMF object"""
    # cdef Current current
    # """View of the simulation Current object"""
    # cdef bint mov_window
    # """Simulation uses a moving window"""

    def __cinit__(self, list ntiles, list nx, list box, double dt, *, 
        species = None, moving_window = False ):

        cdef uint2 _ntiles
        _ntiles.x = ntiles[0]
        _ntiles.y = ntiles[1]

        cdef uint2 _nx
        _nx.x = nx[0]
        _nx.y = nx[1]

        cdef float2 _box
        _box.x = box[0]
        _box.y = box[1]

        self.obj = new simulation.Simulation( _ntiles, _nx, _box, dt )

        self.emf = EMF()
        self.emf.associate( &self.obj.emf )

        self.current = Current()
        self.current.associate( &self.obj.current )       

        if ( isinstance( species, Species )):
            self.obj.add_species( (<Species> species).obj[0] )
        elif ( isinstance( species, (list,tuple) ) ):
            for s in species:
                self.obj.add_species( (<Species> s).obj[0] )
        
        self.mov_window = moving_window
        if ( self.mov_window ):
            self.obj.set_moving_window()

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
            Object to add to the simulation. Currently only Laser objects
            are supported
        """
        if ( isinstance( src, em2d.laser.laser.PlaneWave )):
            (<em2d.laser.laser.PlaneWave> src).add( self.emf )
        elif ( isinstance( src, em2d.laser.laser.Gaussian )):
            (<em2d.laser.laser.Gaussian> src).add( self.emf )
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
        if ( self.mov_window ):
            self.obj.advance_mov_window()
        else:
            self.obj.advance()

    def energy_info( self ):
        """energe_info()

        Print out energy report
        """
        self.obj.energy_info()

