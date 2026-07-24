# distutils: language = c++
# cython: language_level=3

###############################################################################
# Lasers
#

cimport em2d.laser.defs as laser
from em2d.em2d cimport EMF

cdef class PlaneWave:
    """PlaneWave( start = 0, fwhm = 0, rise = 0, flat = 0, fall = 0,
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
    # cdef laser.PlaneWave * obj

    def __cinit__( self, *, float start = 0.0, float fwhm = 0.0,
                   float rise = 0.0, float flat = 0.0, float fall = 0.0,
                   float a0 = 0.0, float omega0 = 0.0, float polarization = 0.0,
                   float cos_pol = 0.0, float sin_pol = 0.0 ):

        self.obj = new laser.PlaneWave()

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

    def add( self, EMF emf ):
        """add( emf )

        Adds laser pulse to EMF object

        Parameters
        ----------
        emf : pyEMF
            EM fields object
        """
        # Plane Wave
        self.obj.add( emf.obj[0] )

cdef class Gaussian:
    """Gaussian( start = 0, fwhm = 0, rise = 0, flat = 0, fall = 0,
                    a0 = 0, omega0 = 0, polarization = 0,
                    cos_pol = 0, sin_pol = 0, W0 = 0, focus = 0, axis = 0 )
    
    Class representing a gaussian laser pulse

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
    # cdef laser.Gaussian * obj

    def __cinit__( self, *, float start = 0.0, float fwhm = 0.0,
                   float rise = 0.0, float flat = 0.0, float fall = 0.0,
                   float a0 = 0.0, float omega0 = 0.0, float polarization = 0.0,
                   float cos_pol = 0.0, float sin_pol = 0.0,
                   float W0 = 0, float focus = 0, float axis = 0 ):

        self.obj = new laser.Gaussian()

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

    def add( self, EMF emf ):
        """add( emf )

        Adds laser pulse to EMF object

        Parameters
        ----------
        emf : pyEMF
            EM fields object
        """
        self.obj.add( emf.obj[0] )