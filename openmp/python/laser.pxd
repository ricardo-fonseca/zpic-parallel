###############################################################################
# Laser pulse classes
#
cdef extern from "../em2d/emf.h":
    cdef cppclass EMF:
        pass

cdef extern from "../em2d/laser.h" namespace "Laser":
    cdef cppclass Pulse:
        float start
        float fwhm
        float rise
        float flat
        float fall
        float a0
        float omega0
        float polarization

        float cos_pol
        float sin_pol

        unsigned int filter

        int validate()
        
        Pulse()
        int add( EMF & )

    cdef cppclass PlaneWave(Pulse):
        PlaneWave()
        int add_plane "add"( EMF & )

    cdef cppclass Gaussian(Pulse):
        float W0
        float focus
        float axis

        Gaussian()
        int add_gaussian "add"( EMF & )