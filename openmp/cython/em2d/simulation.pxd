###############################################################################
# Simulation
#

from libcpp.string cimport string
from libcpp.vector cimport vector

from em2d.vec_types cimport *

from em2d.current cimport Current as current_t
from em2d.emf     cimport EMF     as emf_t
from em2d.species cimport Species as species_t

cdef extern from "../../em2d/simulation.h":
    cdef cppclass Simulation:
        uint2 ntiles
        uint2 nx
        float2 box
        double dt

        emf_t emf
        current_t current
        vector[ species_t ] species

        Simulation( uint2 ntiles, uint2 nx, float2 box, double dt )
        void set_moving_window() 
        void add_species( species_t & sp )

        species_t * get_species( string name )

        void advance()
        void advance_mov_window()

        unsigned int get_iter()
        double get_t()

        void energy_info()
