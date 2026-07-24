###############################################################################
# bnd class
#

cdef cppclass __bnd_pair[T]:
    T lower
    T upper


cdef extern from "../../em2d/bnd.h":
    cdef cppclass bnd[T]:
        __bnd_pair[T] x
        __bnd_pair[T] y
        bnd() except +

ctypedef bnd[unsigned int] bnd_uint

