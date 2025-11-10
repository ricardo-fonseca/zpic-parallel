#ifndef VEC3_CYLGRID_H_
#define VEC3_CYLGRID_H_

#include "vec_types.h"
#include "bnd.h"
#include "zdf-cpp.h"

#include "cyl3grid.h"
#include "cylmodes.h"

#include <iostream>

/**
 * @brief cyl3<> cylindrical modes tiled grid object
 * 
 * @note will create cyl3grids of std::complex<T> type for modes m > 0
 * 
 * @tparam T    Base class type, must be one of float, double or long-double
 */
template <class T>
class Cyl3CylGrid : public CylGrid< T, cyl3grid<T>, cyl3grid< std::complex<T> > > {

    protected:

    using parent = CylGrid< T, cyl3grid<T>, cyl3grid< std::complex<T> > >;

    public:

    using parent :: grid_0;
    using parent :: grid_m;
    using parent :: nmodes;
    using parent :: set_periodic;
    using parent :: get_periodic;

    using CylGrid< T, cyl3grid<T>, cyl3grid< std::complex<T> > > :: CylGrid;

    /**
     * @brief Save mode m grid values to disk
     * 
     * @param m         Cyl. mode (m >= 0)
     * @param fc        Field component to save
     * @param filename  Output file name (includes path)
     */
    void save( unsigned m, const enum fcomp::cyl fc, std::string filename ) {

        if ( m >= nmodes ) {
            std::cerr << "invalid mode (" << m << ") selected.\n";
            exit(1);
        }

        if ( m == 0 ) {
            grid_0 -> save( fc, filename );
        } else {
            grid_m[m-1] -> save( fc, filename );
        }   
    }

    /**
     * @brief Save mode m grid values to disk with full metadata
     * 
     * @param m             Cyl. mode (m >= 0)
     * @param fc            Field component to save
     * @param metadata      Grid metadata
     * @param iter          Iteration metadata
     * @param path          Ouput path
     */
    void save( unsigned m, const enum fcomp::cyl fc, zdf::grid_info &metadata, zdf::iteration &iter, std::string &path ) {

        if ( m >= nmodes ) {
            std::cerr << "invalid mode (" << m << ") selected.\n";
            exit(1);
        }

        if ( m == 0 ) {
            grid_0 -> save( fc, metadata, iter, path );
        } else {
            grid_m[m-1] -> save( fc, metadata, iter, path );
        }   
    }
};

#endif
