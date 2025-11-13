#ifndef CYLMODES_H_
#define CYLMODES_H_

#include "vec_types.h"
#include "bnd.h"
#include "zdf-cpp.h"
#include "grid.h"

#include <iostream>
#include <vector>

/**
 * @brief Scalar cylindrical modes tiled grid object
 * 
 * @tparam T        Base type
 * @tparam G        Real grid type, defaults to grid<T>
 * @tparam cG       Complex grid type, defaults to grid< std::complex<T> > 
 */
template <class T, class G = grid<T>, class cG = grid< std::complex<T> > >
class CylGrid {
    protected:

    /// @brief Number of cylindrical modes (including fundamental mode)
    const int nmodes;

    /// @brief Local number of tiles
    uint2 ntiles;

    /// @brief Consider z boundaries periodic
    int periodic;

    /// Global grid size
    const uint2 dims;

    /// @brief Tile grid size
    const uint2 nx;
    
    /// @brief Tile guard cells
    const bnd<unsigned int> gc;

    /// @brief Object name
    std::string name;

    /// @brief Mode zero (real)
    G * grid_0;

    /// @brief High order modes (complex)
    std::vector< cG * > grid_m;

    private:


    public:

    /**
     * @brief Construct a new Cyl Grid object
     * 
     * @param nmodes    Number of modes (including fundamental mode), must be >= 1
     * @param ntiles    Number of tiles (x,y)
     * @param nx        Individual tile size (x,y)
     * @param gc        Number of guard cells
     */
    CylGrid( int nmodes, uint2 const ntiles, uint2 const nx, bnd<unsigned int> const gc):
        nmodes( nmodes ),
        ntiles( ntiles ),
        periodic( 1 ),
        dims( make_uint2( ntiles.x * nx.x, ntiles.y * nx.y )),
        nx( nx ),
        gc(gc),
        name( "grid" )
    {
        if ( nmodes < 1 ) {
            std::cerr << "Invalid number of cylindrical modes, must be >= 1\n";
            exit(1);
        }

        grid_0 = new G( ntiles, nx, gc );
        grid_0 -> name = name + "-m0";

        if ( nmodes > 1 ) {
            grid_m.reserve( nmodes - 1 );
            for ( auto i = 0; i < nmodes - 1; i++ ) {
                auto * m = new cG( ntiles, nx, gc );
                m -> name = name + "-m" + std::to_string(i+1);
                grid_m.push_back( m );
            }
        }

        set_periodic( periodic );
    };

    /**
     * @brief Destroy CylGrid object
     * 
     */
    ~ CylGrid() {
        for ( auto & m : grid_m ) {
            delete m;
        }
        delete grid_0;
    }

    /**
     * @brief Set the name of a CylGrid object
     * 
     * @param new_name 
     */
    void set_name( std::string new_name ) {
        name = new_name;
        grid_0 -> name = name + "-m0";
        for ( int i = 0; i < nmodes - 1; i++ ) {
            grid_m[i] -> name = name + "-m" + std::to_string(i+1);
        }
    }

    /**
     * @brief Set periodic boundary values for z direction
     * 
     * @param new_periodic      Use periodic boundaries (z)
     */
    void set_periodic( int new_periodic ) {
        periodic = new_periodic;

        // The grids are never periodic along r
        auto p = make_int2( periodic, 0 );

        grid_0 -> set_periodic(p);
        for ( auto & m : grid_m )
            m -> set_periodic(p);
    }

    auto get_nmodes() { return nmodes; }

    /**
     * @brief Get periodic boundary values
     * 
     * @return auto 
     */
    auto get_periodic() { return periodic; }

    /**
     * @brief Get the local number of tiles
     * 
     * @return uint2 
     */
    auto get_ntiles() { return ntiles; };

    /**
     * @brief Get the grid size
     * 
     * @return uint2 
     */
    auto get_dims() { return dims; };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const CylGrid<T,G,cG>& obj) {
        os << obj.name << '{'
           << ' ' << obj.nmodes << " modes"
           << ", " << obj.ntiles.x << "×" << obj.ntiles.y << " tiles"
           << ", " << obj.nx.x << "×" << obj.nx.y << " points/tile "
           << '}';
        return os;
    }

    /**
     * @brief Access mode 0
     * 
     * @return grid<T>&     Mode 0 grid
     */
    auto & mode0() {
        return * grid_0;
    }

    /**
     * @brief Access mode m
     * 
     * @param m     Mode index, must be >= 1
     * @return grid< std::complex(T) >& Mode m grid
     */
    auto & mode( int m ) {
        return * grid_m[m-1];
    }

    /**
     * @brief Sets all values to 0
     * 
     */
    void zero() {
        grid_0 -> zero();
        for ( auto & m : grid_m )
            m -> zero();
    }

    /**
     * @brief Add another CylGrid object to this object
     * 
     * @param rhs 
     */
    void add( const CylGrid<T,G,cG> &rhs ) {
        grid_0 -> add( rhs.grid_0 );
        for( int i = 0; i < nmodes-1; i++ )
            grid_m[i] -> add( rhs.grid_m[i] );
    };

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void copy_to_gc_x() {
        grid_0 -> copy_to_gc_x();
        for ( auto & m : grid_m )
            m -> copy_to_gc_x();
    }

    /**
     * @brief Copies edge values to Y neighboring guard cells
     * 
     */
    void copy_to_gc_y() {
        grid_0 -> copy_to_gc_x();
        for ( auto & m : grid_m )
            m -> copy_to_gc_x();
    }

    /**
     * @brief Copies edge values to neighboring guard cells
     * 
     */
    void copy_to_gc() {
        grid_0 -> copy_to_gc();
        for ( auto & m : grid_m )
            m -> copy_to_gc();
    }

    /**
     * @brief Adds values from neighboring guard cells to local data
     * 
     */
    void add_from_gc() {
        grid_0 -> add_from_gc();
        for ( auto & m : grid_m )
            m -> add_from_gc();
    }

    /**
     * @brief Left shifts data for a specified amount
     * 
     * @note This operation is only allowed if the number of upper x guard cells
     *       is greater or equal to the requested shift
     * 
     * @param shift Number of cells to shift
     */
    void x_shift_left( unsigned int const shift ) {
        grid_0 -> x_shift_left( shift );
        for ( auto & m : grid_m )
            m -> x_shift_left( shift );
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along x
     * 
     * @param a     Kernel value a
     * @param b     Kernel value b
     * @param c     Kernel value c
     */
    template < typename S >
    void kernel3_x( S const a, S const b, S const c ) {
        grid_0 -> kernel3_x( a, b, c );
        for ( auto & m : grid_m )
            m -> kernel3_x( a, b, c );
    }

    /**
     * @brief Perform a convolution with a 3 point kernel [a,b,c] along y
     * 
     * @param a     Kernel value a
     * @param b     Kernel value b
     * @param c     Kernel value c
     */
    template < typename S >
    void kernel3_y( S const a, S const b, S const c ) {
        grid_0 -> kernel3_y( a, b, c );
        for ( auto & m : grid_m )
            m -> kernel3_y( a, b, c );
    }


    /**
     * @brief Save mode m grid values to disk
     * 
     * @param m             Cyl. mode (m >= 0)
     * @param filename      Output file name (includes path)
     */
    void save( int m, std::string filename ) {

        if ( m < 0 || m >= nmodes ) {
            std::cerr << "invalid mode (" << m << ") selected.\n";
            exit(1);
        }

        if ( m == 0 ) {
            grid_0 -> save( filename );
        } else {
            grid_m[m-1] -> save( filename );
        }   
    }

};

#endif
