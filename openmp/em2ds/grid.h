#ifndef GRID_H_
#define GRID_H_

#include "vec_types.h"
#include "bnd.h"
#include "zdf-cpp.h"

#include <iostream>

/**
 * @brief 
 * 
 */
template <class T>
class grid {
    protected:

    private:

    /**
     * @brief Validate grid / parallel parameters. The execution will stop if
     * errors are found.
     * 
     */
    void validate_parameters() {
        // Grid parameters
        if ( ntiles.x == 0 || ntiles.y == 0 ) {
            std::cerr << "Invalid number of tiles " << ntiles << '\n';
            std::exit(1);
        }

        if ( nx.x == 0 || nx.y == 0 ) {
            std::cerr << "Invalid tiles size" << nx << '\n';
            std::exit(1);
        }
    }

    public:

    /// @brief Data buffer
    T * d_buffer;

    /// @brief Local number of tiles
    const uint2 ntiles;

    /// @brief Tile grid size
    const uint2 nx;

    /// @brief Local grid size
    const uint2 dims;

    /// @brief Tile guard cells
    const bnd<unsigned int> gc;
    
    /// @brief Tile grize including guard cells
    const uint2 ext_nx;

    /// @brief Local offset in cells between lower tile corner and position (0,0)
    const unsigned int offset;

    /// @brief Tile volume (may be larger than product of cells for alignment)
    std::size_t tile_vol;

    /// @brief Object name
    std::string name;

    /// @brief Consider grid boundaries periodic
    int2 periodic;


    /**
     * @brief Construct a new grid object
     * 
     * @param ntiles            Number of tiles
     * @param nx                Individual tile size
     * @param gc                Number of guard cells
     */
    grid( uint2 const ntiles, uint2 const nx, bnd<unsigned int> const gc = 0 ):
        d_buffer( nullptr ), 
        ntiles( ntiles ),
        nx( nx ),
        dims( ntiles * nx ),
        gc(gc),
        ext_nx( make_uint2( gc.x.lower + nx.x + gc.x.upper,
                            gc.y.lower + nx.y + gc.y.upper )),
        offset( gc.y.lower * ext_nx.x + gc.x.lower ),
        tile_vol( roundup4( ext_nx.x * ext_nx.y ) ),
        name( "grid" ),
        periodic( make_int2( 1, 1 ) )
    {
        // Validate parameters
        validate_parameters();

        // Allocate main data buffer
        d_buffer = memory::malloc<T>( buffer_size() );
    };

    /**
     * @brief grid destructor
     * 
     */
    ~grid(){
        memory::free( d_buffer );
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const grid<T>& obj) {
        os << obj.name << "{";
        os << "(" << obj.ntiles.x << " x " << obj.ntiles.y << " tiles)";
        os << ", (" << obj.nx.x << " x " << obj.nx.y << " points/tile)";
        os << "}";
        return os;
    }


    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() {
        return (static_cast <std::size_t> (tile_vol)) * ( ntiles.x * ntiles.y ) ;
    };

    /**
     * @brief zero device data on a grid grid
     * 
     * @return int       Returns 0 on success, -1 on error
     */
    int zero() {
        memory::zero( d_buffer, buffer_size() );
        return 0;
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val ){
        #pragma omp parallel for
        for( size_t i = 0; i < buffer_size( ); i++ ) d_buffer[i] = val;
    };

    /**
     * @brief Scalar assigment
     * 
     * @param val       Value
     * @return T        Returns the same value
     */
    T operator=( T const val ) {
        set( val );
        return val;
    }

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const grid<T> &rhs ) {
        size_t const size = buffer_size( );

        #pragma omp parallel for
        for( size_t i = 0; i < size; i++ ) d_buffer[i] += rhs.d_buffer[i];
    };

    /**
     * @brief Operator +=
     * 
     * @param rhs           Other grid to add
     * @return grid<T>& 
     */
    grid<T>& operator+=(const grid<T>& rhs) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Gather local grid into a contiguous grid
     * 
     * Used mostly for diagnostic output
     * 
     * @param out               Output buffer
     * @return unsigned int     Total number of cells
     */
    unsigned int gather( T * const __restrict__ out ) {

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol + offset;

                auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

                // Loop inside tile
                for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                    for( unsigned ix = 0; ix < nx.x; ix ++ ) {
                        auto const gix = tile_idx.x * nx.x + ix;
                        auto const giy = tile_idx.y * nx.y + iy;

                        auto const out_idx = giy * dims.x + gix;

                        out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ];
                    }
                }
            }
        }

        return dims.x * dims.y;
    }

    /**
     * @brief Scatter data from a contiguous grid into tiles
     * 
     * @param in                Intput buffer
     * @return unsigned int     Total number of cells
     */
    unsigned int scatter( T const * const __restrict__ d_in ) {

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol + offset;

                auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

                // Loop inside tile
                for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                    for( unsigned ix = 0; ix < nx.x; ix ++ ) {
                        auto const gix = tile_idx.x * nx.x + ix;
                        auto const giy = tile_idx.y * nx.y + iy;

                        auto const in_idx = giy * dims.x + gix;

                        tile_data[ iy * ext_nx.x + ix ] = d_in[ in_idx ];
                    }
                }
            }
        }

        // Update guard cell values
        copy_to_gc();

        return dims.x * dims.y;
    }

    /**
     * @brief Scatter data from a contiguous grid into tiles and scale
     * 
     * @param d_in              Intput buffer
     * @param scale             Scale value
     * @return unsigned int     Grid volume
     */
    unsigned int scatter( T const * const __restrict__ d_in, T const scale ) {

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol + offset;

                auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

                // Loop inside tile
                for( unsigned iy = 0; iy < nx.y; iy ++ ) {
                    for( unsigned ix = 0; ix < nx.x; ix ++ ) {
                        auto const gix = tile_idx.x * nx.x + ix;
                        auto const giy = tile_idx.y * nx.y + iy;

                        auto const in_idx = giy * dims.x + gix;

                        tile_data[ iy * ext_nx.x + ix ] = d_in[ in_idx ] * scale;
                    }
                }
            }
        }

        // Update guard cell values
        copy_to_gc();

        return dims.x * dims.y;
    }

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void copy_to_gc_x() {

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const int2 tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const int tile_off = tid * tile_vol;
            const int ystride  = ext_nx.x;

            auto * __restrict__ local = & d_buffer[ tile_off ];

            {   // Copy from lower neighbour
                int neighbor_tx = tile_idx.x - 1;
                if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    const auto neighbor_off = (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    auto * __restrict__ x_lower = & d_buffer[ neighbor_off ] ;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.lower; i++ ) {
                            local[ i + j * ystride ] = x_lower[ nx.x + i + j * ystride ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbour
                int neighbor_tx = tile_idx.x + 1;
                if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                    const auto neighbor_off = (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    auto * __restrict__ x_upper =  & d_buffer[ neighbor_off ] ;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + nx.x + i + j * ystride ] = x_upper[ gc.x.lower + i + j * ystride ];
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Copies edge values to Y neighboring guard cells
     * 
     */
    void copy_to_gc_y() {

        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;
            const auto ystride  = ext_nx.x;

            auto * __restrict__ local = & d_buffer[ tile_off ];
            
            {   // Copy from lower neighbour
                int neighbor_ty = tile_idx.y - 1;
                if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    const auto neighbor_off = (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    auto * __restrict__ y_lower = & d_buffer [ neighbor_off ] ;
                    for( unsigned j = 0; j < gc.y.lower; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + j * ystride ] = y_lower[ i + ( nx.y + j ) * ystride ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbour
                int neighbor_ty = tile_idx.y + 1;
                if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                    const auto neighbor_off = (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    auto * __restrict__ y_upper = & d_buffer [ neighbor_off ];
                    for( unsigned j = 0; j < gc.y.upper; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( gc.y.lower + nx.y + j ) * ystride ] = y_upper[ i + ( gc.y.lower + j ) * ystride ];
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Copies edge values to neighboring guard cells, including other
     *        parallel nodes
     * 
     */
    void copy_to_gc()  {

        // Copy along x direction
        copy_to_gc_x();

        // Copy along y direction
        copy_to_gc_y();

    };

    /**
     * @brief Adds values from neighboring x guard cells to local data
     * 
     */
    void add_from_gc_x() {
        // Add along x direction

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;
            const auto ystride  = ext_nx.x;

            auto * __restrict__ local = & d_buffer[ tile_off ];
            
            {   // Add from lower neighbour
                int neighbor_tx = tile_idx.x - 1;
                if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    const auto neighbor_off = (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    T * __restrict__ x_lower = & d_buffer[neighbor_off]; 
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + i + j * ystride ] += x_lower[ gc.x.lower + nx.x + i + j * ystride ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_tx = tile_idx.x + 1;
                if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                    const auto neighbor_off = (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    auto * __restrict__ x_upper = & d_buffer[neighbor_off]; 
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.lower; i++ ) {
                            local[ nx.x + i + j * ystride ] += x_upper[ i + j * ystride ];
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Adds values from neighboring y guard cells to local data
     * 
     */
    void add_from_gc_y(){

        // Add along y direction

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;
            const auto ystride  = ext_nx.x;

            auto * __restrict__ local = & d_buffer[ tile_off ];
            
            {   // Add from lower neighbour
                int neighbor_ty = tile_idx.y - 1;
                if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    const auto neighbor_off = (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    auto * __restrict__ y_lower = & d_buffer [ neighbor_off ]; 
                    for( unsigned j = 0; j < gc.y.upper; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( gc.y.lower + j ) * ystride ] += y_lower[ i + ( gc.y.lower + nx.y + j ) * ystride ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_ty = tile_idx.y + 1;
                if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                    const auto neighbor_off = (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    auto * __restrict__ y_upper = & d_buffer [ neighbor_off ]; 
                    for( unsigned j = 0; j < gc.y.lower; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( nx.y + j ) * ystride ] += y_upper[ i + j * ystride ];
                        }
                    }
                }
            }
        }
    };

    /**
     * @brief Adds values from neighboring guard cells to local data, including
     *        values from other parallel nodes
     * 
     */
    void add_from_gc() {
        // Add along x direction
        add_from_gc_x();

        // Add along y direction
        add_from_gc_y();
    }

    /**
     * @brief Left shifts data for a specified amount
     * 
     * @warning This operation is only allowed if the number of upper x guard cells
     * is greater or equal to the requested shift
     * 
     * @param shift Number of cells to shift
     */
    void x_shift_left( unsigned int const shift ) {

        if ( shift > 0 && shift < gc.x.upper ) {

            const int ystride = ext_nx.x;

            // Loop over tiles
            #pragma omp parallel for
            for( int tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {
                const auto tile_off = tid * tile_vol ;
                
                auto * __restrict__ buffer = & d_buffer[ tile_off ];
                
                for( int iy = 0; iy < ext_nx.y; iy++ ) {
                    for( int ix = 0; ix < ext_nx.x - shift; ix++ ) {
                        buffer[ ix + iy * ystride ] = buffer[ (ix + shift) + iy * ystride ]; 
                    }
                    for( int ix = ext_nx.x - shift; ix < ext_nx.x; ix++ ) {
                        buffer[ ix + iy * ystride ] = T{0};
                    }
                }
            }

            // Copy x guard cells
            copy_to_gc_x();

        } else {
            std::cerr << "x_shift_left(), invalid shift value, must be 0 < shift <= gc.x.upper\n";
            exit(1);
        }
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

        if (( gc.x.lower > 0) && (gc.x.upper > 0)) {

            const int ystride = ext_nx.x;

            // Loop over tiles
            #pragma omp parallel for
            for( int tid = 0; tid < ntiles.y * ntiles.x; tid++ ) {

                // On a GPU these would be on block shared memory
                T A[ tile_vol ];
                T B[ tile_vol ];

                const auto tile_off = tid * tile_vol ;

                auto * __restrict__ buffer = & d_buffer[ tile_off ];

                // Copy data from tile buffer
                for( int i = 0; i < tile_vol; i++ ) {
                    A[i] = B[i] = buffer[i];
                }

                // Apply kernel locally
                for( int iy = 0; iy < ext_nx.y; iy++ ) {
                    for( int ix = gc.x.lower; ix < gc.x.lower + nx.x; ix ++) {
                        B[ iy * ystride + ix ] = A[ iy * ystride + (ix-1) ] * a +
                                                 A[ iy * ystride +  ix    ] * b +
                                                 A[ iy * ystride + (ix+1) ] * c;
                    }
                }

                // Copy data back to tile buffer
                for( int i = 0; i < tile_vol; i++ ) buffer[i] = B[i];
            }

            // Update guard cells
            copy_to_gc_x();

        } else {
            std::cerr << "kernel_x3() requires at least 1 guard cell at both the lower and upper x boundaries.\n";
            exit(1);
        }

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

        if (( gc.y.lower > 0) && (gc.y.upper > 0)) {

            const int ystride = ext_nx.x;

            // Loop over tiles
            #pragma omp for
            for( unsigned int idx = 0; idx < ntiles.y * ntiles.x; idx++ ) {
                auto tx = idx % ntiles.x;
                auto ty = idx / ntiles.x;

                // On a GPU these would be on block shared memory
                T A[ tile_vol ];
                T B[ tile_vol ];

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ buffer = & d_buffer[ tile_off ];

                // Copy data from tile buffer
                for( unsigned i = 0; i < tile_vol; i++ ) {
                    A[i] = B[i] = buffer[i];
                }

                // Apply kernel locally
                for( unsigned iy = gc.y.lower; iy < nx.y + gc.y.lower; iy++ ) {
                    for( unsigned ix = 0; ix < ext_nx.x; ix ++) {
                        B [ iy * ystride + ix ] = A[ (iy-1) * ystride + ix ] * a +
                                                  A[    iy  * ystride + ix ] * b +
                                                  A[ (iy+1) * ystride + ix ] * c;
                    }
                }

                // Copy data back to tile buffer
                for( unsigned i = 0; i < tile_vol; i++ ) buffer[i] = B[i];
            }

            // Update guard cells
            copy_to_gc_y();

        } else {
            std::cerr << "kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries.\n";
            exit(1);
        }

    }

    /**
     * @brief Save grid values to disk with full metadata
     * 
     * The field type <T> must be supported by ZDF file format
     * 
     */
    void save( zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in global grid dimensions
        info.ndims = 2;
        info.count[0] = dims.x;
        info.count[1] = dims.y;

        // Allocate buffer on host to gather data
        T * h_data = memory::malloc<T>( dims.x * dims.y );

        // Gather data on contiguous grid
        gather( h_data );

        // Save data
        zdf::save_grid( h_data, info, iter, path );

        // Free temporary buffer
        memory::free( h_data );
    };

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name
     */
    void save( std::string filename ) {
        // Allocate buffer on host to gather data
        T * h_data = memory::malloc<T>( dims.x * dims.y );

        // Gather data on contiguous grid
        gather( h_data );

        uint64_t gdims[] = { dims.x, dims.y };

        // Save data
        zdf::save_grid( h_data, 2, gdims, name, filename );

        // Free temporary buffer
        memory::free( h_data );
    }

};

#endif