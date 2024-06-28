#ifndef GRID_H_
#define GRID_H_

#include <iostream>

#include "vec_types.h"
#include "bnd.h"
#include "zdf-cpp.h"


/**
 * @brief 
 * 
 */
template <class T>
class grid {
    
    public:

    T * d_buffer;

    /// Number of tiles
    const uint2 ntiles;

    /// Tile grid size
    const uint2 nx;
    
    /// Tile guard cells
    const bnd<unsigned int> gc;

    /// Consider boundaries to be periodic
    int2 periodic;

    /// Global grid size
    const uint2 gnx;
    
    /// Tile grize including guard cells
    const uint2 ext_nx;

    /// Offset in cells between lower tile corner and position (0,0)
    const unsigned int offset;

    /// Tile volume (may be larger than product of cells for alingment)
    const unsigned int tile_vol;

    /// Object name
    std::string name;

    /**
     * @brief Construct a new grid object
     * 
     * @param ntiles    Number of tiles
     * @param nx        Individual tile size
     * @param gc        Number of guard cells
     */
    grid( uint2 const ntiles, uint2 const nx, bnd<unsigned int> const gc):
        d_buffer( nullptr ), 
        ntiles( ntiles ),
        nx( nx ),
        gc(gc),
        periodic( make_int2( 1, 1 ) ),
        gnx( make_uint2( ntiles.x * nx.x, ntiles.y * nx.y )),
        ext_nx( make_uint2( gc.x.lower +  nx.x + gc.x.upper,
                            gc.y.lower +  nx.y + gc.y.upper )),
        offset( gc.y.lower * ext_nx.x + gc.x.lower ),
        tile_vol( roundup4( ext_nx.x * ext_nx.y ) ),
        name( "grid" )
    {
        d_buffer = memory::malloc<T>( buffer_size() );
    };

    /**
     * @brief Construct a new grid object
     * 
     * The number of guard cells is set to 0
     * 
     * @param ntiles    Number of tiles
     * @param nx        Individual tile size
     */
    grid( uint2 const ntiles, uint2 const nx ):
        d_buffer( nullptr ),
        ntiles( ntiles ),
        nx( nx ),
        gc( 0 ),
        periodic( make_int2( 1, 1 ) ),
        gnx( make_uint2( ntiles.x * nx.x, ntiles.y * nx.y )),
        ext_nx( make_uint2( nx.x, nx.y )),
        offset( 0 ),
        tile_vol( roundup4( nx.x * nx.y )),
        name( "grid" )
    {
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
     * @return total size of data buffers
     */
    const std::size_t buffer_size() {
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

        const size_t size = buffer_size( );
        for( size_t i = 0; i < size; i++ ) d_buffer[i] = val;
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

    grid<T>& operator+=(const grid<T>& rhs) {
        for( unsigned i = 0; i < buffer_size(); ++i ) d_buffer[i] += rhs.d_buffer[i];
        return *this;
    }

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     * @return grid&    Reference to local object
     */
    void add( const grid<T> &rhs ) {
        size_t const size = buffer_size( );

        for( size_t i = 0; i < size; i++ ) d_buffer[i] += rhs.d_buffer[i];
    };
    
    /**
     * @brief Gather field into a contiguos grid
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

                        auto const out_idx = giy * gnx.x + gix;

                        out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ];
                    }
                }
            }
        }

        return gnx.x * gnx.y;
    }

#ifdef _OPENMP

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void copy_to_gc_x() {

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;

            auto * __restrict__ local = & d_buffer[ tile_off ];

            {   // Copy from lower neighbour
                int neighbor_tx = tile_idx.x;
                neighbor_tx -= 1;
                if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    auto * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.lower; i++ ) {
                            local[ i + j * ext_nx.x ] = x_lower[ nx.x + i + j * ext_nx.x ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbour
                int neighbor_tx = tile_idx.x;
                neighbor_tx += 1;
                if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                    auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + nx.x + i + j * ext_nx.x ] = x_upper[ gc.x.lower + i + j * ext_nx.x ];
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

            auto * __restrict__ local = d_buffer + tile_off;
            
            {   // Copy from lower neighbour
                int neighbor_ty = tile_idx.y;
                neighbor_ty -= 1;
                if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    for( unsigned j = 0; j < gc.y.lower; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + j * ext_nx.x ] = y_lower[ i + ( nx.y + j ) * ext_nx.x ];
                        }
                    }
                }
            }

            {   // Copy from upper neighbour
                int neighbor_ty = tile_idx.y;
                neighbor_ty += 1;
                if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                    auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    for( unsigned j = 0; j < gc.y.upper; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ] = y_upper[ i + ( gc.y.lower + j ) * ext_nx.x ];
                        }
                    }
                }
            }
        }
    }

#else

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void copy_to_gc_x() {

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];

                {   // Copy from lower neighbour
                    int neighbor_tx = tile_idx.x;
                    neighbor_tx -= 1;
                    if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                    if ( neighbor_tx >= 0 ) {
                        auto * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned j = 0; j < ext_nx.y; j++ ) {
                            for( unsigned i = 0; i < gc.x.lower; i++ ) {
                                local[ i + j * ext_nx.x ] = x_lower[ nx.x + i + j * ext_nx.x ];
                            }
                        }
                    }
                }

                {   // Copy from upper neighbour
                    int neighbor_tx = tile_idx.x;
                    neighbor_tx += 1;
                    if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                    if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                        auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned j = 0; j < ext_nx.y; j++ ) {
                            for( unsigned i = 0; i < gc.x.upper; i++ ) {
                                local[ gc.x.lower + nx.x + i + j * ext_nx.x ] = x_upper[ gc.x.lower + i + j * ext_nx.x ];
                            }
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

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Copy from lower neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty -= 1;
                    if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                    if ( neighbor_ty >= 0 ) {
                        auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned j = 0; j < gc.y.lower; j++ ) {
                            for( unsigned i = 0; i < ext_nx.x; i++ ) {
                                local[ i + j * ext_nx.x ] = y_lower[ i + ( nx.y + j ) * ext_nx.x ];
                            }
                        }
                    }
                }

                {   // Copy from upper neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty += 1;
                    if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                    if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                        auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned j = 0; j < gc.y.upper; j++ ) {
                            for( unsigned i = 0; i < ext_nx.x; i++ ) {
                                local[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ] = y_upper[ i + ( gc.y.lower + j ) * ext_nx.x ];
                            }
                        }
                    }
                }
            }
        }
    }

#endif

    /**
     * @brief Copies edge values to neighboring guard cells
     * 
     */
    void copy_to_gc()  {

        // Copy along x direction
        copy_to_gc_x();

        // Copy along y direction
        copy_to_gc_y();

    };

#ifdef _OPENMP

    /**
     * @brief Adds values from neighboring guard cells to local data
     * 
     */
    void add_from_gc(){

        // Add along x direction

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;

            auto * __restrict__ local = d_buffer + tile_off;
            
            {   // Add from lower neighbour
                int neighbor_tx = tile_idx.x;
                neighbor_tx -= 1;
                if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                if ( neighbor_tx >= 0 ) {
                    T * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.upper; i++ ) {
                            local[ gc.x.lower + i + j * ext_nx.x ] += x_lower[ gc.x.lower + nx.x + i + j * ext_nx.x ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_tx = tile_idx.x;
                neighbor_tx += 1;
                if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                    auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                    for( unsigned j = 0; j < ext_nx.y; j++ ) {
                        for( unsigned i = 0; i < gc.x.lower; i++ ) {
                            local[ nx.x + i + j * ext_nx.x ] += x_upper[ i + j * ext_nx.x ];
                        }
                    }
                }
            }
        }

        // Add along y direction

        // Loop over tiles
        #pragma omp parallel for
        for( unsigned tid = 0; tid < ntiles.y * ntiles.x; tid ++ ) {

            const auto tile_idx = make_int2( tid % ntiles.x, tid / ntiles.x );
            const auto tile_off = tid * tile_vol;

            auto * __restrict__ local = d_buffer + tile_off;
            
            {   // Add from lower neighbour
                int neighbor_ty = tile_idx.y;
                neighbor_ty -= 1;
                if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                if ( neighbor_ty >= 0 ) {
                    auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    for( unsigned j = 0; j < gc.y.upper; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( gc.y.lower + j ) * ext_nx.x ] += y_lower[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ];
                        }
                    }
                }
            }

            {   // Add from upper neighbour
                int neighbor_ty = tile_idx.y;
                neighbor_ty += 1;
                if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                    auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                    for( unsigned j = 0; j < gc.y.lower; j++ ) {
                        for( unsigned i = 0; i < ext_nx.x; i++ ) {
                            local[ i + ( nx.y + j ) * ext_nx.x ] += y_upper[ i + j * ext_nx.x ];
                        }
                    }
                }
            }
        }
    };


#else

    /**
     * @brief Adds values from neighboring guard cells to local data
     * 
     */
    void add_from_gc(){

        // Add along x direction

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Add from lower neighbour
                    int neighbor_tx = tile_idx.x;
                    neighbor_tx -= 1;
                    if ( periodic.x && neighbor_tx < 0 ) neighbor_tx += ntiles.x;

                    if ( neighbor_tx >= 0 ) {
                        T * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned j = 0; j < ext_nx.y; j++ ) {
                            for( unsigned i = 0; i < gc.x.upper; i++ ) {
                                local[ gc.x.lower + i + j * ext_nx.x ] += x_lower[ gc.x.lower + nx.x + i + j * ext_nx.x ];
                            }
                        }
                    }
                }

                {   // Add from upper neighbour
                    int neighbor_tx = tile_idx.x;
                    neighbor_tx += 1;
                    if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) ) neighbor_tx -= ntiles.x;

                    if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                        auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned j = 0; j < ext_nx.y; j++ ) {
                            for( unsigned i = 0; i < gc.x.lower; i++ ) {
                                local[ nx.x + i + j * ext_nx.x ] += x_upper[ i + j * ext_nx.x ];
                            }
                        }
                    }
                }
            }
        }

        // Add along y direction

        // Loop over tiles
        for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
            for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                const auto tile_idx = make_uint2( tx, ty );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Add from lower neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty -= 1;
                    if ( periodic.y && neighbor_ty < 0 ) neighbor_ty += ntiles.y;

                    if ( neighbor_ty >= 0 ) {
                        auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned j = 0; j < gc.y.upper; j++ ) {
                            for( unsigned i = 0; i < ext_nx.x; i++ ) {
                                local[ i + ( gc.y.lower + j ) * ext_nx.x ] += y_lower[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ];
                            }
                        }
                    }
                }

                {   // Add from upper neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty += 1;
                    if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                    if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                        auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned j = 0; j < gc.y.lower; j++ ) {
                            for( unsigned i = 0; i < ext_nx.x; i++ ) {
                                local[ i + ( nx.y + j ) * ext_nx.x ] += y_upper[ i + j * ext_nx.x ];
                            }
                        }
                    }
                }
            }
        }

    };

#endif

    /**
     * @brief Left shifts data for a specified amount
     * 
     * This operation is only allowed if the number of upper x guard cells
     * is greater or equal to the requested shift
     * 
     * @param shift Number of cells to shift
     */
    void x_shift_left( unsigned int const shift ) {

        /*
         * Warning: This version cannot be made parallel inside tiles 
         */
        if ( gc.x.upper >= shift ) {

            const int ystride = ext_nx.x;

            // Loop over tiles
            for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol ;
                    
                    auto * __restrict__ buffer = d_buffer + tile_off;
                    
                    for( unsigned iy = 0; iy < ext_nx.y; iy++ ) {
                        for( unsigned ix = 0; ix < ext_nx.x - shift; ix++ ) {
                            buffer[ ix + iy * ystride ] = buffer[ ix + shift + iy * ystride ]; 
                        }
                        for( unsigned ix = ext_nx.x - shift; ix < ext_nx.x; ix++ ) {
                            buffer[ ix + iy * ystride ] = T{0};
                        }
                    }
                }
            }

            // Copy x guard cells
            copy_to_gc_x();

        } else {
            std::cerr << "x_shift_left(), shift value too large, must be <= gc.x.upper\n";
            exit(1);
        }
    }

    /**
     * @brief Left shifts data for a specified amount
     * 
     * This operation is only allowed if the number of upper x guard cells
     * is greater or equal to the requested shift
     * 
     * @param shift Number of cells to shift
     */
    void x_shift_left_mk2( unsigned int const shift ) {

        if ( gc.x.upper >= shift ) {

            const int ystride = ext_nx.x;

            // Loop over tiles
            for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {
                    
                    // On a GPU this would be on shared memory
                    T local[tile_vol];

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol ;
                    
                    auto * __restrict__ buffer = d_buffer + tile_off;
                    
                    for( int j = 0; j < ext_nx.y; j++ ) {
                        for( int i = 0; i < ext_nx.x - shift; i++ ) {
                            local[ i + j * ystride ] = buffer[ i + shift + j * ystride ]; 
                        }
                        for( int i = ext_nx.x - shift; i < ext_nx.x; i++ ) {
                            local[ i + j * ystride ] = 0;
                        }
                    }

                    // sync ...
                    for( int i = 0; i < tile_vol; i++ ) buffer[i] = local[i];

                }
            }

            // Copy x guard cells
            copy_to_gc_x();

        } else {
            std::cerr << "x_shift_left(), shift value too large, must be <= gc.x.upper\n";
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

            // On a GPU these would be on block shared memory
            T A[ tile_vol ];
            T B[ tile_vol ];

            // Loop over tiles
            for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol ;

                    auto * __restrict__ buffer = d_buffer + tile_off;

                    // Copy data from tile buffer
                    for( unsigned i = 0; i < tile_vol; i++ ) A[i] = buffer[i];

                    // Apply kernel locally
                    for( unsigned iy = 0; iy < ext_nx.y; iy++ ) {
                        for( unsigned ix = gc.x.lower; ix < nx.x + gc.x.lower; ix ++) {
                            B [ iy * ystride + ix ] = A[ iy * ystride + (ix-1) ] * a +
                                                      A[ iy * ystride +  ix    ] * b +
                                                      A[ iy * ystride + (ix+1) ] * c;
                        }
                    }

                    // Copy data back to tile buffer
                    for( unsigned i = 0; i < tile_vol; i++ ) buffer[i] = B[i];
                }
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

            // On a GPU these would be on block shared memory
            T A[ tile_vol ];
            T B[ tile_vol ];

            // Loop over tiles
            for( unsigned ty = 0; ty < ntiles.y; ty ++ ) {
                for( unsigned tx = 0; tx < ntiles.x; tx ++ ) {

                    const auto tile_idx = make_uint2( tx, ty );
                    const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                    const auto tile_off = tid * tile_vol;

                    auto * __restrict__ buffer = d_buffer + tile_off;

                    // Copy data from tile buffer
                    for( unsigned i = 0; i < tile_vol; i++ ) A[i] = buffer[i];

                    // Apply kernel locally
                    for( unsigned iy = 0; iy < ext_nx.y; iy++ ) {
                        for( unsigned ix = gc.x.lower; ix < nx.x + gc.x.lower; ix ++) {
                            B [ iy * ystride + ix ] = A[ (iy-1) * ystride + ix ] * a +
                                                      A[    iy  * ystride + ix ] * b +
                                                      A[ (iy+1) * ystride + ix ] * c;
                        }
                    }

                    // Copy data back to tile buffer
                    for( unsigned i = 0; i < tile_vol; i++ ) buffer[i] = B[i];
                }
            }

            // Update guard cells
            copy_to_gc_y();

        } else {
            std::cerr << "kernel3_y() requires at least 1 guard cell at both the lower and upper y boundaries.\n";
            exit(1);
        }

    }

    /**
     * @brief Save field values to disk
     * 
     * The field type <T> must be supported by ZDF file format
     * 
     */
    void save( zdf::grid_info &info, zdf::iteration &iter, std::string path ) {

        // Fill in grid dimensions
        info.ndims = 2;
        info.count[0] = ntiles.x * nx.x;
        info.count[1] = ntiles.y * nx.y;

        // Allocate buffer on host to gather data
        T * h_data = memory::malloc<T>( info.count[0] * info.count[1] );

        gather( h_data );
        zdf::save_grid( h_data, info, iter, path );

        memory::free( h_data );
    };

};

#endif