#ifndef GRID_H_
#define GRID_H_

#include <sycl/sycl.hpp>
#include <iostream>

#include "bnd.h"
#include "zdf-cpp.h"

/**
 * @brief 
 * 
 */
template <class T>
class grid {

    protected:

    /// @brief Associated sycl queue
    sycl::queue & q;

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
    grid( uint2 const ntiles, uint2 const nx, bnd<unsigned int> const gc, sycl::queue & q ):
        q(q),
        d_buffer( nullptr ), 
        ntiles( ntiles ),
        nx( nx ),
        gc(gc),
        periodic( make_int2(1, 1) ),
        gnx( make_uint2( ntiles.x * nx.x, ntiles.y * nx.y ) ),
        ext_nx( make_uint2( gc.x.lower +  nx.x + gc.x.upper,
                gc.y.lower +  nx.y + gc.y.upper ) ),
        offset( gc.y.lower * ext_nx.x + gc.x.lower ),
        tile_vol( roundup4( ext_nx.x * ext_nx.y ) ),
        name( "grid" )
    {
        d_buffer = device::malloc<T>( buffer_size(), q );
    };

    /**
     * @brief Construct a new grid object
     * 
     * The number of guard cells is set to 0
     * 
     * @param ntiles    Number of tiles
     * @param nx        Individual tile size
     */
    grid( uint2 const ntiles, uint2 const nx, sycl::queue & q ):
        q(q),
        d_buffer( nullptr ),
        ntiles( ntiles ),
        nx( nx ),
        gc( 0 ),
        periodic( make_int2( 1, 1 ) ),
        gnx( make_uint2( ntiles.x * nx.x, ntiles.y * nx.y ) ),
        ext_nx( make_uint2( nx.x, nx.y ) ),
        offset( 0 ),
        tile_vol( roundup4( nx.x * nx.y )),
        name( "grid" )
    {
        d_buffer = device::malloc<T>( buffer_size(), q );
    };

    /**
     * @brief grid destructor
     * 
     */
    ~grid(){
        device::free( d_buffer, q );
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
    int zero( ) {
        device::zero( d_buffer, buffer_size(), q );
        return 0;
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val       Value
     */
    void set( T const & val ){
        device::setval( d_buffer, buffer_size(), val, q );
        q.wait();
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
        std::size_t const size = buffer_size();
        auto * __restrict__ A = d_buffer;
        auto * __restrict__ B = rhs.d_buffer;
        
        q.submit([&](sycl::handler &h) {
            h.parallel_for( size, [=](sycl::id<1> i) { 
                A[i] += B[i];
            });
        });
        q.wait();
    };

#if 0
    void add( const grid<T> &rhs ) {

        q.submit([&](sycl::handler &h) {

            auto * __restrict__ A = d_buffer;
            auto * __restrict__ B = rhs.d_buffer;

            const auto tile_vol = this->tile_vol;

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };

            h.parallel_for( 
                sycl::nd_range{global * local , local},
                [=](sycl::nd_item<2> it) { 

                const auto tid        = it.get_group_linear_id();
                const auto tile_off   = tid * tile_vol ;

                auto * __restrict__ buffer_A = A + tile_off;
                auto * __restrict__ buffer_B = B + tile_off;

                for( unsigned i = it.get_local_id(0); i < tile_vol; i+=it.get_local_range(0) )
                    buffer_A[i] += buffer_B[i];
            });
        });
    };
#endif

    grid<T>& operator+=(const grid<T>& rhs) {
        add( rhs );
        return *this;
    }

    /**
     * @brief Gather field into a contiguos grid
     * 
     * Used mostly for diagnostic output
     * 
     * @param out               Output buffer
     * @return unsigned int     Total number of cells
     */
    unsigned int gather( T * const __restrict__ d_out ) {

        q.submit([&](sycl::handler &h) {

            const auto ntiles   = this->ntiles;
            const auto tile_vol = this->tile_vol;
            const auto nx       = this->nx;
            const auto gnx      = this->gnx;
            const auto ext_nx   = this->ext_nx;
            const auto offset   = this->offset;
            auto * __restrict__ d_buffer = this->d_buffer;

            // 8×1 work items per group
            sycl::range<2> local{ 8, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };

            h.parallel_for( 
                sycl::nd_range{global * local , local},
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol + offset;

                auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

                // Loop inside tile
                for( unsigned idx = it.get_local_id(0); idx < nx.y * nx.x; idx += it.get_local_range(0) ) {
                    const auto iy = idx / nx.x;
                    const auto ix = idx % nx.x;
 
                    auto const gix = tile_idx.x * nx.x + ix;
                    auto const giy = tile_idx.y * nx.y + iy;

                    auto const out_idx = giy * gnx.x + gix;

                    d_out[ out_idx ] = tile_data[ iy * ext_nx.x + ix ];
                }

            });
        });

        q.wait();

        return gnx.x * gnx.y;
    }

    /**
     * @brief Copies edge values to X neighboring guard cells
     * 
     */
    void copy_to_gc_x( ) {

        q.submit([&](sycl::handler &h) {

            const auto ntiles   = this->ntiles;
            const auto tile_vol = this->tile_vol;
            const auto nx       = this->nx;
            const auto ext_nx   = this->ext_nx;
            const auto periodic = this->periodic;
            const auto gc       = this->gc;
            auto * __restrict__ d_buffer = this->d_buffer;

            // 256×1 work items per group
            sycl::range<2> local{ 256, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };
            
            h.parallel_for( 
                sycl::nd_range{global * local , local},
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = & d_buffer[ tile_off ];

                {   // Copy from lower neighbour
                    int neighbor_tx = tile_idx.x - 1;
                    if ( periodic.x && neighbor_tx < 0 )
                        neighbor_tx += ntiles.x;

                    if ( neighbor_tx >= 0 ) {
                        auto * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned idx = it.get_local_id(0); idx < ext_nx.y * gc.x.lower; idx += it.get_local_range(0) ) {
                            const auto i = idx % gc.x.lower;
                            const auto j = idx / gc.x.lower; 
                            local[ i + j * ext_nx.x ] = x_lower[ nx.x + i + j * ext_nx.x ];
                        }
                    }
                }

                {   // Copy from upper neighbour
                    int neighbor_tx = tile_idx.x + 1;
                    if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) )
                        neighbor_tx -= ntiles.x;

                    if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                        auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned idx = it.get_local_id(0); idx < ext_nx.y * gc.x.upper; idx += it.get_local_range(0) ) {
                            const auto i = idx % gc.x.upper;
                            const auto j = idx / gc.x.upper; 
                            local[ gc.x.lower + nx.x + i + j * ext_nx.x ] = x_upper[ gc.x.lower + i + j * ext_nx.x ];
                        }
                    }
                }
            });
        });
        
        q.wait();
    }

    /**
     * @brief Copies edge values to Y neighboring guard cells
     * 
     */
    void copy_to_gc_y( ) {

        q.submit([&](sycl::handler &h) {

            const auto ntiles   = this->ntiles;
            const auto tile_vol = this->tile_vol;
            const auto nx       = this->nx;
            const auto ext_nx   = this->ext_nx;
            const auto periodic = this->periodic;
            const auto gc       = this->gc;
            auto * __restrict__ d_buffer = this->d_buffer;

            // 256×1 work items per group
            sycl::range<2> local{ 256, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };

            h.parallel_for(
                sycl::nd_range{global * local , local},
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Copy from lower neighbour
                    int neighbor_ty = tile_idx.y - 1;
                    if ( periodic.y && neighbor_ty < 0 )
                        neighbor_ty += ntiles.y;

                    if ( neighbor_ty >= 0 ) {
                        auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned idx = it.get_local_id(0); idx < gc.y.lower * ext_nx.x; idx += it.get_local_range(0) ) {
                            const auto i = idx % ext_nx.x;
                            const auto j = idx / ext_nx.x; 
                            local[ i + j * ext_nx.x ] = y_lower[ i + ( nx.y + j ) * ext_nx.x ];
                        }
                    }
                }

                {   // Copy from upper neighbour
                    int neighbor_ty = tile_idx.y + 1;
                    if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) )
                        neighbor_ty -= ntiles.y;

                    if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                        auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned idx = it.get_local_id(0); idx < gc.y.upper * ext_nx.x; idx += it.get_local_range(0) ) {
                            const auto i = idx % ext_nx.x;
                            const auto j = idx / ext_nx.x; 
                            local[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ] = y_upper[ i + ( gc.y.lower + j ) * ext_nx.x ];
                        }
                    }
                }
            });
        });

        q.wait();
    }

    /**
     * @brief Copies edge values to neighboring guard cells
     * 
     */
    void copy_to_gc( )  {

        // Copy along x direction
        copy_to_gc_x( );

        // Copy along y direction
        copy_to_gc_y( );

    };

    /**
     * @brief Adds values from neighboring guard cells to local data
     * 
     */
    void add_from_gc( ){

        // Add along x direction

        // Loop over tiles
        q.submit([&](sycl::handler &h) {
            const auto ntiles   = this->ntiles;
            const auto tile_vol = this->tile_vol;
            const auto nx       = this->nx;
            const auto ext_nx   = this->ext_nx;
            const auto periodic = this->periodic;
            const auto gc       = this->gc;
            auto * __restrict__ d_buffer = this->d_buffer;

            // 256×1 work items per group
            sycl::range<2> local{ 256, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };
            
            h.parallel_for( 
                sycl::nd_range{global * local , local},
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Add from lower neighbour
                    int neighbor_tx = tile_idx.x - 1;
                    if ( periodic.x && neighbor_tx < 0 )
                        neighbor_tx += ntiles.x;

                    if ( neighbor_tx >= 0 ) {
                        T * __restrict__ x_lower = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned idx = it.get_local_id(0); idx < ext_nx.y * gc.x.upper; idx += it.get_local_range(0) ) {
                            const auto i = idx % gc.x.upper;
                            const auto j = idx / gc.x.upper; 
                            local[ gc.x.lower + i + j * ext_nx.x ] += x_lower[ gc.x.lower + nx.x + i + j * ext_nx.x ];
                        }
                    }
                }

                {   // Add from upper neighbour
                    int neighbor_tx = tile_idx.x + 1;
                    if ( periodic.x && neighbor_tx >= static_cast<int>(ntiles.x) )
                        neighbor_tx -= ntiles.x;

                    if ( neighbor_tx < static_cast<int>(ntiles.x) ) {
                        auto * __restrict__ x_upper = d_buffer + (tile_idx.y * ntiles.x + neighbor_tx) * tile_vol;
                        for( unsigned idx = it.get_local_id(0); idx < ext_nx.y * gc.x.lower; idx += it.get_local_range(0) ) {
                            const auto i = idx % gc.x.lower;
                            const auto j = idx / gc.x.lower; 
                            local[ nx.x + i + j * ext_nx.x ] += x_upper[ i + j * ext_nx.x ];
                        }
                    }
                }
            });
        });
        q.wait();

        // Add along y direction

        // Loop over tiles
        q.submit([&](sycl::handler &h) {
            const auto ntiles   = this->ntiles;
            const auto tile_vol = this->tile_vol;
            const auto nx       = this->nx;
            const auto ext_nx   = this->ext_nx;
            const auto periodic = this->periodic;
            const auto gc       = this->gc;
            auto * __restrict__ d_buffer = this->d_buffer;

            // 256×1 work items per group
            sycl::range<2> local{ 256, 1 };

            // ntiles.x × ntiles.y groups
            sycl::range<2> global{ ntiles.x, ntiles.y };
            
            h.parallel_for( 
                sycl::nd_range{global * local , local},
                [=](sycl::nd_item<2> it) { 

                const auto tile_idx = make_uint2( it.get_group(0), it.get_group(1) );
                const auto tid      = tile_idx.y * ntiles.x + tile_idx.x;
                const auto tile_off = tid * tile_vol;

                auto * __restrict__ local = d_buffer + tile_off;
                
                {   // Add from lower neighbour
                    int neighbor_ty = tile_idx.y - 1;
                    if ( periodic.y && neighbor_ty < 0 )
                        neighbor_ty += ntiles.y;

                    if ( neighbor_ty >= 0 ) {
                        auto * __restrict__ y_lower = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned idx = it.get_local_id(0); idx < gc.y.upper * ext_nx.x; idx += it.get_local_range(0) ) {
                            const auto i = idx % ext_nx.x;
                            const auto j = idx / ext_nx.x; 
                            local[ i + ( gc.y.lower + j ) * ext_nx.x ] += y_lower[ i + ( gc.y.lower + nx.y + j ) * ext_nx.x ];
                        }
                    }
                }

                {   // Add from upper neighbour
                    int neighbor_ty = tile_idx.y;
                    neighbor_ty += 1;
                    if ( periodic.y && neighbor_ty >= static_cast<int>(ntiles.y) ) neighbor_ty -= ntiles.y;

                    if ( neighbor_ty < static_cast<int>(ntiles.y) ) {
                        auto * __restrict__ y_upper = d_buffer + (neighbor_ty * ntiles.x + tile_idx.x) * tile_vol;
                        for( unsigned idx = it.get_local_id(0); idx < gc.y.lower * ext_nx.x; idx += it.get_local_range(0) ) {
                            const auto i = idx % ext_nx.x;
                            const auto j = idx / ext_nx.x; 
                            local[ i + ( nx.y + j ) * ext_nx.x ] += y_upper[ i + j * ext_nx.x ];
                        }
                    }
                }
            });
        });
        q.wait();

    };

    /**
     * @brief Left shifts data for a specified amount
     * 
     * This operation is only allowed if the number of upper x guard cells
     * is greater or equal to the requested shift
     * 
     * @param shift Number of cells to shift
     * @param q     Sycl queue
     */
    void x_shift_left( unsigned int const shift ) {

        if ( gc.x.upper >= shift ) {

            // Loop over tiles
            q.submit([&](sycl::handler &h) {

                const auto tile_vol = this->tile_vol;
                const auto ext_nx   = this->ext_nx;
                const auto gc       = this->gc;

                /// @brief [device] grid data
                auto * __restrict__ d_buffer = this->d_buffer;

                const int  ystride    = ext_nx.x;

                // 8×1 work items per group
                sycl::range<2> local{ 8, 1 };

                // ntiles.x × ntiles.y groups
                sycl::range<2> global{ ntiles.x, ntiles.y };
                
                /// @brief [shared] Local copy of grid data
                auto shm = sycl::local_accessor< T, 1 > ( tile_vol, h );

                h.parallel_for( 
                    sycl::nd_range{global * local, local},
                    [=](sycl::nd_item<2> it) { 
                    
                    const auto tid        = it.get_group_linear_id();
                    const auto tile_off   = tid * tile_vol ;

                    auto * __restrict__ buffer = d_buffer + tile_off;
                    
                    for( int idx = it.get_local_id(0); idx < ext_nx.y * ext_nx.x; idx += it.get_local_range(0) ) {
                        const auto i = idx % ext_nx.x;
                        const auto j = idx / ext_nx.x;
                        if ( i < ext_nx.x - shift ) {
                            shm[ i + j * ystride ] = buffer[ (i + shift) + j * ystride ];
                        } else {
                            shm[ i + j * ystride ] = {0};
                        }
                    }

                    it.barrier();

                    for( int i = it.get_local_id(0); i < tile_vol; i  += it.get_local_range(0) ) 
                        buffer[i] = shm[i];

                });
            });
            q.wait();

            // Copy x guard cells
            copy_to_gc_x( );

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

        // Check that local memory can hold up to 2 times the tile buffer
        auto local_mem_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
        if ( local_mem_size < 2 * tile_vol * sizeof( T ) ) {
            std::cerr << "(*error*) Tile size too large " << nx << " (plus guard cells)\n";
            std::cerr << "(*error*) Insufficient local memory (" << local_mem_size << " B) for kernel_x() function.\n";
            abort();
        }

        if (( gc.x.lower > 0) && (gc.x.upper > 0)) {

            const int ystride = ext_nx.x;

            q.submit([&](sycl::handler &h) {
                /// @brief [shared] Local buffer A (original data)
                auto A = sycl::local_accessor< T, 1 > ( tile_vol, h );
                /// @brief [shared] Local buffer B (convolution)
                auto B = sycl::local_accessor< T, 1 > ( tile_vol, h );

                // 8×1 work items per group
                sycl::range<2> local{ 8, 1 };

                // ntiles.x × ntiles.y groups
                sycl::range<2> global{ ntiles.x, ntiles.y };

                const auto tile_vol = this->tile_vol;
                const auto ext_nx   = this->ext_nx;
                const auto nx       = this->nx;
                const auto gc       = this->gc;
                auto * __restrict__ d_buffer = this->d_buffer;

                h.parallel_for( 
                    sycl::nd_range{global * local , local},
                    [=](sycl::nd_item<2> it) { 
                    
                    const auto tid        = it.get_group_linear_id();
                    const auto tile_off   = tid * tile_vol ;

                    auto * __restrict__ buffer = d_buffer + tile_off;

                    // Copy data from tile buffer
                    for( int i = it.get_local_id(0); i < tile_vol; i+=it.get_local_range(0) )
                        A[i] = buffer[i];

                    // Synchronize 
                    it.barrier();

                    // Apply kernel locally
                    for( int idx = it.get_local_id(0); idx < ext_nx.y * nx.x; idx+=it.get_local_range(0) ) {
                        const auto iy = idx / nx.x;
                        const auto ix = idx % nx.x + gc.x.lower;
                        B [ iy * ystride + ix ] = A[ iy * ystride + (ix-1) ] * a +
                                                  A[ iy * ystride +  ix    ] * b +
                                                  A[ iy * ystride + (ix+1) ] * c;
                    }

                    // Synchronize 
                    it.barrier();

                    // Copy data back to tile buffer
                    for( int i = it.get_local_id(0); i < tile_vol; i+=it.get_local_range(0) )
                        buffer[i] = B[i];
                });
            });
            q.wait();

            // Update guard cells
            copy_to_gc_x( );

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

        // Check that local memory can hold up to 2 times the tile buffer
        auto local_mem_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
        if ( local_mem_size < 2 * tile_vol * sizeof( T ) ) {
            std::cerr << "(*error*) Tile size too large " << nx << " (plus guard cells)\n";
            std::cerr << "(*error*) Insufficient local memory (" << local_mem_size << " B) for kernel_y() function.\n";
            abort();
        }

        if (( gc.y.lower > 0) && (gc.y.upper > 0)) {

            const int ystride = ext_nx.x;

            q.submit([&](sycl::handler &h) {

                /// @brief [shared] Local buffer A (original data)
                auto A = sycl::local_accessor< T, 1 > ( tile_vol, h );
                /// @brief [shared] Local buffer B (convolution)
                auto B = sycl::local_accessor< T, 1 > ( tile_vol, h );

                // 8×1 work items per group
                sycl::range<2> local{ 8, 1 };

                // ntiles.x × ntiles.y groups
                sycl::range<2> global{ ntiles.x, ntiles.y };

                const auto tile_vol = this->tile_vol;
                const auto ext_nx   = this->ext_nx;
                const auto nx       = this->nx;
                const auto gc       = this->gc;
                auto * __restrict__ d_buffer = this->d_buffer;

                h.parallel_for(
                    sycl::nd_range{global * local , local},
                    [=](sycl::nd_item<2> it) { 

                    const auto tid        = it.get_group_linear_id();
                    const auto tile_off   = tid * tile_vol ;

                    auto * __restrict__ buffer = d_buffer + tile_off;

                    // Copy data from tile buffer
                    for( int i = it.get_local_id(0); i < tile_vol; i+=it.get_local_range(0) )
                        A[i] = buffer[i];

                    // Synchronize 
                    it.barrier();

                    // Apply kernel locally
                    for( int idx = it.get_local_id(0); idx < nx.y * ext_nx.x; idx+=it.get_local_range(0) ) {
                        const auto iy = idx / ext_nx.x + gc.y.lower;
                        const auto ix = idx % ext_nx.x;

                        B [ iy * ystride + ix ] = A[ (iy-1) * ystride + ix ] * a +
                                                  A[    iy  * ystride + ix ] * b +
                                                  A[ (iy+1) * ystride + ix ] * c;
                    }

                    // Synchronize 
                    it.barrier();

                    // Copy data back to tile buffer
                    for( int i = it.get_local_id(0); i < tile_vol; i+=it.get_local_range(0) )
                        buffer[i] = B[i];
                });
            });
            q.wait();

            // Update guard cells
            copy_to_gc_y( );

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
        info.count[0] = gnx.x;
        info.count[1] = gnx.y;

        const std::size_t bsize = gnx.x * gnx.y;

        T * d_data = device::malloc<T>( bsize, q );
        T * h_data = host::malloc<T>( bsize, q );

        gather( d_data );

        device::memcpy_tohost( h_data, d_data, bsize, q );

        zdf::save_grid( h_data, info, iter, path );

        host::free( h_data, q );
        device::free( d_data, q );
    };

    void save( std::string path ) {
        
        // Prepare file info
        zdf::grid_axis axis[2];
        axis[0] = (zdf::grid_axis) {
            .name = (char *) "x",
            .min = 0.,
            .max = 1. * gnx.x,
            .label = (char *) "x",
            .units = (char *) ""
        };

        axis[1] = (zdf::grid_axis) {
            .name = (char *) "y",
            .min = 0.,
            .max = 1. * gnx.y,
            .label = (char *) "y",
            .units = (char *) ""
        };

        std::string grid_name = "sycl";
        std::string grid_label = "sycl test";

        zdf::grid_info info = {
            .name = (char *) grid_name.c_str(),
            .label = (char *) grid_label.c_str(),
            .units = (char *) "",
            .axis  = axis
        };

        zdf::iteration iter = {
            .name = (char *) "ITERATION",
            .n = 0,
            .t = 0,
            .time_units = (char *) ""
        };

        save( info, iter, path );
    }

};

template<>
struct sycl::is_device_copyable<bnd<unsigned int>> : std::true_type {};

template<>
struct sycl::is_device_copyable<grid<float>> : std::true_type {};

#endif