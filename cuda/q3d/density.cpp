#include "density.h"

namespace kernel {

__global__
/**
 * @brief kernel for injecting a uniform plasma density (mk2)
 * 
 * @note Places contiguous particles in different cells. This minimizes memory
 *       collisions when depositing current, especially for very low
 *       temperatures.
 * 
 * @param range     Cell range (global) to inject particles in 
 * @param ppc       Number of particles per cell (z,r,θ)
 * @param q0        Reference charge
 * @param dr        Radial cell size
 * @param part      Particle data
 */
void uniform( 
    bnd<unsigned int> range,
    uint3 const ppc,
    const float q0,
    double const dr,
    ParticleData const part )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    
    // Tile ID
    int const tid = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    int np = part.np[ tid ];
    block_sync();

    /// @brief angular positions of particles
    auto * pos_th = block::shared_mem<float2>();

    // Find injection range in tile coordinates
    int ri0 = static_cast<int>(range.x.lower) - tile_idx.x * nx.x;
    int ri1 = static_cast<int>(range.x.upper) - tile_idx.x * nx.x;
    int rj0 = static_cast<int>(range.y.lower) - tile_idx.y * nx.y;
    int rj1 = static_cast<int>(range.y.upper) - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);

        const int offset =  part.offset[ tid ];

        auto * __restrict__ ix = &part.ix[ offset ];
        auto * __restrict__ x  = &part.x[ offset ];
        auto * __restrict__ u  = &part.u[ offset ];
        auto * __restrict__ q  = &part.q[ offset ];
        auto * __restrict__ th = &part.th[ offset ];

        double dpcz = 1.0 / ppc.x;
        double dpcr = 1.0 / ppc.y;

        /// @brief radial position shift for this tile
        const int shiftr = tile_idx.y * nx.y;

        // Set angular positions of particles
        pos_th[0] = { 1, 0 };
        
        if ( ppc.z > 1 ) {
            const float Δt = ( 2 * M_PI ) / ppc.z;
            for( int i = block_thread_rank() + 1; i < ppc.z; i += block_num_threads()) {
                pos_th[i] = { std::cos( i * Δt ), std::sin( i * Δt ) };
            }
            block_sync();
        }

        // Check if axial cell is part of injection domain
        int axis = ( rj0 == 0 && shiftr == 0 );
        if ( axis ) rj0 = 1;

        /// @brief grid volume not including axial cell
        int const grid_vol = (rj1 - rj0 + 1 ) * row;

        // Inject particles outside axial cell
        for( unsigned ith = 0; ith < ppc.z; ith++ ) {
            for( unsigned ir = 0; ir < ppc.y; ir++ ) {
                for( unsigned iz = 0; iz < ppc.x; iz++) {
                    float2 const pos = make_float2(
                        dpcz * ( iz + 0.5 ) - 0.5,
                        dpcr * ( ir + 0.5 ) - 0.5
                    );

                    auto ppc_idx = ( ith * ppc.y + ir ) * ppc.x + iz;

                    // Each thread takes 1 cell
                    for( int grid_idx = block_thread_rank(); grid_idx < grid_vol; grid_idx += block_num_threads() ) {
                        int2 const cell = make_int2( 
                            grid_idx % row + ri0,
                            grid_idx / row + rj0
                        );

                        double r = ( shiftr + cell.y ) + double( pos.y );
                        float qnorm = q0 * r * dr;

                        auto part_idx = np + grid_vol * ppc_idx + grid_idx;

                        ix[ part_idx ] = cell;
                        x [ part_idx ] = pos;
                        u [ part_idx ] = make_float3(0,0,0);
                        q [ part_idx ] = qnorm;
                        th[ part_idx ] = pos_th[ ith ];
                    }
                }
            }
        }

        /// add particles already injected in tile
        np += grid_vol * ppc.z * ppc.y * ppc.x ;

        // Axial cells get special treatment
        if ( axis ) {
            
            // Additional charge correction for particles on axial cell
            // see notes
            auto qnorm =  q0 * dr ;
            qnorm *= ( ppc.y % 2 == 0) ? 
                ( 2 * ( ppc.y*ppc.y - 1.) ) / ( 3 * ( ppc.y * ppc.y ) ) :
                (2. / 3.);
            double roff = ( ppc.y % 2 == 0) ? 0.5 : 1.0;

            for( unsigned ith = 0; ith < ppc.z; ith++ ) {
                for( unsigned ir = 0; ir < ppc.y/2; ir++ ) {
                    for( unsigned iz = 0; iz < ppc.x; iz++) {
                        float2 const pos = make_float2(
                            dpcz * ( iz + 0.5 ) - 0.5,
                            dpcr * ( ir + roff )
                        );

                        int ppc_idx = ( ith * (ppc.y/2) + ir ) * ppc.x + iz;

                        // On GPU each thread takes 1 cell
                        for( int grid_idx = block_thread_rank(); grid_idx < row; grid_idx += block_num_threads() ) {
                            int2 const cell = make_int2( grid_idx, 0 );

                            int part_idx = np + row * ppc_idx + grid_idx;

                            ix[ part_idx ] = cell;
                            x [ part_idx ] = pos;
                            u [ part_idx ] = make_float3(0,0,0);
                            q [ part_idx ] = pos.y * dr * qnorm;
                            th[ part_idx ] = pos_th[ ith ];
                        }
                    }
                }
            }
            np += row * ppc.z * ( ppc.y / 2 ) * ppc.x;
        }

        // Update global number of particles in tile
        if ( block_thread_rank() == 0 ) {
            part.np[ tid ] = np;
        }
    }
}

}

/**
 * @brief Injects a uniform density profile
 * 
 * @param part      Particle data object
 * @param norm      Charge normalization constant
 * @param ppc       Number of particles per cell (z,r,θ)
 * @param dx        Cell size (z,r)
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units (z,r)
 * @param range     Cell range in which to inject
 */
void Density::Uniform::inject( Particles & part, const float norm,
    uint3 const ppc, float2 const dx, float2 const ref,
    bnd<unsigned int> range ) const
{
    /// @brief single particle charge
    auto q0 = norm * n0 / (ppc.x*ppc.y*ppc.z);

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 1024 );
    auto shm_size = ppc.z * sizeof( float2 );
    kernel::uniform <<< grid, block, shm_size >>> (
        range, ppc, q0, dx.y, part
    );
}

namespace kernel {

__global__
void uniform_np( 
    bnd<unsigned int> range,
    uint3 const ppc,
    ParticleData const part,
    int * __restrict__ np
) {

    // Tile ID - 1 thread per tile
    const int tile_id = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tile_id < part.ntiles.x * part.ntiles.y ) {

        const uint2 tile_idx = make_uint2(
            tile_id % part.ntiles.x, 
            tile_id / part.ntiles.x
        );

        const int2 nx = make_int2( part.nx.x, part.nx.y );

        // Find injection range in tile coordinates
        int ri0 = static_cast<int>( range.x.lower ) - tile_idx.x * nx.x;
        int ri1 = static_cast<int>( range.x.upper ) - tile_idx.x * nx.x;
        int rj0 = static_cast<int>( range.y.lower ) - tile_idx.y * nx.y;
        int rj1 = static_cast<int>( range.y.upper ) - tile_idx.y * nx.y;

        int local_np;

        const auto ppc_zt = ppc.x * ppc.z;
        const auto ppc_r  = ppc.y;

        // If range overlaps with tile
        if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
            ( rj0 < nx.y ) && ( rj1 >= 0 )) {
            
            // Limit to range inside this tile
            if (ri0 < 0) ri0 = 0;
            if (rj0 < 0) rj0 = 0;
            if (ri1 >= nx.x ) ri1 = nx.x-1;
            if (rj1 >= nx.y ) rj1 = nx.y-1;

            int const row = (ri1-ri0+1);
            if ( tile_idx.y * nx.y + rj0 > 0 ) {
                // Tile does not include axial boundary
                local_np = (rj1-rj0+1) * row * ppc_zt * ppc_r;
            } else {
                // Tile includes axial boundary
                //std::since no particles will be injected for r <= 0, we only inject
                // half the particles in the axial cell
                local_np = row * ppc_zt * ( (rj1-rj0) * ppc_r + ppc_r / 2 );
            }
        } else {
            local_np = 0;
        }

        np[ tile_id ] = local_np;
    }
}

}

/**
 * @brief Returns number of particles per tile that a uniform profile would inject
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void Density::Uniform::np_inject( Particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    const int ntiles = part.ntiles.x * part.ntiles.y;

    int block = ( ntiles > 1024 ) ? 1024 : ntiles;
    int grid  = ( ntiles - 1 ) / block + 1;

    kernel::uniform_np<<< grid, block >>> ( range, ppc, part, np );
}

namespace kernel {

/**
 * @brief Kernel for injecting step profile
 * 
 * @tparam dir      Step direction ( coord::z | coord::r )
 * @param tile_idx  Tile index (x,y)
 * @param range     Cell range to inject particles in
 * @param step      Step position (normalized to node grid coordinates)
 * @param ppc       Number of particles per cell
 * @param dr        Radial cell size
 * @param part      Particle data
 */
template < coord::cyl dir >
__global__
void step( 
    bnd<unsigned int> range,
    const float step, const uint3 ppc, const float q0,
    double const dr,
    ParticleData const part )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    // Tile ID
    int const tid = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile;
    np_tile = part.np[ tid ];

    /// @brief angular positions of particles
    auto * pos_th = block::shared_mem<float2>();

    block_sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if ( ri0 < 0 ) ri0 = 0;
        if ( rj0 < 0 ) rj0 = 0;
        if ( ri1 >= nx.x ) ri1 = nx.x-1;
        if ( rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tid ];

        auto * __restrict__ const ix = &part.ix[ offset ];
        auto * __restrict__ const x  = &part.x[ offset ];
        auto * __restrict__ const u  = &part.u[ offset ];
        auto * __restrict__ const q  = &part.q[ offset ];
        auto * __restrict__ const th = &part.th[ offset ];

        double dpcz = 1.0 / ppc.x;
        double dpcr = 1.0 / ppc.y;

        const int shiftz = tile_idx.x * nx.x;
        const int shiftr = tile_idx.y * nx.y;

        pos_th[0] = { 1, 0 };
        
        if ( ppc.z > 1 ) {
            const float Δt = ( 2 * M_PI ) / ppc.z;
            for( int i = block_thread_rank() + 1; i < ppc.z; i += block_num_threads()) {
                pos_th[i] = { std::cos( i * Δt ), std::sin( i * Δt ) };
            }
            block_sync();
        }

        // Charge normalization
        auto qnorm =  q0 * dr;
        auto α = ( ppc.y % 2 == 0) ? 
            ( 2 * ( ppc.y*ppc.y - 1.) ) / ( 3 * ( ppc.y * ppc.y ) ) :
            (2. / 3.);

        for( unsigned i2 = 0; i2 < ppc.z; i2++ ) {
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcz * ( i0 + 0.5 ) - 0.5,
                        dpcr * ( i1 + 0.5 ) - 0.5
                    );
                    for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads()) {
                        int2 const cell = make_int2(
                            idx % row + ri0,
                            idx / row + rj0
                        );
                        auto r = (shiftr + cell.y) + pos.y;

                        int inj;
                        if constexpr ( dir == coord::z ) inj = ((shiftz + cell.x) + (pos.x + 0.5) > step ) && (r > 0);
                        if constexpr ( dir == coord::r ) inj = r > step && r > 0;

                        int off = block::exscan_add( inj );

                        if ( inj ) {
                            const int k = np_tile + off;
                            ix[ k ] = cell;
                            x [ k ] = pos;
                            u [ k ] = float3{0};
                            q [ k ] = r * ( (shiftr + cell.y == 0) ? α * qnorm : qnorm );
                            th[ k ] = pos_th[ i2 ];
                        }

                        inj = warp::reduce_add( inj );
                        if ( warp::thread_rank() == 0 ) {
                            block::atomic_fetch_add( &np_tile, inj );
                        }
                        block_sync();
                    }
                }
            }
        }

        if ( block_thread_rank() == 0 ) {
            part.np[ tid ] = np_tile;
        }
    }
}

}

/**
 * @brief Inject a step density profile
 * 
 * @param particles     Particle data
 * @param ppc           Number of particles per cell
 * @param dx            Cell size
 * @param ref           Reference for step position
 * @param range         Global cell range for injection
 */
void Density::Step::inject( Particles & part, const float norm,
    uint3 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{

    /// @brief Step position (normalized to node grid coordinates)
    float step_pos;
    /// @brief single particle charge
    auto q0 = norm * n0 / (ppc.x*ppc.y*ppc.z);

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 1024 );
    auto shm_size = ppc.z * sizeof( float2 );

    switch( dir ) {
    case( coord::z ):
        step_pos = (pos - ref.x) / dx.x;
        kernel::step <coord::z> <<< grid, block, shm_size >>> (
            range, step_pos, ppc, q0, dx.y, part
        );
        break;

    case( coord::r ):
        step_pos = (pos - ref.y) / dx.y;
        kernel::step <coord::r> <<< grid, block , shm_size >>> (
            range, step_pos, ppc, q0, dx.y, part
        );
        break;
    }
}

namespace kernel {

template < coord::cyl dir >
__global__
void step_np( 
    bnd<unsigned int> range,
    const float step, const uint3 ppc,
    ParticleData const part, int * np )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );

    // Tile ID
    int const tid = tile_idx.y * part.ntiles.x + tile_idx.x;

    __shared__ int np_tile; np_tile = 0;
    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcz = 1.0 / ppc.x;
        double dpcr = 1.0 / ppc.y;

        const int shiftz = tile_idx.x * nx.x;
        const int shiftr = tile_idx.y * nx.y;

        for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcz * ( i0 + 0.5 ) - 0.5,
                        dpcr * ( i1 + 0.5 ) - 0.5
                    );
                    float r = (shiftr + cell.y) + pos.y;
                    
                    int inj;
                    if constexpr ( dir == coord::z ) inj = ((shiftz + cell.x) + (pos.x + 0.5) > step ) && (r > 0);
                    if constexpr ( dir == coord::r ) inj = r > step;
                    inj_np += inj;
                }
            }
        }
    }

    inj_np = warp::reduce_add( inj_np );
    if ( warp::thread_rank() == 0 ) {
        block::atomic_fetch_add( &np_tile, inj_np );
    } 

    block_sync();

    if ( block_thread_rank() == 0 ) {
        np[ tid ] = np_tile * ppc.z;
    }
}

}

/**
 * @brief Returns number of particles per tile that a step profile would inject
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void Density::Step::np_inject( Particles & part,
    uint3 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 1024 );

    float step_pos;
    switch( dir ) {
    case( coord::z ):
        step_pos = (pos - ref.x) / dx.x;
        kernel::step_np <coord::z> <<< grid, block >>> ( range, step_pos, ppc, part, np );
        break;
    case( coord::r ):
        step_pos = (pos - ref.y) / dx.y;
        kernel::step_np <coord::r> <<< grid, block >>> ( range, step_pos, ppc, part, np );
        break;
    }
}

namespace kernel {

template < coord::cyl dir >
__global__
void slab( 
    bnd<unsigned int> range,
    const float start, const float finish, uint3 ppc, const float q0, 
    double const dr,
    ParticleData const part )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );

    const uint2 ntiles  = part.ntiles;
    const int2 nx = make_int2( part.nx.x, part.nx.y );
 
    // Tile ID
    int const tid = tile_idx.y * ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile; np_tile = part.np[ tid ];

    /// @brief angular positions of particles
    auto * pos_th = block::shared_mem<float2>();

    block_sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tid ];

        auto * __restrict__ const ix = &part.ix[ offset ];
        auto * __restrict__ const x  = &part.x[ offset ];
        auto * __restrict__ const u  = &part.u[ offset ];
        auto * __restrict__ const q  = &part.q[ offset ];
        auto * __restrict__ const th = &part.th[ offset ];

        double dpcz = 1.0 / ppc.x;
        double dpcr = 1.0 / ppc.y;

        const int shiftz = tile_idx.x * nx.x;
        const int shiftr = tile_idx.y * nx.y;

        /// Set angular positions of particles
        pos_th[0] = { 1, 0 };
        
        if ( ppc.z > 1 ) {
            const float Δt = ( 2 * M_PI ) / ppc.z;
            for( int i = block_thread_rank() + 1; i < ppc.z; i += block_num_threads()) {
                pos_th[i] = { std::cos( i * Δt ), std::sin( i * Δt ) };
            }
            block_sync();
        }

        // Charge normalization
        auto qnorm =  q0 * dr;
        auto α = ( ppc.y % 2 == 0) ? 
            ( 2 * ( ppc.y*ppc.y - 1.) ) / ( 3 * ( ppc.y * ppc.y ) ) :
            (2. / 3.);

        for( unsigned i2 = 0; i2 < ppc.z; i2++ ) {
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcz * ( i0 + 0.5 ) - 0.5,
                        dpcr * ( i1 + 0.5 ) - 0.5
                    );
                    for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads()) {
                        int2 const cell = make_int2(
                            idx % row + ri0,
                            idx / row + rj0
                        );
                        auto r = (shiftr + cell.y) + pos.y;

                        float w;
                        if constexpr ( dir == coord::z ) w = (shiftz + cell.x) + (pos.x + 0.5);
                        if constexpr ( dir == coord::r ) w = r;
                        
                        int inj = (w >= start) && (w<finish ) && ( r > 0 );

                        int off = block::exscan_add( inj );
                        
                        if (inj) {
                            const int k = np_tile + off;
                            ix[ k ] = cell;
                            x [ k ] = pos;
                            u [ k ] = make_float3(0,0,0);
                            q [ k ] = r * ( (shiftr + cell.y == 0) ? α * qnorm : qnorm );
                            th[ k ] = pos_th[ i2 ];
                        }

                        inj = warp::reduce_add( inj );
                        if ( warp::thread_rank() == 0 ) {
                            block::atomic_fetch_add( &np_tile, inj );
                        }
                        block_sync();
                    }
                }
            }
        }

        if ( block_thread_rank() == 0 ) {
            part.np[ tid ] = np_tile;
        }
    }
}

}

/**
 * @brief Injects a slab density profile
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void Density::Slab::inject( Particles & part, const float norm,
    uint3 const ppc,float2 const dx, float2 const ref,
    bnd<unsigned int> range ) const
{
    /// @brief Slab start position (normalized to node grid coordinates) 
    float slab_begin;
    /// @brief Slab end position (normalized to node grid coordinates) 
    float slab_end;
    /// @brief single particle charge
    auto q0 = norm * n0 / (ppc.x*ppc.y*ppc.z);

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 1024 );
    auto shm_size = ppc.z * sizeof( float2 );

    switch( dir ) {
    case( coord::z ):
        slab_begin = (begin - ref.x)/ dx.x;
        slab_end = (end - ref.x)/ dx.x;
        kernel::slab <coord::z> <<< grid, block, shm_size >>> (
                range, slab_begin, slab_end, ppc, q0, dx.y,
                part );
        break;

    case( coord::r ):
        slab_begin = (begin - ref.y)/ dx.y;
        slab_end = (end - ref.y)/ dx.y;
        kernel::slab <coord::r> <<< grid, block, shm_size >>> (
                range, slab_begin, slab_end, ppc, q0, dx.y,
                part );
        break;
    }
}

namespace kernel {

/**
 * @brief Kernel for counting how many particles will be injected by slab profie
 * 
 * @tparam dir          Slab direction ( coord::x | coord::y )
 * @param tile_idx      Tile index (x,y)
 * @param range         Cell range to inject particles in
 * @param start         Slab start position (normalized to node grid coordinates)
 * @param finish        Slab end position (normalized to node grid coordinates)
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 * @param np            (out) Number of particles to inject per tile
 */
template < coord::cyl dir >
__global__
void slab_np( 
    bnd<unsigned int> range,
    const float start, const float finish, uint3 ppc,
    ParticleData const part, int * np )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const uint2 ntiles  = part.ntiles;
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    // Tile ID
    int const tid = tile_idx.y * ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile; np_tile = 0;
    block_sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcz = 1.0 / ppc.x;
        double dpcr = 1.0 / ppc.y;

        const int shiftz = tile_idx.x * nx.x;
        const int shiftr = tile_idx.y * nx.y;

        for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcz * ( i0 + 0.5 ) - 0.5,
                        dpcr * ( i1 + 0.5 ) - 0.5
                    );
                    float r = (shiftr + cell.y) + pos.y;

                    float w;
                    if constexpr ( dir == coord::z ) w = (shiftz + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::r ) w = r;
                    
                    int inj = (w >= start) && (w<finish ) && r > 0;
                    inj_np += inj;
                }
            }
        }
    }

    inj_np = warp::reduce_add( inj_np );
    if ( warp::thread_rank() == 0 ) {
        block::atomic_fetch_add( &np_tile, inj_np );
    } 

    block_sync();

    if ( block_thread_rank() == 0 ) {
        np[ tid ] = np_tile* ppc.z;
    }
}

}

/**
 * @brief Returns number of particles per tile that a slab profile would inject
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void Density::Slab::np_inject( Particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref,
    bnd<unsigned int> range,
    int * np ) const
{
    /// @brief Slab start position (normalized to node grid coordinates) 
    float slab_begin;
    /// @brief Slab end position (normalized to node grid coordinates) 
    float slab_end;

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 1024 );

    switch( dir ) {
    case( coord::z ):
        slab_begin = (begin - ref.x)/ dx.x;
        slab_end = (end - ref.x)/ dx.x;
        kernel::slab_np <coord::z> <<< grid, block >>> ( 
            range, slab_begin, slab_end, ppc, part, np
        );
        break;

    case( coord::r ):
        slab_begin = (begin - ref.y)/ dx.y;
        slab_end = (end - ref.y)/ dx.y;
        kernel::slab_np <coord::r> <<< grid, block >>> (
            range, slab_begin, slab_end, ppc, part, np
        );
        break;
    }
}

namespace kernel {

/**
 * @brief Kernel for injecting sphere profile
 * 
 * @param tile_idx      Tile index (x,y)
 * @param range         Cell range to inject particles in
 * @param center        Sphere center in simulation units
 * @param radius        Sphere radius in simulation units
 * @param dx            Cell size
 * @param ppc           Number of particles per cell
 * @param part          Particle data
 */
__global__
void sphere( 
    bnd<unsigned int> range,
    const float2 center, const float radius, const float2 dx, uint3 ppc, const float q0,
    ParticleData const part )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );

    // Tile ID
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile; np_tile = part.np[ tile_id ];

    /// @brief angular positions of particles
    auto * pos_th = block::shared_mem<float2>();

    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];

        auto * __restrict__ const ix = &part.ix[ offset ];
        auto * __restrict__ const x  = &part.x[ offset ];
        auto * __restrict__ const u  = &part.u[ offset ];
        auto * __restrict__ const q  = &part.q[ offset ];
        auto * __restrict__ const th = &part.th[ offset ];

        double dpcz = 1.0 / ppc.x;
        double dpcr = 1.0 / ppc.y;

        const int shiftz = tile_idx.x * nx.x;
        const int shiftr = tile_idx.y * nx.y;

        const float r2 = radius*radius;

        /// get angular positions of particles
        pos_th[0] = { 1, 0 };
        
        if ( ppc.z > 1 ) {
            const float Δt = ( 2 * M_PI ) / ppc.z;
            for( int i = block_thread_rank() + 1; i < ppc.z; i += block_num_threads()) {
                pos_th[i] = { std::cos( i * Δt ), std::sin( i * Δt ) };
            }
            block_sync();
        }

        // Charge normalization
        auto qnorm =  q0;
        auto α = ( ppc.y % 2 == 0) ? 
            ( 2 * ( ppc.y*ppc.y - 1.) ) / ( 3 * ( ppc.y * ppc.y ) ) :
            (2. / 3.);

        for( unsigned i2 = 0; i2 < ppc.z; i2++ ) {
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcz * ( i0 + 0.5 ) - 0.5,
                        dpcr * ( i1 + 0.5 ) - 0.5
                    );
                    for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
                        int2 const cell = make_int2( 
                            idx % row + ri0,
                            idx / row + rj0
                        );
                        float z = ((shiftz + cell.x) + (pos.x+0.5)) * dx.x;
                        float r = ((shiftr + cell.y) + (pos.y) ) * dx.y;
                        
                        int inj = (((z - center.x)*(z - center.x) + (r - center.y)*(r - center.y)) < r2) && (r>0);
                        int off = block::exscan_add( inj );

                        if ( inj ) {
                            const int k = np_tile + off;
                            ix[ k ] = cell;
                            x [ k ] = pos;
                            u [ k ] = make_float3(0,0,0);
                            q [ k ] = r * ( (shiftr + cell.y == 0) ? α * qnorm : qnorm );
                            th[ k ] = pos_th[ i2 ];
                        }
                        
                        inj = warp::reduce_add( inj );
                        if ( warp::thread_rank() == 0 ) {
                            block::atomic_fetch_add( &np_tile, inj );
                        }
                        block_sync();
                    }
                }
            }
        }

        if ( block_thread_rank() == 0 ) {
            part.np[ tile_id ] = np_tile;
        }
    }
}

}

/**
 * @brief Injects a sphere density profile
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 */
void Density::Sphere::inject( Particles & part, const float norm,
    uint3 const ppc, float2 const dx, float2 const ref,
    bnd<unsigned int> range ) const
{

    /// @brief Sphere center position
    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    /// @brief single particle charge
    auto q0 = norm * n0 / (ppc.x*ppc.y*ppc.z);

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 1024 );
    auto shm_size = ppc.z * sizeof( float2 );
    kernel::sphere <<< grid, block, shm_size >>> (
        range, sphere_center, radius, dx, ppc, q0,
        part
    );
}


namespace kernel {

/**
 * @brief Kernel for counting how many particles will be injected by a sphere
 *        profile
 * 
 * @param tile_idx  Tile index (x,y)
 * @param range     Cell range to inject particles in
 * @param center    Sphere center in simulation units
 * @param radius    Sphere radius in simulation units
 * @param dx        Cell size
 * @param ppc       Number of particles per cell
 * @param part      Particle data
 * @param np        (out) Number of particles per tile to inject
 */
__global__
void sphere_np(
    bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint3 ppc,
    ParticleData const part, int * np )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );

    // Tile ID
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    __shared__ int np_tile; np_tile = 0;
    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcz = 1.0 / ppc.x;
        double dpcr = 1.0 / ppc.y;

        const int shiftz = tile_idx.x * nx.x;
        const int shiftr = tile_idx.y * nx.y;

        const float r2 = radius*radius;

        for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
            const int2 cell = make_int2( 
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcz * ( i0 + 0.5 ) - 0.5,
                        dpcr * ( i1 + 0.5 ) - 0.5
                    );
                    float z = ((shiftz + cell.x) + (pos.x+0.5)) * dx.x;
                    float r = ((shiftr + cell.y) + (pos.y)) * dx.y;
                    
                    int inj = (((z - center.x)*(z - center.x) + (r - center.y)*(r - center.y)) < r2) && (r>0);
                    inj_np += inj;
                }
            }
        }
    }

    inj_np = warp::reduce_add( inj_np );
    if ( warp::thread_rank() == 0 ) {
        block::atomic_fetch_add( &np_tile, inj_np );
    } 

    block_sync();

    if ( block_thread_rank() == 0 ) {
        np[ tile_id ] = np_tile * ppc.z;
    }
}

}

/**
 * @brief Returns number of particles per tile that a sphere profile would inject
 * 
 * @param part      Particle data object
 * @param ppc       Number of particles per cell
 * @param dx        Cell size
 * @param ref       Position of local grid on global simulation box in simulation
 *                  units
 * @param range     Cell range in which to inject
 * @param np        (out) Number of particles to inject per tile
 */
void Density::Sphere::np_inject( Particles & part, 
    uint3 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 1024 );
    kernel::sphere_np <<< grid, block >>> ( 
        range, sphere_center, radius, dx, ppc, part, np );
}
