#ifndef UTILS_H_
#define UTILS_H_

#include <cstddef>
#include <cstring>
#include <typeinfo>
#include <iostream>

#include <sycl/sycl.hpp>

#define MEM_ALIGN 64

static inline void print_dev_info( const sycl::queue & q ) {
    const sycl::device & dev = q.get_device();
    std::cout << "Name               : " << dev.get_info<sycl::info::device::name>() << '\n';
    std::cout << "Max. compute units : " << dev.get_info<sycl::info::device::max_compute_units>() << '\n';
    std::cout << "Local mem. size    : " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes \n";
    std::cout << "Max. # sub-groups  : " << dev.get_info<sycl::info::device::max_num_sub_groups>() << '\n';
}

/**
 * @brief Rounds up to a multiple of 4
 * 
 * @tparam T    Value type (must be integer like)
 * @param a     Value to round up
 * @return T    a rounded up to the nearest multiple of 4
 */
template < typename T >
T roundup4( T a ) { return (a + 3) & static_cast<T>(-4);}

/**
 * @brief Rounds up to a multiple of N (where N is a power of 2)
 * 
 * @tparam N    Value will be rounded to a multiple of N. Must be a power of 2.
 * @tparam T    Value type. Must be an integer type (int, long, unsigned, int64_t, etc.)
 * @param a     Value to round up
 * @return T    Value rounded up to a multiple of N
 */
template < int N, typename T >
T roundup( T a ) {
    static_assert( N > 0, "N must be > 0");
    static_assert( !(N & (N-1)), "N must b a power of 2" );
    return ( a + (N-1) ) & static_cast<T>(-N);
};

/**
 * @brief Swaps 2 pointer values
 * 
 * @tparam T    Value type
 * @param a     Value a
 * @param b     Value b
 */
template < typename T >
void swap( T* &a, T* &b ) {
    T * tmp = a; a = b; b = tmp;
}


template < typename T, typename T2 >
inline void print_array( T * __restrict__ buffer, T2 dims ) {
    for( auto i1 = 0; i1 < dims.y; i1 ++ ) {
        std::cout << buffer[i1 * dims.x];
        for( auto i0 = 1; i0 < dims.x; i0 ++ ) {
            std::cout << " " << buffer[i1 * dims.x + i0];
        }
        std::cout << '\n';
    }
}

namespace ops {

/**
 * @brief
 * Multiply-add operation: f = (x * y) + z
 * 
 * @note
 * If the FP_FAST_FMA macro is defined then the routine will call std::fma()
 * which is supposed to implement a (faster) fused multply-add operation.
 * Otherwise, we just do the normal operation to avoid calling the much slower
 * fma operation in libm.
 * 
 * @tparam T 
 * @param x 
 * @param y 
 * @param z 
 * @return auto 
 */
template<typename T>
T fma( T const x, T const y, T const z ) {

#ifdef FP_FAST_FMA
    return std::fma( x, y, z );
#else
    return (x*y)+z;
#endif
}

}

/**
 * @brief Device routines
 * 
 */
namespace device {

/**
 * @brief Sub-group level utility functions
 * 
 */
namespace subgroup {

/**
 * @brief Sub-group level reduction (add)
 * 
 * @warning Only work-item 0 gets the correct result. Also, all work-items in
 *          sub-group must call this.
 * 
 * @tparam T    Template datatype
 * @param sg    Sub-group
 * @param input Input value
 * @return T 
 */
template<typename T>
inline T reduce_add( sycl::sub_group & sg, T const input ) {
    T value = input;
    for( int i = 1; i < sg.get_local_linear_range(); i <<= 1 )
    	value += sycl::permute_group_by_xor(sg, value, i);
    return value;
} 

template<typename T>
inline T reduce_max( sycl::sub_group & sg, T const input ) {
    T value = input;
    for( int i = 1; i < sg.get_local_linear_range(); i <<= 1 ) {
	T tmp = sycl::permute_group_by_xor(sg, value, i);
        if (tmp > value) value = tmp;
    }
    return value;
} 

template<typename T>
inline T reduce_min( sycl::sub_group & sg, T const input ) {
    T value = input;
    for( int i = 1; i < sg.get_local_linear_range(); i <<= 1 ) {
	T tmp = sycl::permute_group_by_xor(sg, value, i); 
        if (tmp < value) value = tmp;
    }
    return value;
} 

template<typename T>
inline T inscan_add( sycl::sub_group & sg, T const input ) {
    T value = input;
    const int laneId = sg.get_local_linear_id();
    for( int i = 1; i < sg.get_local_linear_range(); i <<= 1 ) {
	T tmp = sycl::shift_group_right(sg, value, i);
        if ( laneId >= i ) value += tmp;
    }
    return (laneId > 0) ? value : 0;
} 

template<typename T>
inline T exscan_add( sycl::sub_group & sg, T const input ) {
    T value = input;
    const int laneId = sg.get_local_linear_id();
    for( int i = 1; i < sg.get_local_linear_range(); i <<= 1 ) {
        T tmp = sycl::shift_group_right(sg, value, i);
        if ( laneId >= i ) value += tmp;
    }

    value = sycl::shift_group_right(sg, value);
    return (laneId > 0) ? value : 0;
}

}

namespace group {

/**
 * @brief Group level exclusive scan
 * 
 * @warning tmp must be large enough to hold values from all subgroups
 * 
 * @tparam T        Datatype
 * @tparam dims     Dimension of work item
 * @param it        Work-item
 * @param tmp       Temporary memory for calculations. If possible, use local
 *                  (group) memory
 * @param input     Input value
 * @return T 
 */
template<typename T, int dims>
inline T exscan_add( sycl::nd_item<dims> it, T * tmp, T const input ) {

#if 0
    // Serial implementation
    tmp[ it.get_local_linear_id() ] = input;
    it.barrier();
    if ( it.get_local_linear_id() == 0 ) {
        int prev = 0;
        int range = 0;
        if ( dims == 1 ) range = it.get_local_range(0);
        if ( dims == 2 ) range = it.get_local_range(1) * it.get_local_range(0);
        if ( dims == 3 ) range = it.get_local_range(2) * it.get_local_range(1) * it.get_local_range(0);
        
        for( int k = 0; k < range; k++ ) {
            auto l = tmp[k];
            tmp[k] = prev;
            prev += l;
        }
    }
    it.barrier();
    return tmp[ it.get_local_linear_id() ];
#endif

    auto sg = it.get_sub_group();
    T v = subgroup::exscan_add( sg, input );
    
    // Get group size - depends on number of dimensions
    int range;
    if ( dims == 1 ) range = it.get_local_range(0);
    if ( dims == 2 ) range = it.get_local_range(1) * it.get_local_range(0);
    if ( dims == 3 ) range = it.get_local_range(2) * it.get_local_range(1) * it.get_local_range(0);

    // More than 1 sub-group per group
    if ( range > sg.get_local_linear_range() ) {
        
        if ( sg.get_local_id() == sg.get_local_linear_range() - 1 ) {
            tmp[ sg.get_group_id() ] = v + input;
        }
        it.barrier();

        if ( sg.get_group_id() == 0 )
            tmp[ sg.get_local_id() ] = subgroup::exscan_add( sg, tmp[ sg.get_local_id() ] );
        it.barrier();

        v += tmp[ sg.get_group_id() ];

    }
    return v;
}
}

/**
 * @brief   Allocate memory on device
 * @note    If MEM_ALIGN macro is defined then memory will be aligned to this value
 * 
 * @tparam T    Data type
 * @param size  Size (number of elements) to allocate
 * @param q     Sycl queue
 */
template< typename T >
T * malloc( std::size_t const size, const sycl::queue & q ) {
#ifdef MEM_ALIGN
    T * buffer = sycl::aligned_alloc_device<T>( MEM_ALIGN, size, q );
#else
    T * buffer = sycl::malloc_device<T>( size, q );
#endif

    return buffer;
}

/**
 * @brief   Free device allocated memory
 * 
 * @tparam T    Data type
 * @param ptr   Pointer to allocated memory
 * @param q     Sycl queue
 */
template< typename T >
void free( T * ptr, const sycl::queue & q ) {
    sycl::free( ptr, q );
}

/**
 * @brief Zeroes data buffer
 * 
 * @tparam T    Data type
 * @param ptr   Pointer to data buffer
 * @param size  Buffer size (number of elements)
 * @param q     Sycl queue
 */
template< typename T >
void zero( T * const __restrict__ ptr, std::size_t const size, sycl::queue & q ) {
    q.submit([&](sycl::handler &h) {
        h.memset( (void *) ptr, 0, size * sizeof(T) );
    });
    q.wait();
}

/**
 * @brief Sets buffer to scalar value
 * 
 * @tparam T    Data type
 * @param ptr   Pointer to data buffer
 * @param size  Buffer size (number of elements)
 * @param val   Scalar value to set (passed by copy)
 * @param q     Sycl queue
 */

/**
 * @brief Sets buffer to scalar value
 * 
 * @tparam T    Data type
 * @param ptr   Pointer to data buffer
 * @param size  Buffer size (number of elements)
 * @param val   Scalar value to set (passed by copy)
 * @param q     SYCL queue
 * @return      SYCL event object
 */
template< typename T >
auto setval( T * const __restrict__ ptr, std::size_t const size, const T val, sycl::queue & q ) {
    
    return q.submit([&](sycl::handler &h) {
        h.parallel_for( size, [=](sycl::id<1> i) { ptr[i] = val; });
    });
}

/**
 * @brief Exclusive scan (add) in device memory
 * 
 * @tparam T            Data type
 * @param data          Pointer to data buffer in device memory
 * @param size          Buffer size (number of elements)
 * @param q             SYCL queue
 * @param reduction     (optional) If set the routine will store the total sum in this variable
 */
template< typename T >
void exscan_add( T * __restrict__ data, std::size_t const size, sycl::queue & q, T * __restrict__ reduction = nullptr ) {

    auto group_size = ( size < 1024 ) ? size : 1024;
    sycl::range<1> local{ group_size };

    q.submit([&](sycl::handler &h) {

        /// @brief [shared] Sum of previous group
        auto group_prev = sycl::local_accessor< T, 1 > ( 1, h );

        /// @brief [shared] Sum of previous sub-group
        auto sg_prev = sycl::local_accessor< T, 1 > ( 1, h );

        /// @brief [shared] Temporary results from each sub group
        const int max_num_sub_groups = q.get_device().get_info<sycl::info::device::max_num_sub_groups>();
        auto _tmp = sycl::local_accessor< T, 1 > ( max_num_sub_groups, h );

        h.parallel_for( 
            sycl::nd_range{ local, local },
            [=](sycl::nd_item<1> it) {

            group_prev[0] = 0;
            auto sg = it.get_sub_group();

            for( int i = it.get_local_id(0); i < size; i += it.get_local_range(0) ) {
                T s = data[i];
                T v = device::subgroup::exscan_add( sg, s );

                if ( sg.get_local_id() == sg.get_local_range() - 1 )
                    _tmp[ sg.get_group_linear_id() ] = v + s;

                it.barrier();

                // Only 1 subgroup does this
                if ( sg.get_group_linear_id() == 0 ) {

                    // This is more complex than the CUDA version because the number of subgroups may
                    // be larger than the subgroup size

                    sg_prev[0] = group_prev[0];
                    for( auto j = 0; j < sg.get_group_linear_range(); j += sg.get_local_linear_range() ) {
                        T t = _tmp[ j + sg.get_local_id() ];
                        T e = device::subgroup::exscan_add( sg, t ) + sg_prev[0];
                        _tmp[ j + sg.get_local_id() ] = e;
                        if ( sg.get_local_id() == sg.get_local_linear_range() - 1 ) 
                            sg_prev[0] = e + t;
                    }
                }
                it.barrier();

                // Add in contribution from previous threads
                v += _tmp[ sg.get_group_linear_id() ];
                data[i] = v;

                if (( it.get_local_id(0) == it.get_local_range(0)-1 ) || (i+1 == size)) {
                    group_prev[0] = v+s;
                }
                it.barrier();
            }

            if ( reduction != nullptr ) {
                if ( it.get_global_id(0) == 0 ) *reduction = group_prev[0];
            }
        });
    });
    q.wait();

}


/**
 * @brief Copies data from device to host
 * 
 * @warning The code will wait for the queue to finish before submitting the memcpy action
 * 
 * @tparam T        Data type
 * @param h_out     Output host buffer
 * @param d_in      Input device buffer
 * @param size      Buffer size (number of elements)
 * @param q         Sycl queue
 */
template< typename T >
void memcpy_tohost( T * const __restrict__ h_out, T const * const __restrict__ d_in, size_t const size, sycl::queue & q ) {
    
    // This ensures that the copy does not start until any pending kernels
    // have stopped.
    q.wait();
    
    q.submit([&](sycl::handler &h) {
        h.memcpy(&h_out[0], d_in, size * sizeof(T));
    });
    q.wait();
}

template<typename T>
void print( T * buffer_d, unsigned int size, std::string msg, sycl::queue & q ) {
    T * buffer_h;

    buffer_h = sycl::malloc_host<T>( size, q );
    memcpy_tohost( buffer_h, buffer_d, size, q );
    std::cout << msg << "\n";
    for( int i = 0; i < size; i += 8 ) {
        printf("%8X:", i);
        for( int j = 0; j < 8; j++) {
            int idx = i + j;
            if ( idx < size ) std::cout << " " << buffer_h[idx];
        }
        std::cout << "\n";
    }
    sycl::free( buffer_h, q );
}

/**
 * @brief Class representing a scalar variable in device memory
 * 
 * @tparam T    Variable datatype
 */
template< typename T> class Var {
    private:

    /// @brief Associated sycl queue
    sycl::queue & q;

    /// @brief Pointer to device data
    T * d_data;

    public:

    /**
     * @brief Construct a new Var object
     * 
     * @param q     Sycl queue
     */
    Var( sycl::queue & q ) : q(q) {
#ifdef MEM_ALIGN
        d_data = sycl::aligned_alloc_device<T>( MEM_ALIGN, 1, q );
#else
        d_data = sycl::malloc_device<T>( 1, q );
#endif
    }

    /**
     * @brief Destroy the Var object
     * 
     */
    ~Var() {
        device::free( d_data, q );
    }

    /**
     * @brief Construct a new Var object and initialize it to a specific value
     * 
     * @param q 
     * @param val 
     */
    Var( sycl::queue & q, const T val ) : Var(q) {
        set( val );
    }

    /**
     * @brief Set the value of the variable
     * 
     * @param val       Value
     * @return          Returns the same value 
     */
    T const set( const T val ) {
        q.submit([&](sycl::handler &h) {
            h.memcpy(d_data, &val, sizeof(T));
        });
        q.wait();
        return val;
    }

    /**
     * @brief Overloaded assignment operator for setting the value
     * 
     * @param val   Value
     * @return T    Returns the same value
     */
    T operator= (const T val) {
        return set(val);
    }

    /**
     * @brief Get the value
     *
     * @note This will always copy the data from device memory to host memory.
     *       The queue will be synchronized (`wait()`) first.
     * 
     * @return T const  The value
     */
    T const get() const { 
        T val;
        q.wait();
        q.submit([&](sycl::handler &h) {
            h.memcpy(&val, d_data, sizeof(T));
        });
        q.wait();

        return val;
    }

    /**
     * @brief Gets pointer to device data
     * 
     * @return T*   Device data
     */
    T * ptr() const { return d_data; }

    /**
     * @brief Stream << operator overload for printing variable value
     * 
     * @tparam U                Data type
     * @param os                Output stream
     * @param d                 Variable
     * @return std::ostream&    Output stream
     */
    template< class U >
    friend auto operator<< (std::ostream& os, device::Var<U> const & d) -> std::ostream& { 
        return os << d.get();
    }
};

namespace global {

/**
 * @brief Atomic add operation in global device memory
 * 
 * @tparam T            Data type
 * @param globalAddr    Global address
 * @param val           Value to add
 * @return T            Value at address before operation
 */
template<typename T>
inline T atomicAdd( T * globalAddr, T val ) {
    auto v = sycl::atomic_ref<T,
               sycl::memory_order::relaxed,
               sycl::memory_scope::device,
               sycl::access::address_space::global_space>
               (globalAddr[0]);
    return v.fetch_add(val);
}
}

namespace local {

/**
 * @brief Atomic add operation in local device memory
 * 
 * @tparam T            Data type
 * @param globalAddr    Global address
 * @param val           Value to add
 * @return T            Value at address before operation
 */
template<typename T>
inline T atomicAdd( T * localAddr, T val ) {
    auto v = sycl::atomic_ref<T,
               sycl::memory_order::relaxed,
               sycl::memory_scope::work_group,
               sycl::access::address_space::local_space>
               (localAddr[0]);

    return v.fetch_add(val);
}
}


}

/**
 * @brief Host memory routines
 * 
 */
namespace host {

/**
 * @brief   Allocate memory on host
 * @note    If MEM_ALIGN macro is defined then memory will be aligned to this value
 * 
 * @tparam T    Data type
 * @param size  Size (number of elements) to allocate
 * @param q     Sycl queue
 */
template< typename T >
T * malloc( std::size_t const size, sycl::queue & q ) {
#ifdef MEM_ALIGN
    T * buffer = sycl::aligned_alloc_host<T>( MEM_ALIGN, size, q );
#else
    T * buffer = sycl::malloc_host<T>( size, q );
#endif

    return buffer;
}

/**
 * @brief   Free host allocated memory
 * 
 * @tparam T    Data type
 * @param ptr   Pointer to allocated memory
 * @param q     Sycl queue
 */
template< typename T >
void free( T * ptr, const sycl::queue & q ) {
    sycl::free( ptr, q );
}

/**
 * @brief Sets host buffer to scalar value
 * 
 * @tparam T    Data type
 * @param ptr   Pointer to data buffer
 * @param size  Buffer size (number of elements)
 * @param val   Scalar value to set (passed by copy)
 * @param q     Sycl queue
 */
template< typename T >
void zero( T * const __restrict__ data, std::size_t const size, const sycl::queue & q ) {
    q.submit([&](sycl::handler &h) {
        h.memset( (void *) data, 0, size * sizeof(T) );
    });
}

/**
 * @brief Copies data from host to device
 * 
 * @warning The code will wait for the queue to finish before submitting the memcpy action
 * 
 * @tparam T        Data type
 * @param d_out     Output device buffer
 * @param h_in      Input host buffer
 * @param size      Buffer size (number of elements)
 * @param q         Sycl queue
 */
template< typename T >
void memcpy_todevice( T * const __restrict__ d_out, T const * const __restrict__ h_in, size_t const size, sycl::queue & q ) {
    
    q.submit([&](sycl::handler &h) {
        h.memcpy(&d_out[0], h_in, size * sizeof(T));
    });
}

}

#endif
