#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

namespace omp {

/**
 * @brief Atomic fetch/add operation
 * 
 * @note 
 * If OpenMP support is not enabled this just performs a standard 
 * fetch/add operation
 * 
 * @tparam T    Template data type
 * @param addr  Target value address
 * @param val   Value to be added
 * @return T    Value at address before adding val
 */
template <class T>
inline T atomic_fetch_add( T * addr, T val ) {
    T t;
    #pragma omp atomic capture
    { t = *addr; *addr += val; }
    return t;
}

/**
 * @brief OMP parallel for scheduling options
 *
 * @note Use uppercase because static and auto are C++ keywords
 * 
 */
enum class sched { Static, Dynamic, Guided, Auto };

template < omp::sched sched = omp::sched::Static, int chunk = 0, typename F >
inline void forall( unsigned int const range, F f ) {

#define __FORALL_BODY                                  \
    for( int id = 0; id < static_cast<int>(range); id++ ) \
            f( static_cast<unsigned>(id) );

    if constexpr ( sched == omp::sched::Static ) {
        if constexpr ( chunk > 0 ) {
            #pragma omp parallel for schedule(static, chunk) firstprivate(f)
            __FORALL_BODY
        } else {
            #pragma omp parallel for schedule(static) firstprivate(f)
            __FORALL_BODY
        }
    } else if constexpr ( sched == omp::sched::Dynamic ) {
        if constexpr ( chunk > 0 ) {
            #pragma omp parallel for schedule(dynamic, chunk) firstprivate(f)
            __FORALL_BODY
        } else {
            #pragma omp parallel for schedule(dynamic) firstprivate(f)
            __FORALL_BODY
        }
    } else if constexpr ( sched == omp::sched::Guided ) {
        #pragma omp parallel for schedule(guided) firstprivate(f)
        __FORALL_BODY
    } else {
        #pragma omp parallel for schedule(auto) firstprivate(f)
        __FORALL_BODY
    }
#undef __FORALL_BODY
}

}

