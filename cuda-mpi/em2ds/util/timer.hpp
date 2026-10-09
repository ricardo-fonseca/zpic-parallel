#pragma once

#include <iostream>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "../core/gpu.cuh"

namespace timer {

enum class units { s, ms, us, ns };

/**
 * @brief GPU timer based on CUDA events
 * 
 * Usage: start() -> stop() -> elapsed() / report(). Times are measured on
 * the GPU timeline of the stream(s) where the events are recorded.
 * 
 * @note Under MPI each rank times its own GPU
 */
class clock {
    private:

    /// @brief Timer state
    enum class state { idle, running, stopped, done };

    /// @brief Start / stop events
    cudaEvent_t startev, stopev;

    /// @brief Current state
    /// @note mutable so that elapsed() can be const (it only records that the
    ///       stop event has completed)
    mutable state status;

    /// @brief Object name
    std::string name;

    public:

    /**
     * @brief Construct a new Timer object
     * 
     * @param name      Timer name (used in reports and error messages)
     */
    clock( const std::string & name = "timer" ) : 
        status( state::idle ), name( name ) {

        gpu::check_err( 
            cudaEventCreate( &startev ),
            "unable to create start event"
        );
        
        gpu::check_err(
            cudaEventCreate( &stopev ),
            "unable to create stop event"
        );
    }

    // Owns CUDA events, copying would destroy them twice
    clock( const clock& ) = delete;
    clock& operator=( const clock& ) = delete;

    /**
     * @brief Destroy the Timer object
     * 
     * @note Errors are reported but not fatal: a timer may be destroyed after
     *       the CUDA runtime has been shut down (e.g. a static object)
     */
    ~clock(){
        for( cudaEvent_t ev : { startev, stopev } ) {
            cudaError_t err = cudaEventDestroy( ev );
            if ( err != cudaSuccess && err != cudaErrorCudartUnloading ) {
                std::cerr << "(*warning*) " << name << ": unable to destroy event ("
                          << cudaGetErrorString( err ) << ")\n";
            }
        }
    }

    /**
     * @brief Get timer resolution in ms
     * 
     * @warning Value taken from CUDA documentation
     */
    double resolution() const {
        /**
         * From the cudaEventElapsedTime() documentation:
         * 
         * " Computes the elapsed time between two events (in milliseconds with
         *   a resolution of around 0.5 microseconds)."
         * 
         */

        return 0.5e-3;
    }

    /**
     * @brief Starts the timer
     * 
     * @note Restarts the timer if it was already running
     * 
     * @param stream    Stream on which to record the start event (defaults to
     *                  the default stream)
     */
    void start( cudaStream_t stream = 0 ) { 
        gpu::check_err( 
            cudaEventRecord( startev, stream ),
            "unable to record start event"
        );
        status = state::running;
    }

    /**
     * @brief Stops the timer
     * 
     * @note Does nothing (other than reporting an error) if the timer is not
     *       running
     * 
     * @param stream    Stream on which to record the stop event (defaults to
     *                  the default stream)
     */
    void stop( cudaStream_t stream = 0 ){ 
        if ( status != state::running ) {
            std::cerr << "(*error*) " << name << ": stop() called on a timer that is not running\n";
            return;
        }
        gpu::check_err( 
            cudaEventRecord( stopev, stream ),
            "unable to record stop event"
        );
        status = state::stopped;
    }

    /**
     * @brief Returns elapsed time in nanoseconds
     * 
     * @note Blocks until the stop event has completed
     * 
     * @return uint64_t     Elapsed time (0 if the timer was not complete)
     */
    uint64_t elapsed() const {

        if ( status == state::idle || status == state::running ) {
            std::cerr << "(*error*) " << name << ": timer was not complete\n";
            return 0;
        }

        if ( status == state::stopped ) {
            // Wait for stop event to complete
            gpu::check_err( 
                cudaEventSynchronize( stopev ),
                "unable to synchronize stop event"
            );
            status = state::done;
        }

        float delta;
        gpu::check_err( 
            cudaEventElapsedTime( &delta, startev, stopev ),
            "unable to get elapsed time"
        );
        return static_cast<uint64_t>( 1.e6 * delta );
    }

    /**
     * @brief Returns elapsed time in the specified units
     * 
     * @param u             Desired units (timer::units::s, ::ms, ::us or ::ns)
     * @return double       Elapsed time
     */
    double elapsed( const units u ) const {
        const double ns = elapsed();

        switch( u ) {
        case units::s:  return 1.e-9 * ns;
        case units::ms: return 1.e-6 * ns;
        case units::us: return 1.e-3 * ns;
        case units::ns: return         ns;
        }
        return ns;
    }

    /**
     * @brief Report elapsed time in ms to cout
     * 
     * @param msg   (optional) Message to prepend report, defaults to the
     *              timer name
     */
    void report( const std::string & msg = "" ) const {
        std::cout << ( msg.empty() ? name : msg ) << " elapsed time was " << *this << "." << std::endl;
    }

    /**
     * @brief Stream insertion, prints the elapsed time in ms (e.g. "1.5 ms")
     * 
     * @note Blocks until the stop event has completed
     * 
     * @param os        Output stream
     * @param obj       Timer object
     * @return std::ostream& 
     */
    friend std::ostream& operator<<( std::ostream& os, const clock & obj ) {
        return os << obj.elapsed( units::ms ) << " ms";
    }
};

}
