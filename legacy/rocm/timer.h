#ifndef TIMER_H_
#define TIMER_H_

#include <iostream>
#include <hip/hip_runtime.h>

namespace timer {
    enum units { s, ms, us, ns };
}

class Timer {
    private:

    hipEvent_t startev, stopev;
    int status;

    /// Object name
    std::string name;

    public:

    __host__
    /**
     * @brief Construct a new Timer object
     * 
     */
    Timer( const std::string name = "timer" ) : name(name) {
        auto err = hipEventCreate( &startev );
        CHECK_ERR( err, "Unable to create start event");

        err = hipEventCreate( &stopev );
        CHECK_ERR( err, "Unable to create stop event");

        status = -1;
    }

    /**
     * @brief Destroy the Timer object
     * 
     */
    ~Timer(){
        auto err = hipEventDestroy( startev );
        CHECK_ERR( err, "Unable to destroy start event");

        err = hipEventDestroy( stopev );
        CHECK_ERR( err, "Unable to destroy stop event");
    }

    /**
     * @brief Get timer resolution in ms
     * 
     * @warning Value taken from hip documentation
     */
    constexpr float resolution() {
        
        /**
         * From the hipEventElapsedTime() documentation:
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
     */
    void start() { 
        status = 0;
        auto err = hipEventRecord( startev );
    }

    /**
     * @brief Stops the timer
     * 
     */
    void stop(){ 
        auto err = hipEventRecord( stopev );
        if ( ! status ) {
            status = 1;
        } else {
            std::cerr << "(*error*) Timer was not started\n";
        }
    }

    /**
     * @brief Returns elapsed time in milliseconds
     * 
     * @return float     Elapsed time
     */
    float elapsed(){
        if ( status < 1 ) {
            std::cerr << "(*error*) Timer was not complete\n";
            return -1;
        } else {
            if ( status == 1 ) {
                auto err = hipEventSynchronize( stopev );
                status = 2;
            }
            float delta;
            auto err = hipEventElapsedTime(&delta, startev, stopev );
            return delta;
        }
    }

    /**
     * @brief Returns elapsed time in the specified units
     * 
     * @param units         Desired units (timer::s, timer::ms, timer::us or timer::ns)
     * @return double       Elapsed time
     */
    double elapsed( timer::units units ) {
        double ms = elapsed();

        double t;
        switch( units ) {
        case timer::s:  t = 1.e-3 * ms; break;
        case timer::ms: t =         ms; break;
        case timer::us: t = 1.e+3 * ms; break;
        case timer::ns: t = 1.e+6 * ms; break;
        }
        return t;
    }

    /**
     * @brief Printout timer 
     * 
     */
    void report( std::string msg = "" ) {
        if ( status < 1 ) {
            std::cout << msg << " timer is not complete." << std::endl;
        } else {
            auto time = elapsed();
            std::cout << msg << " elapsed time was " << time << " ms." << std::endl;
        }
    }
};

#endif
