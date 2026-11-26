#ifndef TIMER_H_
#define TIMER_H_

#include <iostream>
#include <chrono>

namespace timer {
    enum units { s, ms, us, ns };
}

class Timer {
    private:

    std::chrono::time_point<std::chrono::steady_clock> startev, stopev;

    /// Object name
    std::string name;

    public:

    /**
     * @brief Construct a new Timer object
     * 
     */
    Timer( const std::string name = "timer" ) : 
        startev(std::chrono::steady_clock::time_point::min()), 
        stopev(std::chrono::steady_clock::time_point::min()),
        name(name) {}

    /**
     * @brief Destroy the Timer object
     * 
     */
    ~Timer(){}

    /**
     * @brief Get timer resolution in ns
     * 
     */
    double resolution() {
        auto period = std::chrono::steady_clock::period();
        return ( period.num * 1.0e9 ) / ( period.den );
    }

    /**
     * @brief Starts the timer
     * 
     */
    void start() { 
        startev = std::chrono::steady_clock::now();
    }

    /**
     * @brief Stops the timer
     * 
     */
    void stop(){ 
        stopev = std::chrono::steady_clock::now();
    }

    /**
     * @brief Returns elapsed time in nanoseconds
     * 
     * @return uint64_t     Elapsed time
     */
    uint64_t elapsed(){
        uint64_t ret;

        if ( stopev < startev ) {
            std::cerr << __func__ << "(): Invalid timer, stop time is less than start time\n";
            ret = 0;
        } else {
            ret = std::chrono::duration_cast<std::chrono::nanoseconds>( stopev - startev ).count();
        }
        return ret;
    }

    /**
     * @brief Returns elapsed time in the specified units
     * 
     * @param units         Desired units (timer::s, timer::ms, timer::us or timer::ns)
     * @return double       Elapsed time
     */
    double elapsed( timer::units units ) {
        double ns = elapsed();

        double t;
        switch( units ) {
        case timer::s:  t = 1.e-9 * ns; break;
        case timer::ms: t = 1.e-6 * ns; break;
        case timer::us: t = 1.e-3 * ns; break;
        case timer::ns: t =         ns; break;
        }
        return t;
    }

    /**
     * @brief Printout timer 
     * 
     */
    void report( std::string msg = "" ) {
        auto time = elapsed( timer::ms );
        std::cout << msg << " elapsed time was " << time << " ms." << std::endl;
    }
};

#endif
