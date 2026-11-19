#pragma once

#include <iostream>
#include <cstdint>
#include <chrono>

namespace timer {

enum class units { s, ms, us, ns };

class clock {
    private:

    std::chrono::time_point<std::chrono::steady_clock> startev, stopev;

    /// Object name
    std::string name;

    public:

    /**
     * @brief Construct a new Timer object
     * 
     */
    clock( const std::string & name = "timer" ) : 
        startev(std::chrono::steady_clock::time_point::min()), 
        stopev(std::chrono::steady_clock::time_point::min()),
        name(name) {}

    /**
     * @brief Destroy the Timer object
     * 
     */
    ~clock(){}

    /**
     * @brief Get timer resolution in ns
     * 
     */
    double resolution() const {
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
    uint64_t elapsed() const{
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
    double elapsed( units units ) const {
        double ns = elapsed();

        double t;
        switch( units ) {
        case units::s:  t = 1.e-9 * ns; break;
        case units::ms: t = 1.e-6 * ns; break;
        case units::us: t = 1.e-3 * ns; break;
        case units::ns: t =         ns; break;
        }
        return t;
    }

    /**
     * @brief Report elapsed time in ms to cout
     * 
     * @param msg   (optional) Message to prepend report
     */
    void report( const std::string & msg = "" ) const {
        auto time = elapsed( units::ms );
        std::cout << msg << " elapsed time was " << time << " ms." << std::endl;
    }
};

}
