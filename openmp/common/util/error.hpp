#pragma once

#include <iostream>
#include <string>
#include <cstdlib>
#include <source_location>

namespace util {

/**
 * @brief Fatal error, outputs message and aborts the code
 * 
 * @note Serial / OpenMP replacement for mpi::fatal(). The program is
 *       terminated with std::abort(), not std::exit(): this may be called
 *       from inside an OpenMP parallel region, where std::exit() is unsafe
 *       (static destructors would run while other threads are still active).
 *       If several threads fail at once, only the first one prints a message.
 * 
 * @param msg       Message to print
 * @param location  (optional) source_location object, defaults to where the function was called
 */
[[noreturn]] inline void fatal(const std::string& msg, 
    const std::source_location location =
          std::source_location::current()) {

    #pragma omp critical (util_fatal)
    {
        std::cerr << "(* fatal *) " << msg << '\n'
                  << "(* fatal *) " << location.file_name() << ':' << location.line()
                  << " " << location.function_name() << '\n'
                  << "(* fatal *) aborting..." << std::endl;
        std::abort();
    }

    // unreachable, silences noreturn analysis
    std::abort();
}

}
