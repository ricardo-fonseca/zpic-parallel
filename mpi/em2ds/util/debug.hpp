#pragma once

#include <execinfo.h>
#include <iostream>
#include <cstdlib>
#include <source_location>

#include "../parallel/mpi.hpp"

namespace debug {

/**
 * @brief Print the callstack
 * 
 */
inline void stack_trace() {
    const int maxFrames = 64;
    void* addrlist[maxFrames];

    auto addrlen = backtrace(addrlist, maxFrames);
    if (addrlen == 0) {
        std::cerr << "No stack frames found.\n";
    } else {
        std::cout << "Stack trace:\n";
        char** symbols = backtrace_symbols(addrlist, addrlen);
        for (int i = 0; i < addrlen; ++i) {
            std::cout << symbols[i] << '\n';
        }
        free(symbols);
    }
}

[[noreturn]] inline void not_implemented( 
        const std::source_location location =
        std::source_location::current() ) {
    std::cerr << "(* fatal *) " << location.file_name() << ':' << location.line()
              << " " << location.function_name()
              << " () not implemented yet, aborting...";
    mpi::abort(1);
}

}
