#pragma once

#include <string>

/**
 * @brief ANSI escape codes for console output
 * 
 */
namespace ansi {
    inline const std::string bold(  "\033[1m" );
    inline const std::string reset( "\033[0m" );

    inline const std::string black   ( "\033[30m" );
    inline const std::string red     ( "\033[31m" );
    inline const std::string green   ( "\033[32m" );
    inline const std::string yellow  ( "\033[33m" );
    inline const std::string blue    ( "\033[34m" );
    inline const std::string magenta ( "\033[35m" );
    inline const std::string cyan    ( "\033[36m" );
    inline const std::string white   ( "\033[37m" );
}
