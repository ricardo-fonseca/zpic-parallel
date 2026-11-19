#pragma once

#include <mpi.h>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string>
#include <source_location>


#include "../core/vec_types.hpp"


namespace mpi {

/**
 * @brief stream class that prepends [ MPI rank ] to every line
 * 
 */
class mpi_ostream : private std::streambuf, public std::ostream
{   
    public:
    mpi_ostream() : std::ostream(this), new_line(true), rank(-1) {}

    private:

    bool new_line;
    int rank;

    int overflow(int c) override
    {
        if (c != std::char_traits<char>::eof() && new_line ) {
            if ( rank < 0 ) {
                if ( MPI_Comm_rank( MPI_COMM_WORLD, &rank ) != MPI_SUCCESS )
                    rank = -1;
            }
            if ( rank >= 0 ) {
                std::cout << "[" << rank << "] ";
            } else {
                std::cout << "[--] ";
            }
        }
        
        new_line = ( c == '\n' );
        std::cout.put(c);

        return std::char_traits<char>::to_int_type(c);
    }

};

/**
 * @brief std::cout replacement, prepends [ MPI rank ] to every line
 * 
 */
inline mpi_ostream cout;

/**
 * @brief Returns MPI_Datatype corresponding to C++ datatype
 * 
 * @tparam T    C++ datatype
 * @return MPI_Datatype 
 */
template< typename T > 
MPI_Datatype data_type () { 
    static_assert( sizeof(T) == 0,"No MPI data type for T"); 
    return MPI_DATATYPE_NULL;
};

// On some MPI implementations (namely OpenMPI 5.*) the MPI_* datatypes are
// not known at compile time so we cannot declare these as constexpr

template<> inline MPI_Datatype data_type<int8_t  >(void) { return MPI_INT8_T; };
template<> inline MPI_Datatype data_type<uint8_t >(void) { return MPI_UNSIGNED_CHAR; };
template<> inline MPI_Datatype data_type<int16_t >(void) { return MPI_INT16_T; };
template<> inline MPI_Datatype data_type<uint16_t>(void) { return MPI_UINT16_T; };
template<> inline MPI_Datatype data_type<int32_t >(void) { return MPI_INT32_T; };
template<> inline MPI_Datatype data_type<uint32_t>(void) { return MPI_UINT32_T; };
template<> inline MPI_Datatype data_type<int64_t >(void) { return MPI_INT64_T; };
template<> inline MPI_Datatype data_type<uint64_t>(void) { return MPI_UINT64_T; };
template<> inline MPI_Datatype data_type<float   >(void) { return MPI_FLOAT; };
template<> inline MPI_Datatype data_type<double  >(void) { return MPI_DOUBLE; };

template<> inline MPI_Datatype data_type<std::complex<float >>(void) { return MPI_C_FLOAT_COMPLEX ; };
template<> inline MPI_Datatype data_type<std::complex<double >>(void) { return MPI_C_DOUBLE_COMPLEX ; };

inline const MPI_Op sum = MPI_SUM;
inline constexpr int proc_null = MPI_PROC_NULL;

namespace type {
    inline MPI_Datatype mpi_int2    = MPI_DATATYPE_NULL;
    inline MPI_Datatype mpi_float2  = MPI_DATATYPE_NULL;
    inline MPI_Datatype mpi_float3  = MPI_DATATYPE_NULL;
    inline MPI_Datatype mpi_double3 = MPI_DATATYPE_NULL;
}

template<> inline MPI_Datatype data_type<int2 >()   { return mpi::type::mpi_int2; };
template<> inline MPI_Datatype data_type<float2 >() { return mpi::type::mpi_float2; };
template<> inline MPI_Datatype data_type<float3 >() { return mpi::type::mpi_float3; };
template<> inline MPI_Datatype data_type<double3>() { return mpi::type::mpi_double3; };

/**
 * @brief Initialize MPI environment and extra MPI types
 * 
 * @param argc      Pointer to command line argument count
 * @param argv      Pointer to command line arguments
 * @return int      MPI_SUCCESS on success, MPI_ERROR on failure
 */
inline int init( int *argc, char ***argv ) {
    #ifdef _OPENMP
    int provided;
    int ierr = MPI_Init_thread( argc, argv, MPI_THREAD_FUNNELED, &provided);
    if ( provided < MPI_THREAD_FUNNELED ) {
        std::cerr << "MPI library does not support MPI_THREAD_FUNNELED\n";
        return -1;
    }
    #else
    int ierr = MPI_Init( argc, argv );
    #endif

    if ( ierr == MPI_SUCCESS ) {
        // Initialize extra types
        MPI_Type_contiguous( 2, MPI_INT,  &mpi::type::mpi_int2 ); 
        MPI_Type_commit( &mpi::type::mpi_int2 );

        MPI_Type_contiguous( 2, MPI_FLOAT,  &mpi::type::mpi_float2 ); 
        MPI_Type_commit( &mpi::type::mpi_float2 );

        MPI_Type_contiguous( 3, MPI_FLOAT,  &mpi::type::mpi_float3 ); 
        MPI_Type_commit( &mpi::type::mpi_float3 );
        
        MPI_Type_contiguous( 3, MPI_DOUBLE, &mpi::type::mpi_double3 );
        MPI_Type_commit( &mpi::type::mpi_double3 );
    } else {
        std::cerr << "Failed to initialize MPI\n";
    }
    return ierr;
}

/**
 * @brief Finialize MPI environment
 * 
 * @return int  MPI_SUCCESS on success, MPI_ERROR on failure
 */
inline int finalize( ) {

    // These aren't strictly necessary
    if ( mpi::type::mpi_int2    != MPI_DATATYPE_NULL ) MPI_Type_free( &mpi::type::mpi_int2 );
    if ( mpi::type::mpi_float2  != MPI_DATATYPE_NULL ) MPI_Type_free( &mpi::type::mpi_float2 );
    if ( mpi::type::mpi_float3  != MPI_DATATYPE_NULL ) MPI_Type_free( &mpi::type::mpi_float3 );
    if ( mpi::type::mpi_double3 != MPI_DATATYPE_NULL ) MPI_Type_free( &mpi::type::mpi_double3 );

    return MPI_Finalize();
}

/**
 * @brief Fatal error, outputs message and aborts the code
 * 
 * @param msg       Message to print
 * @param location  (optional) source_location object, defaults to where the function was called
 */
[[noreturn]] inline void fatal(const std::string& msg, 
    const std::source_location location =
          std::source_location::current()) {
    std::cerr << "(* fatal *) " << msg << '\n'
              << "(* fatal *) " << location.file_name() << ':' << location.line()
              << " " << location.function_name() << '\n'
              << "(* fatal *) aborting..." << std::endl;
    MPI_Abort( MPI_COMM_WORLD, 1 );
    
    // unreachable, silences noreturn analysis
    std::exit(1);
}

/**
 * @brief Returns size of MPI communicator
 * 
 * @param comm  MPI communicator, defaults to MPI_COMM_WORLD
 * @return int 
 */
inline int size( MPI_Comm comm = MPI_COMM_WORLD ) {
    int size;
    if ( MPI_Comm_size( comm, &size ) != MPI_SUCCESS )
        mpi::fatal( "Unable to get communicator size" );
    return size;
}

/**
 * @brief Returns process rank
 * 
 * @param comm  MPI communicator, defaults to MPI_COMM_WORLD
 * @return int 
 */
inline int rank( MPI_Comm comm = MPI_COMM_WORLD ) {
    int rank;
    if ( MPI_Comm_rank( comm, &rank ) != MPI_SUCCESS )
        mpi::fatal( "Unable to get process rank");
    return rank;
}

/**
 * @brief Returns true if the calling node is the root node of the MPI
 *        communicator
 * 
 * @param comm   MPI communicator, defaults to MPI_COMM_WORLD
 * @return bool  1 if the calling node is the root node, 0 otherwise 
 */
inline bool root( MPI_Comm comm = MPI_COMM_WORLD ) {
    int rank;
    MPI_Comm_rank( comm, &rank );
    return rank == 0;
}

/**
 * @brief Performs an MPI_Barrier on the MPI communicator
 * 
 * @param comm   MPI communicator, defaults to MPI_COMM_WORLD
 * @return int 
 */
inline int barrier( MPI_Comm comm = MPI_COMM_WORLD ) {
    return MPI_Barrier( comm );
}


/**
 * @brief Abort the parallel code using an MPI_Abort()
 * 
 * @param errorcode     Error code to return to invoking environment
 * @param comm          MPI communicator, defaults to MPI_COMM_WORLD
 * @return int          MPI_Abort() return value (should not return)
 */
inline int abort( int errorcode, MPI_Comm comm = MPI_COMM_WORLD ) {
    return MPI_Abort( comm, errorcode );
}

}
