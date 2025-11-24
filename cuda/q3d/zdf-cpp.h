#ifndef ZDF_CPP_H_
#define ZDF_CPP_H_

#include <string>
#include <iostream>

namespace zdf {

#include "zdf.h"
#include "complex.h"

using grid_info = t_zdf_grid_info;
using grid_axis = t_zdf_grid_axis;

using iteration = t_zdf_iteration;
using part_info = t_zdf_part_info;

using file = t_zdf_file;
using dataset  = t_zdf_dataset;

using chunk = t_zdf_chunk;

#ifdef USE_MPI
#include "zdf-parallel.h"
using par_file = t_zdf_par_file;
#endif

template< typename T > 
constexpr t_zdf_data_type data_type () { return zdf_null; };

template<> constexpr t_zdf_data_type data_type<int8_t  >() { return zdf_int8;    };
template<> constexpr t_zdf_data_type data_type<uint8_t >() { return zdf_uint8;   };
template<> constexpr t_zdf_data_type data_type<int16_t >() { return zdf_int16;   };
template<> constexpr t_zdf_data_type data_type<uint16_t>() { return zdf_uint16;  };
template<> constexpr t_zdf_data_type data_type<int32_t >() { return zdf_int32;   };
template<> constexpr t_zdf_data_type data_type<uint32_t>() { return zdf_uint32;  };
template<> constexpr t_zdf_data_type data_type<float   >() { return zdf_float32; };
template<> constexpr t_zdf_data_type data_type<double  >() { return zdf_float64; };

// Complex datatypes
template<> constexpr t_zdf_data_type data_type<ops::complex<float >>() { return zdf_complex64; };
template<> constexpr t_zdf_data_type data_type<ops::complex<double>>() { return zdf_complex128; };

/**
 * @brief Save grid data to disk
 * 
 * @tparam T        Datatype
 * @param buffer    Data buffer
 * @param info      Data information
 * @param iter      Iteration information
 * @param path      File path (file name is built from data information)
 * @return int      0 on success, -1 on error
 */
template< typename T>
int save_grid( T *buffer, t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {
    // static_assert( data_type<T>() != zdf_null, "Unsupported data type");
    return zdf_save_grid( (void*) buffer, data_type<T>(), &info, &iter, path.c_str() );
}

/**
 * @brief Save grid data to disk, minimal metadata
 * 
 * @tparam T        Datatype
 * @param buffer    Data buffer
 * @param ndims     Number of dimensions
 * @param dims      Grid dimensions
 * @param name      Grid name
 * @param filename  File name
 * @return int      0 on success, -1 on error
 */
template< typename T >
int save_grid( T *buffer, unsigned ndims, uint64_t dims[], 
               std::string name, std::string filename )
{
    if( data_type<T>() == zdf_null ) {
        std::cerr << "(*error*) Unsupported datatype, aborting.\n";
        return(-1);
    }

    // Create ZDF file
    t_zdf_file zdf;
    if ( !zdf_open_file( &zdf, filename.c_str(), ZDF_CREATE ) ) {
        std::cerr << "(*error*) Unable to open ZDF file, aborting.\n";
        return(-1);
    }

    // Add file type
    if ( !zdf_add_string( &zdf, "TYPE", "grid") ) return(0);

    // Add grid info
    t_zdf_grid_info info;
    info.name  = (char *) name.c_str();
    info.ndims = ndims;
    for( unsigned i = 0; i < ndims; i++ ) info.count[i] = dims[i];
    info.label = nullptr;
    info.units = nullptr;
    info.axis  = nullptr;

    if ( !zdf_add_grid_info( &zdf, &info ) ) return(0);

    // No iteration data

    // Add dataset
    t_zdf_dataset dataset;
    dataset.name = info.name;
    dataset.data_type = data_type<T>();
    dataset.ndims = info.ndims;
    dataset.data = (void *) buffer;
    for( unsigned i = 0; i < info.ndims; i ++) dataset.count[i] = info.count[i];

    if ( !zdf_add_dataset( &zdf, &dataset ) ) return(0);

    // Close ZDF file and return
    return( zdf_close_file( &zdf ) );
}

/**
 * @brief Open particle data file
 * 
 * @param file          File information
 * @param info          Particle data information
 * @param iteration     Interation information
 * @param path          File path (file name is built from particle data information)
 * @return int          0 on success, -1 on error
 */
static inline
int open_part_file( t_zdf_file &file, t_zdf_part_info &info, t_zdf_iteration &iteration, std::string path ) {
    return zdf_open_part_file( &file, &info, &iteration, path.c_str() );
}

static inline
int add_quant_part_file( t_zdf_file &file, std::string name, float * const data, const uint64_t np ) {
    return zdf_add_quant_part_file( &file, name.c_str(), data, np );
}

static inline
int close_file( t_zdf_file &file ) { return zdf_close_file(&file);};


// Parallel interfaces
#ifdef USE_MPI

/**
 * @brief Saves a distributed parallel grid in a zdf file
 * 
 * @tparam T            Template datatype
 * @param chunk         Local data
 * @param info          Global grid metadata
 * @param iter          Iteration metadata
 * @param path          Base path
 * @param comm          MPI communicator
 * @param par_io_mode   Parallel I/O mode, defaults to ZDF_MPI
 * @return int 
 */
template< typename T>
int save_grid( t_zdf_chunk &chunk, t_zdf_grid_info &info,
    t_zdf_iteration &iter, std::string path, MPI_Comm comm,
    t_zdf_parallel_io_mode par_io_mode = ZDF_MPI )
{
    static_assert( data_type<T>() != zdf_null, "Unsupported data type");
    return zdf_par_save_grid( &chunk, data_type<T>(), &info, &iter, path.c_str(), comm, par_io_mode );
}

/**
 * @brief Saves a distributed parallel grid in a zdf file, minimal metadata
 * 
 * @tparam T            Template datatype
 * @param buffer        Data buffer   
 * @param ndims         Number of dimensions
 * @param global        Global grid size
 * @param start         Position of local chunck on global grid
 * @param local         Local grid size
 * @param name          Grid name (metadata)
 * @param filename      File name
 * @param comm          MPI communicator
 * @param par_io_mode   Parallel I/O mode, defaults to ZDF_MPI
 * @return int 
 */
template< typename T >
int save_grid( T *buffer, unsigned ndims,
               uint64_t global[], uint64_t start[], uint64_t local[], 
               std::string name, std::string filename, MPI_Comm comm,
    t_zdf_parallel_io_mode par_io_mode = ZDF_MPI )
{
    static_assert( data_type<T>() != zdf_null, "Unsupported data type");

    t_zdf_grid_info info;
    info.name  = (char *) name.c_str();
    info.ndims = ndims;
    for( unsigned i = 0; i < ndims; i++ ) info.count[i] = global[i];
    info.label = nullptr;
    info.units = nullptr;
    info.axis  = nullptr;

    t_zdf_chunk chunk;
    chunk.data = (void *) buffer;
    for( unsigned i = 0; i < ndims; i++ ) {
        chunk.count[i] = local[i];
        chunk.start[i] = start[i];
        chunk.stride[i] = 1;
    }

    t_zdf_par_file zdf;
    if ( !zdf_par_open_file( &zdf, filename.c_str(), ZDF_CREATE, comm, par_io_mode ) ) {
        std::cerr << "(*error*) Unable to open parallel ZDF file, aborting.\n";
        return(-1);
    }

    // Add file type
    if ( !zdf_par_add_string( &zdf, "TYPE", "grid") ) return(0);

    // No iteration info

    // Add grid info
    if ( !zdf_par_add_grid_info( &zdf, &info ) ) return(0);

    // Add chunked dataset header
    t_zdf_dataset dataset;
    dataset.name = (char *) name.c_str();
    dataset.data_type = data_type<T>();
    dataset.ndims = ndims;
    for( unsigned i = 0; i < ndims; i ++) dataset.count[i] = global[i];

    if ( !zdf_par_start_cdset( &zdf, &dataset ) ) return(0);

    // Write data
    zdf_par_write_cdset( &zdf, &dataset, &chunk, -1 );

    // Close dataset
    zdf_par_end_cdset( &zdf, &dataset );

    // Close ZDF file and return
    return( zdf_par_close_file( &zdf ) );
}

static inline
int open_part_file( t_zdf_par_file &file, t_zdf_part_info &info, t_zdf_iteration &iteration, std::string path,
    MPI_Comm comm, t_zdf_parallel_io_mode par_io_mode = ZDF_MPI ) {
    
    // Create base path if necessary
    int rank; 
    MPI_Comm_rank( comm, & rank );
    if ( rank == 0 ) create_path( path.c_str() );
    MPI_Barrier( comm );

    // Prepare filename
    char filename[1025];
    snprintf( filename, 1024, "%s/%s-%s-%06u.zdf", path.c_str(), "particles", info.name, (unsigned) iteration.n );

    // Create file
    if ( !zdf_par_open_file( &file, filename, ZDF_CREATE, comm, par_io_mode ) ) {
        std::cerr << "(*error*) Unable to open parallel ZDF file, aborting.\n";
        return(-1);
    }

    // Add file type
    if ( !zdf_par_add_string( &file, "TYPE", "particles") ) return(0);

    // Add particle info
    if ( !zdf_par_add_part_info( &file, &info ) ) return(0);

    // Add iteration info
    if ( !zdf_par_add_iteration( &file, &iteration ) ) return(0);

    return(1);
}

static inline
int close_file( t_zdf_par_file &file ) {
    return zdf_par_close_file( & file );
};

static inline
int start_cdset( t_zdf_par_file &file, t_zdf_dataset & dataset ) {
    return zdf_par_start_cdset( & file, & dataset );
}

static inline
int write_cdset( t_zdf_par_file &file, t_zdf_dataset & dataset, t_zdf_chunk & chunk, const int64_t offset ) {
    return zdf_par_write_cdset( &file, &dataset, &chunk, offset );
}

static inline
int end_cdset( t_zdf_par_file &file, t_zdf_dataset & dataset ) {
    return zdf_par_end_cdset( & file, & dataset );
}
#endif

// End namespace zdf
}

#endif
