#pragma once

/**
 * C++ interface to the ZDF library (parallel / MPI)
 * 
 * Includes the serial interface (zdf.hpp) and adds the parallel functions. All parallel
 * functions are collective over the communicator and return the same value on all ranks:
 * 1 on success, 0 on error.
 * 
 * Data buffers must be in host memory.
 */

#include "zdf.hpp"
#include "zdf-parallel.h"

#include <mpi.h>

namespace zdf {

// C interface names, also available in the zdf namespace

using ::zdf_parallel_io_mode;
using ::t_zdf_parallel_io_mode;
using ::ZDF_MPI;
using ::ZDF_INDEPENDENT;
using ::ZDF_MPIIO_INDEPENDENT;
using ::ZDF_MPIIO_COLLECTIVE;
using ::t_zdf_par_file;

using ::zdf_par_create_path;
using ::zdf_par_open_file;
using ::zdf_par_close_file;
using ::zdf_par_add_string;
using ::zdf_par_add_int32;
using ::zdf_par_add_double;
using ::zdf_par_add_iteration;
using ::zdf_par_add_grid_info;
using ::zdf_par_add_part_info;
using ::zdf_par_getoffset;
using ::zdf_par_start_cdset;
using ::zdf_par_write_cdset;
using ::zdf_par_end_cdset;
using ::zdf_par_extend_dataset;
using ::zdf_par_save_grid;

// Short type names

using par_file = t_zdf_par_file;

// Parallel interfaces
// All parallel functions are collective and return the same value on all ranks.

/**
 * @brief Saves a distributed parallel grid in a zdf file
 * 
 * @tparam T        Datatype (must be specified explicitly, e.g. save_grid<float>( ... ) )
 * @param chunk     Local data grid information and data
 * @param info      Global data grid information
 * @param iter      Iteration metadata
 * @param path      Base path to save the file
 * @param comm      MPI communicator
 * @param io_mode   Parallel I/O algorithm to use, defaults to ZDF_MPI, see zdf
 *                  documentation for details
 * @return int      Returns 1 on success, 0 on error
 */
template< typename T>
int save_grid( t_zdf_chunk &chunk, const t_zdf_grid_info &info,
    const t_zdf_iteration &iter, const std::string & path, const MPI_Comm comm,
    const t_zdf_parallel_io_mode par_io_mode = ZDF_MPI )
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
 * @return int          1 on success, 0 on error
 */
template< typename T >
int save_grid( 
    T *buffer, const unsigned ndims,
    const uint64_t global[], const uint64_t start[], const uint64_t local[], 
    const std::string & name, const std::string & filename, const MPI_Comm comm,
    const t_zdf_parallel_io_mode par_io_mode = ZDF_MPI )
{
    static_assert( data_type<T>() != zdf_null, "Unsupported data type");

    int rank; MPI_Comm_rank( comm, &rank );

    if ( ndims < 1 || ndims > zdf_max_dims ) {
        if ( rank == 0 ) std::cerr << "(*error*) Invalid number of dimensions, aborting.\n";
        return 0;
    }

    t_zdf_grid_info info;
    info.name  = name.c_str();
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
        if ( rank == 0 ) std::cerr << "(*error*) Unable to open parallel ZDF file, aborting.\n";
        return 0;
    }

    t_zdf_dataset dataset;
    dataset.name = name.c_str();
    dataset.data_type = data_type<T>();
    dataset.ndims = ndims;
    dataset.data = nullptr;
    for( unsigned i = 0; i < ndims; i ++) dataset.count[i] = global[i];

    // Add file type, grid info (no iteration info) and data. All calls are collective
    // and return the same value on all ranks.
    int ok = zdf_par_add_string( &zdf, "TYPE", "grid") &&
             zdf_par_add_grid_info( &zdf, &info ) &&
             zdf_par_start_cdset( &zdf, &dataset ) &&
             zdf_par_write_cdset( &zdf, &dataset, &chunk, -1 ) &&
             zdf_par_end_cdset( &zdf, &dataset );

    if ( !ok && rank == 0 ) std::cerr << "(*error*) Unable to write parallel ZDF file " << filename << '\n';

    // Close ZDF file and return
    int closed = zdf_par_close_file( &zdf );
    return ok && closed;
}

/**
 * @brief Open parallel particle data file
 * 
 * @param file          Parallel file handle
 * @param info          Particle data information
 * @param iteration     Iteration information
 * @param path          File path (file name is built from particle data information)
 * @param comm          MPI communicator
 * @param par_io_mode   Parallel I/O mode, defaults to ZDF_MPI
 * @return int          1 on success, 0 on error
 */
inline
int open_part_file( t_zdf_par_file &file, t_zdf_part_info &info, const t_zdf_iteration &iteration,
    const std::string & path, MPI_Comm comm, t_zdf_parallel_io_mode par_io_mode = ZDF_MPI ) {
    
    int rank; MPI_Comm_rank( comm, &rank );

    // Create base path if necessary
    if ( !zdf_par_create_path( path.c_str(), comm ) ) return 0;

    // Prepare filename
    char filename[1024];
    int n = snprintf( filename, sizeof(filename), "%s/%s-%s-%06u.zdf", path.c_str(), "particles", info.name, (unsigned) iteration.n );
    if ( n < 0 || (size_t) n >= sizeof(filename) ) {
        if ( rank == 0 ) std::cerr << "(*error*) ZDF file name is too long.\n";
        return 0;
    }

    // Create file
    if ( !zdf_par_open_file( &file, filename, ZDF_CREATE, comm, par_io_mode ) ) {
        if ( rank == 0 ) std::cerr << "(*error*) Unable to open parallel ZDF file, aborting.\n";
        return 0;
    }

    // Add file type, particle info and iteration info
    int ok = zdf_par_add_string( &file, "TYPE", "particles") &&
             zdf_par_add_part_info( &file, &info ) &&
             zdf_par_add_iteration( &file, &iteration );

    if ( !ok ) {
        if ( rank == 0 ) std::cerr << "(*error*) Unable to write metadata to parallel ZDF file " << filename << '\n';
        zdf_par_close_file( &file );
        return 0;
    }

    return 1;
}

/**
 * @brief Closes parallel zdf file
 * 
 * @param file      Parallel file information
 * @return int 
 */
inline
int close_file( t_zdf_par_file &file ) {
    return zdf_par_close_file( & file );
};

inline
int start_cdset( t_zdf_par_file &file, t_zdf_dataset & dataset ) {
    return zdf_par_start_cdset( & file, & dataset );
}

inline
int write_cdset( t_zdf_par_file &file, t_zdf_dataset & dataset, t_zdf_chunk & chunk, const int64_t offset = -1 ) {
    return zdf_par_write_cdset( &file, &dataset, &chunk, offset );
}

inline
int end_cdset( t_zdf_par_file &file, t_zdf_dataset & dataset ) {
    return zdf_par_end_cdset( & file, & dataset );
}

inline
int extend_dataset( t_zdf_par_file &file, t_zdf_dataset & dataset, uint64_t new_count[] ) {
    return zdf_par_extend_dataset( & file, & dataset, new_count );
}

// End namespace zdf
}
