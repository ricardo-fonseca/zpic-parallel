#ifndef ZDF_CPP_H_
#define ZDF_CPP_H_

#include <string>

namespace zdf {

#include "zdf.h"
#include "zdf-parallel.h"

using grid_info = t_zdf_grid_info;
using grid_axis = t_zdf_grid_axis;

using iteration = t_zdf_iteration;
using part_info = t_zdf_part_info;

using file = t_zdf_file;
using par_file = t_zdf_par_file;

using chunk = t_zdf_chunk;

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


template< typename T>
int save_grid( T *buffer, t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {
    static_assert( data_type<T>() != zdf_null, "Unsupported data type");
    return zdf_save_grid( (void*) buffer, data_type<T>(), &info, &iter, path.c_str() );
}


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

/**
 * @brief Saves a distributed parallel grid in a zdf file
 * 
 * @param buffer    Data to save (float)
 * @param chunk     Local data grid information
 * @param info      Global data grid information
 * @param iter      Iteration metadata
 * @param path      Base path to save the file
 * @param comm      MPI communicator
 * @param io_mode   Parallel I/O algorithm to use, defaults to ZDF_MPI, see zdf
 *                  documentation for details
 * @return int      Returns 1 on success, 0 on error
 */
template< typename T>
int save_grid( t_zdf_chunk &chunk, t_zdf_grid_info &info,
    t_zdf_iteration &iter, std::string path, MPI_Comm comm,
    t_zdf_parallel_io_mode par_io_mode = ZDF_MPI )
{
    static_assert( data_type<T>() != zdf_null, "Unsupported data type");
    return zdf_par_save_grid( &chunk, data_type<T>(), &info, &iter, path.c_str(), comm, par_io_mode );
}

template< typename T >
int save_grid( T *buffer, unsigned ndims,
               uint64_t global[], uint64_t start[], uint64_t local[], 
               std::string name, std::string filename, MPI_Comm comm,
    t_zdf_parallel_io_mode par_io_mode = ZDF_MPI )
{
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


// End namespace zdf
}

#endif
