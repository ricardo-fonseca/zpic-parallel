#pragma once

/**
 * C++ interface to the ZDF library (serial)
 * 
 * This header has no MPI dependency. For the parallel (MPI) interface include
 * zdf-parallel.hpp instead, which also includes this file.
 * 
 * The C header is included at global scope. The names are also made available inside the
 * zdf namespace (e.g. zdf::zdf_float32) for compatibility with code that used the previous
 * version of this header.
 * 
 * All functions return 1 on success and 0 on error.
 */

#include <string>
#include <complex>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "zdf.h"

// Forward declaration for the util::complex64 type
namespace util { class complex64; }

namespace zdf {

// C interface names, also available in the zdf namespace

using ::zdf_magic;

using ::zdf_data_type;
using ::t_zdf_data_type;
using ::zdf_null;
using ::zdf_int8;
using ::zdf_uint8;
using ::zdf_int16;
using ::zdf_uint16;
using ::zdf_int32;
using ::zdf_uint32;
using ::zdf_int64;
using ::zdf_uint64;
using ::zdf_float32;
using ::zdf_float64;
using ::zdf_complex64;
using ::zdf_complex128;

using ::zdf_file_access_mode;
using ::ZDF_CREATE;
using ::ZDF_READ;
using ::ZDF_UPDATE;

using ::zdf_axis_type;
using ::zdf_linear;
using ::zdf_log10;
using ::zdf_log2;

using ::t_zdf_file;
using ::t_zdf_dataset;
using ::t_zdf_chunk;
using ::t_zdf_grid_axis;
using ::t_zdf_grid_info;
using ::t_zdf_iteration;
using ::t_zdf_part_info;
using ::t_zdf_track_info;

using ::create_path;
using ::zdf_sizeof;
using ::zdf_sizeof_vector;
using ::zdf_open_file;
using ::zdf_close_file;
using ::zdf_vector_write;
using ::zdf_add_string;
using ::zdf_add_int32;
using ::zdf_add_double;
using ::zdf_add_iteration;
using ::zdf_add_grid_info;
using ::zdf_add_part_info;
using ::zdf_add_track_info;
using ::zdf_add_dataset;
using ::zdf_dataset_header_write;
using ::zdf_start_cdset;
using ::size_zdf_chunk_header;
using ::zdf_write_chunk_header;
using ::zdf_write_cdset;
using ::zdf_end_cdset;
using ::zdf_open_dataset;
using ::zdf_extend_dataset;
using ::zdf_open_grid_file;
using ::zdf_save_grid;
using ::zdf_open_part_file;
using ::zdf_add_quant_part_file;


// Short type names

using grid_info = t_zdf_grid_info;
using grid_axis = t_zdf_grid_axis;

using iteration = t_zdf_iteration;
using part_info = t_zdf_part_info;

using file = t_zdf_file;
using dataset  = t_zdf_dataset;

using chunk = t_zdf_chunk;


// Mapping between C++ types and ZDF data types

template< typename T > 
constexpr t_zdf_data_type data_type () { return zdf_null; };

template<> constexpr t_zdf_data_type data_type<int8_t  >() { return zdf_int8;    };
template<> constexpr t_zdf_data_type data_type<uint8_t >() { return zdf_uint8;   };
template<> constexpr t_zdf_data_type data_type<int16_t >() { return zdf_int16;   };
template<> constexpr t_zdf_data_type data_type<uint16_t>() { return zdf_uint16;  };
template<> constexpr t_zdf_data_type data_type<int32_t >() { return zdf_int32;   };
template<> constexpr t_zdf_data_type data_type<uint32_t>() { return zdf_uint32;  };
template<> constexpr t_zdf_data_type data_type<int64_t >() { return zdf_int64;   };
template<> constexpr t_zdf_data_type data_type<uint64_t>() { return zdf_uint64;  };
template<> constexpr t_zdf_data_type data_type<float   >() { return zdf_float32; };
template<> constexpr t_zdf_data_type data_type<double  >() { return zdf_float64; };

// Complex datatypes (std::complex)
template<> constexpr t_zdf_data_type data_type<std::complex<float >>() { return zdf_complex64; };
template<> constexpr t_zdf_data_type data_type<std::complex<double>>() { return zdf_complex128; };

// Complex datatypes (util)
template<> constexpr t_zdf_data_type data_type<util::complex64>() { return zdf_complex64; };

/**
 * @brief Save grid data to disk
 * 
 * @tparam T        Datatype
 * @param buffer    Data buffer
 * @param info      Data information
 * @param iter      Iteration information
 * @param path      File path (file name is built from data information)
 * @return int      1 on success, 0 on error
 */
template< typename T>
int save_grid( T *buffer, const t_zdf_grid_info &info, const t_zdf_iteration &iter, const std::string & path ) {
    static_assert( data_type<T>() != zdf_null, "Unsupported data type");
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
 * @return int      1 on success, 0 on error
 */
template< typename T >
int save_grid( T *buffer, unsigned ndims, const uint64_t dims[], 
               const std::string & name, const std::string & filename )
{
    static_assert( data_type<T>() != zdf_null, "Unsupported data type");

    if ( ndims < 1 || ndims > zdf_max_dims ) {
        std::cerr << "(*error*) Invalid number of dimensions, aborting.\n";
        return 0;
    }

    // Create ZDF file
    t_zdf_file zdf;
    if ( !zdf_open_file( &zdf, filename.c_str(), ZDF_CREATE ) ) {
        std::cerr << "(*error*) Unable to open ZDF file, aborting.\n";
        return 0;
    }

    // Grid info
    t_zdf_grid_info info;
    info.name  = name.c_str();
    info.ndims = ndims;
    for( unsigned i = 0; i < ndims; i++ ) info.count[i] = dims[i];
    info.label = nullptr;
    info.units = nullptr;
    info.axis  = nullptr;

    // Dataset
    t_zdf_dataset dataset;
    dataset.name = info.name;
    dataset.data_type = data_type<T>();
    dataset.ndims = info.ndims;
    dataset.data = (void *) buffer;
    for( unsigned i = 0; i < info.ndims; i ++) dataset.count[i] = info.count[i];

    // Add file type, grid info and dataset (no iteration data)
    int ok = zdf_add_string( &zdf, "TYPE", "grid") &&
             zdf_add_grid_info( &zdf, &info ) &&
             zdf_add_dataset( &zdf, &dataset );

    if ( !ok ) std::cerr << "(*error*) Unable to write ZDF file " << filename << '\n';

    // Close ZDF file and return
    int closed = zdf_close_file( &zdf );
    return ok && closed;
}

/**
 * @brief Open particle data file
 * 
 * @param file          File information
 * @param info          Particle data information
 * @param iteration     Interation information
 * @param path          File path (file name is built from particle data information)
 * @return int          1 on success, 0 on error
 */
inline
int open_part_file( t_zdf_file &file, t_zdf_part_info &info, const t_zdf_iteration &iteration, const std::string & path ) {
    return zdf_open_part_file( &file, &info, &iteration, path.c_str() );
}

inline
int add_quant_part_file( t_zdf_file &file, const std::string & name, float * const data, const uint64_t np ) {
    return zdf_add_quant_part_file( &file, name.c_str(), data, np );
}

inline
int close_file( t_zdf_file &file ) { return zdf_close_file(&file);};


// End namespace zdf
}
