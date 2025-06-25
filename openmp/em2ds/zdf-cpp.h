#ifndef ZDF_CPP_H_
#define ZDF_CPP_H_

#include <string>

namespace zdf {

#include "zdf.h"

using grid_info = t_zdf_grid_info;
using grid_axis = t_zdf_grid_axis;

using iteration = t_zdf_iteration;
using part_info = t_zdf_part_info;

using file = t_zdf_file;
using dataset  = t_zdf_dataset;

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

// Complex datatypes
template<> constexpr t_zdf_data_type data_type<std::complex<float >>() { return zdf_complex64; };
template<> constexpr t_zdf_data_type data_type<std::complex<double>>() { return zdf_complex128; };

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

// End namespace zdf
}

#endif
