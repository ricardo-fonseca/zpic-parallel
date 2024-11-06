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

static inline
int save_grid( float *buffer, t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {
    return zdf_save_grid( (void*) buffer, zdf_float32, &info, &iter, path.c_str() );
}

static inline
int save_grid( double *buffer, t_zdf_grid_info &info, t_zdf_iteration &iter, std::string path ) {
    return zdf_save_grid( (void*) buffer, zdf_float64, &info, &iter, path.c_str() );
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

}

#endif
