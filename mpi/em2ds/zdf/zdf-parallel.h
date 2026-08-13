/*
 *  zdf_parallel.h
 *
 */

#ifndef ZDF_PARALLEL_H_
#define ZDF_PARALLEL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "zdf.h"
#include <mpi.h>

enum zdf_parallel_io_mode { ZDF_MPI, ZDF_INDEPENDENT, ZDF_MPIIO_INDEPENDENT, ZDF_MPIIO_COLLECTIVE };

typedef enum zdf_parallel_io_mode t_zdf_parallel_io_mode;

typedef struct {
	t_zdf_file zdf_file;				// "parent" class
	enum zdf_parallel_io_mode iomode; 	// Parallel I/O mode
	MPI_Comm comm;						// Parallel Communicator
	MPI_File fh;						// MPIIO Parallel File
	MPI_Offset fpos;					// File position
} t_zdf_par_file;

int zdf_par_open_file( t_zdf_par_file* zdf, const char* filename, enum zdf_file_access_mode amode,
	MPI_Comm comm, t_zdf_parallel_io_mode iomode );

int zdf_par_close_file( t_zdf_par_file* zdf );

int zdf_par_add_string( t_zdf_par_file* zdf, const char* name, const char* str );

int zdf_par_add_int32( t_zdf_par_file* zdf, const char* name, const int32_t value );

int zdf_par_add_double( t_zdf_par_file* zdf, const char* name, const double value );

int zdf_par_add_iteration( t_zdf_par_file* zdf, const t_zdf_iteration* iter );

int zdf_par_add_grid_info( t_zdf_par_file* zdf, const t_zdf_grid_info* grid );

int zdf_par_add_part_info( t_zdf_par_file* zdf,  t_zdf_part_info* part );


int64_t zdf_par_getoffset( t_zdf_chunk* chunk, uint32_t ndims, MPI_Comm comm );


int zdf_par_start_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset );

int zdf_par_write_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk, const int64_t offset );

int zdf_par_end_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset );

int zdf_par_extend_dataset( t_zdf_par_file* zdf, t_zdf_dataset* dataset, uint64_t* new_count );

int zdf_par_save_grid( t_zdf_chunk *chunk, const enum zdf_data_type data_type, const t_zdf_grid_info *info,
    const t_zdf_iteration *iteration, char const path[], MPI_Comm comm, t_zdf_parallel_io_mode par_io_mode );

#ifdef __cplusplus
}
#endif

#endif
