/*
 *  zdf_parallel.h
 *
 *  Parallel (MPI) interface to the ZDF library.
 *
 *  All routines in this interface are collective over the communicator used to open the file,
 *  and return the same value on all ranks: 1 on success, 0 on error. If any rank fails, all
 *  ranks return 0, so callers can safely close the file and abort the operation without
 *  risking a deadlock.
 *
 *  Data pointers (chunk->data) must point to host memory. Device (GPU) buffers must be
 *  copied to (ideally pinned) host memory before being written.
 */

#ifndef ZDF_PARALLEL_H_
#define ZDF_PARALLEL_H_

#include "zdf.h"
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parallel I/O modes
 * 
 * ZDF_MPI                  Data is sent to the root node which writes the file (serial I/O)
 * ZDF_INDEPENDENT          All nodes write their data independently using POSIX I/O. Requires
 *                          a POSIX consistent (parallel) file system, i.e. not NFS.
 * ZDF_MPIIO_INDEPENDENT    MPI/IO, independent writes
 * ZDF_MPIIO_COLLECTIVE     MPI/IO, collective writes
 * 
 * All modes produce identical files.
 */
enum zdf_parallel_io_mode { ZDF_MPI, ZDF_INDEPENDENT, ZDF_MPIIO_INDEPENDENT, ZDF_MPIIO_COLLECTIVE };

typedef enum zdf_parallel_io_mode t_zdf_parallel_io_mode;

typedef struct {
	t_zdf_file zdf_file;				// "parent" class
	enum zdf_parallel_io_mode iomode; 	// Parallel I/O mode
	MPI_Comm comm;						// Parallel Communicator
	MPI_File fh;						// MPIIO Parallel File
	MPI_Offset fpos;					// File position (end of file)
} t_zdf_par_file;

/**
 * @brief Creates path (on root node only) if required
 * @param path  Path to create
 * @param comm  MPI communicator
 * @return      1 on success, 0 on error (on all nodes)
 */
int zdf_par_create_path( const char path[], MPI_Comm comm );

/**
 * @brief Opens a parallel ZDF file
 * 
 * When opening in ZDF_UPDATE mode, new data is appended at the end of the file and new datasets
 * get unique ids.
 * 
 * @param zdf       Parallel ZDF file handle
 * @param filename  File name
 * @param amode     Access mode (ZDF_CREATE, ZDF_READ or ZDF_UPDATE)
 * @param comm      MPI communicator
 * @param iomode    Parallel I/O mode
 * @return          1 on success, 0 on error
 */
int zdf_par_open_file( t_zdf_par_file* zdf, const char* filename, enum zdf_file_access_mode amode,
	MPI_Comm comm, t_zdf_parallel_io_mode iomode );

int zdf_par_close_file( t_zdf_par_file* zdf );

int zdf_par_add_string( t_zdf_par_file* zdf, const char* name, const char* str );

int zdf_par_add_int32( t_zdf_par_file* zdf, const char* name, const int32_t value );

int zdf_par_add_double( t_zdf_par_file* zdf, const char* name, const double value );

int zdf_par_add_iteration( t_zdf_par_file* zdf, const t_zdf_iteration* iter );

int zdf_par_add_grid_info( t_zdf_par_file* zdf, const t_zdf_grid_info* grid );

int zdf_par_add_part_info( t_zdf_par_file* zdf,  t_zdf_part_info* part );

/**
 * @brief Returns the offset (in number of data elements) of the local chunk in the file layout,
 *        i.e. the sum of the number of elements on all lower ranks.
 * @param chunk     Local chunk
 * @param ndims     Number of dimensions
 * @param comm      MPI communicator
 * @return          Offset in elements
 */
int64_t zdf_par_getoffset( t_zdf_chunk* chunk, uint32_t ndims, MPI_Comm comm );


int zdf_par_start_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset );

/**
 * @brief Writes a distributed chunked dataset
 * 
 * Chunks are stored in rank order. The total number of elements over all chunks must not exceed
 * the dataset size (overlapping chunks, e.g. including guard cells, are detected as an error).
 * Empty chunks are allowed.
 * 
 * @param zdf       Parallel ZDF file handle
 * @param dataset   Dataset (global) information
 * @param chunk     Local chunk information and data (host memory)
 * @param offset    Offset of local chunk in the file layout, in elements. Use a negative value
 *                  to have the library calculate it. If specified it must match the value returned
 *                  by zdf_par_getoffset(), otherwise an error is returned.
 * @return          1 on success, 0 on error
 */
int zdf_par_write_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk, const int64_t offset );

int zdf_par_end_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset );

int zdf_par_extend_dataset( t_zdf_par_file* zdf, t_zdf_dataset* dataset, uint64_t* new_count );

int zdf_par_save_grid( t_zdf_chunk *chunk, const enum zdf_data_type data_type, const t_zdf_grid_info *info,
    const t_zdf_iteration *iteration, char const path[], MPI_Comm comm, t_zdf_parallel_io_mode par_io_mode );

#ifdef __cplusplus
}
#endif

#endif
