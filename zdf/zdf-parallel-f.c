/**
 * @file zdf-parallel-f.c
 * @brief Fortran support routines for the parallel ZDF interface
 * 
 * The Fortran parallel file handle is an opaque pointer to a t_zdf_par_file structure
 * allocated here. This is required because the size and layout of MPI_Comm / MPI_File
 * in C are implementation specific (e.g. pointers in Open MPI, integers in MPICH), so
 * the structure cannot be mirrored in Fortran.
 * 
 * Only opening and closing files require these routines; all other parallel routines
 * are called directly from Fortran.
 */

#include "zdf-parallel.h"

#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Opens a parallel ZDF file from Fortran
 * 
 * @param zdf       Pointer to opaque file handle (set to NULL on error)
 * @param filename  File name (null terminated)
 * @param amode     Access mode
 * @param comm      Fortran MPI communicator handle
 * @param iomode    Parallel I/O mode
 * @return          1 on success, 0 on error (on all ranks)
 */
int zdf_par_open_file_f03( t_zdf_par_file** zdf, const char* filename, int amode,
    MPI_Fint comm, int iomode ) {

    MPI_Comm c_comm = MPI_Comm_f2c( comm );

    *zdf = (t_zdf_par_file *) malloc( sizeof( t_zdf_par_file ) );

    // Opening is collective, so make sure all ranks have the handle
    int ok = ( *zdf != NULL ), all;
    MPI_Allreduce( &ok, &all, 1, MPI_INT, MPI_LAND, c_comm );
    if ( !all ) {
        if ( !ok ) fprintf(stderr, "(*error*) Unable to allocate parallel ZDF file handle.\n");
        free( *zdf );
        *zdf = NULL;
        return 0;
    }

    if ( !zdf_par_open_file( *zdf, filename, (enum zdf_file_access_mode) amode, c_comm,
        (t_zdf_parallel_io_mode) iomode ) ) {
        free( *zdf );
        *zdf = NULL;
        return 0;
    }

    return 1;
}

/**
 * @brief Closes a parallel ZDF file from Fortran and frees the handle
 * 
 * @param zdf   Pointer to opaque file handle (set to NULL)
 * @return      1 on success, 0 on error (on all ranks)
 */
int zdf_par_close_file_f03( t_zdf_par_file** zdf ) {

    if ( *zdf == NULL ) return 0;

    int ok = zdf_par_close_file( *zdf );
    free( *zdf );
    *zdf = NULL;
    return ok;
}
