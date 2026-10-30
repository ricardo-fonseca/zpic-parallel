/**
 * @file zdf-parallel.c
 * @brief Parallel (MPI) interface to the ZDF library
 * 
 * Notes:
 * - All public routines are collective and return the same value (1 success, 0 error) on all
 *   ranks of the communicator. Rank-local errors are combined with an MPI_Allreduce.
 * - All I/O modes produce byte-identical files.
 * - Metadata records are serialized into a dynamically sized memory stream, so there is no
 *   limit on metadata size.
 * - Message / I/O sizes above INT_MAX elements are handled using derived datatypes.
 */

#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "zdf-parallel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>
#include <sys/types.h>

#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#if defined(_WIN32) || defined(_WIN64)
  // replace 'fseeko' and 'ftello' with their Windows cognates
  #define fseeko _fseeki64
  #define ftello _ftelli64
  #define ZDF_NO_MEMSTREAM 1
#endif

/**
 * Maximum element count used in a single MPI call. Larger counts use derived datatypes.
 * Can be lowered at compile time for testing.
 */
#ifndef ZDF_MPI_MAX_COUNT
#define ZDF_MPI_MAX_COUNT INT_MAX
#endif

/**
 * Block size (in elements) used for building derived datatypes for large counts
 */
#ifndef ZDF_MPI_BLOCK
#define ZDF_MPI_BLOCK (1 << 30)
#endif

/* -----------------------------------------------------------------------------------------------
  Utilities
-------------------------------------------------------------------------------------------------- */

/**
 * Returns MPI data type from ZDF data type
 * @param  data_type ZDF data type
 * @return           Corresponding MPI data type. Returns MPI_DATATYPE_NULL for unsupported
 *                   datatypes.
 */
MPI_Datatype zdf_mpi_type( int data_type ) {

    switch (data_type) {
        case( zdf_null ):    return( MPI_DATATYPE_NULL );
        case( zdf_float32 ): return( MPI_FLOAT    );
        case( zdf_float64 ): return( MPI_DOUBLE   );
        case( zdf_int8 ):    return( MPI_INT8_T   );
        case( zdf_int16 ):   return( MPI_INT16_T  );
        case( zdf_int32 ):   return( MPI_INT32_T  );
        case( zdf_int64 ):   return( MPI_INT64_T  );
        case( zdf_uint8 ):   return( MPI_UINT8_T  );
        case( zdf_uint16 ):  return( MPI_UINT16_T );
        case( zdf_uint32 ):  return( MPI_UINT32_T );
        case( zdf_uint64 ):  return( MPI_UINT64_T );
        case( zdf_complex64 ):  return( MPI_C_FLOAT_COMPLEX );
        case( zdf_complex128 ):  return( MPI_C_DOUBLE_COMPLEX );
    }
    return( MPI_DATATYPE_NULL );
}

/**
 * @brief Combines a local success flag over all ranks
 * @param comm  MPI communicator
 * @param ok    Local success flag
 * @return      1 if all ranks succeeded, 0 otherwise
 */
static int zdf_par_agree( MPI_Comm comm, int ok ) {
    int local = ( ok != 0 ), all;
    MPI_Allreduce( &local, &all, 1, MPI_INT, MPI_LAND, comm );
    return all;
}

/**
 * @brief (count, datatype) pair for MPI calls that supports more than INT_MAX elements
 */
typedef struct {
    MPI_Datatype type;  ///< Datatype to use
    int count;          ///< Count to use
    int derived;        ///< Set to 1 if type is a derived type that must be freed
} t_zdf_mpi_count;

/**
 * @brief Initializes a (count, datatype) pair for n elements of type base
 * @param c     (count, datatype) pair
 * @param base  Base datatype
 * @param n     Number of elements
 * @return      1 on success, 0 on error
 */
static int zdf_mpi_count_init( t_zdf_mpi_count* c, MPI_Datatype base, uint64_t n ) {

    c -> derived = 0;
    if ( n <= (uint64_t) ZDF_MPI_MAX_COUNT ) {
        c -> type  = base;
        c -> count = (int) n;
        return 1;
    }

    const uint64_t B = ZDF_MPI_BLOCK;
    const uint64_t q = n / B;
    const uint64_t r = n % B;
    if ( q > INT_MAX ) return 0;

    MPI_Datatype block, blocks;
    MPI_Type_contiguous( (int) B, base, &block );
    MPI_Type_contiguous( (int) q, block, &blocks );
    MPI_Type_free( &block );

    if ( r == 0 ) {
        c -> type = blocks;
    } else {
        MPI_Aint lb, extent;
        MPI_Type_get_extent( base, &lb, &extent );
        int          bl[2]    = { 1, (int) r };
        MPI_Aint     disp[2]  = { 0, (MPI_Aint) (q * B) * extent };
        MPI_Datatype types[2] = { blocks, base };
        MPI_Type_create_struct( 2, bl, disp, types, &c -> type );
        MPI_Type_free( &blocks );
    }
    MPI_Type_commit( &c -> type );
    c -> count   = 1;
    c -> derived = 1;
    return 1;
}

/**
 * @brief Frees (count, datatype) pair
 * @param c     (count, datatype) pair
 */
static void zdf_mpi_count_free( t_zdf_mpi_count* c ) {
    if ( c -> derived ) MPI_Type_free( &c -> type );
    c -> derived = 0;
}

/**
 * @brief Writes a byte buffer to an MPI file at the specified offset (independent operation)
 * @param fh        MPI file handle
 * @param offset    File offset
 * @param buf       Buffer
 * @param len       Number of bytes
 * @return          1 on success, 0 on error
 */
static int zdf_mpiio_write_bytes( MPI_File fh, MPI_Offset offset, const void * buf, size_t len ) {
    t_zdf_mpi_count c;
    if ( !zdf_mpi_count_init( &c, MPI_BYTE, len ) ) return 0;
    int err = MPI_File_write_at( fh, offset, (void *) buf, c.count, c.type, MPI_STATUS_IGNORE );
    zdf_mpi_count_free( &c );
    if ( err != MPI_SUCCESS ) {
        fprintf(stderr, "(*error*) MPI/IO write failed.\n");
        return 0;
    }
    return 1;
}

/**
 * @brief Number of elements in chunk
 */
static uint64_t zdf_chunk_size( const t_zdf_chunk* chunk, uint32_t ndims ) {
    uint64_t n = 1;
    for( uint32_t i = 0; i < ndims; i++ ) n *= chunk -> count[i];
    return n;
}

/* -----------------------------------------------------------------------------------------------
  Memory streams, used to serialize metadata / headers before writing them with MPI/IO
-------------------------------------------------------------------------------------------------- */

/**
 * @brief Memory buffer that can be written to with the serial zdf routines
 */
typedef struct {
    t_zdf_file file;    ///< ZDF file object writing to memory
    char * buf;         ///< Data buffer (valid after zdf_membuf_close())
    size_t size;        ///< Data size (valid after zdf_membuf_close())
} t_zdf_membuf;

/**
 * @brief Opens a memory stream
 * @param m             Memory buffer
 * @param ndatasets     Number of datasets to initialize the file object with
 * @return              1 on success, 0 on error
 */
static int zdf_membuf_open( t_zdf_membuf* m, uint32_t ndatasets ) {
    m -> buf = NULL;
    m -> size = 0;
    m -> file.mode = ZDF_CREATE;
    m -> file.ndatasets = ndatasets;

#ifdef ZDF_NO_MEMSTREAM
    m -> file.fp = tmpfile();
#else
    m -> file.fp = open_memstream( &m -> buf, &m -> size );
#endif

    if ( m -> file.fp == NULL ) {
        perror("(*error*) Unable to open memory stream");
        return 0;
    }
    return 1;
}

/**
 * @brief Closes the memory stream. On success m->buf and m->size hold the data written.
 *        zdf_membuf_free() must always be called afterwards.
 * @param m     Memory buffer
 * @return      1 on success, 0 on error
 */
static int zdf_membuf_close( t_zdf_membuf* m ) {
    if ( m -> file.fp == NULL ) return 0;

#ifdef ZDF_NO_MEMSTREAM
    FILE * fp = m -> file.fp;
    int ok = ( fflush( fp ) == 0 );
    long long n = ok ? (long long) ftello( fp ) : -1;
    ok = ok && ( n >= 0 );
    if ( ok ) {
        m -> buf = (char *) malloc( n > 0 ? (size_t) n : 1 );
        ok = ( m -> buf != NULL );
    }
    if ( ok ) {
        rewind( fp );
        ok = ( fread( m -> buf, 1, (size_t) n, fp ) == (size_t) n );
        m -> size = (size_t) n;
    }
    ok = ( fclose( fp ) == 0 ) && ok;
#else
    int ok = ( fclose( m -> file.fp ) == 0 );
#endif

    m -> file.fp = NULL;
    if ( !ok ) fprintf(stderr, "(*error*) Unable to serialize data to memory.\n");
    return ok;
}

/**
 * @brief Frees memory buffer
 * @param m     Memory buffer
 */
static void zdf_membuf_free( t_zdf_membuf* m ) {
    if ( m -> file.fp ) { fclose( m -> file.fp ); m -> file.fp = NULL; }
    free( m -> buf );
    m -> buf = NULL;
    m -> size = 0;
}

/* -----------------------------------------------------------------------------------------------
  Open / Close
-------------------------------------------------------------------------------------------------- */

/**
 * @brief Creates path (on root node only) if required
 * @param path  Path to create
 * @param comm  MPI communicator
 * @return      1 on success, 0 on error (on all nodes)
 */
int zdf_par_create_path( const char path[], MPI_Comm comm ) {
    int rank; MPI_Comm_rank( comm, &rank );
    int ok = 1;
    if ( rank == 0 ) ok = ( create_path( path ) == 0 );
    return zdf_par_agree( comm, ok );
}

/**
 * @brief Shares file position and number of datasets from root node
 * @param zdf   Parallel ZDF file handle
 */
static void zdf_par_bcast_state( t_zdf_par_file* zdf ) {
    uint64_t state[2] = { (uint64_t) zdf -> fpos, zdf -> zdf_file.ndatasets };
    MPI_Bcast( state, 2, MPI_UINT64_T, 0, zdf -> comm );
    zdf -> fpos = (MPI_Offset) state[0];
    zdf -> zdf_file.ndatasets = (uint32_t) state[1];
}

/**
 * Opens file for MPI parallel I/O (only root node opens the file)
 * @param  zdf        Parallel ZDF file handle
 * @param  filename Filename of the ZDF file to open, including path
 * @return          Returns 1 on success, 0 on error
 */
static int zdf_par_open_file_mpi( t_zdf_par_file* zdf, const char* filename ) {

    int rank; MPI_Comm_rank( zdf->comm, &rank );

    int ok = 1;
    if ( rank == 0 ) {
        ok = zdf_open_file( & zdf->zdf_file, filename, zdf -> zdf_file.mode );
        if ( ok ) {
            zdf -> fpos = ftello( zdf -> zdf_file.fp );
            ok = ( zdf -> fpos >= 0 );
            if ( !ok ) zdf_close_file( &zdf -> zdf_file );
        }
    }

    if ( !zdf_par_agree( zdf -> comm, ok ) ) return(0);
    zdf_par_bcast_state( zdf );
    return(1);
}

/**
 * Opens file for Independent parallel I/O
 * @param  zdf        Parallel ZDF file handle
 * @param  filename Filename of the ZDF file to open, including path
 * @return          Returns 1 on success, 0 on error
 */
static int zdf_par_open_file_indep( t_zdf_par_file* zdf, const char* filename ) {

    int rank; MPI_Comm_rank( zdf->comm, &rank );
    int ok = 1;

    // Root opens (and creates, if needed) the file using the serial routine
    if ( rank == 0 ) {
        ok = zdf_open_file( &zdf->zdf_file, filename, zdf->zdf_file.mode );
        if ( ok ) {
            zdf -> fpos = ftello( zdf -> zdf_file.fp );
            ok = ( zdf -> fpos >= 0 );
            if ( !ok ) zdf_close_file( &zdf -> zdf_file );
        }
    }
    if ( !zdf_par_agree( zdf -> comm, ok ) ) return(0);

    // Other nodes open the file after it has been created by root
    if ( rank != 0 ) {
        const char * mode = ( zdf->zdf_file.mode == ZDF_READ ) ? "rb" : "r+b";
        if ( !(zdf->zdf_file.fp = fopen( filename, mode ) ) ) {
            perror("(*error*) Unable to open ZDF file for parallel I/O");
            ok = 0;
        }
    }
    if ( !zdf_par_agree( zdf -> comm, ok ) ) {
        if ( zdf -> zdf_file.fp ) zdf_close_file( &zdf -> zdf_file );
        return(0);
    }

    zdf_par_bcast_state( zdf );
    return(1);
}

/**
 * Opens file for MPI/IO parallel I/O
 * @param  zdf        Parallel ZDF file handle
 * @param  filename Filename of the ZDF file to open, including path
 * @return          Returns 1 on success, 0 on error
 */
static int zdf_par_open_file_mpiio( t_zdf_par_file* zdf, const char* filename ) {

    int rank; MPI_Comm_rank( zdf->comm, &rank );
    int ok = 1;

    // In update mode, have root scan the file for the number of datasets and file size
    // This is done before MPI_File_open because of MPI_MODE_UNIQUE_OPEN
    if ( zdf -> zdf_file.mode == ZDF_UPDATE ) {
        if ( rank == 0 ) {
            t_zdf_file tmp;
            ok = zdf_open_file( &tmp, filename, ZDF_UPDATE );
            if ( ok ) {
                zdf -> fpos = ftello( tmp.fp );
                zdf -> zdf_file.ndatasets = tmp.ndatasets;
                ok = zdf_close_file( &tmp ) && ( zdf -> fpos >= 0 );
            }
        }
        if ( !zdf_par_agree( zdf -> comm, ok ) ) return(0);
        zdf_par_bcast_state( zdf );
    }

    // MPI_MODE_UNIQUE_OPEN mode allows an implementation to optimize access by 
    // eliminating the overhead of file locking.
    int mpiio_amode = MPI_MODE_UNIQUE_OPEN;
    switch ( zdf -> zdf_file.mode ) {
        case( ZDF_CREATE ):
            mpiio_amode |= MPI_MODE_CREATE | MPI_MODE_WRONLY;
            break;
        case( ZDF_READ ):
            mpiio_amode |= MPI_MODE_RDONLY;
            break;
        case( ZDF_UPDATE ):
            mpiio_amode |= MPI_MODE_RDWR;
            break;
    }

    MPI_Info hints;
    MPI_Info_create(&hints);

    // Enable collective buffering optimization
    MPI_Info_set(hints , "romio_cb_write" , "enable" );    

    int err = MPI_File_open( zdf->comm, filename, mpiio_amode, hints, &(zdf -> fh) );
    MPI_Info_free( &hints );

    if ( !zdf_par_agree( zdf -> comm, err == MPI_SUCCESS ) ) {
        if ( rank == 0 ) fprintf(stderr, "(*error*) Unable to open ZDF file %s\n", filename );
        if ( zdf -> fh != MPI_FILE_NULL ) MPI_File_close( &zdf -> fh );
        return(0);
    }

    switch ( zdf -> zdf_file.mode ) {
    case ZDF_CREATE:
        // Truncate existing file upon creation
        ok = ( MPI_File_set_size( zdf -> fh, 0 ) == MPI_SUCCESS );
        // According to the ROMIO documentation we should perform a barrier following a truncation
        MPI_Barrier( zdf->comm );

        // Write magic number
        if ( ok && rank == 0 ) {
            ok = zdf_mpiio_write_bytes( zdf -> fh, 0, zdf_magic, ZDF_MAGIC_LENGTH );
        }
        zdf -> fpos = ZDF_MAGIC_LENGTH;
        break;

    case ZDF_READ:
        // Check magic number
        if ( rank == 0 ) {
            char test_magic[ZDF_MAGIC_LENGTH];
            if ( MPI_File_read_at( zdf -> fh, 0, (void *) test_magic, ZDF_MAGIC_LENGTH, MPI_BYTE,
                MPI_STATUS_IGNORE ) != MPI_SUCCESS ) {
                fprintf(stderr, "(*error*) Unable to read magic number from ZDF file.\n");
                ok = 0;
            } else if ( memcmp( test_magic, zdf_magic, ZDF_MAGIC_LENGTH ) ) {
                fprintf(stderr, "(*error*) Invalid magic number, file is not a proper ZDF file.\n");
                ok = 0;
            }
        }
        zdf -> fpos = ZDF_MAGIC_LENGTH;
        break;

    case ZDF_UPDATE:
        // Magic number was already checked, fpos set to the end of the file
        break;
    }

    if ( !zdf_par_agree( zdf -> comm, ok ) ) {
        MPI_File_close( &zdf -> fh );
        return(0);
    }

    return(1);
}

/**
 * Opens Parallel ZDF file
 * @param  zdf      Parallel ZDF file handle
 * @param  filename Filename of the ZDF file to open, including path
 * @param  mode     Can be one of ZDF_CREATE, ZDF_READ or ZDF_UPDATE
 * @param  comm     MPI Communicator to use
 * @param  iomode   Parallel I/O mode
 * @return          Returns 1 on success, 0 on error
 */
int zdf_par_open_file( t_zdf_par_file* zdf, const char* filename, enum zdf_file_access_mode amode,
    MPI_Comm comm, t_zdf_parallel_io_mode iomode ) {

    zdf -> zdf_file.mode = amode;
    zdf -> zdf_file.ndatasets = 0;
    zdf -> comm = comm;
    zdf -> iomode = iomode;

    zdf -> zdf_file.fp = NULL;
    zdf -> fh = MPI_FILE_NULL;
    zdf -> fpos = ZDF_MAGIC_LENGTH;

    switch ( zdf -> iomode ) {
        case ZDF_MPI:
            return zdf_par_open_file_mpi( zdf, filename );
        case ZDF_INDEPENDENT:
            return zdf_par_open_file_indep( zdf, filename );
        case ZDF_MPIIO_INDEPENDENT:
        case ZDF_MPIIO_COLLECTIVE:
            return zdf_par_open_file_mpiio( zdf, filename );
    }

    fprintf(stderr, "(*error*) Invalid parallel I/O mode.\n");
    return(0);
}


/**
 * Closes parallel ZDF file
 * @param  zdf Parallel ZDF file handle
 * @return     Returns 1 on success, 0 on error
 */
int zdf_par_close_file( t_zdf_par_file* zdf ) {

    int ok = 1;

    switch ( zdf -> iomode ) {
        case ZDF_MPI:
        case ZDF_INDEPENDENT:
            // In ZDF_MPI mode only root has an open file
            if ( zdf -> zdf_file.fp ) ok = zdf_close_file( &zdf -> zdf_file );
            break;
        case ZDF_MPIIO_INDEPENDENT:
        case ZDF_MPIIO_COLLECTIVE: 
            if ( zdf -> fh != MPI_FILE_NULL ) {
                ok = ( MPI_File_close( & (zdf -> fh) ) == MPI_SUCCESS );
            }
            break;
    }

    return zdf_par_agree( zdf -> comm, ok );
}

/* -----------------------------------------------------------------------------------------------
  Metadata
-------------------------------------------------------------------------------------------------- */

/**
 * @brief Serial metadata writer
 */
typedef size_t (*zdf_meta_fn)( t_zdf_file* file, const void* args );

/**
 * @brief Writes a metadata record (or any other record written only by root).
 * 
 * In ZDF_MPI / ZDF_INDEPENDENT modes root writes the record directly to file. In MPI/IO modes
 * all ranks serialize the record to memory (keeping the file position consistent on all ranks)
 * and root writes it to file.
 * 
 * @param zdf   Parallel ZDF file handle
 * @param fn    Serial writer function
 * @param args  Arguments for writer
 * @return      1 on success, 0 on error
 */
static int zdf_par_write_meta( t_zdf_par_file* zdf, zdf_meta_fn fn, const void* args ) {

    int rank; MPI_Comm_rank( zdf->comm, &rank );
    int ok = 1;

    switch ( zdf -> iomode ) {
        case ZDF_MPI:
        case ZDF_INDEPENDENT:
            if ( rank == 0 ) {
                size_t count = fn( &zdf->zdf_file, args );
                if ( count > 0 ) zdf -> fpos += count;
                else ok = 0;
            }
            break;

        case ZDF_MPIIO_INDEPENDENT:
        case ZDF_MPIIO_COLLECTIVE: {
            t_zdf_membuf m;
            size_t count = 0;
            ok = zdf_membuf_open( &m, zdf -> zdf_file.ndatasets );
            if ( ok ) {
                count = fn( &m.file, args );
                ok = zdf_membuf_close( &m ) && ( count > 0 ) && ( count == m.size );
            }
            if ( ok && rank == 0 ) ok = zdf_mpiio_write_bytes( zdf -> fh, zdf -> fpos, m.buf, m.size );
            if ( ok ) zdf -> fpos += count;
            zdf_membuf_free( &m );
            break;
        }
    }

    return zdf_par_agree( zdf -> comm, ok );
}

// Adapters for the serial metadata writers

typedef struct { const char* name; const char* str; } t_meta_string;
static size_t meta_string( t_zdf_file* f, const void* a ) {
    const t_meta_string* p = (const t_meta_string*) a;
    return zdf_add_string( f, p -> name, p -> str );
}

typedef struct { const char* name; int32_t value; } t_meta_int32;
static size_t meta_int32( t_zdf_file* f, const void* a ) {
    const t_meta_int32* p = (const t_meta_int32*) a;
    return zdf_add_int32( f, p -> name, p -> value );
}

typedef struct { const char* name; double value; } t_meta_double;
static size_t meta_double( t_zdf_file* f, const void* a ) {
    const t_meta_double* p = (const t_meta_double*) a;
    return zdf_add_double( f, p -> name, p -> value );
}

static size_t meta_iteration( t_zdf_file* f, const void* a ) {
    return zdf_add_iteration( f, (const t_zdf_iteration*) a );
}

static size_t meta_grid_info( t_zdf_file* f, const void* a ) {
    return zdf_add_grid_info( f, (const t_zdf_grid_info*) a );
}

static size_t meta_part_info( t_zdf_file* f, const void* a ) {
    return zdf_add_part_info( f, (const t_zdf_part_info*) a );
}

static size_t meta_start_cdset( t_zdf_file* f, const void* a ) {
    return zdf_start_cdset( f, (t_zdf_dataset*) a );
}

static size_t meta_end_cdset( t_zdf_file* f, const void* a ) {
    return zdf_end_cdset( f, (t_zdf_dataset*) a );
}

/**
 * Adds string element to parallel ZDF file. Only root node writes data.
 * @param  zdf  Parallel ZDF file handle
 * @param  name Element name
 * @param  str  String value
 * @return      Returns 1 on success, 0 on error
 */
int zdf_par_add_string( t_zdf_par_file* zdf, const char* name, const char* str ) {
    t_meta_string args = { name, str };
    return zdf_par_write_meta( zdf, meta_string, &args );
}

/**
 * Adds int32 element to parallel ZDF file. Only root node writes data.
 * @param  zdf    Parallel ZDF file handle
 * @param  name   Element name
 * @param  value  int32 value
 * @return      Returns 1 on success, 0 on error
 */
int zdf_par_add_int32( t_zdf_par_file* zdf, const char* name, const int32_t value ) {
    t_meta_int32 args = { name, value };
    return zdf_par_write_meta( zdf, meta_int32, &args );
}

/**
 * Adds float64 element to parallel ZDF file. Only root node writes data.
 * @param  zdf    Parallel ZDF file handle
 * @param  name   Element name
 * @param  value  float64 value
 * @return      Returns 1 on success, 0 on error
*/
int zdf_par_add_double( t_zdf_par_file* zdf, const char* name, const double value ) {
    t_meta_double args = { name, value };
    return zdf_par_write_meta( zdf, meta_double, &args );
}

/**
 * Adds iteration metadata group to parallel ZDF file. Only root node writes data.
 * @param  zdf  Parallel ZDF file handle
 * @param  iter Iteration info
 * @return      Returns 1 on success, 0 on error
 */
int zdf_par_add_iteration( t_zdf_par_file* zdf, const t_zdf_iteration* iter ) {
    return zdf_par_write_meta( zdf, meta_iteration, iter );
}

/**
 * Adds parallel grid information metadata group to file. Only root node writes data.
 * @param  zdf  Parallel ZDF file handle
 * @param  grid Grid information
 * @return      Returns 1 on success, 0 on error
 */
int zdf_par_add_grid_info( t_zdf_par_file* zdf, const t_zdf_grid_info* grid ) {
    return zdf_par_write_meta( zdf, meta_grid_info, grid );
}

/**
 * Adds particle information metadata group to parallel ZDF file. Only root node writes data.
 * @param  zdf  Parallel ZDF file handle
 * @param  part Particle information
 * @return      Returns 1 on success, 0 on error
*/
int zdf_par_add_part_info( t_zdf_par_file* zdf, t_zdf_part_info* part ) {
    return zdf_par_write_meta( zdf, meta_part_info, part );
}

/**
 * Adds a chunked dataset header to parallel ZDF file. Only root node writes data.
 * @param  zdf     Parallel ZDF file handle
 * @param  dataset Dataset object
 * @return         1 on success, 0 on error. The dataset object (on all nodes)
 *                 is also modified to include a unique id.
 */
int zdf_par_start_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset ) {

    const MPI_Offset start = zdf -> fpos;
    const uint32_t id = zdf -> zdf_file.ndatasets + 1;

    int ok = zdf_par_write_meta( zdf, meta_start_cdset, dataset );

    zdf -> zdf_file.ndatasets = id;
    dataset -> id = id;

    // In MPI/IO modes the header was written to a memory buffer, convert to file position
    if ( zdf -> iomode == ZDF_MPIIO_INDEPENDENT || zdf -> iomode == ZDF_MPIIO_COLLECTIVE ) {
        dataset -> offset += start;
    }

    return ok;
}

/**
 * Add an end marker for a chunked dataset to a parallel ZDF file. Only root node writes data.
 * @param  zdf     Parallel ZDF file handle
 * @param  dataset Dataset object
 * @return         1 on success, 0 on error
 */
int zdf_par_end_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset ) {
    return zdf_par_write_meta( zdf, meta_end_cdset, dataset );
}

/**
 * Extends dataset dimensions.
 *
 * @param  zdf       Parallel ZDF file handle
 * @param  dataset   Datasete object
 * @param  new_count New dimensions for the dataset, must be >= than the previous
 *                   dimensions.
 * @return           Returns 1 if successful, 0 otherwise.
 */
int zdf_par_extend_dataset( t_zdf_par_file* zdf, t_zdf_dataset* dataset, uint64_t* new_count ) {

    int rank; MPI_Comm_rank( zdf->comm, &rank );
    int ok = 1;

    for( unsigned i = 0; i < dataset -> ndims; i++ ) {
        if ( new_count[i] < dataset -> count[i] ) ok = 0;
    }
    if ( !zdf_par_agree( zdf -> comm, ok ) ) {
        if ( rank == 0 ) {
            fprintf(stderr,"(*error*) Invalid value for zdf_par_extend_dataset.\n");
            fprintf(stderr,"(*error*) New size is smaller than original size.\n");
        }
        return(0);
    }

    switch ( zdf -> iomode ) {
    case ZDF_MPI:
    case ZDF_INDEPENDENT:
        if ( rank == 0 ) {
            ok = zdf_extend_dataset( &(zdf -> zdf_file), dataset, new_count );
            // zdf_extend_dataset() leaves the file pointer at the end of the file as seen by
            // this process; in ZDF_INDEPENDENT mode data from other nodes may still be buffered
            // so move it to the tracked end of file position.
            if ( ok ) ok = ( fseeko( zdf -> zdf_file.fp, zdf -> fpos, SEEK_SET ) == 0 );
        } else {
            for( unsigned i = 0; i < dataset -> ndims; i++ ) dataset -> count[i] = new_count[i];
        }
        break;

    case ZDF_MPIIO_INDEPENDENT:
    case ZDF_MPIIO_COLLECTIVE: {
        for( unsigned i = 0; i < dataset -> ndims; i++ ) dataset -> count[i] = new_count[i];

        // Serialize the new dataset header
        const uint64_t offset = dataset -> offset;
        t_zdf_membuf m;
        ok = zdf_membuf_open( &m, zdf -> zdf_file.ndatasets );
        if ( ok ) ok = zdf_dataset_header_write( &m.file, dataset ) != 0;
        if ( m.file.fp ) ok = zdf_membuf_close( &m ) && ok;
        dataset -> offset = offset;

        // Overwrite dataset header
        if ( ok && rank == 0 ) ok = zdf_mpiio_write_bytes( zdf -> fh, offset, m.buf, m.size );
        zdf_membuf_free( &m );
        break;
    }
    }

    return zdf_par_agree( zdf -> comm, ok );
}

/* -----------------------------------------------------------------------------------------------
  Chunked dataset data
-------------------------------------------------------------------------------------------------- */

/**
 * Returns the local offset in the file layout, in number of data elements, of the parallel chunk. 
 * Note that this does not include the record headers.
 * @param  chunk (parallel) chunk data
 * @param  ndims Number of dimensions of chunk data
 * @param  comm  Parallel communicator
 * @return       Offset in number of data elements
 */
int64_t zdf_par_getoffset( t_zdf_chunk* chunk, uint32_t ndims, MPI_Comm comm ) {

    int64_t len = (int64_t) zdf_chunk_size( chunk, ndims );

    int rank; MPI_Comm_rank( comm, &rank );

    int64_t offset = 0;
    MPI_Exscan( (void *) & len, (void *) & offset, 1, MPI_INT64_T, MPI_SUM, comm );

    // The value on rank 0 from MPI_Exscan is undefined so set it explicitly
    if ( rank == 0 ) offset = 0;

    return(offset);
}

/**
 * @brief Information on the data layout of a parallel write
 */
typedef struct {
    uint64_t count;         ///< Number of elements in local chunk
    uint64_t bytes;         ///< Size of local chunk data in file (including padding)
    uint64_t byte_offset;   ///< Offset of local chunk data in file (excluding headers)
    uint64_t total_bytes;   ///< Total size of chunk data in file (excluding headers)
    uint64_t max_count;     ///< Maximum number of elements in a single chunk
    size_t   headerlen;     ///< Size of chunk header
} t_zdf_par_layout;

/**
 * Write data that is distributed over a parallel universe using MPI and serial I/O. Only root node
 * writes to disk, overlapping communication and file I/O.
 * @param  zdf     Parallel ZDF file handle
 * @param  dataset Dataset to write. Dataset parameters describe global data
 * @param  chunk   Local dataset chunk (includes pointer to data)
 * @param  layout  Data layout
 * @return         Returns 1 if successful, 0 otherwise.
 */
static int zdf_par_write_cdset_mpi( t_zdf_par_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk,
    const t_zdf_par_layout* layout ) {

    enum commm_tags { ping_tag = 1001, tile_tag, data_tag };
    int ping = 1234;

    const unsigned ndims = dataset -> ndims;
    const size_t type_size = zdf_sizeof( dataset -> data_type );
    const MPI_Datatype mpi_type = zdf_mpi_type( dataset -> data_type );

    const MPI_Comm comm = zdf -> comm;
    int comm_rank, comm_size;
    MPI_Comm_rank( comm, &comm_rank );
    MPI_Comm_size( comm, &comm_size );

    // Root allocates communication buffers (sized for the largest chunk)
    void * buf[2] = { NULL, NULL };
    int ok = 1;
    if ( comm_rank == 0 && comm_size > 1 && layout -> max_count > 0 ) {
        buf[0] = malloc( layout -> max_count * type_size );
        buf[1] = malloc( layout -> max_count * type_size );
        if ( !buf[0] || !buf[1] ) {
            fprintf(stderr, "(*error*) Unable to allocate communication buffers for parallel I/O.\n");
            ok = 0;
        }
    }
    if ( !zdf_par_agree( comm, ok ) ) {
        free( buf[0] ); free( buf[1] );
        return(0);
    }

    if ( comm_rank == 0 ) {

        t_zdf_chunk recv[2];
        recv[0].data = buf[0];
        recv[1].data = buf[1];

        t_zdf_chunk * current = chunk;
        uint64_t written = 0;

        for( int src = 0; src < comm_size; src++ ) {
            MPI_Request ping_req = MPI_REQUEST_NULL, data_req = MPI_REQUEST_NULL;
            t_zdf_mpi_count c; c.derived = 0;
            t_zdf_chunk * next = NULL;

            // Start receiving data from next node
            if ( src + 1 < comm_size ) {
                next = &recv[ (src+1) % 2 ];

                MPI_Isend( (void *) &ping, 1, MPI_INT, src + 1, ping_tag, comm, &ping_req );

                uint64_t info[ 3 * zdf_max_dims ];
                MPI_Recv( (void *) info, 3 * ndims, MPI_UINT64_T, src + 1, tile_tag, comm, MPI_STATUS_IGNORE );
                for( unsigned i = 0; i < ndims; i++ ) {
                    next -> count[i]  = info[i];
                    next -> start[i]  = info[ndims + i];
                    next -> stride[i] = info[2*ndims + i];
                }

                uint64_t recv_size = zdf_chunk_size( next, ndims );
                if ( ! zdf_mpi_count_init( &c, mpi_type, recv_size ) ) {
                    fprintf(stderr, "(*error*) Chunk is too large.\n");
                    MPI_Abort( comm, 1 );
                }
                MPI_Irecv( next -> data, c.count, c.type, src + 1, data_tag, comm, &data_req );
            }

            // Write data from current node
            size_t count = zdf_write_cdset( &zdf->zdf_file, dataset, current );
            if ( count ) written += count;
            else ok = 0;

            // Wait for messages to complete
            MPI_Wait( &ping_req, MPI_STATUS_IGNORE );
            MPI_Wait( &data_req, MPI_STATUS_IGNORE );
            zdf_mpi_count_free( &c );

            current = next;
        }

        zdf -> fpos += written;

        // Free communication buffers
        free( buf[0] ); free( buf[1] );

    } else {
        // Wait for ping from root node
        MPI_Recv( (void *) &ping, 1, MPI_INT, 0, ping_tag, comm, MPI_STATUS_IGNORE );

        // Send local chunk information to root node
        uint64_t info[ 3 * zdf_max_dims ];
        for( unsigned i = 0; i < ndims; i++ ) {
            info[i]           = chunk -> count[i];
            info[ndims + i]   = chunk -> start[i];
            info[2*ndims + i] = chunk -> stride[i];
        }
        MPI_Send( (void *) info, 3 * ndims, MPI_UINT64_T, 0, tile_tag, comm );

        // Send data to root node
        t_zdf_mpi_count c;
        if ( ! zdf_mpi_count_init( &c, mpi_type, layout -> count ) ) {
            fprintf(stderr, "(*error*) Chunk is too large.\n");
            MPI_Abort( comm, 1 );
        }
        MPI_Send( chunk -> data, c.count, c.type, 0, data_tag, comm );
        zdf_mpi_count_free( &c );
    }

    return zdf_par_agree( comm, ok );
}


/**
 * Write data that is distributed over a parallel universe, with each parallel node independently
 * writing its part to the file.
 * @param  zdf      Parallel ZDF file handle
 * @param  dataset  Dataset to write. Dataset parameters describe global data
 * @param  chunk    Local dataset chunk (includes pointer to data)
 * @param  layout   Data layout
 * @return          Returns 1 if successful, 0 otherwise.
 */
static int zdf_par_write_cdset_indep( t_zdf_par_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk,
    const t_zdf_par_layout* layout ){

    int rank, size;
    MPI_Comm_rank( zdf->comm, &rank );
    MPI_Comm_size( zdf->comm, &size );

    // Only root node kept track of file position, broadcast it to every node
    int64_t fpos = zdf->fpos;
    MPI_Bcast( &fpos, 1, MPI_INT64_T, 0, zdf -> comm );

    // Get the file offset for this rank
    int64_t file_offset = fpos + layout -> byte_offset + rank * layout -> headerlen;

    int ok = 1;

    // Position file pointer
    if ( fseeko( zdf -> zdf_file.fp, file_offset, SEEK_SET ) ) {
        perror("(*error*) Unable to position file pointer for parallel I/O");
        ok = 0;
    }

    // Write data
    if ( ok && zdf_write_cdset( &zdf -> zdf_file, dataset, chunk ) == 0 ) {
        fprintf(stderr,"(*error*) Unable to write data using parallel I/O.\n");
        ok = 0;
    }

    // Set the file position variable to the end of the dataset on all nodes
    zdf -> fpos = fpos + layout -> total_bytes + (int64_t) size * layout -> headerlen;

    // Set file position to the end of the dataset on root node
    if ( ok && rank == 0 ) {
        if ( fseeko( zdf -> zdf_file.fp, zdf -> fpos, SEEK_SET ) ) {
            perror("(*error*) Unable to position file pointer at the end of the file on root node");
            ok = 0;
        }
    }

    return zdf_par_agree( zdf -> comm, ok );
}

/**
 * Write parallel dataset using MPI/IO. Each rank writes its chunk header and data in a
 * single operation, using an explicit file offset.
 * @param  zdf      Parallel ZDF file handle
 * @param  dataset  Dataset to write. Dataset parameters describe global data
 * @param  chunk    Local dataset chunk (includes pointer to data)
 * @param  layout   Data layout
 * @param  coll     Use collective (_all) write operations if set to true
 * @return          Returns 1 on success, 0 on error.
 */
static int zdf_par_write_cdset_mpiio( t_zdf_par_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk,
    const t_zdf_par_layout* layout, int const coll ){

    int rank, size;
    MPI_Comm_rank( zdf->comm, &rank );
    MPI_Comm_size( zdf->comm, &size );

    // Write chunk header to memory buffer
    t_zdf_membuf m;
    int ok = zdf_membuf_open( &m, zdf -> zdf_file.ndatasets );
    if ( ok ) {
        size_t headerlen = zdf_write_chunk_header( &m.file, dataset, chunk );
        ok = zdf_membuf_close( &m ) && ( headerlen == layout -> headerlen ) && ( m.size == headerlen );
    }

    // Collective write follows, so make sure everyone is ready
    if ( !zdf_par_agree( zdf -> comm, ok ) ) {
        zdf_membuf_free( &m );
        return(0);
    }

    // Create memory datatype with chunk header, chunk data and padding, using absolute addresses
    static const uint8_t pad[BYTES_PER_ZDF_UNIT] = {0};
    const uint64_t data_bytes = layout -> count * zdf_sizeof( dataset -> data_type );
    const int npad = (int) ( layout -> bytes - data_bytes );

    t_zdf_mpi_count c; c.derived = 0;
    int          blocklen[3];
    MPI_Aint     disp[3];
    MPI_Datatype type[3];
    int nblocks = 0;

    blocklen[nblocks] = (int) layout -> headerlen;
    MPI_Get_address( m.buf, &disp[nblocks] );
    type[nblocks] = MPI_BYTE;
    nblocks++;

    if ( layout -> count > 0 ) {
        if ( ! zdf_mpi_count_init( &c, zdf_mpi_type( dataset -> data_type ), layout -> count ) ) {
            fprintf(stderr, "(*error*) Chunk is too large.\n");
            MPI_Abort( zdf -> comm, 1 );
        }
        blocklen[nblocks] = c.count;
        MPI_Get_address( chunk -> data, &disp[nblocks] );
        type[nblocks] = c.type;
        nblocks++;
    }

    if ( npad > 0 ) {
        blocklen[nblocks] = npad;
        MPI_Get_address( (void *) pad, &disp[nblocks] );
        type[nblocks] = MPI_BYTE;
        nblocks++;
    }

    MPI_Datatype mem_type;
    MPI_Type_create_struct( nblocks, blocklen, disp, type, &mem_type );
    MPI_Type_commit( &mem_type );

    // File offset for this rank
    MPI_Offset file_offset = zdf -> fpos + layout -> byte_offset + rank * layout -> headerlen;

    // Write header and data in a single call
    int err;
    if ( coll ) {
        err = MPI_File_write_at_all( zdf -> fh, file_offset, MPI_BOTTOM, 1, mem_type, MPI_STATUS_IGNORE );
    } else {
        err = MPI_File_write_at( zdf -> fh, file_offset, MPI_BOTTOM, 1, mem_type, MPI_STATUS_IGNORE );
    }
    if ( err != MPI_SUCCESS ) {
        fprintf(stderr, "(*error*) MPI/IO write failed.\n");
        ok = 0;
    }

    // Free MPI datatypes and header buffer
    MPI_Type_free( &mem_type );
    zdf_mpi_count_free( &c );
    zdf_membuf_free( &m );

    // Set the file position variable to the end of the dataset on all nodes
    zdf -> fpos += layout -> total_bytes + (MPI_Offset) size * layout -> headerlen;

    return zdf_par_agree( zdf -> comm, ok );
}

/**
 * Adds dataset chunks from a parallel grid to file
 * @param  zdf      Parallel ZDF file handle
 * @param  dataset  Dataset object describing global data
 * @param  chunk    Parallel chunk object with local data
 * @param  offset   Data offset in file layout (in units of data elements). If set to a negative value the code
 *                  will calculate this from individual chunk sizes. If set, it must match the value
 *                  calculated from the chunk sizes.
 * @return          1 on success, 0 on error
 */
int zdf_par_write_cdset( t_zdf_par_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk, const int64_t offset ){

    // Get local rank
    int rank; MPI_Comm_rank( zdf->comm, &rank );

    // Validate dataset
    int ok = ( dataset -> ndims >= 1 && dataset -> ndims <= zdf_max_dims &&
               zdf_mpi_type( dataset -> data_type ) != MPI_DATATYPE_NULL );

    t_zdf_par_layout layout;
    layout.count = ok ? zdf_chunk_size( chunk, dataset -> ndims ) : 0;
    layout.bytes = ok ? zdf_sizeof_vector( dataset -> data_type, layout.count ) : 0;
    layout.headerlen = ok ? size_zdf_chunk_header( dataset ) : 0;

    // Global sums and local offsets
    uint64_t local[3]  = { layout.count, layout.bytes, (uint64_t) ( ok ? 0 : 1 ) };
    uint64_t global[3], prefix[2] = {0, 0};
    MPI_Allreduce( local, global, 3, MPI_UINT64_T, MPI_SUM, zdf -> comm );
    MPI_Exscan( local, prefix, 2, MPI_UINT64_T, MPI_SUM, zdf -> comm );
    if ( rank == 0 ) { prefix[0] = 0; prefix[1] = 0; }

    if ( global[2] ) {
        if ( rank == 0 ) fprintf(stderr, "(*error*) zdf_par_write_cdset: invalid dataset.\n");
        return(0);
    }

    layout.byte_offset = prefix[1];
    layout.total_bytes = global[1];

    // Chunks must not hold more data than the dataset
    uint64_t dataset_count = 1;
    for( unsigned i = 0; i < dataset -> ndims; i++ ) dataset_count *= dataset -> count[i];
    if ( global[0] > dataset_count ) {
        if ( rank == 0 ) {
            fprintf(stderr, "(*error*) zdf_par_write_cdset: chunks hold more elements (%" PRIu64
                ") than the dataset (%" PRIu64 "), do they overlap?\n", global[0], dataset_count );
        }
        return(0);
    }

    // If offset was specified it must be consistent with the file layout
    ok = ( offset < 0 ) || ( (uint64_t) offset == prefix[0] );
    if ( !zdf_par_agree( zdf -> comm, ok ) ) {
        if ( rank == 0 ) fprintf(stderr, "(*error*) zdf_par_write_cdset: invalid chunk offset, "
            "chunks must be stored contiguously in rank order.\n");
        return(0);
    }

    // Largest chunk (used for ZDF_MPI buffers)
    layout.max_count = 0;
    if ( zdf -> iomode == ZDF_MPI ) {
        MPI_Allreduce( &layout.count, &layout.max_count, 1, MPI_UINT64_T, MPI_MAX, zdf -> comm );
    }

    switch( zdf -> iomode ) {
        case ZDF_MPI:
            return zdf_par_write_cdset_mpi( zdf, dataset, chunk, &layout );
        case ZDF_INDEPENDENT:
            return zdf_par_write_cdset_indep( zdf, dataset, chunk, &layout );
        case ZDF_MPIIO_INDEPENDENT:
            return zdf_par_write_cdset_mpiio( zdf, dataset, chunk, &layout, 0 );
        case ZDF_MPIIO_COLLECTIVE:
            return zdf_par_write_cdset_mpiio( zdf, dataset, chunk, &layout, 1 );
    }

    if ( rank == 0 ) fprintf(stderr, "(*error*) Invalid parallel IO mode, unable to write dataset.\n");
    return( 0 );
}


/* -----------------------------------------------------------------------------------------------
  High level interface
-------------------------------------------------------------------------------------------------- */

/**
 * @brief Save a distributed parallel grid zdf file
 * 
 * @param chunk         Local data buffer and information
 * @param data_type     Data type
 * @param info          Global data grid information
 * @param iter          Iteration metadata
 * @param path          Base path to save the file
 * @param comm          MPI communicator
 * @param par_io_mode   Parallel I/O algorithm to use
 * @return int          Returns 1 on success, 0 on error
 */
int zdf_par_save_grid( t_zdf_chunk *chunk, const enum zdf_data_type data_type,
    const t_zdf_grid_info *info, const t_zdf_iteration *iteration,
    char const path[], MPI_Comm comm, const t_zdf_parallel_io_mode par_io_mode )
{
    int rank; MPI_Comm_rank( comm, &rank );

    // Ensure that the path is available
    if ( !zdf_par_create_path( path, comm ) ) return(0);

    // Build filename
    char filename[PATH_MAX];
    int n = snprintf( filename, sizeof(filename), "%s/%s-%06u.zdf", path, info->name, (unsigned) iteration -> n );
    if ( n < 0 || (size_t) n >= sizeof(filename) ) {
        if ( rank == 0 ) fprintf(stderr,"(*error*) ZDF file name is too long.\n");
        return(0);
    }

    // Create parallel ZDF file
    t_zdf_par_file zdf;
    if ( !zdf_par_open_file( &zdf, filename, ZDF_CREATE, comm, par_io_mode ) ) {
        if ( rank == 0 ) fprintf(stderr,"(*error*) Unable to open parallel ZDF file, aborting.\n");
        return(0);
    }

    t_zdf_dataset dataset = {
        .name = info -> name,
        .data_type = data_type,
        .ndims = info->ndims,
        .data = NULL
    };
    for( unsigned i = 0; i < info->ndims && i < zdf_max_dims; i ++) dataset.count[i] = info->count[i];

    // All routines are collective and return the same value on all ranks, so all ranks
    // stop at the same point if there is an error
    int ok = zdf_par_add_string( &zdf, "TYPE", "grid") &&
             zdf_par_add_grid_info( &zdf, info ) &&
             zdf_par_add_iteration( &zdf, iteration ) &&
             zdf_par_start_cdset( &zdf, &dataset ) &&
             zdf_par_write_cdset( &zdf, &dataset, chunk, -1 ) &&
             zdf_par_end_cdset( &zdf, &dataset );

    if ( !ok && rank == 0 ) fprintf(stderr,"(*error*) Unable to write parallel ZDF file %s\n", filename );

    // Close ZDF file and return
    int closed = zdf_par_close_file( &zdf );
    return( ok && closed );
}
