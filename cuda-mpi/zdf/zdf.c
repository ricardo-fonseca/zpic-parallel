/**
 * @file zdf.c
 * @author Ricardo
 * @brief ZDF file library
 * @version 1.1
 * @date 2022-02-04
 * 
 * @copyright Copyright (c) 2022
 * 
 * This ZDF version is totally self contained. It does not depend on XDR or any other
 * external libraries. Data is always stored little endian; big endian systems swap bytes on
 * write / read.
 * This version is not compatible is version 0 (files have the opposite endianess)
 * 
 * Version 1.1 changes:
 * - 8 and 16 bit vectors are padded to BYTES_PER_ZDF_UNIT, and record lengths include
 *   the padding (previously 8 bit records reported the unpadded length and 16 bit vectors
 *   were not padded). Files with 32/64 bit data are unchanged.
 * - Consistent return values (1 / 0 for int functions, bytes / 0 for size_t functions)
 * - ZDF_UPDATE mode scans the file for existing datasets
 * - Binary file modes on all platforms, fixed big endian implementation
 */

/**
 * @brief  Use 2008 edition of the POSIX standard (IEEE Standard 1003.1-2008)
 * 
 */
#include <linux/limits.h>
#define _POSIX_C_SOURCE 200809L

/**
 * @brief  Use 64 bit file interface
 * 
 */
#define _FILE_OFFSET_BITS 64

#include "zdf.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>

#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/**
 * On Windows we cannot use the POSIX.1 mkdir command, so use _mkdir instead
 */
#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#define mkdir(path,mode) _mkdir(path)
#define ZDF_DIR_MODE 0

// Windows also lacks the 'fseeko', 'ftello' functions so replace with Windows
// equivalent.
#define fseeko _fseeki64
#define ftello _ftelli64
typedef long long zdf_off_t;

#define ZDF_IS_PATH_SEP(c) ( (c) == '/' || (c) == '\\' )

#else

#define ZDF_DIR_MODE (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)
typedef off_t zdf_off_t;
#define ZDF_IS_PATH_SEP(c) ( (c) == '/' )

#endif

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

/**
 * Byte order. Windows targets are always little endian and do not define __BYTE_ORDER__
 */
#if defined(__BYTE_ORDER__)
  #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #define ZDF_BIG_ENDIAN 1
  #elif __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    #error "System is neither little endian nor big endian, aborting."
  #endif
#endif

/**
 * Number of bytes required for writing data to ZDF file
 * (round up to multiple of BYTES_PER_ZDF_UNIT)
 */
#define RNDUP(x)  ((((x) + BYTES_PER_ZDF_UNIT - 1) / BYTES_PER_ZDF_UNIT) \
            * BYTES_PER_ZDF_UNIT)

/**
 * Buffer size (bytes) for converting big endian to little endian data
 */
#define ENDIAN_CONV_BUF_SIZE 8192

/**
 * Sizes of datatypes
 */
static const unsigned size_zdf_int32   = 4;    ///< size of zdf_int32
static const unsigned size_zdf_uint32  = 4;    ///< size of zdf_uint32
static const unsigned size_zdf_uint64  = 8;    ///< size of zdf_uint64
static const unsigned size_zdf_double  = 8;    ///< size of zdf_double


/**
 * IDs of ZDF records
 */
#define ZDF_INT32_ID        0x00010000  ///< Int32 record ID
#define ZDF_DOUBLE_ID       0x00020000  ///< Double record ID
#define ZDF_STRING_ID       0x00030000  ///< String record ID

#define ZDF_DATASET_ID      0x00100002  ///< Dataset record ID
#define ZDF_CDSET_START_ID  0x00110000  ///< Chunked dataset start record ID
#define ZDF_CDSET_CHUNK_ID  0x00120000  ///< Chunked dataset data chunk record ID
#define ZDF_CDSET_END_ID    0x00130000  ///< Chunked dataset end record ID

#define ZDF_ITERATION_ID     0x00200001  ///< Iteration record ID
#define ZDF_GRID_INFO_ID     0x00210001  ///< Grid information record ID
#define ZDF_PART_INFO_ID     0x00220002  ///< Particle set information record ID
#define ZDF_TRACK_INFO_ID    0x00230001  ///< Particle tracks information record ID

/* -----------------------------------------------------------------------------------------------
  recursively create path if required
-------------------------------------------------------------------------------------------------- */

/**
 * @brief Creates a single directory, ignoring the error if it already exists
 * @param  dir  Directory to create
 * @return      0 on success, errno value on error
 */
static int zdf_mkdir( const char * dir ) {
    if ( dir[0] == '\0' ) return 0;
    if ( mkdir( dir, ZDF_DIR_MODE ) == 0 ) return 0;

    // mkdir may fail with errors other than EEXIST for existing directories
    // (e.g. EACCES / EROFS on some file systems), so check for the directory
    int err = errno;
    struct stat st;
    if ( stat( dir, &st ) == 0 ) {
        return S_ISDIR( st.st_mode ) ? 0 : ENOTDIR;
    }
    return err;
}

/**
 * @brief Recursively creates a path if required
 * @param  path path to create
 * @return      returns 0 on success, otherwise returns an errno value
 */
int create_path( const char path[] )
{
    if ( path == NULL || path[0] == '\0' ) return 0;

    size_t len = strlen( path );
    char * tmp = (char *) malloc( len + 1 );
    if ( tmp == NULL ) return ENOMEM;
    memcpy( tmp, path, len + 1 );

    // Create each component of the path in turn
    int ierr = 0;
    for( size_t i = 1; i <= len; i++ ) {
        if ( tmp[i] == '\0' || ZDF_IS_PATH_SEP( tmp[i] ) ) {
            char c = tmp[i];
            tmp[i] = '\0';
            ierr = zdf_mkdir( tmp );
            tmp[i] = c;
            if ( ierr ) break;
        }
    }

    if ( ierr ) {
        fprintf( stderr, "(*error*) Unable to create path %s: %s\n", path, strerror( ierr ) );
    }

    free( tmp );
    return ierr;
}


/**
 * Returns size of ZDF datatype
 * @param  data_type Data type id
 * @return           Returns type size in bytes or 0 for an invalid data type
 */
size_t zdf_sizeof( enum zdf_data_type data_type ) {
    switch( data_type ) {
        case zdf_null:
            return(0);
        case zdf_int8:
        case zdf_uint8:
            return(1);
        case zdf_int16:
        case zdf_uint16:
            return(2);
        case zdf_int32:
        case zdf_uint32:
        case zdf_float32:
            return(4);
        case zdf_int64:
        case zdf_uint64:
        case zdf_float64:
        case zdf_complex64:
            return(8);
        case zdf_complex128:
            return(16);
    }
    return(0);
}

/**
 * Returns the number of bytes a vector occupies in the file, including padding
 * @param  data_type Data type id
 * @param  len       Number of elements
 * @return           Size in bytes
 */
size_t zdf_sizeof_vector( enum zdf_data_type data_type, uint64_t len ) {
    return RNDUP( zdf_sizeof( data_type ) * len );
}

/* -----------------------------------------------------------------------------------------------
  Elemental types
-------------------------------------------------------------------------------------------------- */

#ifdef ZDF_BIG_ENDIAN
/**
 * @brief Reverse the byte order of a value in place
 * @param p     Pointer to value
 * @param size  Size of value in bytes
 */
static void zdf_bswap( void * p, size_t size ) {
    uint8_t * b = (uint8_t *) p;
    for( size_t i = 0; i < size / 2; i++ ) {
        uint8_t t = b[i]; b[i] = b[size-1-i]; b[size-1-i] = t;
    }
}
#endif

/**
 * @brief Writes a scalar value (1, 2, 4 or 8 bytes) to file in little endian order
 * @param  zdf  ZDF file descriptor
 * @param  v    Pointer to value
 * @param  size Size of value in bytes
 * @return      Returns number of bytes written on success, 0 on error
 */
static size_t zdf_scalar_write( t_zdf_file* zdf, const void * v, size_t size ) {
    uint8_t tmp[8];
    memcpy( tmp, v, size );
#ifdef ZDF_BIG_ENDIAN
    zdf_bswap( tmp, size );
#endif
    if ( fwrite( (void *) tmp, size, 1, zdf -> fp ) != 1 ) return(0);
    return( size );
}

/**
 * @brief Reads a scalar value (1, 2, 4 or 8 bytes) stored in little endian order from file
 * @param  zdf  ZDF file descriptor
 * @param  v    Pointer to value
 * @param  size Size of value in bytes
 * @return      Returns number of bytes read on success, 0 on error
 */
static size_t zdf_scalar_read( t_zdf_file* zdf, void * v, size_t size ) {
    if ( fread( v, size, 1, zdf -> fp ) != 1 ) return(0);
#ifdef ZDF_BIG_ENDIAN
    zdf_bswap( v, size );
#endif
    return( size );
}

size_t zdf_int32_write( t_zdf_file* zdf, const int32_t i )   { return zdf_scalar_write( zdf, &i, sizeof(int32_t) ); }
size_t zdf_int32_read( t_zdf_file* zdf, int32_t* i )         { return zdf_scalar_read( zdf, i, sizeof(int32_t) ); }
size_t zdf_uint32_write( t_zdf_file* zdf, const uint32_t u ) { return zdf_scalar_write( zdf, &u, sizeof(uint32_t) ); }
size_t zdf_uint32_read( t_zdf_file* zdf, uint32_t* u )       { return zdf_scalar_read( zdf, u, sizeof(uint32_t) ); }
size_t zdf_int64_write( t_zdf_file* zdf, const int64_t i )   { return zdf_scalar_write( zdf, &i, sizeof(int64_t) ); }
size_t zdf_int64_read( t_zdf_file* zdf, int64_t *i )         { return zdf_scalar_read( zdf, i, sizeof(int64_t) ); }
size_t zdf_uint64_write( t_zdf_file* zdf, const uint64_t u ) { return zdf_scalar_write( zdf, &u, sizeof(uint64_t) ); }
size_t zdf_uint64_read( t_zdf_file* zdf, uint64_t *u )       { return zdf_scalar_read( zdf, u, sizeof(uint64_t) ); }
size_t zdf_double_write( t_zdf_file* zdf, const double d )   { return zdf_scalar_write( zdf, &d, sizeof(double) ); }
size_t zdf_double_read( t_zdf_file* zdf, double *d )         { return zdf_scalar_read( zdf, d, sizeof(double) ); }

/**
 * @brief Writes a vector of elements of size elsize (1, 2, 4 or 8 bytes) to file, in little
 *        endian order, adding padding at the end to maintain BYTES_PER_ZDF_UNIT alignment.
 * @param  zdf      ZDF file descriptor
 * @param  data     Pointer to data
 * @param  elsize   Element size in bytes
 * @param  len      Number of elements
 * @return          Returns number of bytes written (including padding), 0 on error
 */
static size_t zdf_vectorN_write( t_zdf_file* zdf, void const * const data, size_t elsize, size_t len ) {

    if ( len == 0 ) return 0;

#ifndef ZDF_BIG_ENDIAN
    if ( fwrite( data, elsize, len, zdf -> fp ) != len ) return(0);
#else
    uint8_t buffer[ENDIAN_CONV_BUF_SIZE];
    const uint8_t * src = (const uint8_t *) data;
    const size_t chunk_max = ENDIAN_CONV_BUF_SIZE / elsize;

    for( size_t offset = 0; offset < len; offset += chunk_max ) {
        size_t n = ( len - offset < chunk_max ) ? len - offset : chunk_max;
        memcpy( buffer, src + offset * elsize, n * elsize );
        if ( elsize > 1 ) {
            for( size_t i = 0; i < n; i++ ) zdf_bswap( buffer + i * elsize, elsize );
        }
        if ( fwrite( (void *) buffer, elsize, n, zdf -> fp ) != n ) return(0);
    }
#endif

    // Add padding if required
    size_t nbytes = len * elsize;
    size_t npad = RNDUP(nbytes) - nbytes;
    if ( npad > 0 ) {
        const uint8_t pad[BYTES_PER_ZDF_UNIT] = {0};
        if ( fwrite( (void *) pad, sizeof(uint8_t), npad, zdf -> fp ) != npad ) return(0);
    }

    return( RNDUP(nbytes) );
}

/**
 * Writes a vector of the specified datatype to file
 * @param  zdf       ZDF file descriptor
 * @param  data      Pointer to data
 * @param  data_type Data type descriptor
 * @param  len       Number of elements in vector
 * @return           Returns number of bytes written on success, 0 on error
 */
size_t zdf_vector_write( t_zdf_file* zdf, const void * data, enum zdf_data_type data_type, size_t len ){

     switch ( data_type ) {
        case zdf_int8:
        case zdf_uint8:
            return zdf_vectorN_write( zdf, data, 1, len );
        case zdf_int16:
        case zdf_uint16:
            return zdf_vectorN_write( zdf, data, 2, len );
        case zdf_int32:
        case zdf_uint32:
        case zdf_float32:
            return zdf_vectorN_write( zdf, data, 4, len );
        case zdf_int64:
        case zdf_uint64:
        case zdf_float64:
            return zdf_vectorN_write( zdf, data, 8, len );

        // Complex types are stored as float pairs (real,img)
        case zdf_complex64:
            return zdf_vectorN_write( zdf, data, 4, 2*len );
        case zdf_complex128:
            return zdf_vectorN_write( zdf, data, 8, 2*len );

        default:
            fprintf(stderr,"(*error*) zdf_vector_write: Unsupported datatype.\n");
     }
    return(0);
 }

/* -----------------------------------------------------------------------------------------------
  Open / Close ZDF file
-------------------------------------------------------------------------------------------------- */

/**
 * Closes ZDF file
 * @param  zdf ZDF file to close
 * @return     Returns 1 on success, 0 on error
 */
int zdf_close_file( t_zdf_file* zdf ) {

    if ( zdf -> fp == NULL ) {
        fprintf(stderr, "(*error*) Unable to close ZDF file, file is not open\n");
        return(0);
    }

    int ok = ( fclose( zdf->fp ) == 0 );
    if ( !ok ) perror("(*error*) Unable to close ZDF file");
    zdf -> fp = NULL;
    return( ok );
}

static int zdf_scan_datasets( t_zdf_file* zdf );

/**
 * Checks magic number at the beginning of the file
 * @param  zdf ZDF file descriptor
 * @return     1 if magic number is valid, 0 otherwise
 */
static int zdf_check_magic( t_zdf_file* zdf ) {
    char test_magic[ZDF_MAGIC_LENGTH];

    if ( fread( (void *) test_magic, sizeof(char), ZDF_MAGIC_LENGTH, zdf->fp )
        != ZDF_MAGIC_LENGTH) {
        fprintf(stderr, "(*error*) Unable to read magic number from ZDF file.\n");
        return(0);
    }

    if ( memcmp( test_magic, zdf_magic, ZDF_MAGIC_LENGTH ) ) {
        fprintf(stderr, "(*error*) Invalid magic number, file is not a proper ZDF file.\n");
        return(0);
    }

    return(1);
}

/**
 * Opens ZDF file
 * @param  zdf      ZDF file to open
 * @param  filename Filename of the ZDF file to open, including path
 * @param  mode     Can be one of ZDF_CREATE, ZDF_READ, or ZDF_UPDATE
 * @return          Returns 1 on success, 0 on error
 */
int zdf_open_file( t_zdf_file* zdf, const char* filename, enum zdf_file_access_mode mode ){

    zdf -> mode = mode;
    zdf -> ndatasets = 0;
    zdf -> fp = NULL;

    // Binary modes ("b") must be used for compatibility with Windows
    switch( mode ) {
        case ZDF_CREATE :
            // Open file for writing
            if (!(zdf->fp = fopen( filename, "w+b"))) {
                perror("(*error*) Unable to open ZDF file for writing");
                fprintf(stderr, "(*error*) file: %s\n", filename );
                return(0);
            }

            // Write magic number
            if ( fwrite( (void *) zdf_magic, sizeof(char), ZDF_MAGIC_LENGTH, zdf->fp )
                 != ZDF_MAGIC_LENGTH ) {
                fprintf(stderr, "(*error*) Unable to write magic number to ZDF file.\n");
                zdf_close_file( zdf );
                return(0);
            }

            break;

        case ZDF_READ :
            // Open file for reading
            if (!(zdf->fp = fopen( filename, "rb"))) {
                perror("(*error*) Unable to open ZDF file for reading");
                fprintf(stderr, "(*error*) file: %s\n", filename );
                return(0);
            }

            if ( ! zdf_check_magic( zdf ) ) {
                zdf_close_file( zdf );
                return(0);
            }
            break;

        case ZDF_UPDATE :
            // Open file for reading and writing
            if (!(zdf->fp = fopen( filename, "r+b"))) {
                perror("(*error*) Unable to open ZDF file for reading / writing");
                fprintf(stderr, "(*error*) file: %s\n", filename );
                return(0);
            }

            if ( ! zdf_check_magic( zdf ) ) {
                zdf_close_file( zdf );
                return(0);
            }

            // Find existing datasets so that new datasets get unique ids
            if ( ! zdf_scan_datasets( zdf ) ) {
                zdf_close_file( zdf );
                return(0);
            }

            // Position the file pointer at the end of the file
            if ( fseeko( zdf->fp, 0, SEEK_END ) ) {
                perror("(*error*) Unable to position file pointer at the end of the file");
                zdf_close_file( zdf );
                return(0);
            }

            break;

        default:
            fprintf(stderr, "(*error*) zdf_open_file: unsupported mode.\n");
            return(0);
    }

    return(1);
}

/* -----------------------------------------------------------------------------------------------
  zdf_string
-------------------------------------------------------------------------------------------------- */

/**
 * Write a string to file
 * @param  zdf ZDF file descriptor
 * @param  str C string (char *) to write
 * @return     Returns number of bytes written on success, 0 on error
 */
size_t zdf_string_write( t_zdf_file* zdf, const char * str ){

    size_t slen = ( str ) ? strlen( str ) : 0;
    if ( slen > UINT32_MAX ) {
        fprintf(stderr, "(*error*) String is too long.\n");
        return(0);
    }

    uint32_t len = (uint32_t) slen;
    if ( ! zdf_uint32_write( zdf, len ) ) return(0);

    size_t count = 0;
    if ( len > 0 ) {
        if ( !(count = zdf_vectorN_write( zdf, (void *) str, 1, len ) ) )
            return(0);
    }

    return( sizeof(uint32_t) + count );
}

/**
 * Read a string from file
 * @param  zdf ZDF file descriptor
 * @param  str C string (char *) to read. The routine will allocate memory using
 *             malloc() for storing the string data. On error *str is set to NULL.
 * @return     Returns number of bytes read on success, 0 on error
 */
size_t zdf_string_read( t_zdf_file* zdf, char * * str ){

    *str = NULL;

    uint32_t len;
    if ( ! zdf_uint32_read( zdf, &len) ) return(0);

    size_t flen = RNDUP( (size_t) len );

    char * buffer = (char *) malloc( flen + 1 );
    if ( buffer == NULL ) {
        fprintf(stderr, "(*error*) Unable to allocate memory for string.\n");
        return(0);
    }

    if ( len > 0 ) {
        if ( fread( (void *) buffer, sizeof(uint8_t), flen, zdf -> fp ) != flen ) {
            free( buffer );
            return(0);
        }
    }

    buffer[len] = 0;

    *str = buffer;

    return( sizeof(uint32_t) + flen );
}

/**
 * Calculate size of string record
 * @param  s C string (char *) to write
 * @return   The number of bytes required for writing the string data
 */
size_t size_zdf_string( const char *s )
{
    size_t len = ( s ) ? strlen( s ) : 0;
    return size_zdf_uint32 + RNDUP(len);
}


/* -----------------------------------------------------------------------------------------------
  zdf records
-------------------------------------------------------------------------------------------------- */

/**
 * @brief ZDF Record
 * 
 */
typedef struct ZDF_Record{
    uint32_t id_version;    ///< id & version
    char*    name;          ///< Name
    uint64_t length;        ///< Record length
} t_zdf_record;

/**
 * Adds ZDF record header to file
 * @param  zdf ZDF file descriptor
 * @param  rec ZDF record
 * @return     Returns number of bytes written on success, 0 on error
 */
size_t zdf_record_write( t_zdf_file* zdf, const t_zdf_record* rec ){

    size_t len;
    if ( ! zdf_uint32_write( zdf, rec -> id_version ) ) return(0);
    if ( ! (len = zdf_string_write( zdf, rec -> name )) ) return(0);
    if ( ! zdf_uint64_write( zdf, rec -> length )     ) return(0);

    return( sizeof(uint32_t) + len + sizeof(uint64_t) );
}

/**
 * Reads ZDF record header from file. The record name is allocated using malloc() and must be
 * freed by the caller (it is set to NULL on error).
 * @param  zdf ZDF file descriptor
 * @param  rec ZDF record
 * @return     Returns number of bytes read on success, 0 on error
 */
size_t zdf_record_read( t_zdf_file* zdf, t_zdf_record* rec ){

    size_t len;
    rec -> name = NULL;
    if ( ! zdf_uint32_read( zdf, & rec -> id_version ) ) return(0);
    if ( ! (len = zdf_string_read( zdf, & rec -> name )) ) return(0);
    if ( ! zdf_uint64_read( zdf, & rec -> length ) ) {
        free( rec -> name ); rec -> name = NULL;
        return(0);
    }

    return( sizeof(uint32_t) + len + sizeof(uint64_t) );
}

/**
 * Skips to the end of ZDF record. File pointer is assumed to be after the record header.
 * @param  zdf ZDF file descriptor
 * @param  rec ZDF record
 * @return     Returns 1 on success, 0 on error
 */
int zdf_record_skip( t_zdf_file* zdf, const t_zdf_record* rec ){

    zdf_off_t offset = (zdf_off_t) rec -> length;
    return ( ! fseeko( zdf -> fp, offset, SEEK_CUR ) ? 1 : 0 );
}

/**
 * @brief Scans the file for existing datasets and sets zdf->ndatasets to the highest
 *        dataset id found. File position is left undefined.
 * @param zdf   ZDF file descriptor
 * @return      1 on success, 0 on error
 */
static int zdf_scan_datasets( t_zdf_file* zdf ) {

    if ( fseeko( zdf -> fp, ZDF_MAGIC_LENGTH, SEEK_SET ) ) return(0);

    uint32_t maxid = 0;
    t_zdf_record rec;
    while( zdf_record_read( zdf, &rec ) ) {
        free( rec.name );
        if ( rec.id_version == ZDF_CDSET_START_ID || rec.id_version == ZDF_DATASET_ID ) {
            // First value in the dataset header is the dataset id
            uint32_t id;
            if ( ! zdf_uint32_read( zdf, &id ) ) break;
            if ( id > maxid ) maxid = id;
            rec.length -= size_zdf_uint32;
        }
        if ( ! zdf_record_skip( zdf, &rec ) ) break;
    }

    if ( ! feof( zdf -> fp ) ) {
        fprintf(stderr, "(*error*) Error while scanning ZDF file for existing datasets.\n");
        return(0);
    }
    clearerr( zdf -> fp );

    zdf -> ndatasets = maxid;
    return(1);
}

/* -----------------------------------------------------------------------------------------------
  zdf basic data tags
-------------------------------------------------------------------------------------------------- */

/**
 * Adds string element to ZDF file
 * @param  zdf  ZDF File handle
 * @param  name Element name
 * @param  str  String value
 * @return      Returns number of bytes written on success, 0 on error
 */
size_t zdf_add_string( t_zdf_file* zdf, const char* name, const char* str ){

    t_zdf_record rec = {
        .id_version = ZDF_STRING_ID,
        .name = (char *)name,
        .length = size_zdf_string( str )
    };

    size_t ok, len;
    len = ( ok = zdf_record_write( zdf, &rec ) );
    if ( !ok ) return(0);

    len += ( ok = zdf_string_write( zdf, str) );
    if ( !ok ) return(0);

    return(len);
}

/**
 * Adds int32 element to ZDF file
 * @param  zdf    ZDF File handle
 * @param  name   Element name
 * @param  value  int32 value
 * @return        Returns number of bytes written on success, 0 on error
 */
size_t zdf_add_int32( t_zdf_file* zdf, const char* name, const int32_t value ){

    t_zdf_record rec = {
        .id_version = ZDF_INT32_ID,
        .name = (char *)name,
        .length = size_zdf_int32
    };

    size_t ok, len;
    len = ( ok = zdf_record_write(zdf, &rec) );
    if (!ok) return(0);

    len += ( ok = zdf_int32_write(zdf, value) );
    if (!ok) return(0);

    return( len );
}

/**
 * Adds float64 element to ZDF file
 * @param  zdf    ZDF file handle
 * @param  name   Element name
 * @param  value  float64 value
 * @return        Returns number of bytes written on success, 0 on error
 */
size_t zdf_add_double( t_zdf_file* zdf, const char* name, const double value )
{
    t_zdf_record rec = {
        .id_version = ZDF_DOUBLE_ID,
        .name = (char *)name,
        .length = size_zdf_double
    };

    size_t ok, len;
    len = (ok = zdf_record_write(zdf, &rec) );    if ( !ok ) return(0);
    len += ( ok = zdf_double_write(zdf, value) ); if ( !ok ) return(0);

    return(len);
}


/* -----------------------------------------------------------------------------------------------
  zdf compound metadata tags
-------------------------------------------------------------------------------------------------- */

/**
 * Adds iteration metadata group to ZDF file
 * @param  zdf  ZDF file handle
 * @param  iter Iteration info
 * @return      Number of bytes written on success, 0 on error
 */
size_t zdf_add_iteration( t_zdf_file* zdf, const t_zdf_iteration* iter ){

    t_zdf_record rec = {
        .id_version = ZDF_ITERATION_ID,
        .name = (char *) iter -> name,
        .length = size_zdf_int32 +
                  size_zdf_double +
                  size_zdf_string( iter -> time_units )
    };

    // Default name
    if ( ! rec.name ) rec.name = (char *) "Iteration";

    size_t ok, len;

    len = ( ok = zdf_record_write( zdf, &rec) );      if ( !ok ) return(0);
    len += ( ok = zdf_int32_write( zdf, iter ->n ) ); if ( !ok ) return(0);
    len += ( ok = zdf_double_write( zdf, iter->t ) ); if ( !ok ) return(0);
    len += ( ok = zdf_string_write( zdf, iter->time_units ) ); if ( !ok ) return(0);

    return(len);
}

/**
 * Returns size of grid information metadata element
 * @param  grid Grid info
 * @return Metadata group size in bytes
 */
size_t size_zdf_grid_info(const t_zdf_grid_info* grid) {
    size_t size;

    size = size_zdf_uint32 + grid -> ndims * size_zdf_uint64 +
            size_zdf_string(grid->label) +  size_zdf_string(grid->units);

    // Includes axis information
    size += size_zdf_int32;
    if ( grid -> axis ) {

        for(unsigned i=0; i<grid -> ndims; i++)
            size += size_zdf_string( grid->axis[i].name ) +
                    size_zdf_int32 +
                    2 * size_zdf_double +
                    size_zdf_string( grid -> axis[i].label ) +
                    size_zdf_string( grid -> axis[i].units );
    }

    return size;
}

/**
 * Adds grid information metadata group to file
 * @param  zdf  File handle
 * @param  grid Grid information
 * @return      Number of bytes written on success, 0 on error
 */
size_t zdf_add_grid_info( t_zdf_file* zdf, const t_zdf_grid_info* grid ){

    if ( grid -> ndims < 1 || grid -> ndims > zdf_max_dims ) {
        fprintf(stderr, "(*error*) Invalid number of grid dimensions (%u)\n", (unsigned) grid -> ndims );
        return(0);
    }

    t_zdf_record rec = {
        .id_version = ZDF_GRID_INFO_ID,
        .name = (char *) grid -> name,
        .length = size_zdf_grid_info( grid )
    };

    size_t reclen;
    if ( !( reclen = zdf_record_write( zdf, &rec ) )  ) return(0);

    if ( ! zdf_uint32_write( zdf, grid -> ndims ) ) return(0);
    for( unsigned i=0; i < grid -> ndims; i++) {
        if ( ! zdf_uint64_write( zdf, grid -> count[i] ) ) return(0);
    }

    if ( !zdf_string_write( zdf, grid->label ) ) return(0);
    if ( !zdf_string_write( zdf, grid->units ) ) return(0);

    int32_t has_axis = ( grid -> axis != NULL);
    if ( !zdf_int32_write( zdf, has_axis ) ) return(0);

    if ( has_axis ) {

        for( unsigned i=0; i < grid -> ndims; i++) {
            if ( !zdf_string_write( zdf, grid -> axis[i].name  ) ) return(0);
            if ( !zdf_int32_write(  zdf, grid -> axis[i].type  ) ) return(0);
            if ( !zdf_double_write( zdf, grid -> axis[i].min   ) ) return(0);
            if ( !zdf_double_write( zdf, grid -> axis[i].max   ) ) return(0);
            if ( !zdf_string_write( zdf, grid -> axis[i].label ) ) return(0);
            if ( !zdf_string_write( zdf, grid -> axis[i].units ) ) return(0);
        }
    }

    return( reclen + size_zdf_grid_info( grid ) );
}

/**
 * Returns size of particles information metadata group
 * @param  part Particles information
 * @return      Metadata group size in bytes
 */
size_t size_zdf_part_info(const t_zdf_part_info* part) {
    size_t size = size_zdf_string(part->label) +    // label
            size_zdf_uint64 +                       // np
            size_zdf_uint32;                        // nquants

    for( unsigned i = 0; i < part -> nquants; i++) {
        size += size_zdf_string( part -> quants[i] ) +
                size_zdf_string( part -> qlabels[i] ) +
                size_zdf_string( part -> qunits[i] );
    }

    return size;
}

/**
 * Adds particle information metadata group to file
 * @param  zdf  File handle
 * @param  part Particle information
 * @return      Number of bytes written on success, 0 on error
 */
size_t zdf_add_part_info( t_zdf_file* zdf, const t_zdf_part_info* part ){

    size_t reclen;

    t_zdf_record rec = {
        .id_version = ZDF_PART_INFO_ID,
        .name = (char *) part->name,
        .length = size_zdf_part_info( part )
    };

    if ( !( reclen = zdf_record_write( zdf, &rec ) ) ) return(0);
    if ( !zdf_string_write( zdf, part -> label )     ) return(0);
    if ( !zdf_uint64_write( zdf, part -> np )        ) return(0);
    if ( !zdf_uint32_write( zdf, part -> nquants )   ) return(0);

    for( unsigned i=0; i < part -> nquants; i++) {
        if ( !zdf_string_write( zdf, part -> quants[i] ) ) return(0);
    }
    for( unsigned i=0; i < part -> nquants; i++) {
        if ( !zdf_string_write( zdf, part -> qlabels[i] ) ) return(0);
    }
    for( unsigned i=0; i < part -> nquants; i++) {
        if ( !zdf_string_write( zdf, part -> qunits[i] )  ) return(0);
    }

    return( reclen + size_zdf_part_info( part ) );
}

/**
 * Returns size of tracks information metadata group
 * @param  tracks Tracks information
 * @return        Metadata group size in bytes
 */
size_t size_zdf_track_info(const t_zdf_track_info* tracks) {
    size_t size = size_zdf_string(tracks->label) +    // label
            4 * size_zdf_uint32; // ntracks, ndump, niter, nquants

    for( unsigned i = 0; i < tracks -> nquants; i++) {
        size += size_zdf_string( tracks -> quants[i] ) +
                size_zdf_string( tracks -> qlabels[i] ) +
                size_zdf_string( tracks -> qunits[i] );
    }

    return size;
}

/**
 * Adds tracks information metadata group to file
 * @param  zdf    File handle
 * @param  tracks Tracks information
 * @return        Number of bytes written on success, 0 on error
 */
size_t zdf_add_track_info( t_zdf_file* zdf, const t_zdf_track_info* tracks ){

    size_t reclen;

    t_zdf_record rec = {
        .id_version = ZDF_TRACK_INFO_ID,
        .name = (char *)tracks -> name,
        .length = size_zdf_track_info( tracks )
    };

    if ( !( reclen = zdf_record_write( zdf, &rec ) ) ) return(0);
    if ( !zdf_string_write( zdf, tracks -> label )   ) return(0);
    if ( !zdf_uint32_write( zdf, tracks -> ntracks ) ) return(0);
    if ( !zdf_uint32_write( zdf, tracks -> ndump )   ) return(0);
    if ( !zdf_uint32_write( zdf, tracks -> niter )   ) return(0);
    if ( !zdf_uint32_write( zdf, tracks -> nquants ) ) return(0);

    for( unsigned i=0; i < tracks -> nquants; i++) {
        if ( !zdf_string_write( zdf, tracks -> quants[i] ) ) return(0);
    }
    for( unsigned i=0; i < tracks -> nquants; i++) {
        if ( !zdf_string_write( zdf, tracks -> qlabels[i] ) ) return(0);
    }
    for( unsigned i=0; i < tracks -> nquants; i++) {
        if ( !zdf_string_write( zdf, tracks -> qunits[i] ) ) return(0);
    }

    return( reclen + size_zdf_track_info( tracks ) );
}
/* -----------------------------------------------------------------------------------------------
  zdf dataset
-------------------------------------------------------------------------------------------------- */

/**
 * Returns the size of the dataset header
 * @param  dataset Dataset object
 * @return         Size of dataset header in bytes
 */
size_t size_zdf_dataset_header( const t_zdf_dataset* dataset ) {
    return size_zdf_uint32 + size_zdf_int32 + size_zdf_uint32 +
        dataset -> ndims * size_zdf_uint64;
}

/**
 * @brief Checks if dataset parameters are valid
 * @param dataset   Dataset object
 * @return          1 if valid, 0 otherwise
 */
static int zdf_valid_dataset( const t_zdf_dataset* dataset ) {
    if ( dataset -> ndims < 1 || dataset -> ndims > zdf_max_dims ) {
        fprintf(stderr, "(*error*) Invalid number of dataset dimensions (%u)\n", (unsigned) dataset -> ndims );
        return(0);
    }
    if ( zdf_sizeof( dataset -> data_type ) == 0 ) {
        fprintf(stderr, "(*error*) Invalid dataset data type (%d)\n", (int) dataset -> data_type );
        return(0);
    }
    return(1);
}

/**
 * Write dataset header to file and modify the dataset object to include the
 * file position where the dataset is stored.
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Number of bytes written on success, 0 on error
 */
size_t zdf_dataset_header_write( t_zdf_file* zdf, t_zdf_dataset* dataset ) {

    // Get the file position for the dataset header and store it
    zdf_off_t offset = ftello( zdf -> fp );
    if ( offset < 0 ) return(0);
    dataset -> offset = offset;

    // Version 0x0001
    uint32_t id = dataset -> id;
    if ( !zdf_uint32_write( zdf, id )                  ) return(0);
    if ( !zdf_int32_write( zdf, dataset -> data_type ) ) return(0);
    if ( !zdf_uint32_write( zdf, dataset -> ndims )    ) return(0);

    for( unsigned i=0; i < dataset -> ndims; i++) {
        if ( !zdf_uint64_write( zdf, dataset -> count[i] ) ) return(0);
    }
    return( size_zdf_dataset_header( dataset ) );
}

/**
 * Read dataset header from file and store the file position where the dataset
 * header is stored.
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Number of bytes read on success, 0 on error
 */
size_t zdf_dataset_header_read( t_zdf_file* zdf, t_zdf_dataset* dataset ) {

    // Get the file position for the dataset header and store it
    zdf_off_t offset = ftello( zdf -> fp );
    if ( offset < 0 ) return(0);
    dataset -> offset = offset;

    // Version 0x0001
    uint32_t id;
    if ( !zdf_uint32_read( zdf, &id ) ) return(0);
    dataset -> id = id;

    int32_t data_type;
    if ( !zdf_int32_read( zdf, &data_type ) ) return(0);
    dataset -> data_type = (enum zdf_data_type) data_type;

    if ( !zdf_uint32_read( zdf, &dataset -> ndims ) ) return(0);
    if ( dataset -> ndims > zdf_max_dims ) {
        fprintf(stderr, "(*error*) Invalid number of dataset dimensions in file (%u)\n", (unsigned) dataset -> ndims );
        return(0);
    }

    for( unsigned i=0; i < dataset -> ndims; i++) {
        if ( !zdf_uint64_read( zdf, &dataset -> count[i] ) ) return(0);
    }
    return( size_zdf_dataset_header( dataset ) );
}

/**
 * Adds dataset to file
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object (includes pointer to data)
 * @return         Number of bytes written on success, 0 on error. The dataset object is also
 *                 modified to include a unique id.
 */
size_t zdf_add_dataset( t_zdf_file* zdf, t_zdf_dataset* dataset ){

    if ( ! zdf_valid_dataset( dataset ) ) return(0);

    uint64_t count = 1;
    for( unsigned i = 0; i < dataset -> ndims; i++ ) count *= dataset -> count[i];

    size_t datalen = zdf_sizeof_vector( dataset -> data_type, count );
    size_t length = size_zdf_dataset_header( dataset ) + datalen;

    t_zdf_record rec = {
        .id_version = ZDF_DATASET_ID,
        .name       = (char *) dataset -> name,
        .length     = length
    };

    size_t reclen;
    if ( !( reclen = zdf_record_write( zdf, &rec ) ) ) return(0);

    dataset -> id = ++ zdf -> ndatasets;
    if ( !zdf_dataset_header_write( zdf, dataset ) ) return(0);

    if ( count > 0 ) {
        if ( zdf_vector_write( zdf, dataset -> data, dataset -> data_type, count ) != datalen ) return(0);
    }

    return( reclen + length );
}

/**
 * Adds a chunked dataset header to file
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Number of bytes written on success, 0 on error. The dataset object is also
 *                 modified to include a unique id.
 */
size_t zdf_start_cdset( t_zdf_file* zdf, t_zdf_dataset* dataset ) {

    if ( ! zdf_valid_dataset( dataset ) ) return(0);

    t_zdf_record rec = {
        .id_version = ZDF_CDSET_START_ID,
        .name       = (char *) dataset -> name,
        .length     = size_zdf_dataset_header( dataset )
    };

    size_t ok, len;

    len = ( ok =  zdf_record_write( zdf, &rec ) );
    if ( !ok ) return(0);

    dataset -> id = ++ zdf -> ndatasets;
    len += (ok = zdf_dataset_header_write( zdf, dataset ) );
    if ( !ok ) return(0);

    return( len );
}

/**
 * Returns size of chunk header
 * 
 * The chunk record name is "%08x-chunk" (14 characters for ids < 2^32), so the
 * name field is always 4 + 16 bytes. The header size is therefore the same for all
 * chunks of a dataset, which is required by the parallel I/O routines.
 * 
 * @param  dataset Dataset object
 * @return         Chunk header size in bytes
 */
size_t size_zdf_chunk_header(const t_zdf_dataset* dataset) {

    return size_zdf_uint32 + (size_zdf_uint32 + 16) + size_zdf_uint64 + 
           size_zdf_uint32 + 3 * dataset -> ndims * size_zdf_uint64;
}

/**
 * Adds a chunk header to file
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @param  chunk   Chunk object
 * @return         Number of bytes written on success, 0 on error.
 */
size_t zdf_write_chunk_header( t_zdf_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk ){

    char name[16];
    snprintf( name, 16, "%08" PRIx32 "-chunk", (uint32_t) dataset -> id);

    uint64_t count = 1;
    for( unsigned i = 0; i < dataset -> ndims; i++ ) count *= chunk -> count[i];

    size_t length = zdf_sizeof_vector( dataset -> data_type, count ) +
                    size_zdf_uint32 + 3 * dataset -> ndims * size_zdf_uint64;

    t_zdf_record rec = {
        .id_version = ZDF_CDSET_CHUNK_ID,
        .name       = (char *)name,
        .length     = length
    };

    size_t reclen;
    if ( !( reclen = zdf_record_write( zdf, &rec ) ) ) return(0);

    // ID of dataset this chunk belongs to
    if ( !zdf_uint32_write( zdf, dataset -> id ) ) return(0);

    // Size of chunk
    for( unsigned i=0; i < dataset -> ndims; i++) {
        if ( !zdf_uint64_write( zdf, chunk -> count[i] ) ) return(0);
    }

    // Start position
    for( unsigned i = 0; i < dataset -> ndims; i++) {
        if ( !zdf_uint64_write( zdf, chunk -> start[i] ) ) return(0);
    }

    // Stride
    for( unsigned i = 0; i < dataset -> ndims; i++) {
        if ( !zdf_uint64_write( zdf, chunk -> stride[i] ) ) return(0);
    }

    return( reclen + size_zdf_uint32 + 3 * dataset -> ndims * size_zdf_uint64 );

}


/**
 * Adds a chunk of dataset data to file
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @param  chunk   Chunk object (includes pointer to data). Data is expected to be contiguos in memory
 * @return         Number of bytes written on success, 0 on error.
 */
size_t zdf_write_cdset( t_zdf_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk ){

    size_t headerlen = zdf_write_chunk_header( zdf, dataset, chunk );
    if ( !headerlen ) return(0);

    // Chunk data
    uint64_t count = 1;
    for( unsigned i = 0; i < dataset -> ndims; i++ ) count *= chunk -> count[i];

    // Empty chunks are valid (e.g. no particles on this node)
    size_t vectorlen = 0;
    if ( count > 0 ) {
        if ( !( vectorlen = zdf_vector_write( zdf, chunk -> data, dataset -> data_type, count ) ) )
            return(0);
    }

    return( headerlen + vectorlen );
}

/**
 * Add an end marker for a chunked dataset to file
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Number of bytes written on success, 0 on error.
 */
size_t zdf_end_cdset( t_zdf_file* zdf, t_zdf_dataset* dataset ){

    char name[16];
    snprintf( name, 16, "%08" PRIx32 "-end", (uint32_t) dataset -> id);

    t_zdf_record rec = {
        .id_version = ZDF_CDSET_END_ID,
        .name       = name,
        .length     = 0
    };

    size_t reclen;
    if ( !( reclen = zdf_record_write( zdf, &rec ) ) ) return(0);

    return(reclen);
}


/**
 * Open a dataset and read the header. The dataset is selected by name.
 * 
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Returns 1 if successful, 0 on error or if the dataset is not found.
 */
size_t zdf_open_dataset( t_zdf_file* zdf, t_zdf_dataset* dataset ) {

    if ( dataset -> name == NULL ) {
        fprintf(stderr,"(*error*) zdf_open_dataset: dataset name not specified\n");
        return(0);
    }

    // Set file position after the magic number
    if ( fseeko( zdf -> fp, ZDF_MAGIC_LENGTH, SEEK_SET ) ) return(0);

    // Locate the requested dataset
    int found = 0;
    t_zdf_record rec;
    while( zdf_record_read( zdf, &rec ) ) {

        int is_dataset = ( rec.id_version == ZDF_CDSET_START_ID || rec.id_version == ZDF_DATASET_ID );
        int match = is_dataset && ( strcmp( dataset -> name, rec.name ) == 0 );
        free( rec.name );

        if ( match ) {
            // Read the dataset header
            if ( !zdf_dataset_header_read( zdf, dataset ) ) return(0);
            found = 1;
            break;
        }

        // Skip the record
        if ( !zdf_record_skip( zdf, &rec ) ) return(0);
    }

    if ( !found ) {
        if ( feof( zdf -> fp ) ) {
            fprintf(stderr,"(*error*) Unable to find dataset %s\n", dataset -> name);
        } else {
            fprintf(stderr,"(*error*) Error reading file while looking for dataset %s\n", dataset -> name);
        }
        clearerr( zdf -> fp );
        return(0);
    }

    // re-position the file pointer at the end of the file
    if ( fseeko( zdf->fp, 0, SEEK_END ) ) return(0);
    return(1);

}

/**
 * Extends dataset dimensions. After this call the file pointer is positioned at the
 * end of the file.
 *
 * @param  zdf       ZDF file handle
 * @param  dataset   Datasete object
 * @param  new_count New dimensions for the dataset, must be >= than the previous
 *                   dimensions.
 * @return           Returns 1 if successful, 0 otherwise.
 */
int zdf_extend_dataset( t_zdf_file* zdf, t_zdf_dataset* dataset, uint64_t* new_count ) {

    for( unsigned i = 0; i < dataset -> ndims; i++ ) {
        if ( new_count[i] < dataset -> count[i] ) {
            fprintf(stderr,"(*error*) Invalid value for zdf_extend_dataset.\n");
            fprintf(stderr,"(*error*) New size is smaller than original size.\n");
            return(0);
        }
    }

    for( unsigned i = 0; i < dataset -> ndims; i++ ) dataset -> count[i] = new_count[i];

    zdf_off_t offset = (zdf_off_t) dataset -> offset;
    if ( fseeko( zdf -> fp, offset , SEEK_SET ) ) return(0);
    if ( ! zdf_dataset_header_write( zdf, dataset ) ) return(0);
    if ( fseeko( zdf->fp, 0, SEEK_END ) ) return(0);

    return(1);
}

/* -----------------------------------------------------------------------------------------------
  zdf high level interface
-------------------------------------------------------------------------------------------------- */

/**
 * Opens ZDF file and adds TYPE, GRID, and ITERATION metadata
 * @param  zdf       File handle
 * @param  info      Grid information
 * @param  iteration Iteration information
 * @param  path      File path
 * @return           Returns 1 on success, 0 on error
 */
int zdf_open_grid_file( t_zdf_file *zdf, const t_zdf_grid_info *info,
    const t_zdf_iteration *iteration, char const path[] ){

    char filename[PATH_MAX];

    // Ensure that the path is available
    if ( create_path( path ) ) return(0);

    // Build filename
    int n = snprintf( filename, sizeof(filename), "%s/%s-%06u.zdf", path, info->name, (unsigned) iteration -> n );
    if ( n < 0 || (size_t) n >= sizeof(filename) ) {
        fprintf(stderr,"(*error*) ZDF file name is too long.\n");
        return(0);
    }

    // Create ZDF file
    if ( !zdf_open_file( zdf, filename, ZDF_CREATE ) ) {
        fprintf(stderr,"(*error*) Unable to open ZDF file, aborting.\n");
        return(0);
    }

    // Add file type, grid info and iteration info
    if ( !zdf_add_string( zdf, "TYPE", "grid") ||
         !zdf_add_grid_info( zdf, info ) ||
         !zdf_add_iteration( zdf, iteration ) ) {
        fprintf(stderr,"(*error*) Unable to write metadata to ZDF file %s\n", filename);
        zdf_close_file( zdf );
        return(0);
    }

    return( 1 );
}


/**
 * @brief Saves a ZDF grid file
 * 
 * @param  data       Pointer to grid data
 * @param  data_type  ZDF Data type of grid data
 * @param  info       Grid information
 * @param  iteration  Iteration information
 * @param  path       File path
 * @return            Returns 1 on success, 0 on error
 */
int zdf_save_grid( const void * data, const enum zdf_data_type data_type, const t_zdf_grid_info *info,
    const t_zdf_iteration *iteration, char const path[] )
{

    t_zdf_file zdf;

    // Open grid file
    if ( !zdf_open_grid_file( &zdf, info, iteration, path ) ) return(0);

    // Add dataset
    t_zdf_dataset dataset = {
        .name = info -> name,
        .data_type = data_type,
        .ndims = info->ndims,
        .data = (void *) data
    };
    for( unsigned i = 0; i < info->ndims; i ++) dataset.count[i] = info->count[i];

    int ok = ( zdf_add_dataset( &zdf, &dataset ) != 0 );
    if ( !ok ) fprintf(stderr,"(*error*) Unable to write dataset to ZDF file.\n");

    // Close ZDF file and return
    ok = zdf_close_file( &zdf ) && ok;
    return( ok );
}

/**
 * Opens ZDF file and adds TYPE, PARTICLES, and ITERATION metadata
 * @param  zdf        File handle
 * @param  info       Particles information
 * @param  iteration  Iteration information
 * @param  path       File path
 * @return            1 on sucess, 0 on error
 */
int zdf_open_part_file( t_zdf_file *zdf, t_zdf_part_info *info,
    const t_zdf_iteration *iteration, char const path[] ){

    char filename[PATH_MAX];

    // Ensure that the path is available
    if ( create_path( path ) ) return(0);

    // Build filename
    int n = snprintf( filename, sizeof(filename), "%s/%s-%s-%06u.zdf", path, "particles", info->name, (unsigned) iteration -> n );
    if ( n < 0 || (size_t) n >= sizeof(filename) ) {
        fprintf(stderr,"(*error*) ZDF file name is too long.\n");
        return(0);
    }

    // Create ZDF file
    if ( !zdf_open_file( zdf, filename, ZDF_CREATE ) ) {
        fprintf(stderr,"(*error*) Unable to open ZDF file, aborting.\n");
        return(0);
    }

    // Add file type, particle info and iteration info
    if ( !zdf_add_string( zdf, "TYPE", "particles") ||
         !zdf_add_part_info( zdf, info ) ||
         !zdf_add_iteration( zdf, iteration ) ) {
        fprintf(stderr,"(*error*) Unable to write metadata to ZDF file %s\n", filename);
        zdf_close_file( zdf );
        return(0);
    }

    return(1);
}

/**
 * Adds individual particle quantity to file
 * @param  zdf  File handle
 * @param  name Quantity name
 * @param  data Quantity data (float32)
 * @param  np   Number of particles
 * @return      1 on success, 0 on error
 */
int zdf_add_quant_part_file( t_zdf_file *zdf, const char *name, const float* data,
    const uint64_t np ) {

    t_zdf_dataset dataset = {
        .name = (char *) name,
        .data_type = zdf_float32,
        .ndims = 1,
        .data = (void *) data
    };

    dataset.count[0] = np;

    return( zdf_add_dataset( zdf, &dataset ) != 0 );

}

#ifdef __TEST_ZDF__

#include <math.h>

int main (int argc, const char * argv[]) {

    const unsigned NX = 128;
    float buf[NX];

    for(unsigned i = 0; i < NX; i++) {
        float x = 8 * (M_PI/NX) * (i+1);
        buf[i] = sin(x)/x;
    }

    t_zdf_grid_axis axis[1];
    axis[0] = (t_zdf_grid_axis) {
        .min = -1.0,
        .max =  1.0,
        .label = "axis label",
        .units = "axis units"
    };

    t_zdf_grid_info info = {
        .ndims = 1,
        .label = "data label",
        .units = "data units",
        .axis = axis
    };

    info.count[0] = NX;

    t_zdf_iteration iter = {
        .n = 123,
        .t = 12.3,
        .time_units = "time units"
    };

    return zdf_save_grid( (void *) buf, zdf_float32, &info, &iter, "V1TEST" ) ? 0 : 1;
}

#endif
