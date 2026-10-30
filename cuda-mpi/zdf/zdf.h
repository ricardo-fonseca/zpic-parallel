#ifndef ZDF_H_
#define ZDF_H_

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !(defined _WIN32 || defined _WIN64)

  #if __SIZEOF_FLOAT__ != 4
  #error This version of ZDF only support systems were sizeof(float) = 4
  #endif

  #if __SIZEOF_DOUBLE__ != 8
  #error This version of ZDF only support systems were sizeof(double) = 8
  #endif

#endif

/**
 * Number of bytes in each ZDF file unit. All records, and all data vectors
 * (including 8 and 16 bit types) are padded to a multiple of this value.
 */
#define BYTES_PER_ZDF_UNIT    (4)

/**
 * @brief Magic byte sequence length
 * 
 */
#define ZDF_MAGIC_LENGTH BYTES_PER_ZDF_UNIT

/**
 * @brief Magic byte sequence identifying ZDF files
 * 
 */
static const char zdf_magic[ZDF_MAGIC_LENGTH] = {'Z','D','F','1'};

/**
 * @brief Maximum array dimensions supported
 * 
 */
#define zdf_max_dims 3

/**
 * @brief ZDF data types
 * 
 */
enum zdf_data_type{ 
    zdf_null,        ///< No type
    zdf_int8,        ///< 8 bit signed intger
    zdf_uint8,       ///< 8 bit unsigned integer
    zdf_int16,       ///< 16 bit signed integer
    zdf_uint16,      ///< 16 bit unsigned integer
    zdf_int32,       ///< 32 bit signed integer
    zdf_uint32,      ///< 32 bit unsigned integer
    zdf_int64,       ///< 64 bit integer
    zdf_uint64,      ///< 64 bit unsigned integer
    zdf_float32,     ///< 32 bit floating point
    zdf_float64,     ///< 64 bit floating point
    zdf_complex64,   ///< 64 bit complex floating point (2 x 32 bit)
    zdf_complex128   ///< 128 bit complex floating point (2 x 64 bit)
};

typedef enum zdf_data_type t_zdf_data_type;

/**
 * @brief File access modes
 * 
 */
enum zdf_file_access_mode {
    ZDF_CREATE,        ///< Create new file
    ZDF_READ,        ///< Read file
    ZDF_UPDATE        ///< Update (read/write) file
};

/**
 * @brief ZDF File
 * 
 */
typedef struct ZDF_File{
    FILE *fp;                          ///< File pointer
    enum zdf_file_access_mode mode;    ///< Access mode
    uint32_t ndatasets;                ///< Number of datasets in file
} t_zdf_file;

/**
 * @brief ZDF Dataset
 * 
 */
typedef struct ZDF_Dataset {
    const char* name;                  ///< Dataset name
    enum zdf_data_type data_type;      ///< Data type
    uint32_t ndims;                    ///< Number of dimensions
    uint64_t count[zdf_max_dims];      ///< Dimension for each direction
    void * data;                       ///< Pointer to data buffer
    uint64_t id;                       ///< Optional integer ID
    uint64_t offset;                   ///< File position for dataset header
} t_zdf_dataset;

/**
 * @brief ZDF data chunk
 * 
 */
typedef struct ZDF_Chunk {
    uint64_t count[zdf_max_dims];      ///< Dimension of chunk
    uint64_t start[zdf_max_dims];      ///< Start position of data chunk
    uint64_t stride[zdf_max_dims];     ///< Chunk stride
    void * data;                       ///< Pointer to chunk data (host memory)
} t_zdf_chunk;

/**
 * @brief Types of axis
 * 
 */
enum zdf_axis_type { 
    zdf_linear, ///< Linear axis
    zdf_log10,  ///< Log10 axis
    zdf_log2    ///< Log2 axis
};

/**
 * @brief Grid data axis information
 * 
 */
typedef struct ZDF_GridAxis {
    const char* name;           ///< Axis name
    enum zdf_axis_type type;    ///< Axis type

    double min;                 ///< Minimum value
    double max;                 ///< Maximum value

    const char* label;          ///< Axis label
    const char* units;          ///< Axis units

} t_zdf_grid_axis;

/**
 * @brief Grid data information
 * 
 */
typedef struct ZDF_GridInfo {
    const char* name;               ///< Grid name
    uint32_t ndims;                 ///< Number of grid dimensions {1..zdf_max_dims}
    uint64_t count[zdf_max_dims];   ///< Grid dimensions
    const char* label;              ///< Grid label
    const char* units;              ///< Grid units
    t_zdf_grid_axis *axis;          ///< Grid axis information
} t_zdf_grid_info;

/**
 * @brief Iteration details
 * 
 */
typedef struct ZDF_Iteration{
    const char* name;           ///< Iteration name
    int32_t n;                  ///< Iteration number
    double t;                   ///< Time (phyical units)
    const char* time_units;     ///< Units for time
} t_zdf_iteration;

/**
 * @brief Particle data information
 * 
 */
typedef struct ZDF_PartInfo {
    const char* name;   ///< Particle species name
    const char* label;  ///< Particle species label
    uint64_t np;        ///< Number of particles
    uint32_t nquants;   ///< Number of quantities stored
    char** quants;      ///< Names of quantities
    char** qlabels;     ///< Labels for quantities
    char** qunits;      ///< Units for quantities
} t_zdf_part_info;

/**
 * @brief Tracks data information
 * 
 */
typedef struct ZDF_TrackInfo {
    const char* name;       ///< Particle species name
    const char* label;      ///< Particle species label
    uint32_t ntracks;       ///< Number of tracks in dataset
    uint32_t ndump;         ///< Frequency at which the file was updated
    uint32_t niter;         ///< Number of iterations between track points

    uint32_t nquants;       ///< Number of quantities in tracks
    const char** quants;    ///< Names of quantities
    const char** qlabels;   ///< Labels for quantities
    const char** qunits;    ///< Units for quantities
} t_zdf_track_info;

/*
 * Return value conventions
 * ------------------------
 * Functions returning `int` return 1 on success and 0 on error.
 * Functions returning `size_t` return the number of bytes written (or read) on success
 * and 0 on error, except where noted.
 * `create_path()` is the exception, it follows the POSIX convention (0 on success).
 */

// Low level interface

/**
 * @brief Recursively creates a path if required
 * 
 * @param path      path to create
 * @return int      returns 0 on success, otherwise returns an errno value
 */
int create_path( const char path[] );

/**
 * @brief Returns size of ZDF datatype
 *
 * @param  data_type Data type id
 * @return           Returns type size in bytes or 0 for an invalid data type
 */
size_t zdf_sizeof( enum zdf_data_type data_type );

/**
 * @brief Returns the number of bytes a vector of the specified datatype occupies in a ZDF file,
 *        including the padding required to keep the file aligned to BYTES_PER_ZDF_UNIT
 *
 * @param  data_type Data type id
 * @param  len       Number of elements in vector
 * @return           Size in bytes
 */
size_t zdf_sizeof_vector( enum zdf_data_type data_type, uint64_t len );

/**
 * @brief Opens ZDF file
 * 
 * When opening a file in ZDF_UPDATE mode the file is scanned for existing datasets, so that
 * new datasets get unique ids, and the file pointer is placed at the end of the file.
 * 
 * @param  zdf      ZDF file to open
 * @param  filename Filename of the ZDF file to open, including path
 * @param  mode     Can be one of ZDF_CREATE, ZDF_READ, or ZDF_UPDATE
 * @return          Returns 1 on success, 0 on error
 */
int zdf_open_file( t_zdf_file* zdf, const char* filename, enum zdf_file_access_mode mode );

/**
 * @brief Closes ZDF file
 * @param  zdf ZDF file to close
 * @return     Returns 1 on success, 0 on error
 */
int zdf_close_file( t_zdf_file* zdf );

/**
 * @brief Writes a vector of the specified datatype to file. Padding is added at the end of the
 *        vector if necessary to keep the file aligned to BYTES_PER_ZDF_UNIT.
 *
 * @param  zdf       ZDF file descriptor
 * @param  data      Pointer to data
 * @param  data_type Data type descriptor
 * @param  len       Number of elements in vector
 * @return           Returns number of bytes written on success, 0 on error. Note that
 *                   writing a 0 length vector also returns 0.
 */
size_t zdf_vector_write( t_zdf_file* zdf, const void * data, enum zdf_data_type data_type, size_t len );

/**
 * @brief Adds string element to ZDF file
 *
 * @param  zdf  ZDF File handle
 * @param  name Element name
 * @param  str  String value
 * @return      Returns number of bytes written on success, 0 on error
 */
size_t zdf_add_string( t_zdf_file* zdf, const char* name, const char* str );

/**
 * @brief Adds int32 element to ZDF file
 *
 * @param  zdf    ZDF File handle
 * @param  name   Element name
 * @param  value  int32 value
 * @return        Returns number of bytes written on success, 0 on error
 */
size_t zdf_add_int32( t_zdf_file* zdf, const char* name, const int32_t value );

/**
 * @brief Adds float64 element to ZDF file
 *
 * @param  zdf    ZDF file handle
 * @param  name   Element name
 * @param  value  float64 value
 * @return        Returns number of bytes written on success, 0 on error
 */
size_t zdf_add_double( t_zdf_file* zdf, const char* name, const double value );

/**
 * @brief Adds iteration metadata group to ZDF file
 *
 * @param  zdf  ZDF file handle
 * @param  iter Iteration info
 * @return      Number of bytes written on success, 0 on error
 */
size_t zdf_add_iteration( t_zdf_file* zdf, const t_zdf_iteration* iter );

/**
 * @brief Adds grid information metadata group to file
 *
 * @param  zdf  File handle
 * @param  grid Grid information
 * @return      Number of bytes written on success, 0 on error
 */
size_t zdf_add_grid_info( t_zdf_file* zdf, const t_zdf_grid_info* grid );

/**
 * @brief Adds particle information metadata group to file
 *
 * @param  zdf  File handle
 * @param  part Particle information
 * @return      Number of bytes written on success, 0 on error
 */
size_t zdf_add_part_info( t_zdf_file* zdf, const t_zdf_part_info* part );

/**
 * @brief Adds tracks information metadata group to file
 *
 * @param  zdf    File handle
 * @param  tracks Tracks information
 * @return        Number of bytes written on success, 0 on error
 */
size_t zdf_add_track_info( t_zdf_file* zdf, const t_zdf_track_info* tracks );

/**
 * @brief Adds dataset to file
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object (includes pointer to data)
 * @return         Number of bytes written on success, 0 on error. The dataset object is also
 *                 modified to include a unique id.
 */
size_t zdf_add_dataset( t_zdf_file* zdf, t_zdf_dataset* dataset );

// Chunked dataset interface

/**
 * @brief Writes a dataset header (id, type, dimensions) at the current file position. The dataset
 *        object is modified to store the file position of the header.
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Number of bytes written on success, 0 on error
 */
size_t zdf_dataset_header_write( t_zdf_file* zdf, t_zdf_dataset* dataset );

/**
 * @brief Adds a chunked dataset header to file
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Number of bytes written on success, 0 on error. The dataset object is also
 *                 modified to include a unique id.
 */
size_t zdf_start_cdset( t_zdf_file* zdf, t_zdf_dataset* dataset );

/**
 * @brief Returns size of chunk header
 *
 * @param  dataset Dataset object
 * @return         Chunk header size in bytes
 */
size_t size_zdf_chunk_header(const t_zdf_dataset* dataset);

/**
 * @brief Adds a chunk header to file
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @param  chunk   Chunk object
 * @return         Number of bytes written on success, 0 on error.
 */
size_t zdf_write_chunk_header( t_zdf_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk );

/**
 * @brief Adds a chunk of dataset data to file
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @param  chunk   Chunk object (includes pointer to data). Data is expected to be contiguos in memory.
 *                 Empty chunks (0 elements) are allowed.
 * @return         Number of bytes written on success, 0 on error.
 */
size_t zdf_write_cdset( t_zdf_file* zdf, t_zdf_dataset* dataset, t_zdf_chunk* chunk );

/**
 * @brief Add an end marker for a chunked dataset to file
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Number of bytes written on success, 0 on error.
 */
size_t zdf_end_cdset( t_zdf_file* zdf, t_zdf_dataset* dataset );

/**
 * @brief Open a dataset and read the header. The dataset is selected by name.
 *
 * @param  zdf     ZDF file handle
 * @param  dataset Dataset object
 * @return         Returns 1 if successful, 0 on error or if the dataset is not found.
 */
size_t zdf_open_dataset( t_zdf_file* zdf, t_zdf_dataset* dataset );

/**
 * @brief Extends dataset dimensions. After this call the file pointer is positioned at the
 *        end of the file.
 *
 * @param  zdf       ZDF file handle
 * @param  dataset   Datasete object
 * @param  new_count New dimensions for the dataset, must be >= than the previous
 *                   dimensions.
 * @return           Returns 1 if successful, 0 otherwise.
 */
int zdf_extend_dataset( t_zdf_file* zdf, t_zdf_dataset* dataset, uint64_t* new_count );


// High level interface

/**
 * @brief Opens ZDF file and adds TYPE, GRID, and ITERATION metadata
 *
 * @param  zdf       File handle
 * @param  info      Grid information
 * @param  iteration Iteration information
 * @param  path      File path
 * @return           Returns 1 on success, 0 on error
 */
int zdf_open_grid_file( t_zdf_file *file, const t_zdf_grid_info *info,
    const t_zdf_iteration *iteration, char const path[] );

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
int zdf_save_grid( const void* data, enum zdf_data_type data_type, const t_zdf_grid_info *info,
    const t_zdf_iteration *iteration, char const path[] );

/**
 * @brief Opens ZDF file and adds TYPE, PARTICLES, and ITERATION metadata
 *
 * @param  zdf        File handle
 * @param  info       Particles information
 * @param  iteration  Iteration information
 * @param  path       File path
 * @return            1 on sucess, 0 on error
 */
int zdf_open_part_file( t_zdf_file *file, t_zdf_part_info *info,
    const t_zdf_iteration *iteration, char const path[] );

/**
 * @brief Adds individual particle quantity to file
 *
 * @param  zdf  File handle
 * @param  name Quantity name
 * @param  data Quantity data (float32)
 * @param  np   Number of particles
 * @return      1 on success, 0 on error
 */
int zdf_add_quant_part_file( t_zdf_file *zdf, const char *name, const float* data,
    const uint64_t np );


#ifdef __cplusplus
}
#endif

#endif
