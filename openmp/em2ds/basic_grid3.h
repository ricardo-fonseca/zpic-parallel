#ifndef BASIC_GRID3_H_
#define BASIC_GRID3_H_

/**
 * @brief 3 contiguous grids
 * 
 * @note Used as 3 contiguous 2D array in device memory
 * 
 * @tparam T    grid datatype
 */
template <class T>
class basic_grid3{

    public:

    /// @brief Data buffer   
    T * d_buffer;    

    /// @brief x component
    T * x;

    /// @brief y component
    T * y;

    /// @brief z component
    T * z;

    /// @brief Grid size
    const uint2 dims;

    /// @brief Object name
    std::string name;
    
    /**
     * @brief Construct a new basic grid object
     * 
     * @param dims  Grid dimensions
     */
    basic_grid3( uint2 dims ) : dims(dims) {
        // Allocate main data buffer on device memory
        d_buffer = memory::malloc<T>( buffer_size() );

        // Get pointers to x, y and z buffers
        x = & d_buffer[                   0 ];
        y = & d_buffer[     dims.x * dims.y ];
        z = & d_buffer[ 2 * dims.x * dims.y ];
    }

    /**
     * @brief Destroy the basic grid object
     * 
     */
    ~basic_grid3() {
        memory::free( d_buffer );
    }

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() {
        return 3 * dims.x * dims.y ;
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const basic_grid3<T>& obj) {
        os << obj.name << "{";
        os << "(" << obj.dims.x << ", " << obj.dims.y << ") x 3";
        return os;
    }

    /**
     * @brief zero device data on a grid grid
     * 
     * @return int       Returns 0 on success, -1 on error
     */
    int zero( ) {
        memory::zero( d_buffer, buffer_size() );
        return 0;
    };

    /**
     * @brief Sets data to a constant value
     * 
     * @param val_x     x value
     * @param val_y     y value
     * @param val_z     z value
     */
    void set( T const & val_x, T const & val_y, T const & val_z  ){
        #pragma omp parallel for
        for( int i = 0; i < dims.y * dims.x; i++ ) {
            x[i] = val_x;
            y[i] = val_y;
            z[i] = val_z;
        }
    };

    /**
     * @brief Adds another grid object on top of local object
     * 
     * @param rhs         Other object to add
     */
    void add( const basic_grid3<T> &rhs ) {
        size_t const size = buffer_size( );

        #pragma omp parallel for
        for( size_t i = 0; i < size; i++ ) d_buffer[i] += rhs.d_buffer[i];
    };

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( fcomp::cart fc, std::string filename ) {

        uint64_t grid_dims[] = {dims.x, dims.y};

        switch( fc ) {
        case fcomp::z : 
            zdf::save_grid( z, 2, grid_dims, name + "-z", filename );
            break;
        case fcomp::y :
            zdf::save_grid( y, 2, grid_dims, name + "-y", filename );
            break;
        case fcomp::x :
            zdf::save_grid( x, 2, grid_dims, name + "-x", filename );
            break;
        default:
            std::cerr << "basic_grid3::save() - Invalid fc\n";
            std::exit(1);
        }
    }

    /**
     * @brief Save grid values to disk
     * 
     * @param fc    Field component to save
     * @param info  Grid metadata (label, units, axis, etc.). Information is used to set file name
     * @param iter  Iteration metadata
     * @param path  Path where to save the file
     */
    void save( fcomp::cart fc, zdf::grid_info &info, zdf::iteration &iter, std::string path ) {
        switch( fc ) {
        case fcomp::z : 
            zdf::save_grid( z, info, iter, path );    
            break;
        case fcomp::y :
            zdf::save_grid( y, info, iter, path );    
            break;
        case fcomp::x :
            zdf::save_grid( x, info, iter, path );    
            break;
        default:
            std::cerr << "basic_grid3::save() - Invalid fc\n";
            std::exit(1);
        }
    }
};

#endif
