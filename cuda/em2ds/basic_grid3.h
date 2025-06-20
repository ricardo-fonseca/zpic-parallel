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
        d_buffer = device::malloc<T>( buffer_size() );

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
        device::free( d_buffer );
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
        device::zero( d_buffer, buffer_size() );
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
        device::setval( x, dims.x * dims.y, val_x );
        device::setval( y, dims.x * dims.y, val_y );
        device::setval( z, dims.x * dims.y, val_z );
    };

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( fcomp::cart fc, std::string filename ) {
        T * h_buffer = host::malloc< T >( dims.x * dims.y );

        switch( fc ) {
        case fcomp::z : 
            device::memcpy_tohost( z, h_buffer, dims.x * dims.y );
            break;
        case fcomp::y :
            device::memcpy_tohost( y, h_buffer, dims.x * dims.y );
            break;
        case fcomp::x :
            device::memcpy_tohost( x, h_buffer, dims.x * dims.y );
            break;
        default:
            ABORT( "basic_grid3::save() - Invalid fc");
        }
        
        uint64_t grid_dims[] = {dims.x, dims.y};
        zdf::save_grid( h_buffer, 2, grid_dims, name, filename );

        host::free( d_buffer );

    }

    /**
     * @brief Copy data from device to host
     * 
     * @param h_buffer  Target host buffer
     */
    void copy_tohost( T * h_buffer ) {
        device::memcpy_tohost( h_buffer, d_buffer, buffer_size() );
    }

    /**
     * @brief Copy data from host to device
     * 
     * @param h_buffer  Source host buffer
     */
    void copy_todevice( T * h_buffer ) {
        host::memcpy_todevice( d_buffer, h_buffer, buffer_size() );
    }
};


#endif
