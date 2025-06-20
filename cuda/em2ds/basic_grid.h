#ifndef BASIC_GRID_H_
#define BASIC_GRID_H_

/**
 * @brief Basic grid class
 * 
 * @note Used as a contiguous 2D array in device memory
 * 
 * @tparam T    grid datatype
 */
template <class T>
class basic_grid{

    public:

    /// @brief Data buffer   
    T * d_buffer;    

    /// @brief Grid size
    const uint2 dims;

    /// @brief Object name
    std::string name;
    
    /**
     * @brief Construct a new basic grid object
     * 
     * @param dims  Grid dimensions
     */
    basic_grid( uint2 dims ) : dims(dims) {
        // Allocate main data buffer on device memory
        d_buffer = device::malloc<T>( buffer_size() );
    }

    /**
     * @brief Destroy the basic grid object
     * 
     */
    ~basic_grid() {
        device::free( d_buffer );
    }

    /**
     * @brief Buffer size
     * 
     * @return total size of data buffers (in elements)
     */
    std::size_t buffer_size() {
        return dims.x * dims.y ;
    };

    /**
     * @brief Stream extraction
     * 
     * @param os 
     * @param obj 
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const basic_grid<T>& obj) {
        os << obj.name << "{";
        os << "(" << obj.dims.x << ", " << obj.dims.y << ")";
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
     * @param val       Value
     */
    void set( T const & val ){
        device::setval( d_buffer, buffer_size(), val );
    };

    /**
     * @brief Save grid values to disk
     * 
     * @param filename      Output file name (includes path)
     */
    void save( std::string filename ) {
        T * h_buffer = host::malloc< T >( dims.x * dims.y );

        copy_tohost( h_buffer );
        
        uint64_t grid_dims[] = {dims.x, dims.y};
        zdf::save_grid( h_buffer, 2, grid_dims, name, filename );

        host::free( h_buffer );

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