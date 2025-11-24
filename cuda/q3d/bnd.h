#ifndef BND_H_
#define BND_H_

/**
 * @brief Edge (lower, upper)
 * 
 */
namespace edge {
    enum pos { lower = 0, upper };
}


/**
 * @brief 2D Boundary information
 * 
 * @note Data is to be addressed using e.g. bnd.y.lower
 * 
 * @tparam T    Typename to use (e.g. int)
 */
template< typename T >
class pair {
    public:

    T lower;
    T upper;

    pair() : lower( static_cast<T>(0) ), upper( static_cast<T>(0) ) {};
    pair( T val ) : lower( val ), upper( val ) {};
    pair( T lower, T upper ) : lower( lower ), upper( upper ) {};

    inline pair<T>& operator=(const pair<T> & rhs) {
        // Guard self assignment
        if (this == &rhs)
            return *this;
        
        lower = rhs.lower;
        upper = rhs.upper;
        return *this;
    }

    friend bool operator==(const pair<T>& lhs, const pair<T>& rhs) {
        return  lhs.lower == rhs.lower &&
                lhs.upper == rhs.upper;
    }

    friend bool operator!=(const pair<T>& lhs, const pair<T>& rhs) {
        return  lhs.lower != rhs.lower ||
                lhs.upper != rhs.upper;
    }

    friend std::ostream& operator<<(std::ostream& os, const pair<T>& obj) {
        os << '(' << obj.lower << ',' << obj.upper << ')';
        return os;
    }
};

/**
 * @brief 2D Boundary information
 * 
 * Data is to be addressed using e.g. bnd.y.lower
 * 
 * @tparam T    Typename to use (e.g. int)
 */
template < typename T >
class bnd {
    public:
    
    /// @brief x boundary values
    pair<T> x;
    /// @brief y boundary values
    pair<T> y;

    /**
     * @brief Construct a new bnd object, sets all values to 0
     * 
     */
    bnd() : x({static_cast<T>(0)}), y({static_cast<T>(0)}) {};

    /**
     * @brief Construct a new bnd object, sets all values to supplied value
     * 
     * @param val   Value used to initialize the bnd values
     */
    bnd( T val ) : x(val), y(val) {};

    bnd( const bnd<T>& rhs ) :
        x( {rhs.x.lower, rhs.x.upper }), 
        y( {rhs.y.lower, rhs.y.upper } )
    { }

    /**
     * @brief Assignment (=) operator
     * 
     * @param rhs           Source value
     * @return bnd<T>&      Return alias to self
     */
    inline bnd<T>& operator=(const bnd<T> & rhs) {
        // Guard self assignment
        if (this == &rhs)
            return *this;

        x = rhs.x;
        y = rhs.y;
        return *this;
    }

    /**
     * @brief Equality (==) operator
     * 
     * @param lhs 
     * @param rhs 
     * @return true 
     * @return false 
     */
    friend bool operator==(const bnd<T>& lhs, const bnd<T>& rhs) {
        return  lhs.x == rhs.x &&
                lhs.y == rhs.y;
    }

    friend bool operator!=(const bnd<T>& lhs, const bnd<T>& rhs) { 
        return  lhs.x != rhs.x ||
                lhs.y != rhs.y;
    }

    friend std::ostream& operator<<(std::ostream& os, const bnd<T>& obj) {
        os << "{";
        os << "x:" << obj.x;
        os << ", ";
        os << "y:" << obj.y;
        os << "}\n";

        return os;
    }
};

#endif
