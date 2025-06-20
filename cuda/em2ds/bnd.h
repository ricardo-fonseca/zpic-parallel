#ifndef BND_H_
#define BND_H_

/**
 * @brief 2D Boundary information
 * 
 * @note Data is to be addressed using e.g. bnd.y.lower
 * 
 * @tparam T    Typename to use (e.g. int)
 */
template < typename T >
class bnd {
    public:
    
    struct {
        T lower;
        T upper;
    } x;
    struct {
        T lower;
        T upper;
    } y;

    bnd() : x({static_cast<T>(0)}), y({static_cast<T>(0)}) {};

    bnd( T val ) : x({val,val}), y({val,val}) {};

    bnd( const bnd<T>& rhs ) :
        x( {rhs.x.lower, rhs.x.upper }), 
        y( {rhs.y.lower, rhs.y.upper } )
    { }

    inline bnd& operator=(const bnd & rhs) {
        // Guard self assignment
        if (this == &rhs)
            return *this;

        x.lower = rhs.x.lower;
        x.upper = rhs.x.upper;
        y.lower = rhs.y.lower;
        y.upper = rhs.y.upper;
        return *this;
    }

    friend bool operator==(const bnd<T>& lhs, const bnd<T>& rhs) {
        return  lhs.x.lower == rhs.x.lower &&
                lhs.x.upper == rhs.x.upper &&
                lhs.y.lower == rhs.y.lower &&
                lhs.y.upper == rhs.y.upper;
    }

    friend bool operator!=(const bnd<T>& lhs, const bnd<T>& rhs) { 
        return  lhs.x.lower != rhs.x.lower ||
                lhs.x.upper != rhs.x.upper ||
                lhs.y.lower != rhs.y.lower ||
                lhs.y.upper != rhs.y.upper;
    }

    friend std::ostream& operator<<(std::ostream& os, const bnd<T>& obj) {
        os << "{";
        os << "x:(" << obj.x.lower << "," << obj.x.upper << ")";
        os << ", ";
        os << "y:(" << obj.y.lower << "," << obj.y.upper << ")";
        os << "}\n";

        return os;
    }
};

#endif
