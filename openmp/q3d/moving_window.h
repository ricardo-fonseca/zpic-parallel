#ifndef __MOVING_WINDOW__
#define __MOVING_WINDOW__


class MovingWindow {
    private:
    bool active_;
    unsigned int n_move_;
    double dx_;

    public:

    MovingWindow() : active_(false), n_move_(0), dx_(0) {};

    /**
     * @brief Turns moving window on
     * 
     */
    void init( float const dx ) {
        active_ = true;
        n_move_ = 0;
        dx_ = dx;
    }

    bool active() const {
        return active_;
    }

    unsigned int n_move() const {
        return n_move_;
    }

    /**
     * @brief Advances the window
     * 
     * @return int  Total number of cells moved
     */
    int advance() {
        if ( active_ ) n_move_++;
        return n_move_;
    }

    /**
     * @brief Total length moved
     * 
     * @return double    Total length moved by the window
     */
    double motion( ) const {
        return n_move_ * dx_;
    }

    bool needs_move( double const t ) const {
        return active_ && (t > dx_*(n_move_+1));
    }
};

#endif
