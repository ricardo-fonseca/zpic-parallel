#pragma once

class moving_window {
    private:
    bool active_;
    unsigned int n_move_;
    double dx_;

    public:

    moving_window() : active_(false), n_move_(0), dx_(0) {};

    /**
     * @brief Turns moving window on
     * 
     */
    void init( float const dx ) {
        active_ = true;
        n_move_ = 0;
        dx_ = dx;
    }

    /**
     * @brief Turns moving window on
     * 
     */
    void init( float const box, unsigned int const ncells ) {
        active_ = true;
        n_move_ = 0;
        dx_ = box / ncells;
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
