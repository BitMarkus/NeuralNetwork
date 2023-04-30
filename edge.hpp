#ifndef EDGE_HPP
#define EDGE_HPP

#include <iostream>

///////////
// Class //
///////////

class Edge
{
    private:

    //Weight of the connection
    double weight = 0;

    public:

    //Constructors
    //Standard constructor without parameters
    Edge();  

    //Methods
    //Setter
    void set_weight(double);
    //Getter
    const double get_weight() const;  
};

///////////////
// Functions //
///////////////

#endif