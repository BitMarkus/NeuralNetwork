#include "edge.hpp"

///////////
// Class //
///////////

//Constructors
//Standard constructor without parameters
Edge::Edge() {};  

//Methods
//Setter
void Edge::set_weight(double val)
{
    weight = val;
}
//Getter
const double Edge::get_weight() const
{
    return weight;
} 


///////////////
// Functions //
///////////////