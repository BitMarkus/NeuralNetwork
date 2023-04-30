#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>
#include "neuron.hpp"
//#include "edge.hpp"

///////////
// Class //
///////////

class Layer
{
    private:

    //Layer index
    int index_layer = 0;
    //Neuron type: 0=Input layer, 1=Hidden layer, 2=Output layer
    int layer_type = 0;  
    //Vector with pointer to all neurons in this layer
    //Allocated memory!!!
    std::vector<Neuron*> neurons_in_layer;  

    public:

    //Constructors
    //Standard constructor without parameters
    Layer();  
    //with parameters (= layer index and layer type)
    Layer(int, int); 

    //Methods
    //Setter
    void set_neuron(Neuron*);

    //Getter
    Neuron* get_ptr_to_neuron(int);  
    const int get_index() const; 
    const int get_no_neurons() const;
    const int get_layer_type() const;  

    //Method to free heap memory used by vector neurons_in_layer
    void clear_neurons(); 
};

///////////////
// Functions //
///////////////

//Overload the output operator << for printing the container
std::ostream& operator<< (std::ostream&, Layer*);

#endif