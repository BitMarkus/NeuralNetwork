#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>
#include <vector>
#include "edge.hpp"
#include <cmath>

///////////
// Class //
///////////

class Neuron
{
    private:

    //Number/index of layer, where the neuron is placed (0 = input layer, end = output layer)
    int index_layer = 0;
    //Number/index of neuron in layer
    int index_neuron = 0; 
    //Neuron type: 0=Input neuron, 1=Hidden neuron, 2=Output neuron
    int neuron_type = 0;  
    //zed value goes into the activation function to calculate activation a
    //not really needed, but nice to know
    double zed = 0;
    //Activation a (0-1)
    double activation = 0;
    //Bias as member variable of a neuron
    double bias = 0;
    //Delta of the neuron (parameter for backward propagation)
    double delta = 0;
    //Vector with pointers to all edges, which are input connected to this neuron
    //All Edges to the left of the neuron "belong to this neuron"
    //Allocated memory!
    std::vector<Edge*> edges;
    //Vector with pointers to all neurons from the previous layer
    std::vector<Neuron*> prev_neurons;
    //edges and prev_neurons share the same index
    //means the edge[0] points to the prev_neurons[0]

    public:

    //Constructors
    //Standard constructor without parameters
    Neuron();  
    //with parameters
    Neuron(int, int, int); 

    //Methods
    //Setter
    void set_edge(Edge*);
    void set_prev_neuron(Neuron*);
    void set_activation(double);
    void set_bias(double);
    void set_neuron_type(int);
    void set_delta(double);
    void set_zed(double);

    //Getter
    //Implement a member function to get the current activation
    const double get_activation() const;  
    const double get_bias() const;  
    const double get_delta() const; 
    const double get_zed() const; 
    const int get_neuron_type() const;     
    const int get_index_layer() const;    
    const int get_index_neuron() const; 
    const int get_no_edges() const;  
    Edge* get_edge_ptr(int) const;
    const Neuron* get_prev_neuron_ptr(int) const;      

    //Method determins the activation function of a neuron
    //First implementation is a sigma function
    //It takes a double value and normalizes it to a value between 0 and 1
    const double activate(const double);

    //Method to free heap memory used by vector edges
    void clear_edges();
};

///////////////
// Functions //
///////////////

//Overload the output operator << for printing the container
std::ostream& operator<< (std::ostream&, Neuron*);

#endif