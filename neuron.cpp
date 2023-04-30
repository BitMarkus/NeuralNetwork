#include "neuron.hpp"

///////////
// Class //
///////////

//Constructors
//Standard constructor without parameters
Neuron::Neuron() {};  
//with parameters (=)
Neuron::Neuron(int il, int in, int nt):
    index_layer(il),
    index_neuron(in),
    neuron_type(nt)
{}

//Methods
//Setter
void Neuron::set_edge(Edge* ptr_edge)
{
    edges.push_back(ptr_edge);
}
void Neuron::set_prev_neuron(Neuron* ptr_neuron)
{
    prev_neurons.push_back(ptr_neuron);
}
void Neuron::set_activation(double a)
{
    activation = a;
}
void Neuron::set_bias(double b)
{
    bias = b;
}
void Neuron::set_neuron_type(int t)
{
    neuron_type = t;
}
void Neuron::set_delta(double d)
{
    delta = d;
}
void Neuron::set_zed(double z)
{
    zed = z; 
}

//Getter
//Implement a member function to get the current activation
const double Neuron::get_activation() const
{
    return activation;
} 
const double Neuron::get_bias() const
{
    return bias;
}
const double Neuron::get_delta() const
{
    return delta;
}
const double Neuron::get_zed() const
{
    return zed;
}
const int Neuron::get_neuron_type() const
{
    return neuron_type;
}
const int Neuron::get_index_layer() const
{
    return index_layer;
}    
const int Neuron::get_index_neuron() const
{
    return index_neuron;
} 
const int Neuron::get_no_edges() const
{
    return edges.size();
}  
Edge* Neuron::get_edge_ptr(int index) const
{
    return edges[index];
}
const Neuron* Neuron::get_prev_neuron_ptr(int index) const
{
    return prev_neurons[index];
}  

//Method determins the activation function of a neuron
//First implementation is a sigma function
//It takes a double value and normalizes it to a value between 0 and 1
const double Neuron::activate(const double z)
{
    return 1.0 / (1.0 + exp(-1.0 * z));
}

//Method to free heap memory used by vector edges
void Neuron::clear_edges()
{
    //Iterate over edges vector and free memory
    for(auto &x:edges)
    {
        delete x;
    }
    //Clear vector
    edges.clear();     
}

///////////////
// Functions //
///////////////

//Overload the output operator << for printing the neuron
std::ostream& operator<< (std::ostream& os, Neuron* neuron)
{
    int index_layer = neuron->get_index_layer();

    os << "::::::::::::NEURON " << neuron->get_index_neuron();
    os << ", LAYER " << index_layer << "::::::::::::" << std::endl;
    //Type of neuron: 0=Input neuron, 1=Hidden neuron, 2=Output neuron
    int neuron_typ = neuron->get_neuron_type();
    if(neuron_typ == 0) {os << ">>INPUT NEURON" << std::endl;}
    else if(neuron_typ == 2) {os << ">>OUTPUT NEURON" << std::endl;}
    else {os << ">>HIDDEN NEURON" << std::endl;}
    os << "Activation: " << neuron->get_activation() << std::endl;
    if(neuron_typ != 0) 
    {
        os << "Bias: " << neuron->get_bias() << std::endl;
        os << "Zed: " << neuron->get_zed() << std::endl;
        os << "Delta: " << neuron->get_delta() << std::endl;
    }
    os << "Number of edges: " << neuron->get_no_edges() << std::endl;    
    //Not fot input neurons
    if(index_layer != 0)
    {
        os << "List of edges with weights:" << std::endl;
        for(int i = 0; i < neuron->get_no_edges(); i++)
        {
            os <<  ">Edge " << i << ": ";
            os << "weight = " << neuron->get_edge_ptr(i)->get_weight();
            os << " -> neuron " << neuron->get_prev_neuron_ptr(i)->get_index_neuron() << ", layer ";
            os << neuron->get_prev_neuron_ptr(i)->get_index_layer() << std::endl;
        }
    }
    return os;  
}