#include "layer.hpp"

///////////
// Class //
///////////

//Constructors
//Standard constructor without parameters
Layer::Layer() {};  
//with parameters (=)
Layer::Layer(int il, int lt):
    index_layer(il),
    layer_type(lt)
{}

//Methods
//Setter
void Layer::set_neuron(Neuron* ptr_neuron)
{
    neurons_in_layer.push_back(ptr_neuron);
}

//Getter
Neuron* Layer::get_ptr_to_neuron(int index)
{
    return neurons_in_layer[index];
}
const int Layer::get_index() const
{
    return index_layer;
}  
const int Layer::get_no_neurons() const
{
    return neurons_in_layer.size();
}
const int Layer::get_layer_type() const
{
    return layer_type;    
}

//Method to free heap memory used by vector neurons_in_layer
void Layer::clear_neurons()
{
    //Iterate over neurons vector and free memory
    for(auto &x:neurons_in_layer)
    {
        delete x;
    }
    //Clear vector
    neurons_in_layer.clear();     
}

///////////////
// Functions //
///////////////

//Overload the output operator << for printing the neuron
std::ostream& operator<< (std::ostream& os, Layer* layer)
{
    int no_neurons_in_layer = layer->get_no_neurons();
    int layer_index = layer->get_index();
    int layer_type = layer->get_layer_type();
    //Layer information
    os << ">> LAYER " << layer_index << ": ";    
    if(layer_type == 0) {os << "INPUT LAYER";}
    else if(layer_type == 2) {os << "OUTPUT LAYER";}
    else {os << "HIDDEN LAYER " << (layer_index - 1);}
    os << " (" << no_neurons_in_layer << " neurons)" << std::endl; 
    //Neurons in layer
    Neuron* neuron = nullptr;
    int no_edges = 0;
    double activation;
    double bias;
    double zed;
    double delta;
    const int no_of_digits = 2;
    for(int i = 0; i < no_neurons_in_layer; i++)
    {
        neuron = layer->get_ptr_to_neuron(i); 
        os << "> Neuron " << neuron->get_index_neuron() << ": ";  
        activation = neuron->get_activation();
        os << "a=" << std::to_string(activation).substr(0, std::to_string(activation).find(".") + no_of_digits + 1); 
        if(layer_type != 0)
        {
            bias = neuron->get_bias();
            os << ", b=" << std::to_string(bias).substr(0, std::to_string(bias).find(".") + no_of_digits + 1);
            zed = neuron->get_zed();
            os << ", z=" << std::to_string(zed).substr(0, std::to_string(zed).find(".") + no_of_digits + 1);
            delta = neuron->get_delta();
            os << ", d=" << std::to_string(delta).substr(0, std::to_string(delta).find(".") + no_of_digits + 1);
        }  
        os << std::endl; 

        //Edges for neuron with weights
        //Not for input layer
        if(layer_type != 0) 
        {
            no_edges = neuron->get_no_edges();
            os << "  Edges (" << no_edges << "): "; 
            const Edge* edge;
            double weight;
            for(int j = 0; j < no_edges; j++)
            {
                edge = neuron->get_edge_ptr(j);
                weight = edge->get_weight();
                os << "w" << j << "=" << std::to_string(weight).substr(0, std::to_string(weight).find(".") + no_of_digits + 1); 
                if(j != (no_edges - 1))
                {
                    os << ", ";
                }
                if((j+1) % 10 == 0 && j != (no_edges - 1))
                {
                    os << std::endl;             
                }
            }
            os << std::endl; 
        }
    }   

    return os; 
}