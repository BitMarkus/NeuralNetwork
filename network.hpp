#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include "neuron.hpp"
#include "edge.hpp"
#include "layer.hpp"
#include "typedefs.hpp"

///////////
// Class //
///////////

class Network
{
    private:

    //Number of input neurons
    int no_input_neurons = 0;
    //Number of output neurons
    int no_output_neurons = 0;
    //Number of hidden layers
    int no_hidden_layers = 0;
    //temporary input for hidden layers
    int input_hidden_layers = 0;
    //Vector with number of neurons in each hidden layer
    //Vector entry [0] is layer next to input
    std::vector<int> no_hidden_neurons;
    //Vector with number of neurons in each layer
    //[0] is input layer, last is output layer
    std::vector<int> no_neurons_in_layer;
    //Vector with objects of different layers
    //Allocated memory!!!
    std::vector<Layer*> layers;
    //Total number of neurons and edges in network
    int no_layers = 0;
    int no_neurons = 0;
    int no_edges = 0;
    int no_biases = 0;

    //Member variable for backpropagation
    //learning rate = step width of gradient descent in backpropagation
    const double learning_rate = 1.0;
    //Threshold value for cost to exit gradient descent
    const double threshold_cost = 0.001;
    //Max amount of iterations if threshold cannot be reached
    //= emergency exit for gradient descent loop
    const int max_iterations = 10000;

    public:

    //Constructors
    //Standard constructor without parameters
    Network();  

    //Getter
    //Get pointer to specific neuron
    Layer* get_pointer_to_layer(int) const;
    Neuron* get_ptr_to_neuron(int, int) const; 
    const int get_no_neurons_in_layer(int);
    const int get_no_layers() const;
    const int get_no_neurons() const;
    const int get_no_edges() const;
    const int get_no_biases() const;
    const double get_learning_rate() const;

    //Methods
    //Form for network generation
    //First parameter is the number of input neurons
    //Second parameter is the nuber of output neurons
    //If they are passed, the form will not ask for these parameters
    void network_input_form(int = 0, int = 0);
    //Method to setup network via model import
    void setup_network_via_import(const std::vector<int>&);
    //Generation of network = initialization of neurons and biases
    void generate_network();
    //Import weights and biases from saved model
    void import_weights_biases(const std::vector<double>&);
    //Method sets all activities, weights and biases in the network to a random value
    //Activity and weights of input neurons are not changed
    //Weights range and bias range can be adjusted (min_w - max_w, min_b - max_b)
    void randomize_network(double, double, double, double);
    //Method generates a formatted string to print the averaged gradient vector
    const std::string print_av_gradient_vector(const std::vector<double>&) const; 
    //Method to set activation values for the neurons of the input layer
    void set_input_layer(const import_format&, int = 0);
    //Method generates output neuron activation
    //dependent on input neuron activation, weights and biases
    void calculate_output();
    //Method exports output in form of a vector
    //= activations of all output neurons
    std::vector<double> export_data();   
    //Print output
    //Returns a string with a list of output neurons and their activity
    const std::string print_output() const;
    //Interpret output
    //Returns a string where output neurons are replaced by their symbols and respective activity
    const std::string print_labeled_output(const labeling_format&) const;
    //Calculate the cost of a single training image
    //Takes the output of the last forward calculation and compare it to desired output of image
    //int: index of training image
    //Returns the cost for a single training image
    double calculate_cost(const import_format&, int = 0);
    //Method to free heap memory used by vector layers
    void clear_layers();
    //Method to reset the current network object
    void reset_network();
    //Backpropagation
    //ALGORITHM:
    // 1) Initialize network with random weights and biases
    // 2) For each training image do:
    //      a) Compute activations for entire network and calculate cost 
    //      b) Compute delta for all neurons in output layer
    //      c) Compute delta for all neurons in previous layers
    //      d) Compute "gradient of cost" with respect to all weights and biases using deltas
    // 3) Compute average cost for whole training set of images 
    // 4) Average the gradient with respect to each weights and biases over the entire set of training images
    // 5) Update weights and biases using gradient descent (using learning rate)
    // 6) Repeat steps 2-4 until cost is below an acceptable level (or if x iterations are done)
    bool backpropagation(const import_format&);

    private:

    //Methods used in backpropagation:

    //Calculate deltas of neurons in the output layer
    void calculate_delta_output_layer(const std::vector<double>&);
    //Calculate and set deltas of neurons in the hidden layers
    void calculate_delta_hidden_layer();
    //Compute gradient of cost for one training image with respect to all weights and biases (using deltas)
    //Gradient of cost vector is added to a vector containing gradients for all training images
    //The order of weights and biases is:
    //- Layers starting at first hidden layer and end at output layer
    //- Neurons in layers start from 0 and end with last neuron
    //- For each neuron first weights of all edges to neurons in the previous layer are saved
    //- The order is the order of indices of neurons in previous layer (0-last)
    //- After weights the bias of each neuron is saved in the gradient vector
    void calculate_gradient_vector(gradient_vectors&);
    //Compute average cost for all training images and output it
    double average_cost(const std::vector<double>&);
    //Method averages gradient vectors for all training images
    //And returns a vector with the mean values
    std::vector<double> average_gradient(const gradient_vectors&);
    //Method updates weights and biases of the network using gradient descent (and learning rate)
    void gradient_descent(const std::vector<double>&); 
};

///////////////
// Functions //
///////////////

//Overload the output operator << for printing the container
std::ostream& operator<< (std::ostream&, Network&);

//Generate random double values between min and max
double generate_random_d(double, double);

#endif