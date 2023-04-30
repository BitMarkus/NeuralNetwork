#ifndef MODEL_HPP
#define MODEL_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "network.hpp"

///////////
// Class //
///////////

class Model
{
    private:

    //Path to folder with models
    std::string path_models = "./models\\";
    //Vector to store the body data (= weights and biases)
    std::vector<double> data_vector;
    //Vector to store the header data (= number of layers/neurons in layers)
    std::vector<int> header_vector;

    public:

    //Constructors
    //Standard constructor without parameters
    Model();  

    //Getter
    //Get header vector
    const std::vector<int>& get_model_header() const; 
    //Get data vector
    const std::vector<double>& get_model_data() const; 

    //Methods

    //Method to export a trained model into a text file
    //The model is specific to a network -> new network is generated when the model data is imported
    //Filename can be choosen, ending is .mdl
    //File consists of header and body
    //Header: Number of layers and neurons per layer (necessary information for network generation)
    //Header data is written in the first line, seperated by |
    //Body: weights ans biases of the network
    //Body data starts at second line, each weight and bias are written in a new line
    //The order of weights and biases is arranged like in the gradient vector
    //- Go through layers starting at first hidden layer and end at output layer
    //- Go through neurons in layers starting at 0 and ending with last neuron
    //- For each neuron first weights of all edges to neurons in the previous layer are saved
    //- The order is the order of indices of neurons in previous layer (0-last)
    //- After all weights of a neuron the bias is saved in the model file
    bool export_model(Network&);

    //Method to import a saved model
    //By importing a model a new network is generated and filled with weights/biases
    bool import_model(Network&);
};

///////////////
// Functions //
///////////////

//Split function for strings using a delimiter
//Returns a vector of delimiter separated strings 
//https://favtutor.com/blogs/split-string-cpp
std::vector<std::string> split_string(std::string, char);

#endif