#include "network.hpp"

///////////
// Class //
///////////

//Standard constructor without parameters
Network::Network() {}; 

//Getter
//Get pointer to specific layer
Layer* Network::get_pointer_to_layer(int index_layer) const
{
    return layers[index_layer];
}
//Get pointer to specific neuron
Neuron* Network::get_ptr_to_neuron(int index_layer, int index_neuron) const
{
    return layers[index_layer]->get_ptr_to_neuron(index_neuron);
}
const int Network::get_no_neurons_in_layer(int index)
{
    return no_neurons_in_layer[index];
}
const int Network::get_no_layers() const
{
    return no_layers;
}   
const int Network::get_no_neurons() const
{
    return no_neurons;
}
const int Network::get_no_edges() const
{
    return no_edges;
}
const int Network::get_no_biases() const
{
    return no_biases;
}
const double Network::get_learning_rate() const
{
    return learning_rate;
}

//Methods
//Form for network generation
//First parameter is the number of input neurons
//Second parameter is the nuber of output neurons
//If they are passed, the form will not ask for these parameters
void Network::network_input_form(int no_in, int no_out)
{
    std::cout << "Enter data for network generation:" << std::endl;
    std::cout << "Number of input neurons: ";
    if(no_in == 0)
    {
        std::cin >> no_input_neurons;
    }
    else
    {
        std::cout << no_in << " (based on training data)" << std::endl;
        no_input_neurons = no_in;
    }
    std::cout << "Number of output neurons: ";
    if(no_out == 0)
    {            
        std::cin >> no_output_neurons;
    }
    else
    {
        std::cout << no_out << " (based on training data)" << std::endl;
        no_output_neurons = no_out;
    }    
    std::cout << "Number of hidden layers: ";
    std::cin >> no_hidden_layers;
    for(int i = 0; i < no_hidden_layers; i++)
    {
        std::cout << "Number of neurons in hidden layer " << (i+1) << ": ";
        std::cin >> input_hidden_layers;
        no_hidden_neurons.push_back(input_hidden_layers);
    }

    //Generate vector no_neurons_in_layer with number of neurons in each layer
    no_neurons_in_layer.push_back(no_input_neurons);
    for(auto it = std::begin(no_hidden_neurons); it != std::end(no_hidden_neurons); ++it) 
    {
        no_neurons_in_layer.push_back(*it);
    }
    no_neurons_in_layer.push_back(no_output_neurons);
    //Total amount of layers
    no_layers = no_neurons_in_layer.size();
    //Total neurons
    for(auto it = std::begin(no_neurons_in_layer); it != std::end(no_neurons_in_layer); ++it) 
    {
        no_neurons += *it;
    }  
    //Total edges
    for(int i = 0; i < (no_neurons_in_layer.size() - 1); i++) 
    {
        no_edges += no_neurons_in_layer[i] * no_neurons_in_layer[(i + 1)];    
    }  
    //Total biases (number of neurons except input layer)
    no_biases = no_neurons - no_neurons_in_layer[0];
}

//Method to setup network via model import
void Network::setup_network_via_import(const std::vector<int>& header_vector) 
{
    no_layers = header_vector.size();
    //Transfer number of neurons in layer and amout of layers to network
    no_neurons_in_layer = header_vector;
    //Set the other parameters
    no_input_neurons = header_vector[0];
    no_output_neurons = header_vector[(no_layers - 1)];
    no_hidden_layers = no_layers - 2;
    for(int i = 1; i < (no_layers - 2); i++)
    {
        no_hidden_neurons.push_back(header_vector[i]);
    }
    //Total neurons
    for(auto it = std::begin(no_neurons_in_layer); it != std::end(no_neurons_in_layer); ++it) 
    {
        no_neurons += *it;
    }  
    //Total edges
    for(int i = 0; i < (no_neurons_in_layer.size() - 1); i++) 
    {
        no_edges += no_neurons_in_layer[i] * no_neurons_in_layer[(i + 1)];    
    }  
    //Total biases (number of neurons except input layer)
    no_biases = no_neurons - no_neurons_in_layer[0];    
}

//Generation of network = initialization of neurons and biases
void Network::generate_network()
{
    std::cout << std::endl << "Generate neural network..." << std::endl;

    //Generate layers
    for(int i = 0; i < no_layers; i++)
    {
        std::cout << "Generate layer " << i << "..." << std::endl;
        //Determine type of layer: 0=Input layer, 1=Hidden layer, 2=Output layer
        int layer_type;
        if(i == 0) {layer_type = 0;}
        else if(i == (no_layers - 1)) {layer_type = 2;}
        else {layer_type = 1;}
        layers.push_back(new Layer(i, layer_type));
        
        //Generate neurons in layer
        Neuron* nptr = nullptr;
        for(int j = 0; j < no_neurons_in_layer[i]; j++)
        {
            //Set neuron type: 0=Input neuron, 1=Hidden neuron, 2=Output neuron
            int neuron_type;
            if(i == 0) {neuron_type = 0;}
            else if(i == (no_layers-1)) {neuron_type = 2;}
            else {neuron_type = 1;}
            //Generate neuron
            layers[i]->set_neuron(new Neuron(i, j, neuron_type));
            //Get pointer to neuron
            nptr = layers[i]->get_ptr_to_neuron(j);
            //Generate edges and complete member variables of neuron objects
            //NOT for neurons in input layer (layer 0)!!!
            if(i != 0)
            {
                //Connection to all Neurons in the previous layer
                for(int k = 0; k < no_neurons_in_layer[(i-1)]; k++)
                {
                    //Complete member variables of neuron objects
                    //Get pointer to neuron from previous layer
                    Neuron* perv_nptr = layers[(i-1)]->get_ptr_to_neuron(k); 
                    //And save them in the vector prev_neuron               
                    nptr->set_prev_neuron(perv_nptr);
                    //Generate edge
                    nptr->set_edge(new Edge);
                }
            }
        }
        nptr = nullptr;
    }
    std::cout << "Network successfully created!" << std::endl;
}

//Import weights and biases from saved model
void Network::import_weights_biases(const std::vector<double>& data_vector)
{
    //The update of weights and biases has to happen in the same order as the gradient vector was generated
    //-> First all weights of the neuron and then the bias   
    //Iterate forward over all layers except the input layer
    int index = 0;
    for(int j = 1; j < no_layers; j++)
    {
        //Iterate over neurons in layer
        for(int k = 0; k < no_neurons_in_layer[j]; k++)
        {
            //Get pointer to neuron
            Neuron* nptr = get_ptr_to_neuron(j, k); 

            //Set WEIGHTS of neuron
            //Iterate over all edges of the neuron
            int no_edges = nptr->get_no_edges();
            for(int l = 0; l < no_edges; l++)
            {
                //Get pointer to edge
                Edge* eptr = nptr->get_edge_ptr(l);
                eptr->set_weight(data_vector[index]);
                //Increment gradient vector index
                index++;
            }

            //Set BIAS of neuron
            //Write new weight to network
            nptr->set_bias(data_vector[index]);
            //Increment gradient vector index
            index++;
        }
    } 
}

//Method sets all weights and biases in the network to a random value
//Activity and weights of input neurons are not changed
//Random range can be adjusted (double)
//Weights range and bias range can be adjusted (min_w - max_w, min_b - max_b)
void Network::randomize_network(double min_w, double max_w, double min_b, double max_b)
{
    std::cout << std::endl << "Randomize network..." << std::endl;

    //Set seed for randomization
    srand((unsigned) time(NULL));

    //Iterate over layers
    //No changes in layer 0 = input layer
    for(int i = 1; i < no_layers; i++)
    {
        //Iterate over neurons in layer
        Neuron* nptr = nullptr;
        for(int j = 0; j < no_neurons_in_layer[i]; j++)
        {
            nptr = layers[i]->get_ptr_to_neuron(j);
            //Randomize bias
            nptr->set_bias(generate_random_d(min_b, max_b));

            //Iterate over edges of neuron
            Edge* eptr = nullptr;
            for(int k = 0; k < nptr->get_no_edges(); k++)
            {
                eptr = nptr->get_edge_ptr(k);
                //Randomize weights
                eptr->set_weight(generate_random_d(min_w, max_w));
            }
            eptr = nullptr;
        }
        nptr = nullptr;
    } 
    std::cout << "Network successfully randomized!" << std::endl; 
}

//Method to set activation values for the neurons of the input layer
void Network::set_input_layer(const import_format& data, int index)
{
    for(int i = 0; i < no_neurons_in_layer[0]; i++)
    {
        get_ptr_to_neuron(0, i)->set_activation(data[index].first[i]);
    }
}

//Method generates a formatted string to print the averaged gradient vector
const std::string Network::print_av_gradient_vector(const std::vector<double>& av_grad_vec) const
{
    std::string output_sting;
    int vec_size = av_grad_vec.size();
    const int no_of_digits = 6;
    output_sting += "::::Averaged gradient vector::::\n"; 
    output_sting += "Size: " + std::to_string(vec_size) + "\n";    
    for(int i = 0; i < vec_size; i++) 
    {
        output_sting += std::to_string(av_grad_vec[i]).substr(0, std::to_string(av_grad_vec[i]).find(".") + no_of_digits + 1) + "\n";        
    } 
    return output_sting;   
}

//Method generates output neuron activation
//dependent on input neuron activation, weights and biases
void Network::calculate_output()
{
    //Iterate over layers, start at layer 1 (not 0!)
    //Neurons of first layer should already be set as input
    for(int i = 1; i < no_layers; i++)
    {
        //Iterate over neurons in layer
        for(int j = 0; j < no_neurons_in_layer[i]; j++)
        {
            Neuron* nptr = get_ptr_to_neuron(i, j);
            double bias = nptr-> get_bias();
            double a; //Activation
            //Iterate over edges (weights) and previous neurons (activations)
            //to calculate z
            double z = 0;
            int no_edges = nptr->get_no_edges();
            for(int k = 0; k < no_edges; k++)
            {
                //Get weights and edges
                double weight = nptr->get_edge_ptr(k)->get_weight();
                double prev_activation = nptr->get_prev_neuron_ptr(k)->get_activation();
                //Sum up z
                z += weight * prev_activation;
            }
            //Add bias
            z += bias;
            //Add z to member variables
            nptr->set_zed(z);
            //Apply activation function to z
            a = nptr->activate(z);
            //Set new activation of neuron
            nptr->set_activation(a);
            nptr = nullptr;
        }        
    }
}

//Method exports output in form of a vector
//= activations of all output neurons
std::vector<double> Network::export_data()
{
    std::vector<double> export_vector;
    int index_output_layer = no_layers - 1;
    //Iterate over neurons in output layer
    for(int i = 0; i < no_neurons_in_layer[index_output_layer]; i++)
    {
        export_vector.push_back(get_ptr_to_neuron(index_output_layer, i)->get_activation());
    }
    return export_vector;
} 

//Print output
const std::string Network::print_output() const
{
    std::string output_sting;
    const int no_of_digits = 3;
    int index_output_layer = no_layers - 1;
    double activation;

    output_sting += "::::Output::::\n";
    //Iterate over neurons in output layer
    for(int i = 0; i < no_neurons_in_layer[index_output_layer]; i++)
    {  
        output_sting += "> Neuron ";
        output_sting += std::to_string(i);
        output_sting += ": ";
        activation = get_ptr_to_neuron(index_output_layer, i)->get_activation(); 
        output_sting += std::to_string(activation).substr(0, std::to_string(activation).find(".") + no_of_digits + 1);
        output_sting += "\n";         
    }  
    return output_sting;
}

//Interpret output
//Returns a string where output neurons are replaced by their symbols and respective activity
const std::string Network::print_labeled_output(const labeling_format& labels) const
{
    std::string output_sting;
    const int no_of_digits = 3;
    int index_output_layer = no_layers - 1;
    double activation;
    int no_labels = labels.size();

    output_sting += "::::Output::::\n";
    //Iterate over neurons in output layer
    for(int i = 0; i < no_neurons_in_layer[index_output_layer]; i++)
    {  
        //Iterate over labeling vector
        for(int j = 0; j < no_labels; j++)
        {
            if(labels[j].second == i)
            {
                output_sting += labels[j].first;                
            }
        }       
        output_sting += ": ";
        activation = get_ptr_to_neuron(index_output_layer, i)->get_activation(); 
        output_sting += std::to_string(activation).substr(0, std::to_string(activation).find(".") + no_of_digits + 1);
        output_sting += "\n";  
    }  
    return output_sting;
}

//Calculate the cost of a single training image
//Takes the output of the last forward calculation and compare it to desired output of image
//int: index of training image
//Returns the cost for a single training image
double Network::calculate_cost(const import_format& training_data, int index)
{
    double cost = 0;
    int index_output_layer = no_layers - 1;
    double count = training_data[index].second.size();
    //Iterate over output vector in pair
    //Sum of square root of desired output minus actual output
    for(int i = 0; i < count; i++)
    {
        cost += pow((training_data[index].second[i] - get_ptr_to_neuron(index_output_layer, i)->get_activation()), 2); 
    }
    return cost;
}

//Method to free heap memory used by vector layers
void Network::clear_layers()
{
    for(auto &x:layers)
    {
        delete x;
    }
    //Clear vector
    layers.clear();     
}

//Method to reset the current network object
void Network::reset_network()
{
    std::cout << std::endl << "Reset neural network..." << std::endl;

    //Free allocated memory:
    //Iterate over layers
    Layer* lptr;
    for(int i = 0; i < no_layers; i++)
    {
        //Iterate over neurons in layer
        Neuron* nptr;
        for(int j = 0; j < no_neurons_in_layer[i]; j++)
        {
            //Get pointer to neuron
            nptr = layers[i]->get_ptr_to_neuron(j);
            //Free allocated memory for edge objects
            nptr->clear_edges();            
        }
        nptr = nullptr;
        //Free allocated memory for neurons of the layer
        lptr = get_pointer_to_layer(i);
        lptr->clear_neurons();
    }
    lptr = nullptr;
    //Free allocated memory for layers
    clear_layers();

    //Reset member variables:
    no_input_neurons = 0;
    no_output_neurons = 0;
    no_hidden_layers = 0;
    input_hidden_layers = 0;
    no_hidden_neurons.clear();
    no_neurons_in_layer.clear();
    no_layers = 0;
    no_neurons = 0;
    no_edges = 0;
    no_biases = 0;

    std::cout << "Reset of network successful!" << std::endl;
}

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
bool Network::backpropagation(const import_format& training_data)
{
    //Create gradient vector: vector with gradient vectors for each training image
    gradient_vectors gradient_vectors;
    //Create a vector with cost for each training image
    std::vector<double> cost_vector;
    //Variable for mean cost
    double mean_cost = 1.0;

    // 1) Initialize network with random weights and biases:
    randomize_network(-1.0, 1.0, -1.0, 1.0);

    // 6) Repeat steps 2-4 until cost is below an acceptable level (or if x iterations are done)
    int no_iter = 0;
    while(mean_cost >= threshold_cost && no_iter <= max_iterations)
    {                                                                                                          
        // 2) For each training image do:
        int no_training_data = training_data.size();
        for(int i = 0; i < no_training_data; i++)
        {  
            // a) Compute activations for entire network and calculate cost
            // Import input data and set activations of the first layer
            set_input_layer(training_data, i);
            calculate_output();
            //Calculate cost of the image and add it to vector for all costs in training set
            cost_vector.push_back(calculate_cost(training_data, i));
                    
            // b) Compute delta for all neurons in output layer
            calculate_delta_output_layer(training_data[i].second);

            // c) Compute delta for all neurons in previous layers (except input layer)     <- Segemtation fault bei get_delta()
            calculate_delta_hidden_layer();

            // d) Compute gradient of cost with respect to all weights and biases (using deltas)
            // Adds the gradient vector for one image to a vector for all images (gradient_vectors)
            calculate_gradient_vector(gradient_vectors);
        }
        
        // 3) Compute average cost for all training images and output it
        mean_cost = average_cost(cost_vector);
        std::cout << "Cost: " << mean_cost << std::endl;
        
        // 4) Average the gradients with respect to each weights and biases over the entire set of training images
        //Generate vector gradient_vector_mean containing the mean values of all gradient vectors
        std::vector<double> gradient_vector_mean = average_gradient(gradient_vectors);
        //std::cout << print_av_gradient_vector(gradient_vector_mean);

        // 5) Update weights and biases using gradient descent (using learning rate)
        gradient_descent(gradient_vector_mean);

        //Clear vectors
        gradient_vectors.clear();
        cost_vector.clear();

        //Increment numbers of iterations
        no_iter++;
    }
   
    return true;
}

//Methods used in backpropagation
//Calculate and set deltas of neurons in the output layer
void Network::calculate_delta_output_layer(const std::vector<double>& exp_a_output_neuron)
{
    //Iterate over neurons in output layer
    int index_output_layer = no_layers - 1;
    //Iterate over neurons in output layer
    for(int i = 0; i < no_neurons_in_layer[index_output_layer]; i++)
    {  
        //Activation of neuron i in layer 0
        Neuron* nptr_out = get_ptr_to_neuron(index_output_layer, i);
        double a_out = nptr_out->get_activation();
        //Compute delta
        double delta = 2 * (exp_a_output_neuron[i] - a_out) * a_out * (1 - a_out);
        //Set delta as member variable of output neuron
        nptr_out->set_delta(delta);
    }
}

//Calculate and set deltas of neurons in the hidden layers
void Network::calculate_delta_hidden_layer()
{
    //Iterate backwards over all hidden layers
    int layer_start = no_layers - 2;
    int layer_end = 1;
    int layer_curr = layer_start;
    while(layer_curr >= layer_end)
    {
        Layer* ptr_l = get_pointer_to_layer(layer_curr);
        int no_neurons_layer_l = ptr_l->get_no_neurons();
        //Iterate over neurons in layer l
        for(int j = 0; j < no_neurons_layer_l; j++)
        {            
            Neuron* ptr_neuron_l = ptr_l->get_ptr_to_neuron(j);
            //Get activation of neuron in layer l
            double act_neuron_l = ptr_neuron_l->get_activation();
            
            double delta = 0;
            //Iterate over neurons in next layer l+1
            Layer* ptr_lp1 = get_pointer_to_layer(layer_curr + 1);
            int no_neurons_layer_lp1 = ptr_lp1->get_no_neurons();
            for(int k = 0; k < no_neurons_layer_lp1; k++)
            {  
                Neuron* ptr_neuron_lp1 = ptr_lp1->get_ptr_to_neuron(k);
                //Get delta of neuron in layer l+1
                double delta_neuron_lp1 = ptr_neuron_lp1->get_delta();
                //Get weight of connecting edge of neuron in layer l+1
                //pointing to neuron in layer l
                Edge* ptr_edge_kj = ptr_neuron_lp1->get_edge_ptr(j);
                double weight_kj = ptr_edge_kj->get_weight();
                
                //Comput deltas for neurons in layer l
                delta += delta_neuron_lp1 * weight_kj * act_neuron_l * (1 - act_neuron_l);
            }

            //Set delta for neuron in layer l
            ptr_neuron_l->set_delta(delta);
        }
        layer_curr--;
    }
}

//Compute gradient of cost for one training image with respect to all weights and biases (using deltas)
//Gradient of cost vector is added to a vector containing gradients for all training images
//The order of weights and biases is:
//- Layers starting at first hidden layer and end at output layer
//- Neurons in layers start from 0 and end with last neuron
//- For each neuron first weights of all edges to neurons in the previous layer are saved
//- The order is the order of indices of neurons in previous layer (0-last)
//- After weights the bias of each neuron is saved in the gradient vector
void Network::calculate_gradient_vector(gradient_vectors& grad_vectors)
{
    std::vector<double> gradient_vector_image;

    //Iterate forward over all layers except input layer
    for(int j = 1; j < no_layers; j++)
    {
        //Iterate over neurons in layer
        for(int k = 0; k < no_neurons_in_layer[j]; k++)
        {
            //Get pointer to neuron in layer l
            Neuron* nptr_l = get_ptr_to_neuron(j, k); 
            //Get delta of the neuron
            double delta_l = nptr_l->get_delta();

            //Set WEIGHT gradients
            //Iterate over neurons in previous layer l-1
            for(int l = 0; l < no_neurons_in_layer[(j - 1)]; l++)
            {
                //Get pointer to neuron in layer l-1
                Neuron* nptr_lm1 = get_ptr_to_neuron((j - 1), l);  
                //Get activation of previous neuron
                double activation_lm1 = nptr_lm1->get_activation();

                //Set weight gradient
                double gradient_weight = delta_l * activation_lm1;
                //copy gradients in gradient vector
                gradient_vector_image.push_back(gradient_weight);
            }  

            //Set BIAS gradient
            gradient_vector_image.push_back(delta_l);
        }
    } 
    //Add gradient vector for single training image to vector for all images
    grad_vectors.push_back(gradient_vector_image);
}

//Compute average cost for all training images and output it
double Network::average_cost(const std::vector<double>& cost_vector)
{
    int cost_vector_size = cost_vector.size();
    double mean_cost = 0;
    for(int i = 0; i < cost_vector_size; i++)
    {
        mean_cost += cost_vector[i];
    }
    return mean_cost / (double) cost_vector_size;
}

//Method averages gradient vectors for all training images
//And returns a vector with the mean values
std::vector<double> Network::average_gradient(const gradient_vectors& gradient_vectors)
{
    std::vector<double> gradient_vector_mean;
    int gradient_vectors_size = gradient_vectors.size();
    int gradient_vector_size = gradient_vectors[0].size();
    for(int j = 0; j < gradient_vector_size; j++)  
    {    
        double sum = 0;
        for(int k = 0; k < gradient_vectors_size; k++)
        {
            sum += gradient_vectors[k][j];
        } 
        gradient_vector_mean.push_back((sum / (double)gradient_vectors_size));   
    }
    return gradient_vector_mean;
}

//Method updates weights and biases of the network using gradient descent (and learning rate)
void Network::gradient_descent(const std::vector<double>& gradient_vector_mean)
{
    //The update of weights and biases has to happen in the same order as the gradient vector was generated
    //-> First all weights of the neuron and then the bias   
    //Iterate forward over all layers except the input layer
    int gradient_vector_index = 0;
    for(int j = 1; j < no_layers; j++)
    {
        //Iterate over neurons in layer
        for(int k = 0; k < no_neurons_in_layer[j]; k++)
        {
            //Get pointer to neuron
            Neuron* nptr = get_ptr_to_neuron(j, k); 

            //Set WEIGHTS of neuron
            //Iterate over all edges of the neuron
            int no_edges = nptr->get_no_edges();
            for(int l = 0; l < no_edges; l++)
            {
                //Get pointer to edge
                Edge* eptr = nptr->get_edge_ptr(l);
                //Get old edge weight
                double old_weight = eptr->get_weight();
                //Compute new weight
                //////////////////////////////////////////////////////////////////////////////////////////////////
                double new_weight = old_weight + (learning_rate * gradient_vector_mean[gradient_vector_index]);
                //Write new weight to network
                eptr->set_weight(new_weight);
                //Increment gradient vector index
                gradient_vector_index++;
            }

            //Set BIAS of neuron
            //Get old bias
            double old_bias = nptr->get_bias();
            //Compute new bias
            //////////////////////////////////////////////////////////////////////////////////////////////////
            double new_bias = old_bias + (learning_rate * gradient_vector_mean[gradient_vector_index]);
            //Write new weight to network
            nptr->set_bias(new_bias);
            //Increment gradient vector index
            gradient_vector_index++;
        }
    } 
}

///////////////
// Functions //
///////////////

//Overload the output operator << for printing the neuron
std::ostream& operator<< (std::ostream& os, Network& nn)
{
    int no_layers = nn.get_no_layers();

    os << "::::::::::::NEURAL NETWORK::::::::::::" << std::endl; 
    if(no_layers == 0)
    {
        std::cout<< std::endl << "Network is not set up yet!" << std::endl;        
    }
    else
    {
        //Print network information
        std::cout << "Number of layers: " << no_layers << std::endl;
        std::cout << "Number of hidden layers: " << (no_layers - 2) << std::endl;  
        std::cout << "Number of neurons: " << nn.get_no_neurons() << std::endl; 
        std::cout << "Number of edges: " << nn.get_no_edges() << std::endl; 
        std::cout << "Number of biases: " << nn.get_no_biases() << std::endl << std::endl;
        //Print layer information
        Layer* layer_ptr;
        for(int i = 0; i < no_layers; i++)
        {
            layer_ptr = nn.get_pointer_to_layer(i);
            std::cout << layer_ptr << std::endl;
        }
    }
    return os; 
}

//Generate random double values between min and max
double generate_random_d(double min, double max)
{
    return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}