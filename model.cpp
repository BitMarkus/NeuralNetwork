#include "model.hpp"

///////////
// Class //
///////////

//Standard constructor without parameters
Model::Model() {}; 

//Getter
//Get header vector
const std::vector<int>& Model::get_model_header() const
{
    return header_vector;
}
//Get data vector
const std::vector<double>& Model::get_model_data() const
{
    return data_vector;
}

//Methods

//Method to export a trained model into a text file
//The model is specific to a network -> new network is generated when the model data is imported
//Filename can be choosen, ending is .mdl
//File consists of header and body
//Header: Number of neurons per layer (necessary information for network generation)
//Header data is written in the first line, seperated by | = delimiter
//Body: weights ans biases of the network
//Body data starts at second line, each weight and bias are written in a new line
//The order of weights and biases is arranged like in the gradient vector
//- Go through layers starting at first hidden layer and end at output layer
//- Go through neurons in layers starting at 0 and ending with last neuron
//- For each neuron first weights of all edges to neurons in the previous layer are saved
//- The order is the order of indices of neurons in previous layer (0-last)
//- After all weights of a neuron the bias is saved in the model file
bool Model::export_model(Network& net)
{
    //Number of layers in the network
    int no_layers = net.get_no_layers();
    //Number of digits for weights and biases
    const int no_of_digits = 6;
    //Filename of model
    std::string filename;

    //Generate header file = Neurons per layer
    std::string header;
    for(int i = 0; i < no_layers; i++)
    {
       header += std::to_string(net.get_no_neurons_in_layer(i));
       if(i != (no_layers - 1))
       {
            header += "|";        
       }
       else
       {
            header += "\n";           
       }
    }

    //Genertae body with weights and biases of the network
    std::string body;
    for(int j = 1; j < no_layers; j++)
    {
        int no_neurons_in_layer = net.get_no_neurons_in_layer(j);
        //Iterate over neurons in layer
        for(int k = 0; k < no_neurons_in_layer; k++)
        {
            //Get pointer to neuron
            Neuron* nptr = net.get_ptr_to_neuron(j, k); 
            //Save WEIGHTS of neuron
            //Iterate over all edges of the neuron
            int no_edges = nptr->get_no_edges();
            for(int l = 0; l < no_edges; l++)
            {
                //Get pointer to edge
                Edge* eptr = nptr->get_edge_ptr(l);
                //Get edge weight
                double weight = eptr->get_weight();
                //Set string
                body += std::to_string(weight).substr(0, std::to_string(weight).find(".") + no_of_digits + 1) + "\n"; 
            }
            //Save BIAS of neuron
            double bias = nptr->get_bias();
            body += std::to_string(bias).substr(0, std::to_string(bias).find(".") + no_of_digits + 1) + "\n";
        }
    } 

    //Form for filename
    std::cout << "Save model as .mdl file" << std::endl;
    std::cout << "Filename (without extension): ";
    std::cin >> filename;
    //Add path and extension to filename
    std::string path_name_ext = path_models + filename + ".mdl";

    //Open file for writing with check
    std::ofstream file(path_name_ext);
    if(file.is_open())
    {
        //Write header and body
        file << header << body;        
        //Close file
        file.close();
        return true;
    }
    //If file cannot be opened
    else 
    {
        std::cout << "Unable to open file"; 
        return false;
    }
}

//Method to import a saved model
//By importing a model a new network is generated and filled with weights/biases
bool Model::import_model(Network&)
{
    //Clear vectors for new import
    data_vector.clear();
    header_vector.clear();
    
    //Choose file to open
    std::string filename;
    //Open model file for reading
    //The name of the desired file needs to be typed in (not the folder nor the path ot the extension)
    //Later a list of all files in the model folder should be displayed
    std::cout << "Load model and create an appropriate network:" << std::endl;
    std::cout << "Filename (without extension): ";
    std::cin >> filename; 
    std::string path_name_ext = path_models + filename + ".mdl";

    //Open file with check
    std::ifstream file(path_name_ext);
    if(file.is_open())
    {
        //String for storing lines (line separated version)
        std::string line;
        //Separate string to store the header line
        std::string header_line;
        //Read line by line
        int nr_line = 1;
        while(getline(file, line))
        {
            //Read header line
            if(nr_line == 1)
            {
                header_line = line;                
            }
            //Read body lines
            else
            {
                //Cast string to double and append it to the vector
                data_vector.push_back(std::stod(line));
            }
            nr_line++;
        }
        //Close file
        file.close();
        
        //Parse header line: Split header string by delimiter |
        std::vector<std::string> header_split = split_string(header_line, '|');
        //Check size of vector
        int header_split_size = header_split.size();
        if(header_split_size < 2)
        {
            std::cout << "Import of heade data failed!"; 
            return false;            
        }
        else
        {
            //Convert strings to ints and add them to the vector header_vector<int>
            for(int i = 0; i < header_split_size; i++)
            {
                header_vector.push_back(stoi(header_split[i]));                
            }
            /*
            for(int i = 0; i < header_split_size; i++)
            {std::cout << header_vector[i] << std::endl;}
            */
            return true;
        }
    }
    //If file cannot be opened
    else 
    {
        std::cout << "Unable to open file!"; 
        return false;
    }
}

///////////////
// Functions //
///////////////

//Split function for strings using a delimiter
//Returns a vector of delimiter separated strings 
//https://favtutor.com/blogs/split-string-cpp
std::vector<std::string> split_string(std::string str, char delimiter) 
{
    std::vector<std::string> return_vector;
    int startIndex = 0;
    int endIndex = 0;
    for (int i = 0; i <= str.size(); i++) 
    {        
        //If end of the word or the end of the input was reached
        if(str[i] == delimiter || i == str.size()) 
        {
            endIndex = i;
            std::string temp;
            temp.append(str, startIndex, endIndex - startIndex);
            return_vector.push_back(temp);
            startIndex = endIndex + 1;
        }
    }
    return return_vector;
}