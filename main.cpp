#include <iostream>
#include "network.hpp"
#include "typedefs.hpp"
#include "data.hpp"
#include "model.hpp"

int main()
{ 
    //Generate a object for training data handling
    Data dat;
    //Generate a object for model handling (im-/export)
    Model mod;
    //Generate a empty network object
    Network net;

    ///////////////
    // MAIN MENU //
    ///////////////

    int menu0_choice = 0;
    bool menu0_stay = true;
    while(menu0_stay)
    {
        std::cout << std::endl << "::::::::Main Menu::::::::" << std::endl;
        std::cout << "1) Import training data and generate new network" << std::endl;
        std::cout << "2) Import model and generate saved network" << std::endl;
        std::cout << "3) Train network with training images" << std::endl;
        std::cout << "4) Test network with test images" << std::endl;
        std::cout << "5) Export model" << std::endl;
        std::cout << "6) Reset network" << std::endl;
        std::cout << "7) Print network" << std::endl;
        std::cout << "8) Print training data" << std::endl;
        std::cout << "9) Exit program" << std::endl;
        std::cout << "Please choose an option: ";
        std::cin >> menu0_choice;
        std::cout << std::endl;

        switch (menu0_choice)
        {
            case 1:
            {
                //1) Import training data and generate new network
                dat.import_img_data(dat.get_path_training());
                std::cout << "Training data successfully imported." << std::endl; 
                //Read number of input and output neurons from training data
                int no_input_neurons = dat.get_no_input_neurons();
                int no_output_neurons = dat.get_no_output_neurons();
                //Fill network with necessary data
                net.network_input_form(no_input_neurons, no_output_neurons);
                //Generate network
                net.generate_network();
                break;
            }
            case 2:
            {
                //2) Import model and generate saved network
                //Import header and data (weights and biases)
                mod.import_model(net);
                //Set network parameters for new network generation
                net.setup_network_via_import(mod.get_model_header());
                //Generate network
                net.generate_network();
                //Import weights and biases
                net.import_weights_biases(mod.get_model_data());
                std::cout << "Import of weights and biases successful!" << std::endl;                
                break;
            }
            case 3:
            {
                //3) Train network with training images
                //Backpropagation
                net.backpropagation(dat.get_training_data());
                std::cout << "Training of model finished!" << std::endl;          
                break;
            }
            case 4:
            {
                //4) Test network with test images
                //Test network
                std::string filename = dat.test_data_form();
                dat.import_img_data(dat.get_path_testing(), filename);
                net.set_input_layer(dat.get_test_data());
                net.calculate_output();
                std::cout << net.print_labeled_output(dat.get_labels());
                std::cout << "Cost: " << net.calculate_cost(dat.get_test_data()) << std::endl; 
                break;
            }
            case 5:
            {
                //5) Export model
                mod.export_model(net);
                break;
            }
            case 6:
            {
                //6) Reset network
                net.reset_network();
                break;
            }
            case 7:
            {
                //7) Print network
                std::cout << net;
                break;
            }
            case 8:
            {
                //8) Print training data
                std::cout << dat << std::endl;                 
                break;
            }
            case 9:
            {
                std::cout << "Exit program..." << std::endl << std::endl;
                menu0_stay = false;
                break;
            }
            default:
            {
                //9) Exit program
                std::cout << "Not a valid choice!" << std::endl;
            }
        }
    }

    return 0;
}