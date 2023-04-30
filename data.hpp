#ifndef DATA_HPP
#define DATA_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cmath> 
#include "typedefs.hpp"
#include <filesystem>
namespace fs = std::filesystem;

///////////
// Class //
///////////

class Data
{
    private:

    //Generate vector for training data
    import_format training_data;
    //Vector for testing images
    import_format test_data;    
    //Symbols for training and expected output (index of output neuron, which is 1.0)
    //Four types of symbols are used for training:
    //1) circle
    //2) square
    //3) cross (Andreskreuz)
    //4) triangle
    labeling_format vector_labels
    {
        std::make_pair("circle", 0),
        std::make_pair("square", 1),
        std::make_pair("cross", 2),
        std::make_pair("triangle", 3)
    };
    //Path to folder with training images
    std::string path_training = "./training_data";
    //Path to folder with test images
    std::string path_testing = "./test_data";

    public:

    //Constructors
    //Standard constructor without parameters
    Data();  

    //Getter
    //Returns a const reference to the taining and test data vector
    const import_format& get_training_data() const;  
    const import_format& get_test_data() const; 
    //Returns a const reference to the vector with labels
    const labeling_format& get_labels() const; 
    //Get number of input neurons from training data
    //= number of pixels of one training image
    //As all images should have the same size it is determined for the first picture
    const int get_no_input_neurons() const;  
    //Number of output neurons from training data
    //= number of symbols for training
    const int get_no_output_neurons() const;  
    //Get path strings
    const std::string get_path_training() const;  
    const std::string get_path_testing() const;      

    //Method to import training data
    //Read image: https://stackoverflow.com/questions/9296059/read-pixel-value-in-bmp-file (modified)
    //Convert to grayscale: https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-color-space-conversion/
    //Number of symbols determines the number of output neurons
    //The names of the symbols must be in the label of the training pictures! (e.g. circle(1).bmp)
    //Pictures are automatically read from a directory and stored in the training image vector
    //Pictures must have the following requirements:
    //1) Windows bitmap format .bmp (uncompressed)
    //2) 24-bit RGB color
    //3) First training set pictures are 25x25 pixels = 625 pixel in total
    //   -> Number of total pixels must be equal to number of input neurons in the neural network
    //Pictures are automatically converted from RGB to grayscale to reduce the number of input neurons
    //First parameter is the folder name and path as string
    //The second parameter is the name of the image to process as string
    //If this parameter is an empy string, all images of the folder will be processed
    bool import_img_data(const std::string&, const std::string& = "");
    //Overload the output operator << for printing the training
    //Implemented as friend function because access of data over getter too complicated
    friend std::ostream& operator<< (std::ostream&, const Data&);
    //Method to choose and test new images
    std::string test_data_form();
};

///////////////
// Functions //
///////////////

//Arduino map() function for double values -> normalization of data
//In: range of data to be normalized
//Out: range to which input is supposed to be normalized
double mapd(double, double, double, double, double);

#endif