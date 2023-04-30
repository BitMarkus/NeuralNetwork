#include "data.hpp"

///////////
// Class //
///////////

//Constructors
//Standard constructor without parameters
Data::Data() {};  

//Methods

//Getter
//Returns a const reference to the taining and test data vector
const import_format& Data::get_training_data() const
{
    return training_data;
}
const import_format& Data::get_test_data() const
{
    return test_data;
}
//Returns a const reference to the vector with labels
const labeling_format& Data::get_labels() const
{
    return vector_labels;
}
//Get number of input neurons from training data
//= number of pixels of one training image
//As all images should have the same size it is determined for the first picture
const int Data::get_no_input_neurons() const
{
    if(training_data.size() > 0)
    {return training_data[0].first.size();}
    else
    {return 0;}
}  
//Number of output neurons from training data
//= number of symbols for training
const int Data::get_no_output_neurons() const
{
    if(training_data.size() > 0)
    {return training_data[0].second.size();}
    else
    {return 0;}
}  
//Get path strings
const std::string Data::get_path_training() const
{
    return path_training;
} 
const std::string Data::get_path_testing() const
{
    return path_testing;
} 

//Method to import training data
//Read image: https://stackoverflow.com/questions/9296059/read-pixel-value-in-bmp-file
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
//If this parameter is an empty string, all images of the folder will be processed
bool Data::import_img_data(const std::string& path, const std::string& name)
{
    //Header size of a windows bitmap file
    static constexpr size_t HEADER_SIZE = 54;

    //Loop over images in folder
    //https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
    for(const auto & entry : fs::directory_iterator(path))
    {
        //Read filename for labels
        auto fn = fs::path(entry.path()).filename();
        //Convert filename to string 
        std::string filename{fn.u8string()};

        if(name == "" || (name != "" && name == filename))
        {
            //std::cout << filename << std::endl;

            //Output vector with desired output for one image
            std::vector<double> output_vector;
            //Input vector with all grayscale values of one training image
            //normalized to 1.0 (0 = 0.0, 255 = 1.0)
            std::vector<double> input_vector; 

            ///////////////////
            // Output vector //
            ///////////////////

            int index_label_found = -1;
            //Iterate over vector labels and check, if *label is in the filename
            for(int i = 0; i < vector_labels.size(); i++)
            {
                //search for label substring in filename
                //https://cplusplus.com/reference/string/string/find/
                std::size_t found = filename.find(vector_labels[i].first);
                if(found != std::string::npos)
                {
                    index_label_found = i;              
                }
            }
            //If no label is found, output vector remains empty
            if(index_label_found >= 0)
            {
                for(int i = 0; i < vector_labels.size(); i++)
                { 
                    //Fill vector with 0.0 or 1.0
                    double x;
                    if(i == vector_labels[index_label_found].second) {x = 1.0;}
                    else {x = 0;}
                    output_vector.push_back(x);                 
                }           
            }

            //////////////////
            // Input vector //
            //////////////////

            //Read file in binary mode
            std::ifstream f(entry.path(), std::ios::binary);
            std::array<char, HEADER_SIZE> header;
            f.read(header.data(), header.size());

            //Read bmp header file info
            auto fileSize = *reinterpret_cast<uint32_t *>(&header[2]);
            auto dataOffset = *reinterpret_cast<uint32_t *>(&header[10]);
            auto width = *reinterpret_cast<uint32_t *>(&header[18]);
            auto height = *reinterpret_cast<uint32_t *>(&header[22]);
            auto depth = *reinterpret_cast<uint16_t *>(&header[28]);

            //Every row is filled with 0's at the end until the number of bytes can be devided by 4 (= padding)
            //One byte per color channel = RGB = 3 bytes per pixel
            int row_padded = (width*3 + 3) & (~3);
            //Actual number of bytes for pixel data (with color and padding)
            int dataSize = row_padded * height;
            //Vector to store bytes in a image row
            std::vector<char> data_row(row_padded);

            //By following code the image is read from bottom left to top right
            //reading in format BGR and not RGB!
            for(int i = 0; i < height; i++)
            {
                f.seekg(HEADER_SIZE + (row_padded * i));
                f.read(data_row.data(), (width*3));
                //Line length for iteration
                int line_length_it = (width*3) - 2;
                //Iterate over line
                for(int j = 0; j < line_length_it; j += 3)
                {
                    //Convert RGB values to grayscale values
                    //Grayscale = 0.299R + 0.587G + 0.114B
                    double gray_val = (0.114 * int(data_row[j] & 0xff)) +      //B
                                    (0.587 * int(data_row[j+1] & 0xff)) +    //G 
                                    (0.299 * int(data_row[j+2] & 0xff));     //R
                    //Normalize the gray value to 0-1.0
                    double norm_grey_val = mapd(gray_val, 0, 255.0, 0, 1.0);
                    //Add the value to the grayscale vector
                    input_vector.push_back(norm_grey_val);
                }        
            }

            //Reverse the vector: https://stackoverflow.com/questions/8877448/how-do-i-reverse-a-c-vector
            //Just for cosmetic reasons: Afterwards up and down of the picture is correct, however, it is mirrored
            //Without it, left and right sides are correct, but up and down are flipped
            std::reverse(input_vector.begin(), input_vector.end());

            //Create input vector for training with a pair of vectors
            //First vector contains the normalized grey values for input neurons
            //Second vector the desired output pattern for output neurons
            if(name == "")
            {
                training_data.push_back(std::make_pair(input_vector, output_vector));
            }
            else if(name != "" && name == filename)
            {
                test_data.clear();
                test_data.push_back(std::make_pair(input_vector, output_vector));                
            }
        }
    }

    return true;
}

//Method to choose and test new images
//Returns the name of the test image as string
std::string Data::test_data_form()
{
    std::cout << "::::::::::::TEST DATA::::::::::::" << std::endl;
    //Show all images in respective folder "test_data"
    //Generate a vector with a pair containing filename and access number
    std::vector<std::pair<std::string, int>> test_files;
    int no = 0;
    //Loop over images in folder
    //https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
    for(const auto & entry : fs::directory_iterator(path_testing))
    {
        no++;
        //Read filename for labels
        auto fn = fs::path(entry.path()).filename();
        //Convert filename to string 
        std::string filename{fn.u8string()}; 
        //Create vector with pairs
        test_files.push_back(std::make_pair(filename, no));
    }
    //Determine number of files in folder
    int no_test_files = test_files.size();
    //std::cout << no_test_files << std::endl;
    //Print filenames
    if(no_test_files == 0)
    {
        std::cout << std::endl << "No files in folder detected!" << std::endl;
    }
    else
    {
        //Iterate over test_files vector
        for(int i = 0; i < no_test_files; i++)
        {    
            std::cout << test_files[i].second << ") ";
            std::cout << test_files[i].first << std::endl;
        }
    }
    int image_nr_choice = 0;
    std::cout << "Please choose an image: ";
    std::cin >> image_nr_choice;
    std::cout << std::endl;

    if(test_files.size() > 0)
    {
        return test_files[(image_nr_choice-1)].first;
    }
    else
    {
        return "";        
    }
}

///////////////
// Functions //
///////////////

//Overload the output operator << for printing the training data
std::ostream& operator<< (std::ostream& os, const Data& imp)
{
    int no_training_data = imp.training_data.size();
    const int no_of_digits = 2;

    os << "::::::::::::TRAINING DATA::::::::::::" << std::endl;
    os << "Number of images: " << no_training_data << std::endl << std::endl;

    //Iterate over training set
    for(int i = 0; i < no_training_data; i++)
    {
        os << std::endl << ">> Training image " << i << ":" << std::endl;

        //Iterate over input vector
        int input_vector_size = imp.training_data[i].first.size();
        os << "> Input vector (" << input_vector_size << "):" << std::endl;
        int counter = 0;
        for(auto it = imp.training_data[i].first.cbegin(); it != imp.training_data[i].first.cend(); ++it)
        {
            //os << *it;
            os << std::to_string(*it).substr(0, std::to_string(*it).find(".") + no_of_digits + 1);
            if(it != (imp.training_data[i].first.cend() - 1))
            {
                os << ", ";                 
            } 
            //Make a line break every line in picture
            //For that purpose the square root  of the input vector can be used
            //picture must be squared!
            int no_lines = sqrt(input_vector_size);  
            counter++;  
            if(counter % no_lines == 0 && it != (imp.training_data[i].first.cend() - 1))
            {
                os << std::endl;               
            } 
        }
        os << std::endl; 

        //Iterate over output vector
        int output_vector_size = imp.training_data[i].second.size();
        os << "> Output vector (" << std::to_string(output_vector_size) << "):" << std::endl;
        counter = 0;
        for(auto it = imp.training_data[i].second.cbegin(); it != imp.training_data[i].second.cend(); ++it)
        {
            //os << *it;
            os << std::to_string(*it).substr(0, std::to_string(*it).find(".") + no_of_digits + 1);
            if(it != (imp.training_data[i].second.cend() - 1))
            {
                os << ", ";                 
            }
            counter++;
            if(counter % 25 == 0)
            {
                os << std::endl;              
            }
        }
        os << std::endl;
    }
    os << std::endl;

    return os;
}

/*
//Overload the output operator << for printing the training data
std::ostream& operator<< (std::ostream& os, const Import& imp)
{
    int no_test_data = imp.test_data.size();
    const int no_of_digits = 2;

    os << "::::::::::::TEST DATA::::::::::::" << std::endl;
    os << "Number of test images: " << no_test_data << std::endl << std::endl;

    //Iterate over training set
    for(int i = 0; i < no_test_data; i++)
    {
        os << ">> Test image " << i << ":" << std::endl;

        //Iterate over input vector
        int input_vector_size = imp.test_data[i].first.size();
        os << "> Input vector (" << input_vector_size << "):" << std::endl;
        int counter = 0;
        for(auto it = imp.test_data[i].first.cbegin(); it != imp.test_data[i].first.cend(); ++it)
        {
            //os << *it;
            os << std::to_string(*it).substr(0, std::to_string(*it).find(".") + no_of_digits + 1);
            if(it != (imp.test_data[i].first.cend() - 1))
            {
                os << ", ";                 
            } 
            //Make a line break every line in picture
            //For that purpose the square root  of the input vector can be used
            //picture must be squared!
            int no_lines = sqrt(input_vector_size);  
            counter++;  
            if(counter % no_lines == 0 && it != (imp.test_data[i].first.cend() - 1))
            {
                os << std::endl;               
            } 
        }
        os << std::endl; 

        //Iterate over output vector
        int output_vector_size = imp.test_data[i].second.size();
        os << "> Output vector (" << std::to_string(output_vector_size) << "):" << std::endl;
        counter = 0;
        for(auto it = imp.test_data[i].second.cbegin(); it != imp.test_data[i].second.cend(); ++it)
        {
            //os << *it;
            os << std::to_string(*it).substr(0, std::to_string(*it).find(".") + no_of_digits + 1);
            if(it != (imp.test_data[i].second.cend() - 1))
            {
                os << ", ";                 
            }
            counter++;
            if(counter % 25 == 0)
            {
                 os << std::endl;              
            }
        }
        os << std::endl;
    }
    os << std::endl;

    return os;
}
*/

//Arduino map() function for double values -> normalization of data
//In: range of data to be normalized
//Out: range to which input is supposed to be normalized
double mapd(double val, double in_min, double in_max, double out_min, double out_max) 
{
  return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
