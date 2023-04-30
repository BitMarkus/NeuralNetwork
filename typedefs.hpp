#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <vector>

///////////////
// Typedefs ///
///////////////

//Typedef for format of training data
typedef std::vector<std::pair<std::vector<double>,std::vector<double>>> import_format;

//Typedef for vector with gradient vectors for each training image
typedef std::vector<std::vector<double>> gradient_vectors;

//Typedef for format of data labeling
typedef std::vector<std::pair<std::string, int>> labeling_format;

#endif
