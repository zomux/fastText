#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "src/matrix.h"
#include "src/qmatrix.h"
#include "src/vector.h"

/**
* Quantize embedding matrix in Glove format using NPQ.
* and export the quantized matrix in Glove format.
*
*/

int main(int argc, char const * const argv[])
{
    if (argc != 5) {
        std::cout << "Usage:" << std::endl;
        std::cout << "./export_glove <num_word> <num_dim> <num_dsub> <glove_path>" << std::endl;
        return -1;
    }

    int n_word = std::atoi(argv[1]);
    int n_dim = std::atoi(argv[2]);
    int n_dsub = std::atoi(argv[3]);
    std::string input_path = argv[4];
    if (n_word <= 0 || n_dim <= 0 || n_dsub <= 0) {
        std::cout << "Incorrect arguments." << std::endl;
        return -1;
    }

    // Load the Glove matrix
    std::cout << "load " << input_path << std::endl;
    std::ifstream infile(input_path);
    fasttext::Matrix mat;
    mat.load_glove(infile, n_word, n_dim);

    // Quantize the matrix and produce QMatrix
    std::cout << "matrix loaded, quantize" << std::endl;
    fasttext::QMatrix qmat(mat, n_dsub, true);

    // Export the quantized vectors
    std::string out_path = input_path + ".pq." + std::to_string(n_dsub);
    std::ofstream out(out_path);
    std::cout << "export to " << out_path << std::endl;

    fasttext::Vector vec(n_dim);
    for (int i=0; i < n_word; ++i) {
        vec.zero();
        vec.addRow(qmat, i);
        out << vec << std::endl;
    }
    out.close();
    return 0;
}
