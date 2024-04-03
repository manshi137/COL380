#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <string>
#include <time.h>	
#include <dirent.h> // For directory traversalusing namespace std;
#define BLOCK_SIZE 32
using namespace std;

void readImage(const std::string& filename, float** image){
        // std::cout<<filename;
    FILE* file = fopen(filename.c_str(), "r");

    if (!file) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    // *image = (float*)malloc(28 * 28 * sizeof(float));

    for (int i = 0; i < 28; ++i) {
        // Read pixel values
        for (int j = 0; j < 28; ++j) {
            if (fscanf(file, "%f", &((*image)[i * 28 + j])) != 1) {
                std::cerr << "Error: Unable to read pixel values from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);    
}

void readFC1(const std::string& filename, float** kernelWeights, float** biasValues){
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int inputChannel = 50;
    int outputChannel = 500; //=number of filters
    int kernelSize = 4;
    int numFilters = outputChannel;
    // each filter is 4 x 4
    int totalWeights = kernelSize*kernelSize*inputChannel;//per filter
    int totalBias = 1;//per filter
    // *kernelWeights = (float*)malloc(numFilters * totalWeights * sizeof(float));
    // *biasValues = (float*)malloc(numFilters * totalBias * sizeof(float));

    for (int i = 0; i < numFilters; ++i) {
        // Read kernel weights
        for (int j = 0; j < totalWeights; ++j) {
            if (fscanf(file, "%f", &((*kernelWeights)[i * totalWeights + j])) != 1) {
                std::cerr << "Error: Unable to read kernel weights from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < numFilters; ++i) {
        // Read bias values
        for (int j = 0; j < totalBias; ++j) {
            if (fscanf(file, "%f", &((*biasValues)[i*totalBias])) != 1) {
                std::cerr << "Error: Unable to read bias values from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);    
}

void readFC2(const std::string& filename, float** kernelWeights, float** biasValues){
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int inputChannel = 500;
    int outputChannel = 10; //=number of filters
    int kernelSize = 1;
    int numFilters = outputChannel;
    // each filter is 4 x 4
    int totalWeights = kernelSize*kernelSize*inputChannel;//per filter
    int totalBias = 1;//per filter
    // *kernelWeights = (float*)malloc(numFilters * totalWeights * sizeof(float));
    // *biasValues = (float*)malloc(numFilters * totalBias * sizeof(float));

    for (int i = 0; i < numFilters; ++i) {
        // Read kernel weights
        for (int j = 0; j < totalWeights; ++j) {
            if (fscanf(file, "%f", &((*kernelWeights)[i * totalWeights + j])) != 1) {
                std::cerr << "Error: Unable to read kernel weights from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < numFilters; ++i) {
        // Read bias values
        for (int j = 0; j < totalBias; ++j) {
            if (fscanf(file, "%f", &((*biasValues)[i*totalBias+j])) != 1) {
                std::cerr << "Error: Unable to read bias values from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);    
}

void readKernelWeightsAndBiasconv1(const std::string& filename, float** kernelWeights, float** biasValues) {
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    // int inputSize = 28;
    int inputChannel = 1;
    // int outputChannel = 20;
    int numFilters = 20;
    int kernelSize = 5;
    // Calculate the total number of weights and biases for each filter
    int totalWeights = kernelSize * kernelSize * inputChannel; //per filter
    int totalBias = 1; //per filter

    // Allocate memory for kernel weights and bias values
    // *kernelWeights = (float*)malloc(numFilters * totalWeights * sizeof(float));
    // *biasValues = (float*)malloc(20*sizeof(float));

    // Read kernel weights and bias values for each filter
    for (int i = 0; i < numFilters; ++i) {
        // Read kernel weights
           for (int j = 0; j < totalWeights; ++j) {
            if (fscanf(file, "%f", &((*kernelWeights)[i * totalWeights + j])) != 1) {
                std::cerr << "Error: Unable to read kernel weights from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < numFilters; ++i) {
        // Read bias values
        for (int j = 0; j < totalBias; ++j) {
            if (fscanf(file, "%f", &((*biasValues)[i*totalBias+j])) != 1) {
                std::cerr << "Error: Unable to read bias values from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    // cout<<"afvblujsbdjvbsbvlsfabjlvblsbivfbaibvfuovboas";
    // for (int i = 0; i < 20; ++i) {
    //     std::cout << (*biasValues)[i] << " ";
    // }
    // cout<<"\n";
    fclose(file);
}

void readKernelWeightsAndBiasconv2(const std::string& filename, float** kernelWeights, float** biasValues) {
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    // int inputSize = 12;
    int inputChannel = 20;
    int outputChannel = 50;
    int numFilters = outputChannel;
    int kernelSize = 5;
    // Calculate the total number of weights and biases for each filter
    int totalWeights = kernelSize * kernelSize * inputChannel; //per filter
    int totalBias = 1; 
    for (int i = 0; i < numFilters; ++i) {
        // Read kernel weights
        for (int j = 0; j < totalWeights; ++j) {
            if (fscanf(file, "%f", &((*kernelWeights)[i * totalWeights + j])) != 1) {
                std::cerr << "Error: Unable to read kernel weights from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < numFilters; ++i) {
        // Read bias values
        for (int j = 0; j < totalBias; ++j) {
            if (fscanf(file, "%f", &((*biasValues)[i*totalBias + j])) != 1) {
                std::cerr << "Error: Unable to read bias values from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }


    

    fclose(file);
}
