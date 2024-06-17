#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <algorithm>

std::vector<std::vector<float> > convolution(const std::vector<std::vector<float> >& input, const std::vector<std::vector<float> >& kernel) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    int outputSize = inputSize - kernelSize + 1;

    std::vector<std::vector<float> > output(outputSize, std::vector<float>(outputSize, 0.0f));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

std::vector<std::vector<float> > convolutionWithPadding(std::vector<std::vector<float> > &image, std::vector<std::vector<float> > &kernel) {
    int imgHeight = image.size();
    int imgWidth = image[0].size();
    int kernelHeight = kernel.size();
    int kernelWidth = kernel[0].size();

    // Initialize the output image with zeros
    std::vector<std::vector<float> > out(imgHeight, std::vector<float>(imgWidth, 0.0));

    // Pad the image
    int padHeight = kernelHeight / 2;
    int padWidth = kernelWidth / 2;
    std::vector<std::vector<float> > paddedImage(imgHeight + 2 * padHeight, std::vector<float>(imgWidth + 2 * padWidth, 0.0));
    for (int i = 0; i < imgHeight; ++i) {
        for (int j = 0; j < imgWidth; ++j) {
            paddedImage[i + padHeight][j + padWidth] = image[i][j];
        }
    }

    // Perform convolution
    for (int i = 0; i < imgHeight; ++i) {
        for (int j = 0; j < imgWidth; ++j) {
            float sum = 0.0;
            for (int k = 0; k < kernelHeight; ++k) {
                for (int l = 0; l < kernelWidth; ++l) {
                    sum += paddedImage[i + k][j + l] * kernel[k][l];
                }
            }
            out[i][j] = sum;
        }
    }
    return out;
}


// Non-linear activation functions
float relu(float x) {
    return std::max(0.0f, x);
}

float tanh(float x) {
    return std::tanh(x);
}


// Function to perform max pooling
std::vector<std::vector<float> > maxPooling(const std::vector<std::vector<float> >& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;

    std::vector<std::vector<float> > output(outputSize, std::vector<float>(outputSize, 0.0f));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float maxVal = input[i * poolSize][j * poolSize];
            for (int pi = 0; pi < poolSize; ++pi) {
                for (int pj = 0; pj < poolSize; ++pj) {
                    maxVal = std::max(maxVal, input[i * poolSize + pi][j * poolSize + pj]);
                }
            }
            output[i][j] = maxVal;
        }
    }
    return output;
}

// Function to perform average pooling
std::vector<std::vector<float> > averagePooling(const std::vector<std::vector<float> >& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;

    std::vector<std::vector<float> > output(outputSize, std::vector<float>(outputSize, 0.0f));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float sum = 0.0f;
            for (int pi = 0; pi < poolSize; ++pi) {
                for (int pj = 0; pj < poolSize; ++pj) {
                    sum += input[i * poolSize + pi][j * poolSize + pj];
                }
            }
            output[i][j] = sum / (poolSize * poolSize);
        }
    }
    return output;
}


// Softmax function
std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    float maxVal = *std::max_element(input.begin(), input.end());
    float sumExp = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = std::exp(input[i] - maxVal);
        sumExp += result[i];
    }
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] /= sumExp;
    }
    return result;
}

// Sigmoid function
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [task of choice 1=convolution, 2=non-linear-activations, 3=subsampling, 4=converting a vector]" << std::endl;
        return 1;
    }
    
    int task = std::stoi(argv[1]);
    
    if (task == 1) {
        int N = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);
        int P = std::stoi(argv[4]);

        std::vector<std::vector<float> > input(N, std::vector<float>(N));
        std::vector<std::vector<float> > kernel(M, std::vector<float>(M));

        float val;
        int count = 0;
        while (count!=N*N) {
            val=std::stof(argv[count+5]);
            input[count / N][count % N] = val;
            ++count;
        }

        while (count<N*N+M*M) {
            val=std::stof(argv[count+5]);
            kernel[(count-N*N) / M][(count-N*N) % M] = val;
            ++count;
        }

        if (P==0){
            std::vector<std::vector<float> > result = convolution(input, kernel);

            // std::cout << "Convolution without padding:" << std::endl;
            for (const auto& row : result) {
                for (float val : row) {
                    std::cout << val << " ";
                }
            // std::cout << std::endl;
            }

        }
        else{
            std::vector<std::vector<float> > result = convolutionWithPadding(input, kernel);

            // std::cout << "\nConvolution with padding:" << std::endl;
            for (const auto& row : result) {
                for (float val : row) {
                    std::cout << val << " ";
                }
                // std::cout << std::endl;
            }
        }


    } else if (task == 2) {
        int choice=std::stoi(argv[2]);
                int N = std::stoi(argv[3]); 

        int M = std::stoi(argv[4]);
        
                std::vector<std::vector<float> > input(N, std::vector<float>(M));
                        float val;
        int count = 0;
        while (count!=N*M) {
            val=std::stof(argv[count+5]);
            input[count / M][count % M] = val;
            ++count;
        }
 if(choice==0){
// std::cout << "\nReLU activation:" << std::endl;
    for (const auto& row : input) {
        for (float val : row) {
            std::cout << relu(val) << " ";
        }
        // std::cout << std::endl;
    }
 }
 else{
//  std::cout << "\nTanh activation:" << std::endl;
    for (const auto& row : input) {
        for (float val : row) {
            std::cout << tanh(val) << " ";
        }
        // std::cout << std::endl;
    }

 }


    } else if (task == 3) {
        int choice = std::stoi(argv[2]);
        int N = std::stoi(argv[3]);
        int poolsize = std::stoi(argv[5]);

        std::vector<std::vector<float> > input(N, std::vector<float>(N));
        float val;
        int count = 0;
        while (count!=N*N) {
            val=std::stof(argv[count+5]);
            input[count / N][count % N] = val;
            ++count;
        }
        if(choice==0){
            std::vector<std::vector<float> > result  = maxPooling(input, poolsize);

    // std::cout << "\nMax Pooling:" << std::endl;
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        // std::cout << std::endl;
    }
        }
        else{
                std::vector<std::vector<float> > result  = averagePooling(input, poolsize);

    // std::cout << "\nAverage Pooling:" << std::endl;
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        // std::cout << std::endl;
    }
        }



    } else if (task == 4) {
int choice = std::stoi(argv[2]);
        int N = std::stoi(argv[3]);

        std::vector<float> input(N);
        float val;
        int count = 0;
        while (count!=N) {
            val=std::stof(argv[count+4]);
            input[count] = val;
            ++count;
        }
        if (choice==0){
    // std::cout << "\nSigmoid result:" << std::endl;
    for (float val : input) {
    std::cout << sigmoid(val) << " ";

    }
        std::cout<<std::endl;

        }else{
    std::vector<float> softmaxResult = softmax(input);

                // std::cout << "\nSoftmax result:" << std::endl;
    for (float val : softmaxResult) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
        }

    } else {
        std::cout << "Invalid task number. Please provide a valid task number." << std::endl;
        return 1;
    }

    return 0;
}