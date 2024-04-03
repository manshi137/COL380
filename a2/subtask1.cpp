#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<std::vector<float>> convolution(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& kernel) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    int outputSize = inputSize - kernelSize + 1;

    std::vector<std::vector<float>> output(outputSize, std::vector<float>(outputSize, 0.0f));

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

std::vector<std::vector<float>> convolutionWithPadding(std::vector<std::vector<float>> &image, std::vector<std::vector<float>> &kernel) {
    int imgHeight = image.size();
    int imgWidth = image[0].size();
    int kernelHeight = kernel.size();
    int kernelWidth = kernel[0].size();

    // Initialize the output image with zeros
    std::vector<std::vector<float>> out(imgHeight, std::vector<float>(imgWidth, 0.0));

    // Pad the image
    int padHeight = kernelHeight / 2;
    int padWidth = kernelWidth / 2;
    std::vector<std::vector<float>> paddedImage(imgHeight + 2 * padHeight, std::vector<float>(imgWidth + 2 * padWidth, 0.0));
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
std::vector<std::vector<float>> maxPooling(const std::vector<std::vector<float>>& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;

    std::vector<std::vector<float>> output(outputSize, std::vector<float>(outputSize, 0.0f));

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
std::vector<std::vector<float>> averagePooling(const std::vector<std::vector<float>>& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;

    std::vector<std::vector<float>> output(outputSize, std::vector<float>(outputSize, 0.0f));

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

int main() {
    // Example usage
    std::vector<std::vector<float>> input = {{-1.0f, 2.0f, 3.0f,3.0f},
                                             {4.0f, -5.0f, 6.0f,3.0f},
                                             {7.0f, 8.0f, 9.0f,-3.0f},
                                             {71.0f, 10.0f, -8.0f,-3.0f}};

    std::vector<std::vector<float>> kernel = {{0.0f, 1.0f},
                                              {1.0f, 0.0f}};

    std::vector<std::vector<float>> result = convolution(input, kernel);

    std::cout << "Convolution without padding:" << std::endl;
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    result = convolutionWithPadding(input, kernel);

    std::cout << "\nConvolution with padding:" << std::endl;
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nReLU activation:" << std::endl;
    for (const auto& row : input) {
        for (float val : row) {
            std::cout << relu(val) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTanh activation:" << std::endl;
    for (const auto& row : input) {
        for (float val : row) {
            std::cout << tanh(val) << " ";
        }
        std::cout << std::endl;
    }

    result = maxPooling(input, 2);

    std::cout << "\nMax Pooling:" << std::endl;
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    result = averagePooling(input, 2);

    std::cout << "\nAverage Pooling:" << std::endl;
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> softmaxResult = softmax({1.0f, 2.0f, 3.0f});
    std::cout << "\nSoftmax result:" << std::endl;
    for (float val : softmaxResult) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "\nSigmoid result:" << std::endl;
    std::cout << sigmoid(0.5f) << std::endl;

    return 0;
}

