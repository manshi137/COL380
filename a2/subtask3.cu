#include <iostream>
#include <vector>
#include <fstream>
#include <string>
using namespace std;
#define BLOCK_SIZE 16



__global__ void softmaxKernel(float *input, float *output, float maxVal, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = expf(input[idx] - maxVal);
    }
}

__global__ void avgPoolingKernel(float *input, float *output, int inputSize, int poolSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputSize && col < inputSize) {
        int outputRow = row / poolSize;
        int outputCol = col / poolSize;

        // int inputIdx = row * inputSize + col;
        int outputIdx = outputRow * (inputSize / poolSize) + outputCol;

        float sum = 0.0f;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int curRow = outputRow * poolSize + i;
                int curCol = outputCol * poolSize + j;
                if (curRow < inputSize && curCol < inputSize) {
                    int curInputIdx = curRow * inputSize + curCol;
                    sum += input[curInputIdx];
                }
            }
        }

        output[outputIdx] = sum / (poolSize * poolSize);
    }
}


__global__ void maxPoolingKernel(float *input, float *output, int inputSize, int poolSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputSize && col < inputSize) {
        int outputRow = row / poolSize;
        int outputCol = col / poolSize;

        int outputIdx = outputRow * (inputSize / poolSize) + outputCol;

        float maxVal = -INFINITY;  // Initialize maxVal to negative infinity

        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int curRow = outputRow * poolSize + i;
                int curCol = outputCol * poolSize + j;
                if (curRow < inputSize && curCol < inputSize) {
                    int inputIdx = curRow * inputSize + curCol;
                    maxVal = fmaxf(maxVal, input[inputIdx]);
                }
            }
        }

        output[outputIdx] = maxVal;
    }
}


__global__ void tanhKernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = tanh(input[index]);
    }
}


__global__ void reluKernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = fmaxf(0.0f, input[index]);
    }
}


__global__ void convolutionKernel(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize - kernelSize +1;

    if (col < outputSize && row < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                int inputIdx = (row + i) * inputSize + (col + j);
                int kernelIdx = i * kernelSize + j;
                sum += input[inputIdx] * kernel[kernelIdx];
            }
        }
        output[row * outputSize + col] = sum;
    }
}



void convolutionCUDA(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    float *d_input, *d_kernel, *d_output;
    int outputSize = inputSize - kernelSize +1;
    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((inputSize + blockSize.x - 1) / blockSize.x, (inputSize + blockSize.y - 1) / blockSize.y);

    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, inputSize, kernelSize);

    cudaMemcpy(output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}


void reluCUDA(float *input, float *output, int rows, int cols) {
    float *d_input, *d_output;

    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    reluKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


void tanhCUDA(float *input, float *output, int rows, int cols) {
    float *d_input, *d_output;

    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    tanhKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


void maxPoolingCUDA(float* input, float* output, int inputSize, int poolSize) {
    // int inputSize = input.size();
    int outputSize = inputSize / poolSize;

    float *d_input, *d_output;

    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, outputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    maxPoolingKernel<<<gridSize, blockSize>>>(d_input, d_output, inputSize, poolSize);

    cudaMemcpy(output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


void avgPoolingCUDA(float* input, float* output, int inputSize, int poolSize) {
    // int inputSize = input.size();
    int outputSize = inputSize / poolSize;

    float *d_input, *d_output;

    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, outputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    avgPoolingKernel<<<gridSize, blockSize>>>(d_input, d_output, inputSize, poolSize);

    cudaMemcpy(output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}




void softmaxCUDA(float* input, float* output, int inputSize) {
    int size = inputSize;
    // std::vector<float> result(size);
    float maxVal = input[0];
    for (size_t i = 1; i < inputSize; ++i) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }
    float sumExp = 0.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    softmaxKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, maxVal, size);

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    for (int i = 0; i < size; ++i) {
        sumExp += output[i];
    }

    for (int i = 0; i < size; ++i) {
        output[i] /= sumExp;
    }
}

void readKernelWeightsAndBiasconv1(const std::string& filename, int numFilters, int kernelSize, float** kernelWeights, float** biasValues) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Calculate the total number of weights and biases for each filter
    int totalWeights = kernelSize * kernelSize;
    int totalBias = 20;

    // Allocate memory for kernel weights and bias values
    *kernelWeights = (float*)malloc(numFilters * totalWeights * sizeof(float));
    *biasValues = (float*)malloc(numFilters * totalBias * sizeof(float));

    // Read kernel weights and bias values for each filter
    for (int i = 0; i < numFilters; ++i) {
        // Read kernel weights
        for (int j = 0; j < totalWeights; ++j) {
            if (!(file >> (*kernelWeights)[i * totalWeights + j])) {
                std::cerr << "Error: Unable to read kernel weights from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
        // Read bias values
        for (int j = 0; j < totalBias; ++j) {
            if (!(file >> (*biasValues)[j])) {
                std::cerr << "Error: Unable to read bias values from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    

    file.close();
}


void readKernelWeightsAndBiasconv2(const std::string& filename, int numFilters, int inputchannels, int kernelSize, float** kernelWeights, float** biasValues) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Calculate the total number of weights and biases for each filter
    int totalWeights = kernelSize * kernelSize * inputchannels;
    int totalBias = 50;

    // Allocate memory for kernel weights and bias values
    *kernelWeights = (float*)malloc(numFilters * totalWeights * sizeof(float));
    *biasValues = (float*)malloc(numFilters * totalBias * sizeof(float));

    // Read kernel weights and bias values for each filter
    for (int i = 0; i < numFilters; ++i) {
        // Read kernel weights
        for (int j = 0; j < totalWeights; ++j) {
            if (!(file >> (*kernelWeights)[i * totalWeights + j])) {
                std::cerr << "Error: Unable to read kernel weights from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
        // Read bias values
        for (int j = 0; j < totalBias; ++j) {
            if (!(file >> (*biasValues)[j])) {
                std::cerr << "Error: Unable to read bias values from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    

    file.close();
}

int main() {
    const int inputSize = 28;
    const int kernelSize = 5;
    const int poolSize = 2;
    const int conv1numkernel = 20; //numFilters
    const int conv2numkernel = 50; //numFilters
    const int conv1inputchannelnum = 1;
    const int conv2inputchannelnum = 20;
    int conv1OutputSize = inputSize - kernelSize + 1;
    int pool1OutputSize = conv1OutputSize / poolSize;
    int conv2OutputSize = pool1OutputSize - kernelSize + 1;
    int pool2OutputSize = conv2OutputSize / poolSize;
    // int fc1InputSize = pool2OutputSize * pool2OutputSize * 50; 
    // const int fc1OutputSize = 500;
    // int fc2InputSize = fc1OutputSize;
    // const int fc2OutputSize = 10;

    // Allocate memory for input, output, and weights
    float *inputSoft = (float*)malloc(inputSize * inputSize * sizeof(float));
    float *outputSoft1 = (float*)malloc(conv1OutputSize * conv1OutputSize * sizeof(float));
    float *outputSoft2 = (float*)malloc(pool1OutputSize * pool1OutputSize * sizeof(float));
    float *outputSoft3 = (float*)malloc(conv2OutputSize * conv2OutputSize * sizeof(float));
    float *outputSoft4 = (float*)malloc(pool2OutputSize * pool2OutputSize * sizeof(float));
    float *outputSoft5 = (float*)malloc(fc1OutputSize * sizeof(float));
    float *outputSoft6 = (float*)malloc(fc2OutputSize * sizeof(float));
    float *conv1Weights = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    float *conv2Weights = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    float *conv1Bias = (float*)malloc(conv1numkernel * sizeof(float));
    float *conv2Bias = (float*)malloc(conv2numkernel * sizeof(float));
    // float *fc1Weights = (float*)malloc(fc1InputSize * fc1OutputSize * sizeof(float));
    // float *fc2Weights = (float*)malloc(fc2InputSize * fc2OutputSize * sizeof(float));


 
    // Initialize weights
    string filename="/home/cse/dual/cs5200534/a2col380/trained_weights/conv1.txt";
    readKernelWeightsAndBiasconv1(filename, conv1numkernel, kernelSize, &conv1Weights, &conv1Bias);
    cout << "Conv1 weights read from file:" << endl;
    for (int i = 0; i < conv1numkernel; ++i) {
        cout << "Filter " << i + 1 << ":" << endl;
        for (int j = 0; j < kernelSize * kernelSize; ++j) {
            cout << conv1Weights[i * kernelSize * kernelSize + j] << " ";
            if((j+1)%5==0){cout<<"\n";}
        }
        cout << endl;
    }
    std::cout << "Bias Values:" << std::endl;
    for (int i = 0; i < conv1numkernel; ++i) {
        std::cout << conv1Bias[i] << " ";
    }
    std::cout << std::endl;

    filename="/home/cse/dual/cs5200534/a2col380/trained_weights/conv2.txt";
    readKernelWeightsAndBiasconv2(filename, conv2numkernel, conv2inputchannelnum, kernelSize, &conv2Weights, &conv2Bias);
    cout << "Conv2 weights read from file:" << endl;
    for (int i = 0; i < 2; ++i) {
        cout << "Filter " << i + 1 << ":" << endl;
        for (int j = 0; j < kernelSize * kernelSize*conv2inputchannelnum; ++j) {
            cout << conv2Weights[i * kernelSize * kernelSize * conv2inputchannelnum + j] << " ";
            if((j+1)%5==0){cout<<"\n";}
        }
        cout << endl;
    }
    std::cout << "Bias Values:" << std::endl;
    for (int i = 0; i < conv2numkernel; ++i) {
        std::cout << conv2Bias[i] << " ";
    }
    std::cout << std::endl;


    // Perform forward pass
    // convolution(inputSoft, outputSoft1, conv1Weights, inputSize, kernelSize);
    // maxPooling(outputSoft1, outputSoft2, conv1OutputSize, poolSize);
    // convolution(outputSoft2, outputSoft3, conv2Weights, pool1OutputSize, kernelSize);
    // maxPooling(outputSoft3, outputSoft4, conv2OutputSize, poolSize);
    // fullyConnected(outputSoft4, outputSoft5, fc1Weights, fc1InputSize, fc1OutputSize);
    // relu(outputSoft5, fc1OutputSize);
    // fullyConnected(outputSoft5, outputSoft6, fc2Weights, fc2InputSize, fc2OutputSize);
    // softmax(outputSoft6, fc2OutputSize);

    // Free allocated memory
    free(inputSoft);
    free(outputSoft1);
    free(outputSoft2);
    free(outputSoft3);
    free(outputSoft4);
    free(outputSoft5);
    free(outputSoft6);
    free(conv1Weights);
    free(conv2Weights);
    free(fc1Weights);
    free(fc2Weights);

    return 0;
}