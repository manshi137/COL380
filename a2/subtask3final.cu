#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <string>
using namespace std;
#define BLOCK_SIZE 32

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
    int inputSize = 28;
    int inputChannel = 1;
    int outputChannel = 20;
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
    int inputSize = 12;
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
                    // cout<<input[inputIdx]<<" "<<maxVal;
                    maxVal = fmaxf(maxVal, input[inputIdx]);
                }
            }
        }

        output[outputIdx] = maxVal;
    }
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

__global__ void reluKernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = fmaxf(0.0f, input[index]);
    }
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


__global__ void softmaxKernel(float *input, float *output, float maxVal, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = expf(input[idx] - maxVal);
    }
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
    // cout<<"\n jghvvhjjg   "<<maxVal<<"\n";
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
        // cout<<output[i]<<" ";

    }
}

int main() {
    // conv1
    const int conv1InputSize = 28;
    const int convkernelSize = 5;
    const int conv1numkernel = 20; //numFilters
    const int conv1inputchannelnum = 1;
    const int conv1OutputSize = conv1InputSize - convkernelSize + 1;
    // pool1
    const int poolSize = 2;
    const int pool1OutputSize = conv1OutputSize / poolSize;
    //  conv2
    const int conv2inputSize = conv1OutputSize;
    const int conv2inputchannelnum = 20;
    const int conv2numkernel = 50; //numFilters
    const int conv2OutputSize = pool1OutputSize - convkernelSize + 1;
    // pool2
    const int pool2OutputSize = conv2OutputSize / poolSize;
    // fc1
    const int fc1InputSize = pool2OutputSize; //4
    const int fckernelSize = 4;
    const int fc1OutputSize = fc1InputSize - fckernelSize +1 ;
    const int fc1InputChannel = 50;
    const int fc1OutputChannel = 500;
    // fc2
    const int fc2InputSize = fc1OutputSize;
    const int fc2InputChannel = fc1OutputChannel;
    const int fc2OutputSize = fc2InputSize - fckernelSize +1 ;
    const int fc2OutputChannel = 10;

    // --------------------------read----------------------------
    string conv1file = "trained_weights/conv1.txt";
    string conv2file = "trained_weights/conv2.txt";
    string fc1file = "trained_weights/fc1.txt";
    string fc2file = "trained_weights/fc2.txt";  float *conv1Weights = (float*)malloc(20*5*5*1 * sizeof(float));
    float *conv1Bias = (float*)malloc(20*sizeof(float)); 
    float *conv1output = (float*)malloc(20*24*24 * sizeof(float));
    float *pool1output = (float*)malloc(20*12*12 * sizeof(float));
    float *conv2Weights = (float*)malloc(50*5*5*20 * sizeof(float));
    float *conv2Bias= (float*)malloc(50*sizeof(float)); 
    float *conv2output = (float*)malloc(50*8*8 * sizeof(float));
    float *pool2output = (float*)malloc(50*4*4 * sizeof(float));
    float *fc1Weights = (float*)malloc(500*4*4*50 * sizeof(float));
    float *fc1Bias = (float*)malloc(500*sizeof(float)); 
    float *fc1output = (float*)malloc(500 * sizeof(float));
    float *fc2Weights = (float*)malloc(10*1*1*500 * sizeof(float));
    float* fc2Bias= (float*)malloc(10*sizeof(float)); 
    float *fc2output = (float*)malloc(10 * sizeof(float));
    float *fc2softmaxoutput = (float*)malloc(10 * sizeof(float));
    
    memset(conv1Weights, 0, 20*5*5*1 * sizeof(float));
    memset(conv1Bias, 0, 20 * sizeof(float));
    memset(conv1output, 0, 20*24*24 * sizeof(float));
    memset(pool1output, 0, 20*12*12 * sizeof(float));
    memset(conv2Weights, 0, 50*5*5*20 * sizeof(float));
    memset(conv2Bias, 0, 50 * sizeof(float));
    memset(conv2output, 0, 50*8*8 * sizeof(float));
    memset(pool2output, 0, 50*4*4 * sizeof(float));
    memset(fc1Weights, 0, 500*4*4*50 * sizeof(float));
    memset(fc1Bias, 0, 500 * sizeof(float));
    memset(fc1output, 0, 500 * sizeof(float));
    memset(fc2Weights, 0, 10*1*1*500 * sizeof(float));
    memset(fc2Bias, 0, 10 * sizeof(float));
    memset(fc2output, 0, 10 * sizeof(float));
    memset(fc2softmaxoutput, 0, 10 * sizeof(float));

    readKernelWeightsAndBiasconv1(conv1file, &conv1Weights, &conv1Bias);
    readKernelWeightsAndBiasconv2(conv2file, &conv2Weights, &conv2Bias);
    readFC1(fc1file, &fc1Weights, &fc1Bias);
    readFC2(fc2file, &fc2Weights, &fc2Bias);

    // cout << "Conv1 weights read from file:" << endl;
    // for(int i=0; i<conv1numkernel; i++){
    //     cout << "Filter " << i + 1 << ":" << endl;
    //     int tmp = i*convkernelSize*convkernelSize;
    //     for(int j=0; j<convkernelSize; j++){
    //         for(int k=0; k<convkernelSize; k++){
    //             cout<< conv1Weights[tmp + j*convkernelSize + k ]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    // }

    // std::cout << "conv1 Bias Values:" << std::endl;
    // for (int i = 0; i < conv1numkernel; ++i) {
    //     std::cout << conv1Bias[i] << " ";
    // }

    // std::cout << std::endl;
    // cout << "Conv2 weights read from file:" << endl;
    // for (int i = 0; i < conv2numkernel; i++) {
    //     cout << "Filter " << i + 1 << ":" << endl;
    //     int tmp = i * convkernelSize * convkernelSize * 20; // Adjusting the indexing
    //     for (int depth = 0; depth < 20; depth++) {
    //         cout << "Depth " << depth + 1 << "     Filter "<< i+1 << ":" << endl;
    //         for (int j = 0; j < convkernelSize; j++) {
    //             for (int k = 0; k < convkernelSize; k++) {
    //                 cout << conv2Weights[tmp + depth * convkernelSize * convkernelSize + j * convkernelSize + k] << " ";
    //             }
    //             cout << "\n";
    //         }
    //     }
    // }

    //     std::cout << "conv2 Bias Values:" << std::endl;
    //     for (int i = 0; i < conv2numkernel; ++i) {
    //         std::cout << conv2Bias[i] << " ";
    //     }

    // cout << "fc1 weights read from file:" << endl;
    // for(int i=0; i<fc1OutputChannel; i++){
    //     cout << "Filter " << i + 1 << ":" << endl;
    //     int tmp = i*fckernelSize*fckernelSize;
    //     for(int j=0; j<fckernelSize; j++){
    //         for(int k=0; k<fckernelSize; k++){
    //             cout<< fc1Weights[tmp + j*fckernelSize + k ]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    // }
    // cout << "fc1 Bias Values:" << std::endl;
    // for (int i = 0; i < fc1OutputChannel; ++i) {
    //     std::cout << fc1Bias[i] << " ";
    // }

    // cout << "fc2 weights read from file:" << endl;
    // for(int i=0; i<10; i++){
    //     cout << "Filter " << i + 1 << ":" << endl;
    //     int tmp = i*500;
    //     for(int j=0; j<500; j++){
    //             cout<< fc2Weights[tmp + j ]<<" ";
    //     }
    //     cout<<"\n";
    // }
    // cout << "fc2 Bias Values:" << std::endl;
    // for (int i = 0; i < fc2OutputChannel; ++i) {
    //     std::cout << fc2Bias[i] << " ";
    // }

//     std::string testFolder = "testtext/";
//     // Number of test images
//     int numImages = 10;
//       for (int i = 0; i < numImages; ++i) 
// {
    string imageFile = "testtext/num7.txt";
    float* image = (float*)malloc(28*28*sizeof(float));
    readImage(imageFile, &image);
    cout << "Image read from file:" << endl;
    for(int i=0; i<28; i++){
        for(int j=0; j<28; j++){
            cout<< image[i*28 + j]<<" ";
        }
        cout<<"\n";
    }


    // ------------------conv1---------------------

    for(int i=0; i<conv1numkernel; i++){
        convolutionCUDA(image, conv1Weights + i*convkernelSize*convkernelSize, conv1output + i*24*24, conv1InputSize, convkernelSize);
        for(int j=0; j<24*24; j++){
            conv1output[i*24*24 + j] += conv1Bias[i];
        }
    }
    // print conv1 output
    // cout << "Conv1 output:" << endl;
    // for(int i=0; i<20; i++){
    //     cout << "Filter " << i + 1 << ":" << endl;
    //     for(int j=0; j<24; j++){
    //         for(int k=0; k<24; k++){
    //             cout<< conv1output[i*24*24 + j*24 + k]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    // }

    // ------------------pool1---------------------

    for(int i=0; i<conv1numkernel; i++){
        maxPoolingCUDA(conv1output + i*24*24, pool1output + i*12*12, 24, poolSize);
    }
    // print pool1 output
    // cout << "Pool1 output:" << endl;
    // for(int i=0; i<20; i++){
    //     cout << "Filter " << i + 1 << ":" << endl;
    //     for(int j=0; j<12; j++){
    //         for(int k=0; k<12; k++){
    //             cout<< pool1output[i*12*12 + j*12 + k]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    // }

    // ------------------conv2---------------------

    for(int i=0; i< 50; i++){
        float* tmp = (float*)malloc(8*8* sizeof(float));
for (int p = 0; p < 64; ++p) {
    tmp[p] = 0.0f;
}
        // cout<<"\nhere i = "<<i<<"\n";

        for(int j=0; j<20; j++){
            float* tmp2 = (float*)malloc(8*8* sizeof(float));

for (int p = 0; p < 64; ++p) {
    tmp2[p] = 0.0f;
}
            convolutionCUDA(pool1output + j*12*12, conv2Weights + i*5*5*20 + j*5*5, tmp2, 12, 5);
            // add tmp2 in tmp1
            for(int k=0; k<8*8; k++){
                tmp[k] += tmp2[k];
            }
            // free(tmp2);
        }
        for(int j=0; j<8*8; j++){
            tmp[j] += conv2Bias[i];
        }
        for(int j=0; j<8*8; j++){
            conv2output[i*8*8+j] = tmp[j];
        }
        // free(tmp);
    }
    
    // cout << "Conv2 output:" << endl;
    // for(int i=0; i<50; i++){
    //     cout << "Filter " << i + 1 << ":" << endl;
    //     for(int j=0; j<8; j++){
    //         for(int k=0; k<8; k++){
    //             cout<< conv2output[i*8*8 + j*8 + k]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    // }


    // ------------------pool2---------------------

    for(int i=0; i<conv2numkernel; i++){
        maxPoolingCUDA(conv2output + i*8*8, pool2output + i*4*4, 8, poolSize);
    }
    // print pool2 output
    // cout << "Pool2 output:" << endl;
    // for(int i=0; i<50; i++){
    //     cout << "Filter " << i + 1 << ":" << endl;
    //     for(int j=0; j<4; j++){
    //         for(int k=0; k<4; k++){
    //             cout<< pool2output[i*4*4 + j*4 + k]<<" ";
    //         }
    //         cout<<"\n";
    //     }
    // }
   

    // ------------------fc1---------------------

for(int i=0; i< 500; i++){
        float* tmp4 = (float*)malloc(1*1* sizeof(float));
for (int p = 0; p < 1; ++p) {
    tmp4[p] = 0.0f;
}
        // cout<<"\nhere i = "<<i<<"\n";

        for(int j=0; j<50; j++){
            float* tmp2 = (float*)malloc(1*1* sizeof(float));
for (int p = 0; p < 1; ++p) {
    tmp2[p] = 0.0f;
}
            convolutionCUDA(pool2output + j*4*4, fc1Weights + i*4*4*50 + j*4*4, tmp2, 4, 4);
            // add tmp2 in tmp1
            for(int k=0; k<1*1; k++){
                tmp4[k] += tmp2[k];
            }
            // free(tmp2);
        }
        
        for(int j=0; j<1*1; j++){
            tmp4[j] += fc1Bias[i];
        }
        for(int j=0; j<1*1; j++){
            fc1output[i*1*1+j] = tmp4[j];
        }
        // free(tmp);
    }

    // cout << "FC1 output:" << endl;
    // for(int i=0; i<fc1OutputChannel; i++){
    //     cout << fc1output[i] << " ";
    // }

 float *fc1reluoutput = (float*)malloc(500 * sizeof(float));
    reluCUDA(fc1output, fc1reluoutput, 1, 500);

    // cout << "\n";
    // cout << "FC1 relu output:" << endl;
    // for(int i=0; i<fc1OutputChannel; i++){
    //     cout << fc1reluoutput[i] << " ";
    // }
    // cout << "\n";


    // ------------------fc2---------------------

    for(int i=0; i< 10; i++){
        float* tmp3 = (float*)malloc(1*1* sizeof(float));
        for (int p = 0; p < 1; ++p) {
    tmp3[p] = 0.0f;
}
        // cout<<"\nhere i = "<<i<<"\n";
        for(int j=0; j<500; j++){
            float* tmp2 = (float*)malloc(1*1* sizeof(float));
            for (int p = 0; p < 1; ++p) {
    tmp2[p] = 0.0f;
}
            convolutionCUDA(fc1reluoutput + j*1*1, fc2Weights + i*1*1*500 + j*1*1, tmp2, 1,1);
// cout<<fc1output[j]<<" "<<fc2Weights[j]<<" "<<tmp2[0]<<"\n";
            // add tmp2 in tmp1
            for(int k=0; k<1*1; k++){
                tmp3[k] += tmp2[k];
            }
            // free(tmp2);
        }
        
        for(int j=0; j<1*1; j++){
            tmp3[j] += fc2Bias[i];
        }
        for(int j=0; j<1*1; j++){
            fc2output[i*1*1+j] = tmp3[j];
            // cout<< fc2output[i*1*1+j]<<" ";
        }
        // free(tmp);
    }

    // cout << "FC2 output:" << endl;
    // for(int i=0; i<10; i++){
    //     cout << fc2output[i] << " ";
    // }



    // ------------------softmax---------------------


    // softmax layer

    softmaxCUDA(fc2output, fc2softmaxoutput, 10);

    // cout << "\n";
    // cout << "FC2 softmax output:" << endl;
    // for(int i=0; i<10; i++){
    //     cout << fc2softmaxoutput[i] << " ";
    // }

    // ------------------prediction---------------------

    int maxIndex = 0;
    for(int i=0; i<10; i++){
        if(fc2softmaxoutput[i] > fc2softmaxoutput[maxIndex]){
            maxIndex = i;
        }
    }
    cout << "The number is: " << maxIndex << endl;


// }


    free(conv1output);
    free(conv2Weights);
    free(conv2Bias);
    free(pool1output);
    free(fc1Weights);
    free(fc1Bias);
    free(pool2output);
    free(fc1output);
    free(fc2Weights);
    free(fc2Bias);
    free(fc1reluoutput);
    free(fc2output);
    free(conv2output);
    free(fc2softmaxoutput);
    free(conv1Weights);
    free(conv1Bias);
    free(image);
    

    


    return 0;
}
