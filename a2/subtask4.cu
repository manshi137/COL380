#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <string>
#include <time.h>	
#include <dirent.h> // For directory traversalusing namespace std;
#include "read.c"
#define BLOCK_SIZE 32
using namespace std;

__global__ void conv2kernel(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize - kernelSize +1;

    if (col < outputSize && row < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                __shared__ float sharedInput[12][12];
                __shared__ float sharedKernel[5][5];
                sharedInput[i][j] = input[(row + i) * inputSize + (col + j)];
                sharedKernel[i][j] = kernel[i * kernelSize + j];
                __syncthreads();
                sum += sharedInput[i][j] * sharedKernel[i][j];
            }
        }
        __syncthreads();
        output[row * outputSize + col] = sum;
    }
}

void conv2(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    float *d_input, *d_kernel, *d_output;
    int outputSize = inputSize - kernelSize +1;
    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((inputSize + blockSize.x - 1) / blockSize.x, (inputSize + blockSize.y - 1) / blockSize.y);

    conv2kernel<<<gridSize, blockSize, 0>>>(d_input, d_kernel, d_output, inputSize, kernelSize);

    cudaMemcpy(output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}



__global__ void conv1kernel(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize - kernelSize +1;

    if (col < outputSize && row < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                __shared__ float sharedInput[28][28];
                __shared__ float sharedKernel[5][5];
                sharedInput[i][j] = input[(row + i) * inputSize + (col + j)];
                sharedKernel[i][j] = kernel[i * kernelSize + j];
                __syncthreads();
                sum += sharedInput[i][j] * sharedKernel[i][j];
            }
        }
        __syncthreads();
        output[row * outputSize + col] = sum;
    }
}

void conv1(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    float *d_input, *d_kernel, *d_output;
    int outputSize = inputSize - kernelSize +1;
    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((inputSize + blockSize.x - 1) / blockSize.x, (inputSize + blockSize.y - 1) / blockSize.y);

    conv1kernel<<<gridSize, blockSize, 0>>>(d_input, d_kernel, d_output, inputSize, kernelSize);

    cudaMemcpy(output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}



__global__ void fc1kernel(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize - kernelSize +1;

    if (col < outputSize && row < outputSize) {
        float sum = 0.0f;
        __shared__ float sharedInput[4][4];
        __shared__ float sharedKernel[4][4];
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                sharedInput[i][j] = input[(row + i) * inputSize + (col + j)];
                sharedKernel[i][j] = kernel[i * kernelSize + j];
                __syncthreads();
                sum += sharedInput[i][j] * sharedKernel[i][j];
            }
        }
        __syncthreads();
        output[row * outputSize + col] = sum;
    }
}

void fc1(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    float *d_input, *d_kernel, *d_output;
    int outputSize = inputSize - kernelSize +1;
    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((inputSize + blockSize.x - 1) / blockSize.x, (inputSize + blockSize.y - 1) / blockSize.y);

    fc1kernel<<<gridSize, blockSize, 0 >>>(d_input, d_kernel, d_output, inputSize, kernelSize);

    cudaMemcpy(output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
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

void convolutionCUDA(float *input, float *kernel, float *output, int inputSize, int kernelSize, cudaStream_t* stream) {
    float *d_input, *d_kernel, *d_output;
    int outputSize = inputSize - kernelSize +1;
    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((inputSize + blockSize.x - 1) / blockSize.x, (inputSize + blockSize.y - 1) / blockSize.y);

    convolutionKernel<<<gridSize, blockSize, 0, *stream>>>(d_input, d_kernel, d_output, inputSize, kernelSize);

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
    clock_t start, stop;	
    start = clock();
    int correct = 0;
    int total = 0;	
    // conv1
    const int conv1InputSize = 28;
    const int convkernelSize = 5;
    const int conv1numkernel = 20; //numFilters
    // const int conv1inputchannelnum = 1;
    const int conv1OutputSize = conv1InputSize - convkernelSize + 1;
    // pool1
    const int poolSize = 2;
    const int pool1OutputSize = conv1OutputSize / poolSize;
    //  conv2
    // const int conv2inputSize = conv1OutputSize;
    // const int conv2inputchannelnum = 20;
    const int conv2numkernel = 50; //numFilters
    const int conv2OutputSize = pool1OutputSize - convkernelSize + 1;
    // pool2
    const int pool2OutputSize = conv2OutputSize / poolSize;
    // fc1
    const int fc1InputSize = pool2OutputSize; //4
    // const int fckernelSize = 4;
    // const int fc1OutputSize = fc1InputSize - fckernelSize +1 ;
    // const int fc1InputChannel = 50;
    // const int fc1OutputChannel = 500;
    // fc2
    // const int fc2InputSize = fc1OutputSize;
    // const int fc2InputChannel = fc1OutputChannel;
    // const int fc2OutputSize = fc2InputSize - fckernelSize +1 ;
    // const int fc2OutputChannel = 10;

    // ----------------------------------malloc---------------------------
    float *conv1Weights = (float*)malloc(20*5*5*1 * sizeof(float));
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

    // --------------------------read----------------------------
    string conv1file = "trained_weights/conv1.txt";
    string conv2file = "trained_weights/conv2.txt";
    string fc1file = "trained_weights/fc1.txt";
    string fc2file = "trained_weights/fc2.txt";  
    readKernelWeightsAndBiasconv1(conv1file, &conv1Weights, &conv1Bias);
    readKernelWeightsAndBiasconv2(conv2file, &conv2Weights, &conv2Bias);
    readFC1(fc1file, &fc1Weights, &fc1Bias);
    readFC2(fc2file, &fc2Weights, &fc2Bias);

    std::string folderPath = "testtext2";
    std::vector<std::string> filePaths;

    DIR* dir = opendir(folderPath.c_str());
    if (dir == NULL) {
        std::cerr << "Error opening directory" << std::endl;
        return 1;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string fileName = entry->d_name;

        if (fileName != "." && fileName != "..") {
            std::string filePath = folderPath + "/" + fileName;
            filePaths.push_back(filePath);
        }
    }

    closedir(dir);

    for (int x=0;x<filePaths.size();x++) {
        clock_t start1, stop1;	
        start1 = clock();
        total = total+1;
        float* image = (float*)malloc(28 * 28 * sizeof(float));
        cout << "Processing file: " << filePaths[x] << "--------------------------yy-----------------------------" << endl;
        readImage(filePaths[x], &image);
    
    // }
    

    cudaStream_t streams[500];
    // ------------------conv1---------------------
    for(int i=0; i<20; i++){
        cudaStreamCreate(&streams[i]);
    }
    
    cout<<"conv1\n";
    for(int i=0; i<conv1numkernel; i++){
        convolutionCUDA(image, conv1Weights + i*convkernelSize*convkernelSize, conv1output + i*24*24, conv1InputSize, convkernelSize, &streams[i]);
        // conv1(image, conv1Weights + i*convkernelSize*convkernelSize, conv1output + i*24*24, conv1InputSize, convkernelSize);
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

    cout<<"pool1\n";
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
    cout<<"conv2\n";
    for(int i=0; i<50; i++){
        cudaStreamCreate(&streams[i]);
    }   
    float* tmp = (float*)malloc(8*8* sizeof(float));
    float* tmp2 = (float*)malloc(8*8* sizeof(float));
    for(int i=0; i< 50; i++){
        for (int p = 0; p < 64; ++p) {
            tmp[p] = 0.0f;
        }

        for(int j=0; j<20; j++){

            for (int p = 0; p < 64; ++p) {
                tmp2[p] = 0.0f;
            }

            convolutionCUDA(pool1output + j*12*12, conv2Weights + i*5*5*20 + j*5*5, tmp2, 12, 5, &streams[i]);
            // conv2(pool1output + j*12*12, conv2Weights + i*5*5*20 + j*5*5, tmp2, 12, 5);
            // add tmp2 in tmp1
            for(int k=0; k<8*8; k++){
                tmp[k] += tmp2[k];
            }
        }
        for(int j=0; j<8*8; j++){
            tmp[j] += conv2Bias[i];
        }
        for(int j=0; j<8*8; j++){
            conv2output[i*8*8+j] = tmp[j];
        }
    }
    free(tmp2);
    free(tmp);
    
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
    cout<<"pool2\n";
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
    cout<<"fc1\n";
    // for (int i = 0; i < 500; i++) {
    //     cudaStreamCreate(&streams[i]);
    // }
    float* tmp4 = (float*)malloc(1*1* sizeof(float));
    float* tmp5 = (float*)malloc(1*1* sizeof(float));
    for(int i=0; i< 500; i++){
            // for (int p = 0; p < 1; ++p) {
            //     tmp4[p] = 0.0f;
            // }
            tmp4[0] = 0.0f;
            for(int j=0; j<50; j++){
                for (int p = 0; p < 1; ++p) {
                    tmp5[p] = 0.0f;
                }
                // convolutionCUDA(pool2output + j*4*4, fc1Weights + i*4*4*50 + j*4*4, tmp5, 4, 4);
                fc1(pool2output + j*4*4, fc1Weights + i*4*4*50 + j*4*4, tmp5, 4, 4); //correct
                // add tmp5 in tmp1
                for(int k=0; k<1*1; k++){
                    tmp4[k] += tmp5[k];
                }
            }
            for(int j=0; j<1*1; j++){
                tmp4[j] += fc1Bias[i];
            }
            for(int j=0; j<1*1; j++){
                fc1output[i*1*1+j] = tmp4[j];
            }
        }
    free(tmp5);
    free(tmp4);

    // cout << "FC1 output:" << endl;
    // for(int i=0; i<fc1OutputChannel; i++){
    //     cout << fc1output[i] << " ";
    // }
    cout    << "relu\n";
    float *fc1reluoutput = (float*)malloc(500 * sizeof(float));
    reluCUDA(fc1output, fc1reluoutput, 1, 500);

    // cout << "\n";
    // cout << "FC1 relu output:" << endl;
    // for(int i=0; i<fc1OutputChannel; i++){
    //     cout << fc1reluoutput[i] << " ";
    // }
    // cout << "\n";


    // ------------------fc2---------------------
    float* tmp3 = (float*)malloc(1*1* sizeof(float));
    float* tmp6 = (float*)malloc(1*1* sizeof(float));
    cout<<"fc2\n";
    for(int i=0; i< 10; i++){
        // for (int p = 0; p < 1; ++p) {
        //     tmp3[p] = 0.0f;
        // }
        tmp3[0] = 0.0f;
        // cout<<"\nhere i = "<<i<<"\n";
        for(int j=0; j<500; j++){
            for (int p = 0; p < 1; ++p) {
                tmp6[p] = 0.0f;
            }
            // convolutionCUDA(fc1reluoutput + j*1*1, fc2Weights + i*1*1*500 + j*1*1, tmp6, 1,1);
            tmp6[0]= fc1reluoutput[j]*fc2Weights[i*1*1*500 + j*1*1];
            // add tmp6 in tmp1
            for(int k=0; k<1*1; k++){
                tmp3[k] += tmp6[k];
            }
        }
        
        for(int j=0; j<1*1; j++){
            tmp3[j] += fc2Bias[i];
        }
        for(int j=0; j<1*1; j++){
            fc2output[i*1*1+j] = tmp3[j];
            // cout<< fc2output[i*1*1+j]<<" ";
        }
    }
    free(tmp6);
    free(tmp3);

    // cout << "FC2 output:" << endl;
    // for(int i=0; i<10; i++){
    //     cout << fc2output[i] << " ";
    // }

    // ------------------softmax---------------------
    cout<<"softmax\n";
    softmaxCUDA(fc2output, fc2softmaxoutput, 10);

    // cout << "\n";
    cout << "FC2 softmax output:" << endl;
    for(int i=0; i<10; i++){
        cout << fc2softmaxoutput[i] << " ";
    }

    // ------------------prediction---------------------

    int maxIndex = 0;
    for(int i=0; i<10; i++){
        if(fc2softmaxoutput[i] > fc2softmaxoutput[maxIndex]){
            maxIndex = i;
        }
    }
    cout << "\n";
    cout << "The number is: " << maxIndex << endl;
    cout << filePaths[x] << endl;
    cout << "The number should be: " << filePaths[x][filePaths[x].size() - 5]  << endl;
    if(maxIndex == (int)(filePaths[x][filePaths[x].size() - 5]  - '0')){
        correct++;
    }
    stop1 = clock();
    float time_taken1 = (float) (stop1-start1)/(CLOCKS_PER_SEC) ;
    cout << "Time taken: " << time_taken1 << " seconds" << endl;
    cout << "correct: " << correct << " total: " << total << endl;
    free(image);
}
    stop = clock();
    float time_taken = (float) (stop-start)/(CLOCKS_PER_SEC) ;
    cout << "Total Time taken: " << time_taken << " seconds" << endl;
    cout << "Accuracy: " << (float)correct/total*100 << "%" << endl;


    return 0;
}
