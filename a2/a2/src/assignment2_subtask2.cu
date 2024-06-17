#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#define BLOCK_SIZE 16

__global__ void convolutionKernel(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    // printf("uvvwykvhyvkhwg");
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
        // printf("%d",sum);
        output[row * outputSize + col] = sum;
    }
}

void convolutionCUDA(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    // std::cout<<"hjhj"<<inputSize<<" "<<kernelSize;
    float *d_input, *d_kernel, *d_output;
    int outputSize = inputSize - kernelSize +1;
    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((inputSize + blockSize.x - 1) / blockSize.x, (inputSize + blockSize.y - 1) / blockSize.y);
// std::cout<<"here"<<gridSize<<blockSize;
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, inputSize, kernelSize);
// printf("there");
    cudaMemcpy(output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
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

__global__ void tanhKernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = tanh(input[index]);
    }
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

__global__ void avgPoolingKernel(float *input, float *output, int inputSize, int poolSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputSize && col < inputSize) {
        int outputRow = row / poolSize;
        int outputCol = col / poolSize;

        int inputIdx = row * inputSize + col;
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


__global__ void sigmoidKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = 1.0f / (1.0f + std::exp(-(input[idx])));
        // output[idx] = expf(input[idx] - maxVal);
    }
}

void sigmoidCUDA(float* input, float* output, int inputSize) {
    int size = inputSize;
    // std::vector<float> result(size);
    // float maxVal = input[0];
    // for (size_t i = 1; i < inputSize; ++i) {
    //     if (input[i] > maxVal) {
    //         maxVal = input[i];
    //     }
    // }
    // float sumExp = 0.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    sigmoidKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, size);

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    // for (int i = 0; i < size; ++i) {
    //     sumExp += output[i];
    // }

    // for (int i = 0; i < size; ++i) {
    //     output[i] /= sumExp;
    // }
}

int stringToInt(const char* str) {
    int result = 0;
    int sign = 1;
    if (*str == '-') {
        sign = -1;
        ++str;
    }
    while (*str) {
        result = result * 10 + (*str - '0');
        ++str;
    }
    return sign * result;
}

float stringToFloat(const char* str) {
    float result = 0.0f;
    float sign = 1.0f;
    if (*str == '-') {
        sign = -1.0f;
        ++str;
    }
    while (*str && *str != '.') {
        result = result * 10.0f + (*str - '0');
        ++str;
    }
    if (*str == '.') {
        float factor = 0.1f;
        ++str;
        while (*str) {
            result = result + (*str - '0') * factor;
            factor *= 0.1f;
            ++str;
        }
    }
    return sign * result;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [task of choice 1=convolution, 2=non-linear-activations, 3=subsampling, 4=converting a vector]" << std::endl;
        return 1;
    }
    
    int task = stringToInt(argv[1]);
    // std::cout<<"task "<<task<<"endl\n";
    if (task == 1) {

    const int inputSize = stringToInt(argv[2]);
        const int rows = inputSize;
    const int cols = inputSize;
    int kernelSize = stringToInt(argv[3]);
    const int outputsize= inputSize-kernelSize+1;
    const int N=inputSize;
const int M=kernelSize;
    float *input;
    float *kernel;
    float *output;
    size_t bytes = (rows*cols)*sizeof(float);
    size_t bytesoutput = (outputsize*outputsize)*sizeof(float);
    size_t byteskernel = (kernelSize*kernelSize)*sizeof(float);
    // printf("%d",bytes);


    input = (float*)malloc(bytes);
    kernel = (float*)malloc(byteskernel);
    output= (float*)malloc(bytesoutput);
        int P = stringToInt(argv[4]);

        float val;
        int count = 0;
        while (count!=N*N) {
            
            val=stringToFloat(argv[count+5]);
            input[count]= val;
            ++count;
        }
// std::cout<<count;
        while (count<N*N+M*M) {
            val=stringToFloat(argv[count+5]);
            kernel[count-N*N] = val;
            ++count;
        }

// Print input array
    // std::cout << ("Input:\n");
    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         std::cout << ("%.2f ", input[i * cols + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

    // Print kernel array
    // std::cout << ("\nKernel:\n");
    // for (int i = 0; i < kernelSize; ++i) {
    //     for (int j = 0; j < kernelSize; ++j) {
    //         std::cout << ("%.2f ", kernel[i * kernelSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

    // // Print output array
    // std::cout << ("\nOutput:\n");
    // for (int i = 0; i < outputsize; ++i) {
    //     for (int j = 0; j < outputsize; ++j) {
    //         std::cout << ("%.2f ", output[i * (cols - 1) + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

        if (P==0){
// std::cout<<inputSize<<kernelSize;
    convolutionCUDA(input, kernel, output, inputSize, kernelSize);



    // std::cout << "Convolution with no padding result:" << std::endl;
    for (int i = 0; i < outputsize; ++i) {
        for (int j = 0; j < outputsize; ++j) {
            std::cout << output[i * outputsize + j] << " ";
        }
        // std::cout << std::endl;
    }

        }
        else{
  int paddedSize = inputSize + kernelSize-1;
    float *input2;
    float *output2;
    input2 = (float*)malloc(paddedSize*paddedSize*sizeof(float));
    output2 = (float*)malloc(inputSize*inputSize*sizeof(float));

int inputind=0;
    for(int i = 0; i < paddedSize * paddedSize; ++i) {
        
        input2[i] = 0.0f;
        int x = i/paddedSize;
        int y = i%paddedSize;
        if(x>=kernelSize/2 && x< inputSize+ kernelSize/2 && y>=kernelSize/2 && y< inputSize+ kernelSize/2){
            // std::cout <<x<<" "<<y<<"\n";
            input2[i] = input[inputind];
            inputind=inputind+1;
        }
    }
    
    for(int i = 0; i < inputSize * inputSize; ++i) {
        output2[i] = 0.0f;
    }

    // std::cout << ("\nInput2:\n");
    // for (int i = 0; i < paddedSize; ++i) {
    //     for (int j = 0; j < paddedSize; ++j) {
    //         std::cout << ("%.2f ", input2[i * paddedSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

    // std::cout << ("\nKernel:\n");
    // for (int i = 0; i < kernelSize; ++i) {
    //     for (int j = 0; j < kernelSize; ++j) {
    //         std::cout << ("%.2f ", kernel[i * kernelSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

    // std::cout << ("\nOutput2:\n");
    // for (int i = 0; i < inputSize; ++i) {
    //     for (int j = 0; j < inputSize; ++j) {
    //         std::cout << ("%.2f ", output2[i * inputSize + j]) <<" ";
    //     }
        // std::cout << ("\n");
    // }

    convolutionCUDA(input2, kernel, output2, paddedSize, kernelSize);

    // std::cout << "Convolution after padding result:" << std::endl;
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            std::cout << ("%.2f ", output2[i * inputSize + j]) <<" ";
        }
        // std::cout << std::endl;
    }
        }
    }

    else if (task == 2) {
        int choice=stringToInt(argv[2]);
        int N = stringToInt(argv[3]); 
        int M = stringToInt(argv[4]);
    size_t bytes = (N*M)*sizeof(float);
            float *input;
            input = (float*)malloc(bytes);

        float val;
        int count = 0;
        while (count!=N*M) {
            val=stringToFloat(argv[count+5]);
            input[count]= val;
            ++count;
        }
 if(choice==0){

    float* outputRelu;
    outputRelu = (float*)malloc(N*M*sizeof(float));

    
    for(int i = 0; i < N * M; ++i) {
        outputRelu[i] = 0.0f;
    }
    reluCUDA(input, outputRelu, N, M);

    // std::cout << ("\nOutput relu:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << ("%.2f ", outputRelu[i * N + j]) <<" ";
        }
        // std::cout << ("\n");
    }

 }
 else{

    float* outputTanh;
    outputTanh = (float*)malloc(N*M*sizeof(float));

    
    for(int i = 0; i < N * M; ++i) {
        outputTanh[i] = 0.0f;
    }
    tanhCUDA(input, outputTanh, N, M);

    // std::cout << ("\nOutput tanh:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << ("%.2f ", outputTanh[i * N + j]) <<" ";
        }
        // std::cout << ("\n");
    }

 }
    }


    else if (task == 3) {

        int choice=stringToInt(argv[2]);
        int N = stringToInt(argv[3]); 
        // std::cout<<choice;
    size_t bytes = (N*N)*sizeof(float);
            float *input;
            input = (float*)malloc(bytes);

        float val;
        int count = 0;
        while (count!=N*N) {
            val=stringToFloat(argv[count+4]);
            input[count]= val;
            ++count;
        }
        if(choice==0){
            int poolSize = 2;
    float *outpoolMaxPool;
    int outputMaxSize = N/poolSize;
    outpoolMaxPool = (float*)malloc(outputMaxSize*outputMaxSize*sizeof(float));

    maxPoolingCUDA(input, outpoolMaxPool, N, poolSize);
    
    // std::cout << ("\nOutput maxpool:\n");
    for (int i = 0; i < outputMaxSize; ++i) {
        for (int j = 0; j < outputMaxSize; ++j) {
            std::cout << ("%.2f ", outpoolMaxPool[i * outputMaxSize + j]) <<" ";
        }
        // std::cout << ("\n");
    }
        }
        else{
              int poolSize = 2;
    float *outpoolMaxPool;
    int outputMaxSize = N/poolSize;
    outpoolMaxPool = (float*)malloc(outputMaxSize*outputMaxSize*sizeof(float));

    avgPoolingCUDA(input, outpoolMaxPool, N, poolSize);
    
    // std::cout << ("\nOutput maxpool:\n");
    for (int i = 0; i < outputMaxSize; ++i) {
        for (int j = 0; j < outputMaxSize; ++j) {
            std::cout << ("%.2f ", outpoolMaxPool[i * outputMaxSize + j]) <<" ";
        }
        // std::cout << ("\n");
    }  

    }
    }


    else if (task == 4) {

        int choice=stringToInt(argv[2]);
        int N = stringToInt(argv[3]); 
        // std::cout<<choice;
    size_t bytes = (N)*sizeof(float);
            float *input;
            input = (float*)malloc(bytes);

        float val;
        int count = 0;
        while (count!=N) {
            val=stringToFloat(argv[count+4]);
            input[count]= val;
            ++count;
        }

        if (choice==0){
    float *outputsoft;
    size_t bytessoft = (N)*sizeof(float);
    outputsoft= (float*)malloc(bytessoft);


    sigmoidCUDA(input, outputsoft, N);

    // std::cout << ("\nOutput softmax:\n");
    for (int i = 0; i < N; ++i) {
            std::cout << ("%.2f ", outputsoft[i]) <<" ";
    }
        }else{

    float *outputsoft;
    size_t bytessoft = (N)*sizeof(float);
    outputsoft= (float*)malloc(bytessoft);


    softmaxCUDA(input, outputsoft, N);

    // std::cout << ("\nOutput sigmoid:\n");
    for (int i = 0; i < N; ++i) {
            std::cout << ("%.2f ", outputsoft[i]) <<" ";
    }


        }
    

    } else {
        std::cout << "Invalid task number. Please provide a valid task number." << std::endl;
        return 1;
    }

    return 0;
}


// int main() {
//     // Example usage
//     const int rows = 6;
//     const int cols = 6;
//     const int inputSize = rows;
//     int kernelSize = 2;
//     const int outputsize= inputSize-kernelSize+1;
    

//     float *input;
//     float *kernel;
//     float *output;
//     size_t bytes = (rows*cols)*sizeof(float);
//     size_t bytesoutput = (outputsize*outputsize)*sizeof(float);
//     size_t byteskernel = (kernelSize*kernelSize)*sizeof(float);
//     input = (float*)malloc(bytes);
//     kernel = (float*)malloc(byteskernel);
//     output= (float*)malloc(bytesoutput);

//     float value = -9.0f;
//     for(int i = 0; i < rows * cols; ++i) {
//         input[i] = value;
//         value += 1.0f;
//     }
//     value=1.0f;
//     for(int i = 0; i < kernelSize * kernelSize; ++i) {
//         kernel[i] = value;
//         // value += 1.0f;
//     }
//     value=0.0f;
//     for(int i = 0; i < (rows-1) * (cols-1); ++i) {
//         output[i] = value;
//     }
    // // Print input array
    // std::cout << ("Input:\n");
    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         std::cout << ("%.2f ", input[i * cols + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

    // // Print kernel array
    // std::cout << ("\nKernel:\n");
    // for (int i = 0; i < kernelSize; ++i) {
    //     for (int j = 0; j < kernelSize; ++j) {
    //         std::cout << ("%.2f ", kernel[i * kernelSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

    // // Print output array
    // std::cout << ("\nOutput:\n");
    // for (int i = 0; i < outputsize; ++i) {
    //     for (int j = 0; j < outputsize; ++j) {
    //         std::cout << ("%.2f ", output[i * (cols - 1) + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }


//     convolutionCUDA(input, kernel, output, inputSize, kernelSize);



//     std::cout << "Convolution with no padding result:" << std::endl;
//     for (int i = 0; i < outputsize; ++i) {
//         for (int j = 0; j < outputsize; ++j) {
//             std::cout << output[i * outputsize + j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     // =====================================================convulution with padding=============================================
//       // Print input2 array
    // int paddedSize = inputSize + kernelSize-1;
    // float *input2;
    // float *output2;
    // input2 = (float*)malloc(paddedSize*paddedSize*sizeof(float));
    // output2 = (float*)malloc(inputSize*inputSize*sizeof(float));
    // value = 1.0f;
    // for(int i = 0; i < paddedSize * paddedSize; ++i) {
    //     input2[i] = 0.0f;
    //     int x = i/paddedSize;
    //     int y = i%paddedSize;
    //     if(x>=kernelSize/2 && x< inputSize+ kernelSize/2 && y>=kernelSize/2 && y< inputSize+ kernelSize/2){
    //         // std::cout <<x<<" "<<y<<"\n";
    //         input2[i] = value;
    //         value = value+1.0f;
    //     }
    // }
    
    // for(int i = 0; i < inputSize * inputSize; ++i) {
    //     output2[i] = 0.0f;
    // }

    // std::cout << ("\nInput2:\n");
    // for (int i = 0; i < paddedSize; ++i) {
    //     for (int j = 0; j < paddedSize; ++j) {
    //         std::cout << ("%.2f ", input2[i * paddedSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }
    // std::cout << ("\nKernel:\n");
    // for (int i = 0; i < kernelSize; ++i) {
    //     for (int j = 0; j < kernelSize; ++j) {
    //         std::cout << ("%.2f ", kernel[i * kernelSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }
    // std::cout << ("\nOutput2:\n");
    // for (int i = 0; i < inputSize; ++i) {
    //     for (int j = 0; j < inputSize; ++j) {
    //         std::cout << ("%.2f ", output2[i * inputSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

    // convolutionCUDA(input2, kernel, output2, paddedSize, kernelSize);

    // std::cout << "Convolution after padding result:" << std::endl;
    // for (int i = 0; i < inputSize; ++i) {
    //     for (int j = 0; j < inputSize; ++j) {
    //         std::cout << ("%.2f ", output2[i * inputSize + j]) <<" ";
    //     }
    //     std::cout << std::endl;
    // }

// // ============================================================relu========================================================

    // float* outputRelu;
    // outputRelu = (float*)malloc(inputSize*inputSize*sizeof(float));

    
    // for(int i = 0; i < inputSize * inputSize; ++i) {
    //     outputRelu[i] = 0.0f;
    // }
    // reluCUDA(input, outputRelu, inputSize, inputSize);

    // std::cout << ("\nOutput relu:\n");
    // for (int i = 0; i < inputSize; ++i) {
    //     for (int j = 0; j < inputSize; ++j) {
    //         std::cout << ("%.2f ", outputRelu[i * inputSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

// // =====================================================tanh=================================================================
    
    // float* outputTanh;
    // outputTanh = (float*)malloc(inputSize*inputSize*sizeof(float));

    
    // for(int i = 0; i < inputSize * inputSize; ++i) {
    //     outputTanh[i] = 0.0f;
    // }
    // tanhCUDA(input, outputTanh, inputSize, inputSize);

    // std::cout << ("\nOutput tanh:\n");
    // for (int i = 0; i < inputSize; ++i) {
    //     for (int j = 0; j < inputSize; ++j) {
    //         std::cout << ("%.2f ", outputTanh[i * inputSize + j]) <<" ";
    //     }
    //     std::cout << ("\n");
    // }

//     // =================================================maxpool=======================================================

//     int poolSize = 2;
//     float *outpoolMaxPool;
//     int outputMaxSize = inputSize/poolSize;
//     outpoolMaxPool = (float*)malloc(outputMaxSize*outputMaxSize*sizeof(float));

//     maxPoolingCUDA(input, outpoolMaxPool, inputSize, poolSize);
    
//     // Print input array
//     std::cout << ("Input:\n");
//     for (int i = 0; i < inputSize; ++i) {
//         for (int j = 0; j < inputSize; ++j) {
//             std::cout << ("%.2f ", input[i * inputSize + j]) <<" ";
//         }
//         std::cout << ("\n");
//     }
//     std::cout << ("\nOutput maxpool:\n");
//     for (int i = 0; i < outputMaxSize; ++i) {
//         for (int j = 0; j < outputMaxSize; ++j) {
//             std::cout << ("%.2f ", outpoolMaxPool[i * outputMaxSize + j]) <<" ";
//         }
//         std::cout << ("\n");
//     }
    
//     // =================================================avgpool=======================================================


// //    int poolSize = 2;
//     float *outpoolAvgPool;
//     int outputAvgSize = inputSize/poolSize;
//     outpoolAvgPool = (float*)malloc(outputAvgSize*outputAvgSize*sizeof(float));

//     avgPoolingCUDA(input, outpoolAvgPool, inputSize, poolSize);
    
//     // Print input array
//     std::cout << ("Input:\n");
//     for (int i = 0; i < inputSize; ++i) {
//         for (int j = 0; j < inputSize; ++j) {
//             std::cout << ("%.2f ", input[i * inputSize + j]) <<" ";
//         }
//         std::cout << ("\n");
//     }
//     std::cout << ("\nOutput avgpool:\n");
//     for (int i = 0; i < outputAvgSize; ++i) {
//         for (int j = 0; j < outputAvgSize; ++j) {
//             std::cout << ("%.2f ", outpoolAvgPool[i * outputAvgSize + j]) <<" ";
//         }
//         std::cout << ("\n");
//     }

//     // =================================================softmax=======================================================


//     float *inputsoft;
//     float *outputsoft;
//     size_t bytessoft = (inputSize)*sizeof(float);
//     inputsoft = (float*)malloc(bytessoft);
//     outputsoft= (float*)malloc(bytessoft);


//     value = 1.0f;
//     for(int i = 0; i < inputSize; ++i) {
//         inputsoft[i] = value;
//         value += 1.0f;
//     }
//     std::cout << ("\nOutput softmax:\n");
//     for (int i = 0; i < inputSize; ++i) {
//             std::cout << ("%.2f ", inputsoft[i]) <<" ";
//     }


//     softmaxCUDA(inputsoft, outputsoft, inputSize);

//     std::cout << ("\nOutput softmax:\n");
//     for (int i = 0; i < inputSize; ++i) {
//             std::cout << ("%.2f ", outputsoft[i]) <<" ";
//     }


//     // std::cout << "Softmax Output:" << std::endl;
//     // for (float val : softmaxOutput) {
//     //     std::cout << val << " ";
//     // }
//     // std::cout << std::endl;



//     return 0;
// }
