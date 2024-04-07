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

__global__ void computeFC1Output(float* d_fc1Output, float* d_tmp6, float* bias) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 500) {
        d_fc1Output[tid] = 0.0f;
        for (int j = 0; j < 50; ++j) {
            d_fc1Output[tid] += d_tmp6[tid * 50 + j];
        }
        d_fc1Output[tid]+=bias[tid];
    }
}

__global__ void fc1kernel(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize - kernelSize +1;
    float sum = 0.0f;
    int i,j;
    if (col < outputSize && row < outputSize) {
        sum = 0.0f;
        // __shared__ float sharedInput[4][4];
        // __shared__ float sharedKernel[4][4];
        for (i = 0; i < kernelSize; ++i) {
            for (j = 0; j < kernelSize; ++j) {
                // sharedInput[i][j] = input[(row + i) * inputSize + (col + j)];
                // sharedKernel[i][j] = kernel[i * kernelSize + j];
                // __syncthreads();
                // sum += sharedInput[i][j] * sharedKernel[i][j];
                sum += input[(row + i) * inputSize + (col + j)] * kernel[i * kernelSize + j];
            }
        }
        // __syncthreads();
        output[row * outputSize + col] = sum;
    }
}



__global__ void conv2kernel_again(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int outputSize = inputSize - kernelSize +1;
    float sum = 0.0f;
    int i,j;
    if (col < outputSize && row < outputSize) {
        sum = 0.0f;
        // __shared__ float sharedInput[4][4];
        // __shared__ float sharedKernel[4][4];
        for (i = 0; i < kernelSize; ++i) {
            for (j = 0; j < kernelSize; ++j) {
                // sharedInput[i][j] = input[(row + i) * inputSize + (col + j)];
                // sharedKernel[i][j] = kernel[i * kernelSize + j];
                // __syncthreads();
                // sum += sharedInput[i][j] * sharedKernel[i][j];
                sum += input[(row + i) * inputSize + (col + j)] * kernel[i * kernelSize + j];
            }
        }
        // __syncthreads();
        output[row * outputSize + col] = sum;
    }
}

__global__ void computeConv2Output(float* d_conv2Output, float* d_tmp7, float *bias) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 50) {
        for(int i=0; i<64; i++){
            d_conv2Output[tid*64+i] = 0.0f;
            for (int j = 0; j < 20; ++j) {
                d_conv2Output[tid*64+i] += d_tmp7[tid * 20*64 + j*64+ i];
            }
            d_conv2Output[tid*64 +i]+= bias[tid];
        }
        // d_conv2Output[tid]+= bias[tid];
    }
} 


__global__ void convolution1(float *input, float *kernel, float *output, int inputSize, int kernelSize, float bias) {
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
        output[row * outputSize + col] = sum + bias;
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
                    // cout<<input[inputIdx]<<" "<<maxVal;
                    maxVal = fmaxf(maxVal, input[inputIdx]);
                }
            }
        }
        output[outputIdx] = maxVal;
    }
}


void maxPoolingCUDA(float* d_input, float* d_output, int inputSize, int poolSize) {
    int outputSize = inputSize / poolSize;

    // float *d_output;
    // cudaMalloc(&d_output, outputSize * outputSize * sizeof(float));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    maxPoolingKernel<<<gridSize, blockSize>>>(d_input, d_output, inputSize, poolSize);

    // cudaMemcpy(output, d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaFree(d_output);
}

__global__ void reluKernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        output[index] = fmaxf(0.0f, input[index]);
    }
}

void reluCUDA(float *d_input, float *output, int rows, int cols) {
    float  *d_output;

    // cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    reluKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // cudaFree(d_input);
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

    float* d_conv1Weights;
    cudaMalloc(&d_conv1Weights,  20*5*5*1 * sizeof(float));
    float* image = (float*)malloc(28 * 28 * sizeof(float));
    float* d_image;
    cudaMalloc(&d_image, conv1InputSize * conv1InputSize * sizeof(float));
    float* d_conv1output;
    cudaMalloc(&d_conv1output, 20*24*24* sizeof(float));
    float* d_conv2Weights;
    cudaMalloc(&d_conv2Weights,  50*5*5*20 * sizeof(float));
    float* d_pool1output;
    cudaMalloc(&d_pool1output,  20*12*12 * sizeof(float));
    float* tmp = (float*)malloc(8*8* sizeof(float));
    float* tmp2 = (float*)malloc(8*8* sizeof(float));
    float* d_conv2output;
    cudaMalloc(&d_conv2output,  50*8*8 * sizeof(float));
    float* d_fc1Weights;
    cudaMalloc(&d_fc1Weights, 500*4*4*50 * sizeof(float));
    float* d_conv2bias;
    cudaMalloc(&d_conv2bias, 50 * sizeof(float));
    float* d_fc1bias;
    cudaMalloc(&d_fc1bias, 500 * sizeof(float));
    float* d_pool2output;
    cudaMalloc(&d_pool2output,  50*4*4 * sizeof(float));
    float* tmp4 = (float*)malloc(1*1* sizeof(float));
    float* tmp6 = (float*)malloc(500*50*1*1* sizeof(float));
    float *fc1reluoutput = (float*)malloc(500 * sizeof(float));
    float* tmp3 = (float*)malloc(1*1* sizeof(float));
    float* tmp5 = (float*)malloc(1*1* sizeof(float));

    float *d_fc1Output, *d_tmp6, *d_tmp7;
    cudaMalloc(&d_fc1Output, 500 * sizeof(float));
    cudaMalloc(&d_tmp6, 500 * 50 * sizeof(float)); // Assuming 500 elements per 50 elements
    cudaMalloc(&d_tmp7, 50 * 20*8*8 * sizeof(float));
    cudaMemcpy(d_fc1Weights, fc1Weights,  500*4*4*50 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2bias, conv2Bias,  50 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1bias, fc1Bias,  500 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1Weights, conv1Weights,   20*5*5*1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2Weights, conv2Weights,   50*5*5*20 * sizeof(float), cudaMemcpyHostToDevice);

    for (int x=0;x<filePaths.size();x++) {
        clock_t start1, stop1, start2;	
        start1 = clock();
    
        total = total+1;
        cout << "Processing file: " << filePaths[x] << endl;
        readImage(filePaths[x], &image);
        cudaMemcpy(d_image, image, conv1InputSize * conv1InputSize * sizeof(float), cudaMemcpyHostToDevice);
        // ------------------conv1---------------------
        cout<<"conv1\n";
        start2 = clock();
        for(int i=0; i<conv1numkernel; i++){
            dim3 blockSize(16, 16);
            dim3 gridSize((conv1InputSize + blockSize.x - 1) / blockSize.x, (conv1InputSize + blockSize.y - 1) / blockSize.y);
            convolution1<<<gridSize, blockSize, 0>>>(d_image, d_conv1Weights + i*convkernelSize*convkernelSize, d_conv1output + i*24*24 , conv1InputSize, convkernelSize, conv1Bias[i]);
            // cudaMemcpy(conv1output + i*24*24, d_conv1output, conv1OutputSize * conv1OutputSize * sizeof(float), cudaMemcpyDeviceToHost);
        }
        end1 = clock();
        float time_taken1 = (float) (end1-start2)/(CLOCKS_PER_SEC) ;
        cout << "Time taken conv1: " << time_taken1 << " seconds" << endl;
        // ------------------pool1---------------------
        cout<<"pool1\n";
        // cudaMemcpy(d_conv1output, conv1output,  20*24*24 * sizeof(float), cudaMemcpyHostToDevice);
        start2 = clock();
        for(int i=0; i<conv1numkernel; i++){
            dim3 blockSizepool1(16, 16);
            dim3 gridSizepool1((24 + blockSizepool1.x - 1) / blockSizepool1.x, (24 + blockSizepool1.y - 1) / blockSizepool1.y);
            maxPoolingKernel<<<gridSizepool1, blockSizepool1, 0>>>(d_conv1output + i*24*24, d_pool1output + i*12*12, 24, poolSize);
        }
        end1 = clock();
        time_taken1 = (float) (end1-start2)/(CLOCKS_PER_SEC) ;
        cout << "Time taken pool1: " << time_taken1 << " seconds" << endl;
        // ------------------conv2---------------------
        cout<<"conv2\n";
        start2 = clock();
        for(int i=0; i< 50; i++){
            for(int j=0; j<20; j++){

                dim3 blockSizeconv2(16, 16);
                dim3 gridSizeconv2((12 + blockSizeconv2.x - 1) / blockSizeconv2.x, (12 + blockSizeconv2.y - 1) / blockSizeconv2.y);
                conv2kernel_again<<<gridSizeconv2, blockSizeconv2, 0>>>(d_pool1output + j*12*12, d_conv2Weights + i*5*5*20 + j*5*5, d_tmp7 + i*20*64 + j*64 , 12, 5);

                
            }  
        }
        int blockSizeconv21 = 256;
        int gridSizeconv21 = (50 + blockSizeconv21 - 1) / blockSizeconv21;
        computeConv2Output<<<gridSizeconv21, blockSizeconv21>>>(d_conv2output, d_tmp7, d_conv2bias);

        for(int i=0; i<conv2numkernel; i++){
            dim3 blockSizepool2(16, 16);
            dim3 gridSizepool2((8 + blockSizepool2.x - 1) / blockSizepool2.x, (24 + blockSizepool2.y - 1) / blockSizepool2.y);
            maxPoolingKernel<<<gridSizepool2, blockSizepool2, 0>>>(d_conv2output + i*8*8, d_pool2output + i*4*4, 8, poolSize);
            // maxPoolingCUDA(d_conv2output + i*8*8, d_pool2output + i*4*4, 8, poolSize);
        }
        end1 = clock();
        time_taken1 = (float) (end1-start2)/(CLOCKS_PER_SEC) ;
        cout << "Time taken conv2: " << time_taken1 << " seconds" << endl;
        // ------------------fc1---------------------
        cout<<"fc1 --\n";
        
        // cudaMemcpy(d_pool2output, pool2output,  50*4*4 * sizeof(float), cudaMemcpyHostToDevice);
        start2 = clock();
        for(int i=0; i< 500; i++){
            for(int j=0; j<50; j++){
                // convolutionCUDA(pool2output + j*4*4, fc1Weights + i*4*4*50 + j*4*4, tmp6, 4, 4);
                // fc1(d_pool2output + j*4*4, d_fc1Weights + i*4*4*50 + j*4*4, d_tmp6 + i*50 + j, 4, 4);
                dim3 blockSizefc1(16, 16);
                dim3 gridSizefc1((4 + blockSizefc1.x - 1) / blockSizefc1.x, (4 + blockSizefc1.y - 1) / blockSizefc1.y);
                fc1kernel<<<gridSizefc1, blockSizefc1, 0>>>(d_pool2output + j*4*4, d_fc1Weights + i*4*4*50 + j*4*4, d_tmp6 + i*50 + j , 4, 4);

                
            }  
        }
        int blockSize = 256;
        int gridSize = (500 + blockSize - 1) / blockSize;
        computeFC1Output<<<gridSize, blockSize>>>(d_fc1Output, d_tmp6, d_fc1bias);

        end1 = clock();
        time_taken1 = (float) (end1-start2)/(CLOCKS_PER_SEC) ;
        cout << "Time taken fc1: " << time_taken1 << " seconds" << endl;

        // computeFC1Output<<<gridSize, blockSize>>>(d_fc1Output, d_tmp6);
        // cudaMemcpy(fc1output, d_fc1Output, 500 * sizeof(float), cudaMemcpyDeviceToHost);
        // for(int i=0; i<500; i++){
        //     d_fc1Output[i]= d_fc1Output[i]+ fc1Bias[i];
        // }

        // ----------------------------------relu---------------------------
        cout << "relu\n";
        start2 = clock();
        reluCUDA(d_fc1Output, fc1reluoutput, 1, 500);
        end1 = clock();
        time_taken1 = (float) (end1-start2)/(CLOCKS_PER_SEC) ;
        cout << "Time taken relu: " << time_taken1 << " seconds" << endl;

        // ------------------fc2---------------------
        cout<<"fc2\n";
        start2 = clock();

        for(int i=0; i< 10; i++){
            tmp3[0] = 0.0f;
            for(int j=0; j<500; j++){
                // convolutionCUDA(fc1reluoutput + j*1*1, fc2Weights + i*1*1*500 + j*1*1, tmp5, 1,1);
                tmp3[0]+= fc1reluoutput[j]*fc2Weights[i*1*1*500 + j*1*1];
            }
            
            tmp3[0] += fc2Bias[i];
            fc2output[i*1*1] = tmp3[0];
        }
        end1 = clock();
        time_taken1 = (float) (end1-start2)/(CLOCKS_PER_SEC) ;
        cout << "Time taken fc2: " << time_taken1 << " seconds" << endl;


        // ------------------softmax---------------------
        cout<<"softmax\n";
        start2 = clock();
        softmaxCUDA(fc2output, fc2softmaxoutput, 10);

        cout << "FC2 softmax output:" << endl;
        for(int i=0; i<10; i++){
            cout << fc2softmaxoutput[i] << " ";
        }
        cout << endl;
        end1 = clock();
        time_taken1 = (float) (end1-start2)/(CLOCKS_PER_SEC) ;
        cout << "Time taken softmax: " << time_taken1 << " seconds" << endl;

        // ------------------prediction---------------------
        start2 = clock();
        int maxIndex = 0;
        for(int i=0; i<10; i++){
            if(fc2softmaxoutput[i] > fc2softmaxoutput[maxIndex]){
                maxIndex = i;
            }
        }
        cout << "The number is: " << maxIndex << endl;
        cout << filePaths[x] << endl;
        cout << "The number should be: " << filePaths[x][filePaths[x].size() - 5]  << endl;
        if(maxIndex == (int)(filePaths[x][filePaths[x].size() - 5]  - '0')){
            correct++;
        }
        end1 = clock();
        time_taken1 = (float) (end1-start2)/(CLOCKS_PER_SEC) ;
        cout << "Time taken prediction: " << time_taken1 << " seconds" << endl;

        stop1 = clock();
        float time_taken1 = (float) (stop1-start1)/(CLOCKS_PER_SEC) ;
        cout << "Time taken per image: " << time_taken1 << " seconds" << endl;
        cout << "correct: " << correct << " total: " << total << endl;
    }
    stop = clock();
    float time_taken = (float) (stop-start)/(CLOCKS_PER_SEC) ;
    cout << "Total Time taken: " << time_taken << " seconds" << endl;
    cout << "Accuracy: " << (float)correct/total*100 << "%" << endl;
    return 0;
}
