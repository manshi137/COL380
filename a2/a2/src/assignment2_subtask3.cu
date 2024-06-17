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



int printTop5SoftmaxProb(float* softmaxOutput, int size, const char * filename) {
    // Array to store softmax probabilities
    float probs[10];

    // Copy softmax probabilities to probs array
    for (int i = 0; i < 10; ++i) {
        probs[i] = softmaxOutput[i];
    }

    // Sort probabilities and their corresponding labels
    float sortedProbs[10];
    int sortedIndices[10];
    for (int i = 0; i < 10; ++i) {
        sortedProbs[i] = softmaxOutput[i];
        sortedIndices[i] = i;
    }
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9 - i; ++j) {
            if (sortedProbs[j] < sortedProbs[j + 1]) {
                // Swap probabilities
                float tempProb = sortedProbs[j];
                sortedProbs[j] = sortedProbs[j + 1];
                sortedProbs[j + 1] = tempProb;
                // Swap corresponding indices
                int tempIndex = sortedIndices[j];
                sortedIndices[j] = sortedIndices[j + 1];
                sortedIndices[j + 1] = tempIndex;
            }
        }
    }
// //cout<<filename;
    // Write the top 5 softmax probabilities and their corresponding labels to a text file
    FILE *file;
    file = fopen(filename, "w+");
    if (file == NULL) {
        printf("Error opening file.\n");
        return -1;
    }
    // fprintf(file, "Top 5 softmax probabilities:\n");
    for (int i = 0; i < 5; ++i) {
        fprintf(file, "%f ", sortedProbs[i]);
    }
    fclose(file);
    int maxLabel = sortedIndices[0];
    return maxLabel;
}


__global__ void computeFC2Output(float* d_fc2Output, float* d_tmp8, float* bias) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 10) {
        d_fc2Output[tid] = 0.0f;
        for (int j = 0; j < 500; ++j) {
            d_fc2Output[tid] += d_tmp8[tid * 500 + j];
        }
        d_fc2Output[tid]+=bias[tid];
    }
}
__global__ void fc2kernel(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int index = row*32 + col;
    int startOffset = index*1*1;
    if (index< 500){
        float sum = 0.0f;

                // sum += input[i*4 + j] * kernel[i*4 + j];
                sum += input[startOffset] * kernel[startOffset];

        output[index] = sum;
    }
}
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
    int index = row*8 + col;
    int startOffset = index*4*4;
    if (index< 50){
        float sum = 0.0f;
        for(int i=0; i<kernelSize; i++){
            for(int j=0; j<kernelSize; j++){
                // sum += input[i*4 + j] * kernel[i*4 + j];
                sum += input[startOffset + i*kernelSize + j] * kernel[startOffset + i*kernelSize + j];
            }
        }
        output[index] = sum;
    }
}

__global__ void conv2kernel(float *input, float *kernel, float *output, int inputSize, int kernelSize) {
    // printf("hello");
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = blockIdx.x;
    // printf("x: %d, y: %d, z: %d\n", x, y, z);

    if(z<20 && x<8 && y<8){
        int inputOffset = z * inputSize * inputSize;
        int kernelOffset = z * kernelSize * kernelSize;

        float sum = 0.0f;
        for( int i=0; i<kernelSize; i++){
            for(int j=0; j<kernelSize; j++){
                sum+= input[inputOffset + (x+i)*inputSize + (y+j)] * kernel[kernelOffset + i*kernelSize + j];     
            }
        }
        output[z*8*8 + x*8 + y] = sum;
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
                    // //cout<<input[inputIdx]<<" "<<maxVal;
                    maxVal = fmaxf(maxVal, input[inputIdx]);
                }
            }
        }
        output[outputIdx] = maxVal;
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

void reluCUDA(float *d_input, float *d_output, int rows, int cols) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    reluKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
}


__global__ void softmaxKernel(float *input, float *output, float maxVal, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float maxInput = input[0];
        for (int i = 1; i < size; ++i) {
            maxInput = max(maxInput, input[i]);
        }

        float expSum = 0.0f;
        for (int i = 0; i < size; ++i) {
            expSum += expf(input[i] - maxInput);
        }

        output[idx] = expf(input[idx] - maxInput) / expSum;
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
    string conv1file = "../weights/conv1.txt";
    string conv2file = "../weights/conv2.txt";
    string fc1file = "../weights/fc1.txt";
    string fc2file = "../weights/fc2.txt";  
    readKernelWeightsAndBiasconv1(conv1file, &conv1Weights, &conv1Bias);
    readKernelWeightsAndBiasconv2(conv2file, &conv2Weights, &conv2Bias);
    readFC1(fc1file, &fc1Weights, &fc1Bias);
    readFC2(fc2file, &fc2Weights, &fc2Bias);

    std::string folderPath = "../pre-proc-img";
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
    float* d_fc2Weights;
    cudaMalloc(&d_fc2Weights, 10*1*1*500 * sizeof(float));
    float* d_conv2bias;
    cudaMalloc(&d_conv2bias, 50 * sizeof(float));
    float* d_fc1bias;
    cudaMalloc(&d_fc1bias, 500 * sizeof(float));
    float* d_fc2bias;
    cudaMalloc(&d_fc2bias, 10 * sizeof(float));
    float* d_pool2output;
    cudaMalloc(&d_pool2output,  50*4*4 * sizeof(float));
    float* tmp4 = (float*)malloc(1*1* sizeof(float));
    float* tmp6 = (float*)malloc(500*50*1*1* sizeof(float));
    float *fc1reluoutput = (float*)malloc(500 * sizeof(float));
    float *d_fc1reluoutput;
    cudaMalloc(&d_fc1reluoutput, 500 * sizeof(float));
    float *d_fc2output;
    cudaMalloc(&d_fc2output, 10 * sizeof(float));

    float* tmp3 = (float*)malloc(1*1* sizeof(float));
    float* tmp5 = (float*)malloc(1*1* sizeof(float));

    float *d_fc1Output, *d_tmp6, *d_tmp7, *d_tmp8;
    cudaMalloc(&d_fc1Output, 500 * sizeof(float));
    cudaMalloc(&d_tmp6, 500 * 50 * sizeof(float)); // Assuming 500 elements per 50 elements
    cudaMalloc(&d_tmp7, 50 * 20*8*8 * sizeof(float));
    cudaMalloc(&d_tmp8, 500 * 10 * sizeof(float));
    cudaMemcpy(d_fc1Weights, fc1Weights,  500*4*4*50 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2Weights, fc2Weights,  10*1*1*500 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2bias, conv2Bias,  50 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1bias, fc1Bias,  500 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2bias, fc2Bias,  10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1Weights, conv1Weights,   20*5*5*1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2Weights, conv2Weights,   50*5*5*20 * sizeof(float), cudaMemcpyHostToDevice);
    float* d_fc2softmaxoutput;
    cudaMalloc(&d_fc2softmaxoutput, 10 * sizeof(float));


    for (int x=0;x<filePaths.size();x++) {
        clock_t start1, stop1, start2, end1;	
    
        total = total+1;
        cout << "Processing file: " << filePaths[x] <<  " --------------------------------------------------------- "<<endl;
        readImage(filePaths[x], &image);
        cudaMemcpy(d_image, image, conv1InputSize * conv1InputSize * sizeof(float), cudaMemcpyHostToDevice);

        start1 = clock();
        // ------------------conv1---------------------
        //cout<<"conv1\n";
        start2 = clock();
        for(int i=0; i<conv1numkernel; i++){
            dim3 blockSize(16, 16);
            dim3 gridSize((conv1InputSize + blockSize.x - 1) / blockSize.x, (conv1InputSize + blockSize.y - 1) / blockSize.y);
            convolution1<<<gridSize, blockSize, 0>>>(d_image, d_conv1Weights + i*convkernelSize*convkernelSize, d_conv1output + i*24*24 , conv1InputSize, convkernelSize, conv1Bias[i]);
            // cudaMemcpy(conv1output + i*24*24, d_conv1output, conv1OutputSize * conv1OutputSize * sizeof(float), cudaMemcpyDeviceToHost);
        }
        end1 = clock();
        double time_taken1 =  (end1-start2) * 1000.0f/ CLOCKS_PER_SEC;
        //cout << "Time taken conv1: " << time_taken1 << " milli seconds" << endl;
        // ------------------pool1---------------------
        //cout<<"pool1\n";
        // cudaMemcpy(d_conv1output, conv1output,  20*24*24 * sizeof(float), cudaMemcpyHostToDevice);
        start2 = clock();
        for(int i=0; i<conv1numkernel; i++){
            dim3 blockSizepool1(16, 16);
            dim3 gridSizepool1((24 + blockSizepool1.x - 1) / blockSizepool1.x, (24 + blockSizepool1.y - 1) / blockSizepool1.y);
            maxPoolingKernel<<<gridSizepool1, blockSizepool1, 0>>>(d_conv1output + i*24*24, d_pool1output + i*12*12, 24, poolSize);
        }
        end1 = clock();
        time_taken1 = (end1-start2)* 1000.0f/ CLOCKS_PER_SEC ;
        //cout << "Time taken pool1: " << time_taken1 << "milli seconds" << endl;
        // ------------------conv2---------------------
        //cout<<"conv2\n";
        start2 = clock();
        for(int i=0; i< 50; i++){
            dim3 blockSizeconv2(8, 8);
            // dim3 gridSizeconv2(1, 1, 1);
            int gridSizeconv2 = 20;
            conv2kernel<<<gridSizeconv2, blockSizeconv2>>>(d_pool1output, d_conv2Weights + i*5*5*20, d_tmp7 + i*20*8*8, 12, 5);
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
        // time in milliseconds
        time_taken1 = (end1-start2)* 1000.0f/ CLOCKS_PER_SEC;
        //cout << "Time taken conv2: " << time_taken1 << " milliseconds" << endl;
        // ------------------fc1---------------------
        //cout<<"fc1 --\n";
        
        // cudaMemcpy(d_pool2output, pool2output,  50*4*4 * sizeof(float), cudaMemcpyHostToDevice);

        start2 = clock();
        for(int i=0; i< 500; i++){
                dim3 blockSizefc1(2, 2);
                dim3 gridSizefc1(4, 4);
                fc1kernel<<<gridSizefc1, blockSizefc1, 0>>>(d_pool2output , d_fc1Weights + i*4*4*50, d_tmp6 + i*50, 4, 4);
        }
        end1 = clock();
        time_taken1 =  (end1-start2)* 1000.0f/ CLOCKS_PER_SEC ;
        //cout << "Time taken fc1: " << time_taken1 << " milliseconds" << endl;
        start2 = clock();


        int blockSize = 256;
        int gridSize = (500 + blockSize - 1) / blockSize;
        computeFC1Output<<<gridSize, blockSize>>>(d_fc1Output, d_tmp6, d_fc1bias);
        end1 = clock();
        time_taken1 =  (end1-start2)* 1000.0f/ CLOCKS_PER_SEC ;
        //cout << "Time taken computefc1: " << time_taken1 << " milliseconds" << endl;

        // ----------------------------------relu---------------------------
        //cout << "relu\n";
        start2 = clock();
        reluCUDA(d_fc1Output, d_fc1reluoutput, 1, 500);
        end1 = clock();
        time_taken1 = (end1-start2)  * 1000.0f/ CLOCKS_PER_SEC ;
        //cout << "Time taken relu: " << time_taken1 << " milliseconds" << endl;

        // ------------------fc2---------------------
        //cout<<"fc2\n";
        start2 = clock();

        // for(int i=0; i< 10; i++){
        //     tmp3[0] = 0.0f;
        //     for(int j=0; j<500; j++){
        //         // convolutionCUDA(fc1reluoutput + j*1*1, fc2Weights + i*1*1*500 + j*1*1, tmp5, 1,1);
        //         tmp3[0]+= fc1reluoutput[j]*fc2Weights[i*1*1*500 + j*1*1];
        //     }
        //     tmp3[0] += fc2Bias[i];
        //     fc2output[i*1*1] = tmp3[0];
        // }

        // fc2Kernel<<<10, 1>>>(d_fc1reluoutput, d_fc2Weights, d_fc2bias, d_fc2output);

        for(int i=0; i< 10; i++){
                dim3 blockSizefc2(32, 32);
                int gridSizefc2= 1;
                fc2kernel<<<gridSizefc2, blockSizefc2, 0>>>(d_fc1reluoutput , d_fc2Weights + i*1*1*500, d_tmp8 + i*500, 1, 1);
        }
        end1 = clock();
        time_taken1 =  (end1-start2)* 1000.0f/ CLOCKS_PER_SEC ;
        //cout << "Time taken fc2: " << time_taken1 << " milliseconds" << endl;
        start2 = clock();


        int blockSizefc2 = 256;
        int gridSizefc2 = (10 + blockSizefc2 - 1) / blockSizefc2;
        computeFC2Output<<<gridSizefc2, blockSizefc2>>>(d_fc2output, d_tmp8, d_fc2bias);
        end1 = clock();
        time_taken1 =  (end1-start2)* 1000.0f/ CLOCKS_PER_SEC ;
        //cout << "Time taken fc2compute: " << time_taken1 << " milliseconds" << endl;


        // ------------------softmax---------------------
        //cout<<"softmax\n";
        start2 = clock();
        // softmaxCUDA(d_fc2output, fc2softmaxoutput, 10);
        softmaxKernel<<<1, 10>>>(d_fc2output, d_fc2softmaxoutput, 0, 10);
        cudaMemcpy(fc2softmaxoutput, d_fc2softmaxoutput, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        end1 = clock();
        time_taken1 = (end1-start2) * 1000.0f / CLOCKS_PER_SEC;
        //cout << "Time taken softmax: " << time_taken1 << "milli seconds" << endl;

        // ------------------prediction---------------------
        std::string filepath="../output/"+filePaths[x];

        std::string searchString = "../pre-proc-img/";
        size_t found = filepath.find(searchString);
        if (found != std::string::npos) {
            // Erase the substring starting from 'found' position till the length of searchString
            filepath.erase(found, searchString.length());
        }
        const char* filename = filepath.c_str();
        // //cout<<filepath<<"\n";
        int maxIndexdummy = printTop5SoftmaxProb(fc2softmaxoutput, 10,filename);
        start2 = clock();
        int maxIndex = 0;
        for(int i=0; i<10; i++){
            if(fc2softmaxoutput[i] > fc2softmaxoutput[maxIndex]){
                maxIndex = i;
            }
        }
        if(maxIndex == (int)(filePaths[x][filePaths[x].size() - 5]  - '0')){
            correct++;
        }
        end1 = clock();
        time_taken1 = (end1-start2)* 1000.0f/ CLOCKS_PER_SEC ;
        //cout << "Time taken prediction: " << time_taken1 << " milliseconds" << endl;

        stop1 = clock();
        time_taken1 = (stop1-start1)* 1000.0f/ CLOCKS_PER_SEC ;
        //cout << "Time taken per image: " << time_taken1 << " miliseconds" << endl;
        //cout << "correct: " << correct << " total: " << total << endl;
    }
    stop = clock();
    double time_taken = (stop-start)* 1000.0f/ CLOCKS_PER_SEC ;
    cout << "Total Time taken: " << time_taken << "milli seconds" << endl;
    cout << "Accuracy: " << (float)correct/total*100 << "%" << endl;
    return 0;
}