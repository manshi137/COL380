// Implement the following functions using 32 bit float as datatype.

// convolution of a square input matrix and a square kernel, both matrices of any size and the kernel smaller than the input

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <bits/stdc++.h>

using namespace std;

// Function to perform convolution
vector<vector<float>> convolution(vector<vector<float>> input, vector<vector<float>> kernel) {
    int input_size = input.size();
    int kernel_size = kernel.size();
    int output_size = input_size - kernel_size + 1;
    vector<vector<float>> output(output_size, vector<float>(output_size, 0));
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    output[i][j] += input[i + k][j + l] * kernel[k][l];
                }
            }
        }
    }
    return output;
}
// convolution with padding
// n * n input matrix and f*f padding matrix
vector<vector<float>> convolution_padding(vector<vector<float>> input, vector<vector<float>> kernel, int padding) {
    int input_size = input.size();
    int kernel_size = kernel.size();
    int output_size = input_size - kernel_size + 1;
    vector<vector<float>> output(input_size + 2 * padding, vector<float>(input_size + 2 * padding, 0));
    for (int i = padding; i < input_size + padding; i++) {
        for (int j = padding; j < input_size + padding; j++) {
            output[i][j] = input[i - padding][j - padding];
        }
    }
    vector<vector<float>> result = convolution(output, kernel);
    return result;
}