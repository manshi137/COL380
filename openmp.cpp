#include <iostream>
#include <vector>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include <omp.h>
#include <chrono>
#include <cmath>
using namespace std;
#define MAX_THREADS 4
static int num_threads;

/*
inputs: a(n,n)
    outputs: π(n), l(n,n), and u(n,n)
    // parallelise initialisation of π, l, and u
    initialize π as a vector of length n
    initialize u as an n x n matrix with 0s below the diagonal
    initialize l as an n x n matrix with 1s on the diagonal and 0s above the diagonal

    for i = 1 to n
        π[i] = i

    for k = 1 to n
        max = 0
        // start threads
        for i = k to n
            if max < |a(i,k)|
            max = |a(i,k)|
            k' = i
        // join threads
        if max == 0
            error (singular matrix)
        swap π[k] and π[k']
        swap a(k,:) and a(k',:)
        swap l(k,1:k-1) and l(k',1:k-1)
        u(k,k) = a(k,k)
        for i = k+1 to n
            l(i,k) = a(i,k)/u(k,k)
            u(k,i) = a(k,i)
        for i = k+1 to n
            for j = k+1 to n
            a(i,j) = a(i,j) - l(i,k)*u(k,j)
*/

// vector<int> pi(n);
// vector<int> pi(n);
vector<vector<double>> lu_decomposition(vector<vector<double>> a, vector<int>& pi){
    cout<<"calling lu decomposition"<<endl;
    int n = a.size();
    cout<<n<<"=n "<<endl;
    vector<vector<double>> l(n, vector<double>(n, 0.0));
    vector<vector<double>> u(n, vector<double>(n, 0.0));

    auto start = std::chrono::high_resolution_clock::now();
    // initialize u as an n x n matrix with 0s below the diagonal
    // initialize l as an n x n matrix with 1s on the diagonal and 0s above the diagonal
    #pragma omp parallel for schedule(dynamic, n/(num_threads))
    for(int i = 0; i < n; i++){
        l[i][i] = 1;
    }

    // Initialize π as a vector of length n
    // thread start
    #pragma omp parallel for schedule(dynamic, n/(num_threads))
    for (int i = 0; i < n; ++i)
        pi[i] = i;
    // thread join
    
    for (int k = 0; k < n; ++k) {
        double max_val = 0;
        int k_prime = 0;
        // Find the maximum absolute value in column k
        // thread start
        #pragma omp parallel for schedule(dynamic, n/(num_threads))
        for (int i = k; i < n; ++i) {
            if (abs(a[i][k]) > max_val) {
                max_val = abs(a[i][k]);
                k_prime = i;
            }
        }
        // thread join
      if (max_val == 0) {
            cerr << "Error: Singular matrix" << endl;
            exit(1);
        }
        
        swap(pi[k], pi[k_prime]);
        // thread start
        swap(a[k], a[k_prime]);
        // thread join

        // thread start
        #pragma omp parallel for schedule(dynamic, n/(num_threads))
        for (int i = 0; i < k; ++i)
            swap(l[k][i], l[k_prime][i]);
        // thread join

        u[k][k] = a[k][k];
        
        // thread start
        #pragma omp parallel for schedule(dynamic, n/(num_threads))
        for (int i = k + 1; i < n; ++i) {
            l[i][k] = a[i][k] / u[k][k];
            u[k][i] = a[k][i];
        }
        // #pragma omp barrier
        // thread join
        
        // thread start
        #pragma omp parallel for schedule(dynamic, 10)
        for(int ind =0 ; ind<((n-k-1)*(n-k-1)); ind++){
            int i = ind/(n-k-1) + k+1;
            int j = ind%(n-k-1) + k+1;
            a[i][j] -= l[i][k] * u[k][j];
        }
        //   for (int i = k + 1; i < n; ++i) {
        //     for (int j = k + 1; j < n; ++j) {
        //         a[i][j] -= l[i][k] * u[k][j];
        //     }
        // }
        // thread join
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Computation time: " << duration.count() << " milliseconds" << std::endl;

    // calculate L*U
    vector<vector<double>> lu(n, vector<double>(n, 0.0));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            double sum = 0;
            for(int k = 0; k < n; k++){
                sum += l[i][k] * u[k][j];
            }
            lu[i][j] = sum;
        }
    }
    return lu;
}
double computeL21Norm(const std::vector<std::vector<double>>& matrix) {
    int n = matrix.size(); // Size of the square matrix
    double l21Norm = 0.0;

    // Iterate through each column of the matrix
    for (int j = 0; j < n; ++j) {
        double columnSumOfSquares = 0.0;

        // Calculate the sum of squares of elements in the current column
        for (int i = 0; i < n; ++i) {
            columnSumOfSquares += matrix[i][j] * matrix[i][j];
        }

        // Take the square root of the sum of squares
        double columnNorm = sqrt(columnSumOfSquares);

        // Add the square root to the L2,1 norm
        l21Norm += columnNorm;
    }

    return l21Norm;
}

int main(int argc, char *argv[])  {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        return 1;
    }

    // Convert command-line arguments to integers
    int size = std::atoi(argv[1]); // Size of the matrix
    num_threads = std::atoi(argv[2]); // Number of threads
    cout<<"size= "<<size<<endl;
    cout<<"num_threads= "<<num_threads<<endl;

    // Set the number of threads to be used
    omp_set_num_threads(num_threads);

    srand(time(nullptr));
    // int size = 10;
    // Declare a 2D vector of size 100x100
    vector<vector<double>> A_original(size, vector<double>(size));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A_original[i][j] = rand() % 10 + 1; // Generates a random integer between 1 and 10
        }
    }
    vector<int> pi(size);
    vector<vector<double>> L, U;
    cout<<"before"<<endl;
    vector<vector<double>> luprod= lu_decomposition(A_original, pi);
    
    vector<vector<double>> A_permuted(size, vector<double>(size, 0.0));
    for (int i = 0; i < pi.size(); ++i) {
            A_permuted[pi[i]] = luprod[i];
    }
    vector<vector<double>> diff(size, vector<double>(size, 0.0));
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++)
            diff[i][j] = A_permuted[i][j]-A_original[i][j];
    }
    double l21Norm = computeL21Norm(diff);
    std::cout << "L2,1 Norm: " << l21Norm << std::endl;

    // Output results
    // cout << "pi :"<<endl;
    // for (int i = 0; i < pi.size(); ++i) cout << pi[i] << " ";
    // cout << endl;
    // print A_permuted
    // cout << "a after applying permutation pi:" << endl;
    // for(int i = 0; i < a.size(); i++){
    //     for(int j = 0; j < a.size(); j++){
    //         cout << A_permuted[i][j] << " ";
    //     }
    //     cout << endl;
    // }
}
// g++ serial.cpp  -std=c++11 