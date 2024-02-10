#include <iostream>
#include <vector>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()

using namespace std;

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
vector<vector<double>> lu_decomposition(vector<vector<double>> a, vector<int>& pi){
    int n = a.size();
    vector<vector<double>> l(n, vector<double>(n, 0.0));
    vector<vector<double>> u(n, vector<double>(n, 0.0));
    // initialize u as an n x n matrix with 0s below the diagonal
    // initialize l as an n x n matrix with 1s on the diagonal and 0s above the diagonal
    for(int i = 0; i < n; i++){
        l[i][i] = 1;
    }

    // Initialize π as a vector of length n
    for (int i = 0; i < n; ++i)
        pi[i] = i;
    
    for (int k = 0; k < n; ++k) {
        double max_val = 0;
        int k_prime = 0;
        
        // Find the maximum absolute value in column k
        for (int i = k; i < n; ++i) {
            if (abs(a[i][k]) > max_val) {
                max_val = abs(a[i][k]);
                k_prime = i;
            }
        }
      if (max_val == 0) {
            cerr << "Error: Singular matrix" << endl;
            exit(1);
        }
        
        swap(pi[k], pi[k_prime]);
        // thread start
        swap(a[k], a[k_prime]);
        // thread join

        // thread start
        for (int i = 0; i < k; ++i)
            swap(l[k][i], l[k_prime][i]);
        // thread join

        u[k][k] = a[k][k];
        
        // thread start
        for (int i = k + 1; i < n; ++i) {
            l[i][k] = a[i][k] / u[k][k];
            u[k][i] = a[k][i];
        }
        // thread join
        
        // thread start
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                a[i][j] -= l[i][k] * u[k][j];
            }
        }
        // thread join
    }
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
int main() {
    srand(time(nullptr));
    int size = 10;
    // Declare a 2D vector of size 100x100
    vector<vector<double>> a(size, vector<double>(size));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i][j] = rand() % 10 + 1; // Generates a random integer between 1 and 10
        }
    }
    cout<<"original matrix a: \n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
    vector<int> pi(a.size());
    vector<vector<double>> L, U;
    vector<vector<double>> luprod= lu_decomposition(a, pi);
    
    // Output results
    cout << "π:";
    for (int i = 0; i < pi.size(); ++i) cout << " " << pi[i];
    cout << endl;
    cout << "a after applying permutation pi:" << endl;
    vector<vector<double>> A_permuted(a.size(), vector<double>(a[0].size(), 0.0));
    for (int i = 0; i < pi.size(); ++i) {
            A_permuted[pi[i]] = luprod[i];
    }
    // print A_permuted
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a.size(); j++){
            cout << A_permuted[i][j] << " ";
        }
        cout << endl;
    }
}
// g++ serial.cpp  -std=c++11 