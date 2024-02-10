#include <iostream>
#include <vector>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include <pthread.h>

using namespace std;
const int size = 10;
const int num_threads = 4;
const int MAX_THREADS=4;
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
void* thread_start(void* arg) {
    int thread_id = *((int*)arg); // Get the thread ID
    int chunk_size = n / MAX_THREADS; // Calculate the chunk size for each thread
    int start_index = thread_id * chunk_size; // Calculate the start index for this thread
    int end_index = (thread_id == MAX_THREADS - 1) ? n : (start_index + chunk_size); // Calculate the end index

    // Assign values to the array elements within the thread's range
    for (int i = start_index; i < end_index; ++i) {
        pi[i] = i;
    }

    pthread_exit(NULL); // Exit the thread
}

void* max_finder(void* arg) {
    int thread_id = *((int*)arg->thread_id); // Get the thread ID
    int chunk_size = (n)/ MAX_THREADS; // Calculate the chunk size for each thread
    int start_index = (thread_id * chunk_size) ; // Calculate the start index for this thread
    int end_index = (thread_id == MAX_THREADS - 1) ? n : (start_index + chunk_size); // Calculate the end index
    int k= arg->val
    // Assign values to the array elements within the thread's range
    int k_prime=-1;
    for(int j=start_index;j<end_index; j++){
        for (int i = k; i < n; ++i) {
            if (abs(a[i][j]) > max_val) {
                max_val = abs(a[i][j]);
                k_prime = i;
            }
        }
    }

    pthread_exit(NULL); // Exit the thread
    return (void*) k_prime;
}

struct info{
int* thread_id = malloc(sizeof(int)); // Allocate memory for the thread ID
int val;
};

// vector<int> pi(n);
vector<vector<double>> lu_decomposition(vector<vector<double>> a, vector<int>& pi){

    pthread_t threads[MAX_THREADS]; // Declare an array of thread IDs
    // struct info args;
    struct info thread_info[NUM_THREADS];
    int n = a.size();
    vector<vector<double>> l(n, vector<double>(n, 0.0));
    vector<vector<double>> u(n, vector<double>(n, 0.0));
    for(int i = 0; i < n; i++){
        l[i][i] = 1;
    }

    // Create threads
    for (int i = 0; i < MAX_THREADS; ++i) {
        // int* thread_id = malloc(sizeof(int)); // Allocate memory for the thread ID
        *thread_id = i; // Assign the thread ID
        pthread_create(&threads[i], NULL, thread_start, (void*)&args); // Create the thread
    }

    // Join threads
    for (int i = 0; i < MAX_THREADS; ++i) {
        pthread_join(threads[i], NULL); // Wait for the thread to finish
    }

    // Print the array elements
    for (int i = 0; i < n; ++i) {
        printf("%d ", pi[i]);
    }
    printf("\n");


    // // thread start
    // for (int i = 0; i < n; ++i)
    //     pi[i] = i;
    // // thread join

    for (int k = 0; k < n; ++k) {
        double max_val = 0;
        int k_prime = 0;
        
         for (int i = 0; i < NUM_THREADS; ++i) {
            // Initialize thread data
            *thread_info[i].thread_id = i;
            thread_info[i].val = k;

            // Create thread
            pthread_create(&threads[i], NULL, findMaxValue, &thread_info[i]);
        }
       
    


        for (int i = 0; i < NUM_THREADS; ++i) {
            void *thread_return_value;
            pthread_join(threads[i], &thread_return_value);
            int local_max_index = (int)thread_return_value;
            if (local_max_index != -1 && fabs(a[local_max_index][k]) > fabs(a[k_prime][k])) {
                k_prime = local_max_index;
            }
        }
        max_val= a[k_prime][k];

        // thread start
        // for (int i = k; i < n; ++i) {
        //     if (abs(a[i][k]) > max_val) {
        //         max_val = abs(a[i][k]);
        //         k_prime = i;
        //     }
        // }
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
    // print L
    // cout << "L:" << endl;
    // for(int i = 0; i < n; i++){
    //     for(int j = 0; j < n; j++){
    //         cout << l[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // print U
    // cout << "U:" << endl;
    // for(int i = 0; i < n; i++){
    //     for(int j = 0; j < n; j++){
    //         cout << u[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    
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
    // calculate difference matrix between A_permuted and a
    vector<vector<double>> diff(a.size(), vector<double>(a[0].size(), 0.0));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a.size(); j++){
            diff[i][j] = A_permuted[i][j] - a[i][j];
        }
    }
    // calculate L2,1  matrix induced norm of diff
    double l2norm = 0;
    for(int i = 0; i < a.size(); i++){
        double sum = 0;
        for(int j = 0; j < a.size(); j++){
            sum += diff[i][j] * diff[i][j];
        }
        l2norm += sqrt(sum);
    }
    cout << "L2,1 norm of difference matrix: " << l2norm << endl;

}
// g++ pthread.cpp  -std=c++11 