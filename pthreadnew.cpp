#include <iostream>
#include <vector>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include <pthread.h>
#define MAX_SIZE 1000   
using namespace std;
static int num_threads;
int curr_size ;
vector<vector<double>> A_global(MAX_SIZE, vector<double>(MAX_SIZE, 100.0));
vector<vector<double>> l_global(MAX_SIZE, vector<double>(MAX_SIZE,0.0));
vector<vector<double>> u_global(MAX_SIZE, vector<double>(MAX_SIZE, 0.0));
vector<int> pi_global(MAX_SIZE);
pthread_mutex_t mutex1;
static int k_global = 0;
static int k_prime = -1;
static double max_val = -1e9;

void* assign_a(void* rank){
    long my_rank = (long) rank;
    long loopsize = (curr_size-k_global)*(curr_size-k_global);
    int chunk_size = (loopsize) / num_threads; 
    int start_index = (my_rank * chunk_size) ; // Calculate the start index for this thread
    int end_index = (my_rank == num_threads - 1) ? loopsize : (start_index + chunk_size); // Calculate the end index
    if (loopsize<num_threads){
        if(my_rank!=0) return NULL;
        start_index = 0 ; // Calculate the start index for this thread
        end_index = loopsize; // Calculate the end index
    }
    for(int ind = start_index; ind<end_index; ind++){
        int i = ind/(curr_size-k_global) + k_global+1;
        int j = ind%(curr_size-k_global) + k_global+1;
        A_global[i][j] -= l_global[i][k_global] * u_global[k_global][j];
    }
    // for(int ind =0 ; ind<(curr_size-k_global)*(curr_size-k_global); ind++){
    //     int i = ind/(curr_size-k_global) + k_global+1;
    //     int j = ind%(curr_size-k_global) + k_global+1;
    //     a[i][j] -= l_global[i][k_global] * u_global[k_global][j];
    // }
    return NULL;
}

void* assign_l(void* rank){
    long my_rank = (long) rank;
    int chunk_size = (curr_size-(k_global+1)) / num_threads; // Calculate the chunkSIZE for each thread
    int start_index = (my_rank * chunk_size)+(k_global+1) ; // Calculate the start index for this thread
    int end_index = (my_rank == num_threads - 1) ? curr_size : (start_index + chunk_size); // Calculate the end index
    if (curr_size-(k_global+1)<num_threads){
        if(my_rank!=0) return NULL;
        start_index = (k_global+1) ; // Calculate the start index for this thread
        end_index = curr_size; // Calculate the end index
    }
    for(int i=start_index; i<end_index; i++){
        // l_global[i][k_global] = a[i][k_global] / u_global[k_global][k_global];
        l_global[i][k_global] = A_global[i][k_global] / u_global[k_global][k_global];
    }
    return NULL;
}

void* assign_u(void* rank){
    long my_rank = (long) rank;
    int chunk_size = (curr_size-(k_global+1)) / num_threads; // Calculate the chunkSIZE for each thread
    int start_index = (my_rank * chunk_size)+(k_global+1) ; // Calculate the start index for this thread
    int end_index = (my_rank == num_threads - 1) ? curr_size : (start_index + chunk_size); // Calculate the end index
    if (curr_size-(k_global+1)<num_threads){
        if(my_rank!=0) return NULL;
        start_index = (k_global+1) ; // Calculate the start index for this thread
        end_index = curr_size; // Calculate the end index
    }
    for(int i=start_index; i<end_index; i++){
        u_global[k_global][i] = A_global[k_global][i];
    }
    return NULL;
}

void* get_k_prime(void* rank){
    
    long my_rank = (long) rank;
    int chunk_size = (curr_size-k_global) / num_threads; // Calculate the chunkSIZE for each thread
    int start_index = (my_rank * chunk_size)+k_global ; // Calculate the start index for this thread
    int end_index = (my_rank == num_threads - 1) ? curr_size : (start_index + chunk_size); // Calculate the end index
    if (curr_size-k_global<num_threads){
        if(my_rank!=0) return NULL;
        start_index = k_global ; // Calculate the start index for this thread
        end_index = curr_size; // Calculate the end index
    }
    long local_max_val = -1e9;
    long local_k_prime = -1;
    for(int i=start_index; i<end_index; i++){
        if (abs(A_global[i][k_global]) > local_max_val) {
            local_max_val = abs(A_global[i][k_global]);
            local_k_prime = i;
        }
    }
    pthread_mutex_lock(&mutex1);
    if(local_max_val>max_val){
        max_val = local_max_val;
        k_prime = local_k_prime;
    }
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

vector<vector<double>> lu_decomposition(vector<vector<double>>& a, vector<int>& pi){
    cout<<"calling lu_decomposition"<<endl;
    pthread_t* thread_handles1;
    pthread_t* thread_handles2;
    pthread_t* thread_handles3;
    pthread_t* thread_handles4;
    // vector<vector<double>> l(curr_size, vector<double>(curr_size, 0.0));
    // vector<vector<double>> u(curr_size, vector<double>(curr_size, 0.0));
    for(int i = 0; i < curr_size; i++){
        l_global[i][i] = 1;
    }
    for (int i = 0; i < curr_size; ++i)
        pi[i] = i;

    for ( k_global = 0; k_global < curr_size; ++k_global) {
        // pthread start loop1
        // for (int i = k; i < n; ++i) {
        //     if (abs(a[i][k]) > max_val) {
        //         max_val = abs(a[i][k]);
        //         k_prime = i;
        //     }
        // }
        k_prime = -1;
        max_val = -1e9;
        thread_handles1= (pthread_t*)malloc(num_threads * sizeof(pthread_t));

        pthread_mutex_init(&mutex1, NULL);
        for (int thread=0; thread<num_threads; thread++) {
            pthread_create(&thread_handles1[thread], NULL, get_k_prime, (void *) thread);
        }
        for(int thread=0; thread<num_threads; thread++) {
            pthread_join(thread_handles1[thread], NULL);
        }
        pthread_mutex_destroy(&mutex1);
        free(thread_handles1);
        // join loop1
        if (max_val == 0) {
            cerr << "Error: Singular matrix" << endl;
            exit(1);
        }
        
        swap(pi[k_global], pi[k_prime]);
        // thread start
        swap(a[k_global], a[k_prime]);
        // thread join

        // thread start
        for (int i = 0; i < k_global; ++i)
            swap(l_global[k_global][i], l_global[k_prime][i]);
        // thread join

        u_global[k_global][k_global] = a[k_global][k_global];

        // thread start loop2.1
        // for (int i = k_global + 1; i < curr_size; ++i) {
        //     l_global[i][k_global] = a[i][k_global] / u_global[k_global][k_global];
        // }
        thread_handles2= (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        for (int thread=0; thread<num_threads; thread++) {
            pthread_create(&thread_handles2[thread], NULL, assign_l, (void *) thread);
        }
        for(int thread=0; thread<num_threads; thread++) {
            pthread_join(thread_handles2[thread], NULL);
        }
        free(thread_handles2);
        // thread join loop2.1

        // thread start loop2.2
        // for (int i = k_global + 1; i < curr_size; ++i) {
        //     u_global[k_global][i] = a[k_global][i];
        // }
        thread_handles3= (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        for (int thread=0; thread<num_threads; thread++) {
            pthread_create(&thread_handles3[thread], NULL, assign_u, (void *) thread);
        }
        for(int thread=0; thread<num_threads; thread++) {
            pthread_join(thread_handles3[thread], NULL);
        }
        free(thread_handles3);
        // thread join loop2.2

        // thread start loop3
        // for (int i = k_global + 1; i < curr_size; ++i) {
        //     for (int j = k_global + 1; j < curr_size; ++j) {
        //         a[i][j] -= l_global[i][k_global] * u_global[k_global][j];
        //     }
        // }
        // for(int ind =0 ; ind<(curr_size-k_global)*(curr_size-k_global); ind++){
        //     int i = ind/(curr_size-k_global) + k_global+1;
        //     int j = ind%(curr_size-k_global) + k_global+1;
        //     a[i][j] -= l_global[i][k_global] * u_global[k_global][j];
        // }
        thread_handles4= (pthread_t*)malloc(num_threads * sizeof(pthread_t));
        for(int thread=0; thread<num_threads; thread++) {
            pthread_create(&thread_handles4[thread], NULL, assign_a, (void *) thread);
        }
        for(int thread=0; thread<num_threads; thread++) {
            pthread_join(thread_handles4[thread], NULL);
        }
        free(thread_handles4);
        // thread join loop3
    }
    // calculate L*U
    vector<vector<double>> lu(curr_size, vector<double>(curr_size, 0.0));
    for(int i = 0; i < curr_size; i++){
        for(int j = 0; j < curr_size; j++){
            double sum = 0;
            for(int k = 0; k < curr_size; k++){
                sum += l_global[i][k] * u_global[k][j];
            }
            lu[i][j] = sum;
        }
    }
    return lu;
}

double computeL21Norm(const std::vector<std::vector<double>>& matrix) {
    double l21Norm = 0.0;
    for (int j = 0; j < curr_size; ++j) {
        double columnSumOfSquares = 0.0;
        for (int i = 0; i < curr_size; ++i) {
            columnSumOfSquares += matrix[i][j] * matrix[i][j];
        }
        double columnNorm = sqrt(columnSumOfSquares);
        l21Norm += columnNorm;
    }
    return l21Norm;
}

int main(int argc, char *argv[])  {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]); // Size of the matrix
    curr_size = n;
    num_threads = std::atoi(argv[2]); // Number of threads
    cout<<"curr_size= "<<curr_size<<endl;
    cout<<"num_threads= "<<num_threads<<endl;
    srand(time(nullptr));
    vector<vector<double>> A_original(curr_size, vector<double>(curr_size));

    for (int i = 0; i < curr_size; ++i) {
        for (int j = 0; j < curr_size; ++j) {
            A_original[i][j] = rand() % 10 + 1; // Generates a random integer between 1 and 10
            A_global[i][j] = A_original[i][j];
        }
    }
    // print a
    for (int i = 0; i < curr_size; ++i) {
        for (int j = 0; j < curr_size; ++j) {
            std::cout << A_global[i][j] << " ";
        }
        std::cout << std::endl;
    }
    //main
    vector<vector<double>> luprod= lu_decomposition(A_global, pi_global);
    //mainover
    vector<vector<double>> A_permuted(curr_size, vector<double>(curr_size, 0.0));
    for (int i = 0; i < curr_size; ++i) {
            A_permuted[pi_global[i]] = luprod[i];
    }
    // print a_original and a+permuted
    for (int i = 0; i < curr_size; ++i) {
        for (int j = 0; j < curr_size; ++j) {    std::cout << A_original[i][j] << " ";}
        std::cout << " | ";
        for (int j = 0; j < curr_size; ++j) {    std::cout << A_permuted[i][j] << " ";}
        std::cout << std::endl;
    }
    vector<vector<double>> diff(curr_size, vector<double>(curr_size, 0.0));
    for(int i=0;i<curr_size;i++){
        for(int j=0;j<curr_size;j++)
            diff[i][j] = A_permuted[i][j]-A_original[i][j];
    }
    double l21Norm = computeL21Norm(diff);
    std::cout << "L2,1 Norm: " << l21Norm << std::endl;

}
// g++ serial.cpp  -std=c++11 