#include <iostream>
#include <vector>
#include <cstdlib> // for rand() and srand()
#include <stdlib.h>
#include <ctime>   // for time()
#include <pthread.h>
#include <stdint.h>
using namespace std;
#define MAX_THREADS 3
#define SIZE 3
// pthread_barrier_t barrier;

struct info{
    // int* thread_id; // Allocate memory for the thread ID
    // int val;
    vector<int>* pi;
    vector<vector<double> >* a;
    vector<vector<double> >* l;
    vector<vector<double> >* u;
};
struct threaddata{
    // int thread_id;
    vector<vector<pair<info,pair<int,int> > > > data;
};

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
// */
// void* thread_start(void* arg) {
//     struct info* thread_info = (struct info*)arg; // Cast arg to struct info pointer
//     int thread_id = *(thread_info->thread_id); // Get the thread ID
//     // vector<int> pi = *(thread_info->pi); // Get the thread ID
//     printf("%d thread_start\n",thread_id);

//     // int thread_id = *((int*)arg); // Get the thread ID
//     // int chunk_size =SIZE / MAX_THREADS; // Calculate the chunkSIZE for each thread
//     // int start_index = thread_id * chunk_size; // Calculate the start index for this thread
//     // int end_index = (thread_id == MAX_THREADS - 1) ?SIZE : (start_index + chunk_size); // Calculate the end index
//     // vector<int> pi(SIZE);
//     // Assign values to the array elements within the thread's range
//     for (int i = 0; i < (*(thread_info->pi)).size(); ++i) {
//         //initialize π as a vector of length n
//         (*(thread_info->pi))[i] = i;
//     }

//     printf("%d thread_end\n",thread_id);

//     pthread_exit(NULL); // Exit the thread
    
// }

void* max_finder(void* arg) {
        if (!arg) {
        cerr << "Error: Null argument\n";
        pthread_exit(NULL);
    }
    // vector<pair<info,pair<int,int> > >* td = reinterpret_cast<vector<pair<info,pair<int,int> > >* > (arg);
    pair<info, pair<int, int> >* td = reinterpret_cast<pair<info, pair<int, int> >* >(arg);

    // pair<info,pair<int,int> >* td = (pair<info,pair<int,int> >*)arg;
    //     if (td->empty()) {
    //     cerr << "Error: Empty vector\n";
    //     pthread_exit(NULL);
    // }

    // printf("k=%d\n",pthread_self());
    // vector<pair<info,pair<int,int> > > data=*(td);
    // printf("2k=%d\n",pthread_self());
    int k = (*td).second.second;
    vector<vector<double> > a=*((*td).first.a);
    // printf(" ......................################÷##### %d thread_start #\n",k);
    //print a of each thread
    // for (int i = 0; i < SIZE; ++i) {
    //     for (int j = 0; j < SIZE; ++j) {
    //         cout << a[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // pr
    // printf(" ##################### %d thread_start #\n",pthread_self());    
    // if (SIZE-k<){
    //     pthread_exit(NULL); // Exit the thread
    // }
// printf("k=%d\n",pthread_self());
    // struct info* thread_info = (struct info*)arg; // Cast arg to struct info pointer
    // int thread_id = static_cast<int>(reinterpret_cast<uintptr_t>(pthread_self())); // Get the thread ID
    int thread_id = (*td).second.first;
    // printf("******************* %d %d %d %d\n",k,thread_id,pthread_self(),(*td).second.first);
    // printf(" # %d thread_start #\n",thread_id);
    // int thread_id=(int)(pthread_self());
    // int k = ...;
    
     // Calculate the chunkSIZE for each thread
    int chunk_size = (SIZE-k) / MAX_THREADS; // Calculate the chunkSIZE for each thread
    int start_index = (thread_id * chunk_size)+k ; // Calculate the start index for this thread
    int end_index = (thread_id == MAX_THREADS - 1) ? SIZE : (start_index + chunk_size); // Calculate the end index
    if (SIZE-k<MAX_THREADS){
        start_index = k ; // Calculate the start index for this thread
        end_index = SIZE; // Calculate the end index
    }
    printf("^^^^^^^^^^^^^^^^^^^^^^^^ %d %d %d %d\n",start_index,end_index,thread_id,pthread_self());
    // int k= thread_info->val;
    // Assign values to the array elements within the thread's range
    int k_prime=-1;
    double max_val = -1e9;
    // for (int i = k; i < SIZE; ++i) {
        for(int j=start_index;j<end_index; j++){
            if (abs(a[j][k]) > max_val) {
                max_val = abs(a[j][k]);
                k_prime = j;
            }
        // }
    }
    printf("$$$$$$$$$$$$$$$$$$$$$$ %d %d %d %d\n",k,k_prime,thread_id,pthread_self());
    pthread_exit(NULL); // Exit the thread
    return (void*) k_prime;
}



vector<vector<double> > lu_decomposition(vector<vector<double> > a, vector<int>& pi){
    // cout<<"87";
    pthread_t threads[MAX_THREADS]; // Declare an array of thread IDs
    // struct info args;
    struct info thread_info;
    thread_info.pi = &pi;
    thread_info.a = &a;
    int n = a.size();
    printf("%d\n",n);
    vector<vector<double> > l(n, vector<double>(n, 0.0));
    vector<vector<double> > u(n, vector<double>(n, 0.0));
    for(int i = 0; i < n; i++){
        l[i][i] = 1;
    }
    thread_info.l = &l;
    thread_info.u = &u;
    cout<<"97";
    // Create threads
    // for (int i = 0; i < MAX_THREADS; ++i) {
    //     // thread_info[i].thread_id = (int*)malloc(sizeof(int)); // Allocate memory for the thread ID
    //     // *thread_info[i].thread_id = i; // Assign the thread ID
    //     // thread_info[i].pi = &pi;
    //     // thread_info[i].a = &a;
    //     pthread_create(&threads[i], NULL, thread_start, &thread_info[i]); // Create the thread
    // }
    // //print pi of each thread
    // //print pi of each thread
    
    // // Join threads
    // cout<<"105";
    // for (int i = 0; i < MAX_THREADS; ++i) {
    //     pthread_join(threads[i], NULL); // Wait for the thread to finish
    // }

    // for (int i = 0; i < MAX_THREADS; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << (*thread_info[i].pi)[j] << " m";
    //     }
    //     cout << endl;
    // }
//CLEARRRR!!!

    for (int k = 0; k < n; ++k) {
        double max_val = 0;
        int k_prime = 0;
        vector<vector<pair<info,pair<int,int> > > > data(MAX_THREADS);
        
            vector<pair<info,pair<int,int> > > datadummy(MAX_THREADS);
        for (int i = 0; i < MAX_THREADS; ++i) {

            datadummy[i]=(make_pair(thread_info,make_pair(i,k)));
            data[i]=datadummy;
            //print datadummy[i]   
                cout << "klklkl"<<data[i][i].second.first << "klklkl ";
        
            cout << endl;

            for (int j = 0; j < MAX_THREADS; ++j) {  
                cout << data[i][j].second.first << " ";
            }
            cout << endl;

            pthread_create(&threads[i], NULL, max_finder, &data[i][i]); // Create the thread
            printf("thread %d created\n",i);
        }

    // pthread_barrier_init(&barrier, NULL, MAX_THREADS);
        cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%130";
        for (int i = 0; i < MAX_THREADS; ++i) {
            void *thread_return_value;
            pthread_join(threads[i], &thread_return_value);
            int local_max_index = *((int*)thread_return_value);
            if (local_max_index != -1 && fabs(a[local_max_index][k]) > fabs(a[k_prime][k])) {
                k_prime = local_max_index;
            }
        }
        // print
        max_val= a[k_prime][k];
        cout<<max_val;

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
    // calculate L*U
    vector<vector<double> > lu(n, vector<double>(n, 0.0));
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
    vector<vector<double> > adummy;
    vector<int> pidummy;
    adummy.resize(SIZE, vector<double>(SIZE, 0.0));
    pidummy.resize(SIZE);
    for (int i = 0; i <SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            adummy[i][j] = rand() % 10 + 1; // Generates a random integer between 1 and 10
        }
    }
    cout<<"original matrix a: \n";
    for (int i = 0; i <SIZE; ++i) {
        for (int j = 0; j <SIZE; ++j) {
            cout << adummy[i][j] << " ";
        }
        cout << endl;
    }
    vector<vector<double> > L, U;
    vector<vector<double> > luprod=lu_decomposition(adummy, pidummy);
    
    // Output results
    cout << "π:";
    for (int i = 0; i < pidummy.size(); ++i) cout << " " << pidummy[i];
    cout << endl;
    cout << "a after applying permutation pi:" << endl;
    vector<vector<double> > A_permuted(adummy.size(), vector<double>(adummy[0].size(), 0.0));
    for (int i = 0; i < pidummy.size(); ++i) {
            A_permuted[pidummy[i]] = luprod[i];
    }
    // print A_permuted
    for(int i = 0; i < adummy.size(); i++){
        for(int j = 0; j < adummy.size(); j++){
            cout << A_permuted[i][j] << " ";
        }
        cout << endl;
    }
    // calculate difference matrix between A_permuted and a
    vector<vector<double> > diff(adummy.size(), vector<double>(adummy[0].size(), 0.0));
    for(int i = 0; i < adummy.size(); i++){
        for(int j = 0; j < adummy.size(); j++){
            diff[i][j] = A_permuted[i][j] - adummy[i][j];
        }
    }
    // calculate L2,1  matrix induced norm of diff
    double l2norm = 0;
    for(int i = 0; i < adummy.size(); i++){
        double sum = 0;
        for(int j = 0; j < adummy.size(); j++){
            sum += diff[i][j] * diff[i][j];
        }
        l2norm += sqrt(sum);
    }
    cout << "L2,1 norm of difference matrix: " << l2norm << endl;

}
// g++ pthread.cpp  -std=c++11 