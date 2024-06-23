#include <iostream>
#include <string>
#include <cstdlib>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]) {

    int my_rank, p;
    MPI_Comm comm;
    MPI_Datatype blk_col_mpi_t;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &p);
    if(my_rank ==0){
        if (argc != 5) {
            cerr << "Usage: mpirun -np 4 ./maze.out -g [bfs/kruskal] -s [dfs/dijkstra]" << endl;
            // MPI_Finalize();
            return 1;
        }

        string g_flag= argv[1];
        // cout<<"algo graph=="<<g_flag<<endl;
        string s_flag = argv[3];
        // cout<<"algo search =="<<s_flag<<endl;

        // Check the graph algorithm
        if (g_flag== "-g" && s_flag == "-s") {
            string gen = argv[2];

            string solve = argv[4];

            

            // Run the appropriate algorithm based on the tags
            string command = "./mazegen.out " + gen;
            system(command.c_str());
            MPI_Finalize();

            // cout<<"generation done"<<endl;


            string command2 = "./mazesolve.out " + solve;
            system(command2.c_str());
            // cout<<"solving done"<<endl;

            
        } else {
            cerr << "Invalid syntax. Please provide graph algorithm." << endl;
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();

    return 0;
}
