#include <iostream>
#include <string>
#include <cstdlib>


using namespace std;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        return 1;
    }
    string solve_algo= argv[1];
    // cout<<"maze solving algo=="<<solve_algo<<endl;

    if (solve_algo == "dijkstra") {
        system("mpirun -np 4 ./dijkstra.out maze.txt output.txt");
    } else if (solve_algo== "dfs") {
        system("mpirun -np 4 ./dfs.out maze.txt output.txt");
    } else {
        cerr << "Invalid maze gen algo given" << endl;
        return 1;
    }
       
    return 0;
}
