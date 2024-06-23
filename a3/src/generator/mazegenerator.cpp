#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>

#include <numeric>

using namespace std;

// Method for checking boundaries
bool isSafe(int i, int j, int N)
{
    if (i >= 0 && i < N && j >= 0 && j < N)
        return true;
    return false;
}
 
// Returns true if there is a
// path from a source (a
// cell with value 1) to a
// destination (a cell with
// value 2)
bool isaPath(vector<vector<int>>& matrix, int i, int j,
            vector<vector<bool>>& visited)
{
    // Checking the boundaries, walls and
    // whether the cell is unvisited
    int N = matrix.size();
    if (isSafe(i, j, (int)matrix.size()) && matrix[i][j] != 0
        && !visited[i][j]) {
        // Make the cell visited
        visited[i][j] = true;
 
        // if the cell is the required
        // destination then return true
        if (i==63 && j==0)
            return true;
 
        // traverse up
        bool up = isaPath(matrix, i - 1, j, visited);
 
        // if path is found in up
        // direction return true
        if (up)
            return true;
 
        // traverse left
        bool left = isaPath(matrix, i, j - 1, visited);
 
        // if path is found in left
        // direction return true
        if (left)
            return true;
 
        // traverse down
        bool down = isaPath(matrix, i + 1, j, visited);
 
        // if path is found in down
        // direction return true
        if (down)
            return true;
 
        // traverse right
        bool right = isaPath(matrix, i, j + 1, visited);
 
        // if path is found in right
        // direction return true
        if (right)
            return true;
    }
 
    // no path has been found
    return false;
}
 
// Method for finding and printing
// whether the path exists or not
bool isPath(vector<vector<int>>& matrix)
{
    int N = matrix.size();
    // Defining visited array to keep
    // track of already visited indexes
    // bool visited[N][N];
    // memset(visited, 0, sizeof(visited));
    vector<vector<bool>>visited(64, vector<bool>(64, 0));
    // Flag to indicate whether the
    // path exists or not
    bool flag = false;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // if matrix[i][j] is source
            // and it is not visited
            if (i==0 && j==63 && !visited[i][j])
 
                // Starting from i, j and
                // then finding the path
                if (isaPath(matrix, i, j, visited)) {
 
                    // if path exists
                    flag = true;
                    break;
                }
        }
    }
    // if (flag){
    //     cout << "YES";
    //     return true;}
    // else{
    //     cout << "NO";
    //     return false;}
    return false;

}
 

int main(int argc, char* argv[]) {

    if (argc != 2) {
        return 1;
    }
    string gen_algo= argv[1];
    // cout<<"maze generation algo=="<<gen_algo<<endl;
    if (gen_algo == "bfs") {
        system("mpirun -np 4 ./bfs.out maze.txt");
        // // read maze.txt
        // ifstream infile;
        // infile.open("maze.txt");
        // if (!infile.is_open()) {
        //     cerr << "Error: Unable to open input file." << endl;
        //     return 1;
        // }
        // string line;
        // vector<vector<int>> maze(64, vector<int>(64, 5));

        // int x=0, y=0;
        // while (getline(infile, line)) {

        //     for (char c : line) {

        //         if(c=='S' || c=='E' || c==' '){
        //             maze[x][y] = 1;
        //         }
        //         else{
        //             maze[x][y]= -1;
        //         }

        //         y++;
        //     }
            
        //     x++;
        //     y=0;
        // }

        // // Close the input file
        // infile.close();

        // while(!isPath(maze)){
        //     // cout<<"NOT PERFECT"<<endl;
        //     system("mpirun -np 4 ./bfs.out maze.txt");
        //     // read maze.txt
        //     ifstream infile1;
        //     infile1.open("maze.txt");
        //     if (!infile1.is_open()) {
        //         cerr << "Error: Unable to open input file." << endl;
        //         return 1;
        //     }
        //     maze.resize(64, vector<int> (64, 0));
        //     string line1;

        //     int x=0, y=0;
        //     while (getline(infile1, line1)) {

        //         for (char c : line1) {

        //             if(c=='S' || c=='E' || c==' '){
        //                 maze[x][y] = 1;
        //             }
        //             else{
        //                 maze[x][y]= -1;
        //             }

        //             y++;
        //         }
                
        //         x++;
        //         y=0;
        //     }
        //     infile1.close();

        // }



    } else if (gen_algo== "kruskal") {
        system("mpirun -np 4 ./kruskal.out maze.txt");
    } else {
        cerr << "Invalid maze gen algo given" << endl;
        // cout<<"generation from mazegenerator not done"<<endl;

        return 1;
    }

    // cout<<"generation from mazegenerator"<<endl;
       
    return 0;
}
